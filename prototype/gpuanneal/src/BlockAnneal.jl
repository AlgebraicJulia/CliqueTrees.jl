module BlockAnneal

# v2: one annealing chain per thread block, many simultaneous proposals per chain.
#
# Each round, every thread of the block proposes one tree rotation at a random internal node P
# (P = (X, Y), Y = (B, C)  ->  P = (B, Y'), Y' = (X, C); see GPUAnneal.jl for the algebra). A rotation
# reads the legs of P, X, Y, B, C and writes only L(Y), CH(P), CH(Y). Threads claim the internal nodes
# among {P, Y, X, B, C} with atomic compare-and-swap on a per-node round stamp (P in a first phase,
# behind a barrier, the rest in a second phase); a proposal whose claims fail is dropped. Proposals that
# hold their claims touch pairwise disjoint node sets and none reads what another writes, so they commute:
# applying their Metropolis decisions simultaneously equals applying them one after another
# (checkerboard-style parallel Metropolis). Acceptance uses
#     dE = (1-mix) * log2((C + dC)/C) + mix * [lse2(new local costs) - lse2(old local costs)] + lam * dpen
# with C the total cost at the start of the round.
# Every `evalevery` rounds the block re-evaluates the chain exactly (block-parallel), records the best
# feasible tree, and tries `nslice` slice moves (add / remove / swap one sliced index, exact evaluation).
#
# Leg sets live in shared memory when they fit (device attribute), otherwise in global memory.
# Nothing is specific to a GPU model: block size, chain count and the shared/global choice are computed
# from the instance size and device attributes.

using CUDA
using Random
using Printf

include("hostio.jl")

@inline function xs(s::UInt64)
    s ⊻= s >> 12; s ⊻= s << 25; s ⊻= s >> 27
    s, s * 0x2545F4914F6CDD1D
end
@inline u01(x::UInt64) = Float32(x >> 40) * Float32(5.9604645f-8)
@inline rint(x::UInt64, n::Integer) = Int32((UInt64(x >> 32) * UInt64(n)) >> 32)

@inline function lse2(a::Float32, b::Float32)
    m = max(a, b)
    m + log2(exp2(a - m) + exp2(b - m))
end

@inline function claim!(OWN, v, stamp::Int32)
    o = @inbounds OWN[v]
    o == stamp && return false
    p = pointer(OWN, v)
    CUDA.atomic_cas!(p, o, stamp) == o
end

@inline function wset(x::UInt64, w, CM, cw, K)
    s = 0.0f0
    for c in Int32(1):K
        @inbounds s += cw[c] * Float32(count_ones(x & CM[w, c]))
    end
    s
end

# block reductions: sum (Float32), max (Float32), sum (Float64) through shared scratch
@inline function bsum(x::T, RED, tid, nth) where {T}
    @inbounds RED[tid] = x
    sync_threads()
    s = nth >> Int32(1)
    while s > Int32(0)
        if tid <= s
            @inbounds RED[tid] = RED[tid] + RED[tid+s]
        end
        sync_threads()
        s >>= Int32(1)
    end
    r = @inbounds RED[1]
    sync_threads()
    r
end
@inline function bmax(x::T, RED, tid, nth) where {T}
    @inbounds RED[tid] = x
    sync_threads()
    s = nth >> Int32(1)
    while s > Int32(0)
        if tid <= s
            @inbounds RED[tid] = max(RED[tid], RED[tid+s])
        end
        sync_threads()
        s >>= Int32(1)
    end
    r = @inbounds RED[1]
    sync_threads()
    r
end

@inline function bmin(x::T, RED, tid, nth) where {T}
    @inbounds RED[tid] = x
    sync_threads()
    s = nth >> Int32(1)
    while s > Int32(0)
        if tid <= s
            @inbounds RED[tid] = min(RED[tid], RED[tid+s])
        end
        sync_threads()
        s >>= Int32(1)
    end
    r = @inbounds RED[1]
    sync_threads()
    r
end

# TreeSASlicer's leg scores, block-parallel: for every index, the number of maximum-size intermediates
# (size > mw - 0.99) whose legs contain it. Returns (best unsliced index to add: most frequent; best sliced
# index to remove: least frequent), 0 if none; ties broken at random.
@inline function slicescores(L, SM, CM, cw, K, W, nt, nn, mw, tid, nth, CNT, g, RU64, rs)
    i = tid
    while i <= W * Int32(64)
        @inbounds CNT[i, g] = Int32(0)
        i += nth
    end
    sync_threads()
    v = nt + tid
    while v <= nn
        sz = 0.0f0
        for w in Int32(1):W
            @inbounds sz += wset(L[w, v] & ~SM[w], w, CM, cw, K)
        end
        if sz > mw - 0.99f0
            for w in Int32(1):W
                @inbounds x = L[w, v]
                while x != UInt64(0)
                    b = Int32(trailing_zeros(x))
                    CUDA.atomic_add!(pointer(CNT, (g - Int32(1)) * W * Int32(64) + (w - Int32(1)) * Int32(64) + b + Int32(1)), Int32(1))
                    x &= x - UInt64(1)
                end
            end
        end
        v += nth
    end
    sync_threads()
    ka = UInt64(0); kr = typemax(UInt64)
    i = tid
    while i <= W * Int32(64)
        @inbounds c = CNT[i, g]
        @inbounds insl = ((SM[(i-Int32(1))>>6+Int32(1)] >> ((i - Int32(1)) & Int32(63))) & UInt64(1)) == UInt64(1)
        rs, r = xs(rs)
        key = (UInt64(c) << 40) | (((r >> 48) & UInt64(0xffff)) << 24) | UInt64(i)
        if insl
            kr = min(kr, key)
        elseif c > Int32(0)
            ka = max(ka, key)
        end
        i += nth
    end
    ka = bmax(ka, RU64, tid, nth)
    kr = bmin(kr, RU64, tid, nth)
    ia = Int32(ka & UInt64(0xffffff))
    ir = kr == typemax(UInt64) ? Int32(0) : Int32(kr & UInt64(0xffffff))
    ia, ir, rs
end

# exact evaluation with slice mask SM (shared, W words): returns (mx, s, exc, ws) on every thread
@inline function fulleval(L, CH, SM, CM, cw, K, W, nt, nn, Wt, tid, nth, RF, RD)
    # pass 1: max node cost
    mx = -Inf32
    v = nt + tid
    while v <= nn
        @inbounds a = CH[1, v]; @inbounds b = CH[2, v]
        wi = 0.0f0
        for w in Int32(1):W
            @inbounds wi += wset((L[w, a] | L[w, b]) & ~SM[w], w, CM, cw, K)
        end
        mx = max(mx, wi)
        v += nth
    end
    mx = bmax(mx, RF, tid, nth)
    s = 0.0; exc = 0.0f0; mw = 0.0f0
    v = nt + tid
    while v <= nn
        @inbounds a = CH[1, v]; @inbounds b = CH[2, v]
        wi = 0.0f0; wl = 0.0f0
        for w in Int32(1):W
            @inbounds sm = ~SM[w]
            @inbounds wi += wset((L[w, a] | L[w, b]) & sm, w, CM, cw, K)
            @inbounds wl += wset(L[w, v] & sm, w, CM, cw, K)
        end
        s += Float64(exp2(wi - mx))
        exc += max(0.0f0, wl - Wt)
        mw = max(mw, wl)
        v += nth
    end
    s = bsum(s, RD, tid, nth)
    exc = bsum(exc, RF, tid, nth)
    mw = bmax(mw, RF, tid, nth)
    ws = 0.0f0
    if tid == Int32(1)
        for w in Int32(1):W
            @inbounds ws += wset(SM[w], w, CM, cw, K)
        end
    end
    ws = bsum(ws, RF, tid, nth)
    mx, s, exc, ws, mw
end

# uniformly random set bit of the bitset given by getword (sequential, one thread); 0 if empty
@inline function randbit1(getword, W, r::UInt64)
    tot = Int32(0)
    for w in Int32(1):W
        tot += Int32(count_ones(getword(w)))
    end
    tot == Int32(0) && return Int32(0)
    k = rint(r, tot)
    for w in Int32(1):W
        x = getword(w); c = Int32(count_ones(x))
        if k < c
            for _ in 1:k
                x &= x - UInt64(1)
            end
            return (w - Int32(1)) * Int32(64) + Int32(trailing_zeros(x)) + Int32(1)
        end
        k -= c
    end
    Int32(0)
end

# Exact subtree reconfiguration (block-parallel): grow a random subtree from root R to nf <= kmax frontier
# pieces, find the cheapest binary tree over them by subset DP (each DP level spread over the threads),
# and install it if it is strictly cheaper. Costs are the exact local terms 2^{w(inv \ S) - ref}; with a
# finite width target, intermediates wider than Wt are forbidden. Returns thread 1's RNG state.
@inline function reconf!(L, CH, SM, CM, cw, K, W, nt, nn, Wt, tid, nth, RU, RL, g, DPB, DPC, FR, rs, ref, kmax)
    ni = nn - nt
    if tid == Int32(1)
        rs, r = xs(rs)
        R = nt + Int32(1) + rint(r, ni)
        if (r >> 20) & 1 == 1          # half of the time: the most expensive of 4 random nodes
            bsz = -1.0f0
            for _ in 1:4
                rs, r = xs(rs)
                v = nt + Int32(1) + rint(r, ni)
                @inbounds a = CH[1, v]; @inbounds b = CH[2, v]
                sz = 0.0f0
                for w in Int32(1):W
                    @inbounds sz += wset((L[w, a] | L[w, b]) & ~SM[w], w, CM, cw, K)
                end
                if sz > bsz
                    bsz = sz; R = v
                end
            end
        end
        nf = Int32(1); @inbounds FR[3] = R; nI = Int32(0)
        while nf < kmax
            ci = Int32(0)
            for i in Int32(1):nf
                @inbounds ci += FR[2+i] > nt ? Int32(1) : Int32(0)
            end
            ci == Int32(0) && break
            rs, r = xs(rs)
            k = rint(r, ci); j = Int32(0)
            for i in Int32(1):nf
                @inbounds v = FR[2+i]
                if v > nt
                    if j == k
                        nI += Int32(1); @inbounds FR[10+nI] = v
                        @inbounds FR[2+i] = CH[1, v]; nf += Int32(1); @inbounds FR[2+nf] = CH[2, v]
                        break
                    end
                    j += Int32(1)
                end
            end
        end
        @inbounds FR[1] = nf; @inbounds FR[2] = R
    end
    sync_threads()
    @inbounds nf = FR[1]; @inbounds R = FR[2]
    nf < Int32(3) && return rs
    full = (Int32(1) << nf) - Int32(1)
    m = tid
    while m <= full
        for w in Int32(1):W
            u = UInt64(0)
            for i in Int32(0):nf-Int32(1)
                ((m >> i) & Int32(1)) == Int32(1) && (@inbounds u |= L[w, FR[3+i]])
            end
            @inbounds RU[w, m, g] = u
        end
        m += nth
    end
    sync_threads()
    m = tid
    while m <= full
        cm = full ⊻ m
        for w in Int32(1):W
            @inbounds o = (cm == Int32(0) ? UInt64(0) : RU[w, cm, g]) | L[w, R]
            @inbounds RL[w, m, g] = RU[w, m, g] & o
        end
        if count_ones(m) == 1
            @inbounds DPB[m] = 0.0; @inbounds DPC[m] = Int32(0)
        end
        m += nth
    end
    sync_threads()
    for p in Int32(2):nf
        m = tid
        while m <= full
            if count_ones(m) == p
                feas = true
                if isfinite(Wt) && m != full
                    sz = 0.0f0
                    for w in Int32(1):W
                        @inbounds sz += wset(RL[w, m, g] & ~SM[w], w, CM, cw, K)
                    end
                    feas = sz <= Wt + 1.0f-3
                end
                bestc = Inf; bestA = Int32(0)
                if feas
                    low = m & (-m)
                    a = (m - Int32(1)) & m
                    while a > Int32(0)
                        if (a & low) != Int32(0)
                            b = m ⊻ a
                            @inbounds cab = DPB[a] + DPB[b]
                            if cab < bestc
                                wi = 0.0f0
                                for w in Int32(1):W
                                    @inbounds wi += wset((RL[w, a, g] | RL[w, b, g]) & ~SM[w], w, CM, cw, K)
                                end
                                c = cab + Float64(exp2(wi - ref))
                                if c < bestc
                                    bestc = c; bestA = a
                                end
                            end
                        end
                        a = (a - Int32(1)) & m
                    end
                end
                @inbounds DPB[m] = bestc; @inbounds DPC[m] = bestA
            end
            m += nth
        end
        sync_threads()
    end
    if tid == Int32(1)
        old = 0.0
        for i in Int32(1):nf-Int32(1)
            @inbounds v = FR[10+i]; @inbounds a = CH[1, v]; @inbounds b = CH[2, v]
            wi = 0.0f0
            for w in Int32(1):W
                @inbounds wi += wset((L[w, a] | L[w, b]) & ~SM[w], w, CM, cw, K)
            end
            old += Float64(exp2(wi - ref))
        end
        @inbounds newc = DPB[full]
        if newc < old * (1.0 - 1.0e-6)
            pool = Int32(2); sp = Int32(1)
            @inbounds FR[19] = full; @inbounds FR[20] = R
            while sp > Int32(0)
                @inbounds mm = FR[17+2sp]; @inbounds v = FR[18+2sp]; sp -= Int32(1)
                @inbounds a = DPC[mm]; b = mm ⊻ a
                ida = Int32(0); idb = Int32(0)
                if count_ones(a) == 1
                    @inbounds ida = FR[3+trailing_zeros(a)]
                else
                    @inbounds ida = FR[10+pool]; pool += Int32(1)
                    sp += Int32(1); @inbounds FR[17+2sp] = a; @inbounds FR[18+2sp] = ida
                end
                if count_ones(b) == 1
                    @inbounds idb = FR[3+trailing_zeros(b)]
                else
                    @inbounds idb = FR[10+pool]; pool += Int32(1)
                    sp += Int32(1); @inbounds FR[17+2sp] = b; @inbounds FR[18+2sp] = idb
                end
                @inbounds CH[1, v] = ida; @inbounds CH[2, v] = idb
                for w in Int32(1):W
                    @inbounds L[w, v] = RL[w, mm, g]
                end
            end
        end
    end
    sync_threads()
    rs
end

function block_kernel!(Lg, CHg, SLg, STg, RNGg, BESTE, BESTCH, BESTS, NACC, NPROP, CMg, cwg, LBETA, RU, RL, CNT,
                       ::Val{SH}, K::Int32, W::Int32, nt::Int32, nn::Int32, nrounds::Int32,
                       beta0::Float32, lbstep::Float32, lam::Float32, Wt::Float32, mix::Float32,
                       evalevery::Int32, nslice::Int32, stamp0::Int32, kmax::Int32, nreconf::Int32, Wf::Float32, pheur::Float32) where {SH}
    tid = Int32(threadIdx().x); nth = Int32(blockDim().x); g = Int32(blockIdx().x)
    ni = nn - nt
    # ---- shared layout
    off = 0
    RD = CuDynamicSharedArray(Float64, nth, off); off += 8 * nth
    RF = CuDynamicSharedArray(Float32, nth, off); off += 4 * nth
    off = (off + 7) & ~7
    RU64 = CuDynamicSharedArray(UInt64, nth, off); off += 8 * nth
    MISC = CuDynamicSharedArray(Float64, 4, off); off += 32
    SM = CuDynamicSharedArray(UInt64, W, off); off += 8 * W
    SM2 = CuDynamicSharedArray(UInt64, W, off); off += 8 * W
    CM = CuDynamicSharedArray(UInt64, (W, K), off); off += 8 * W * K
    cw = CuDynamicSharedArray(Float32, K, off); off += 4 * K + 4 * ((4 - K % 4) % 4)
    off = (off + 7) & ~7
    CH = CuDynamicSharedArray(Int32, (2, nn), off); off += 8 * nn
    OWN = CuDynamicSharedArray(Int32, nn, off); off += 4 * nn
    off = (off + 7) & ~7
    DPB = CuDynamicSharedArray(Float64, 256, off); off += 8 * 256
    DPC = CuDynamicSharedArray(Int32, 256, off); off += 4 * 256
    FR = CuDynamicSharedArray(Int32, 64, off); off += 4 * 64
    off = (off + 7) & ~7
    L = SH ? CuDynamicSharedArray(UInt64, (W, nn), off) : view(Lg, :, :, g)
    # ---- load
    i = tid
    while i <= nn
        @inbounds CH[1, i] = CHg[1, i, g]; @inbounds CH[2, i] = CHg[2, i, g]; @inbounds OWN[i] = Int32(0)
        if SH
            for w in Int32(1):W
                @inbounds L[w, i] = Lg[w, i, g]
            end
        end
        i += nth
    end
    i = tid
    while i <= W
        @inbounds SM[i] = SLg[i, g]
        for c in Int32(1):K
            @inbounds CM[i, c] = CMg[i, c]
        end
        i += nth
    end
    tid <= K && (@inbounds cw[tid] = cwg[tid])
    @inbounds rs = RNGg[tid, g]
    sync_threads()
    mx, s, exc, ws, mw = fulleval(L, CH, SM, CM, cw, K, W, nt, nn, Wt, tid, nth, RF, RD)
    # chain scalars (identical on all threads): C relative to 2^ref
    C = s; ref = mx
    nacc = Int32(0); nprop = Int32(0)
    lbeta = @inbounds LBETA[g]
    for round in Int32(1):nrounds
        beta = exp2(lbeta); lbeta += lbstep
        stamp = stamp0 + round
        rs, r = xs(rs)
        P = nt + Int32(1) + rint(r, ni)
        ok = claim!(OWN, P, stamp)
        sync_threads()
        dC = 0.0; dpen = 0.0f0; acc = false
        if ok
            sd = Int32(r & 1); t = Int32((r >> 1) & 1)
            @inbounds Y = CH[1+sd, P]; @inbounds X = CH[2-sd, P]
            if Y <= nt
                X, Y = Y, X
            end
            ok = Y > nt && claim!(OWN, Y, stamp)
            if ok
                @inbounds B = CH[1+t, Y]; @inbounds Cc = CH[2-t, Y]
                ok = (X <= nt || claim!(OWN, X, stamp)) && (B <= nt || claim!(OWN, B, stamp)) &&
                     (Cc <= nt || claim!(OWN, Cc, stamp))
                if ok
                    nprop += Int32(1)
                    ao = 0.0f0; bo = 0.0f0; an = 0.0f0; bn = 0.0f0; sy = 0.0f0; syn = 0.0f0
                    for w in Int32(1):W
                        @inbounds sm = ~SM[w]
                        @inbounds lx = L[w, X]; @inbounds lb = L[w, B]; @inbounds lc = L[w, Cc]
                        @inbounds lp = L[w, P]; @inbounds ly = L[w, Y]
                        ny = (lx | lc) & (lb | lp)
                        for c in Int32(1):K
                            @inbounds m = CM[w, c] & sm
                            @inbounds wc = cw[c]
                            ao += wc * Float32(count_ones((lb | lc) & m))
                            bo += wc * Float32(count_ones((lx | ly) & m))
                            an += wc * Float32(count_ones((lx | lc) & m))
                            bn += wc * Float32(count_ones((lb | ny) & m))
                            sy += wc * Float32(count_ones(ly & m))
                            syn += wc * Float32(count_ones(ny & m))
                        end
                    end
                    dc = Float64(exp2(an - ref)) + Float64(exp2(bn - ref)) - Float64(exp2(ao - ref)) - Float64(exp2(bo - ref))
                    Cn = C + dc
                    dE = Cn > 0.0 ? log2(Float32(Cn / C)) : -Inf32
                    if mix > 0.0f0
                        dE = (1.0f0 - mix) * dE + mix * (lse2(an, bn) - lse2(ao, bo))
                    end
                    dp = max(0.0f0, syn - Wt) - max(0.0f0, sy - Wt)
                    dE += lam * dp
                    rs, r = xs(rs)
                    if dE <= 0.0f0 || u01(r) < exp2(-beta * dE)
                        for w in Int32(1):W
                            @inbounds lx = L[w, X]; @inbounds lb = L[w, B]; @inbounds lc = L[w, Cc]; @inbounds lp = L[w, P]
                            @inbounds L[w, Y] = (lx | lc) & (lb | lp)
                        end
                        @inbounds CH[1, P] = B; @inbounds CH[2, P] = Y
                        @inbounds CH[1, Y] = X; @inbounds CH[2, Y] = Cc
                        dC = dc; dpen = dp; acc = true
                        nacc += Int32(1)
                    end
                end
            end
        end
        # combine the accepted changes of this round
        dCt = bsum(dC, RD, tid, nth)
        dpt = bsum(dpen, RF, tid, nth)
        Cn = C + dCt
        doev = (round % evalevery == Int32(0)) || round == nrounds
        if doev && nreconf > Int32(0)
            for _ in Int32(1):nreconf
                rs = reconf!(L, CH, SM, CM, cw, K, W, nt, nn, Wt, tid, nth, RU, RL, g, DPB, DPC, FR, rs, ref, kmax)
            end
        end
        if doev || !(Cn > C * 0.004)
            # exact re-evaluation (also repairs cancellation), best tracking, slice moves
            mx, s, exc, ws, mw = fulleval(L, CH, SM, CM, cw, K, W, nt, nn, Wt, tid, nth, RF, RD)
            C = s; ref = mx
            if round % evalevery == Int32(0) || round == nrounds
                for _ in Int32(1):nslice
                    ia = Int32(0); ir = Int32(0)
                    if pheur > 0.0f0
                        ia, ir, rs = slicescores(L, SM, CM, cw, K, W, nt, nn, mw, tid, nth, CNT, g, RU64, rs)
                    end
                    # thread 1 proposes S' in SM2
                    if tid == Int32(1)
                        rs2 = rs
                        rs2, r2 = xs(rs2)
                        nS = Int32(0)
                        for w in Int32(1):W
                            @inbounds SM2[w] = SM[w]; @inbounds nS += Int32(count_ones(SM[w]))
                        end
                        op = nS == 0 ? Int32(0) : rint(r2, 3)
                        good = true
                        rs2, r3 = xs(rs2)
                        if u01(r3) < pheur && ia > Int32(0)
                            # TreeSASlicer moves: add the most frequent leg of the widest nodes while over
                            # the cap; at the cap: remove the least useful slice, or swap it for the best leg
                            if mw > Wt + 1.0f-3 || nS == 0 || ir == Int32(0)
                                op = Int32(-1)
                            else
                                op = (r3 & 1) == 1 ? Int32(-2) : Int32(-3)
                            end
                            if op == Int32(-1) || op == Int32(-3)
                                @inbounds SM2[(ia-Int32(1))>>6+Int32(1)] |= UInt64(1) << ((ia - Int32(1)) & Int32(63))
                            end
                            if op == Int32(-2) || op == Int32(-3)
                                @inbounds SM2[(ir-Int32(1))>>6+Int32(1)] &= ~(UInt64(1) << ((ir - Int32(1)) & Int32(63)))
                            end
                        end
                        if op == Int32(1) || op == Int32(2)
                            rs2, r2 = xs(rs2)
                            j = randbit1(ww -> (@inbounds SM[ww]), W, r2)
                            j > 0 && (@inbounds SM2[(j-Int32(1))>>6+Int32(1)] &= ~(UInt64(1) << ((j - Int32(1)) & Int32(63))))
                        end
                        if op == Int32(0) || op == Int32(2)
                            bv = Int32(0); bsz = -1.0f0
                            for _ in 1:8
                                rs2, r2 = xs(rs2)
                                v = nt + Int32(1) + rint(r2, ni - Int32(1))
                                sz = 0.0f0
                                for w in Int32(1):W
                                    @inbounds sz += wset(L[w, v] & ~SM2[w], w, CM, cw, K)
                                end
                                if sz > bsz
                                    bsz = sz; bv = v
                                end
                            end
                            rs2, r2 = xs(rs2)
                            vb = bv
                            j = randbit1(ww -> (@inbounds L[ww, vb] & ~SM2[ww]), W, r2)
                            if j > 0
                                @inbounds SM2[(j-Int32(1))>>6+Int32(1)] |= UInt64(1) << ((j - Int32(1)) & Int32(63))
                            else
                                good = false
                            end
                        end
                        rs2, r2 = xs(rs2)
                        @inbounds MISC[1] = good ? 1.0 : 0.0
                        @inbounds MISC[2] = Float64(u01(r2))
                        rs = rs2
                    end
                    sync_threads()
                    @inbounds good = MISC[1] > 0.5
                    @inbounds uu = Float32(MISC[2])
                    if good
                        mx2, s2, exc2, ws2, mw2 = fulleval(L, CH, SM2, CM, cw, K, W, nt, nn, Wt, tid, nth, RF, RD)
                        dE = (ws2 + mx2 + log2(Float32(s2)) + lam * exc2) - (ws + ref + log2(Float32(C)) + lam * exc)
                        if dE <= 0.0f0 || uu < exp2(-beta * dE)
                            if tid <= W
                                @inbounds SM[tid] = SM2[tid]
                            end
                            C = s2; ref = mx2; exc = exc2; ws = ws2; mw = mw2
                        end
                    end
                    sync_threads()
                end
                if mw <= Wf + 1.0f-3
                    E = ws + ref + log2(Float32(C))
                    if E < @inbounds BESTE[g]
                        sync_threads()
                        tid == Int32(1) && (@inbounds BESTE[g] = E)
                        v = nt + tid
                        while v <= nn
                            @inbounds BESTCH[1, v-nt, g] = CH[1, v]; @inbounds BESTCH[2, v-nt, g] = CH[2, v]
                            v += nth
                        end
                        tid <= W && (@inbounds BESTS[tid, g] = SM[tid])
                    end
                end
            end
        else
            C = Cn; exc += dpt
        end
        if C > 1.0e9 || C < 1.0e-9
            k = floor(log2(C)); C *= exp2(-k); ref += Float32(k)
        end
        sync_threads()
    end
    # ---- store
    i = tid
    while i <= nn
        @inbounds CHg[1, i, g] = CH[1, i]; @inbounds CHg[2, i, g] = CH[2, i]
        if SH
            for w in Int32(1):W
                @inbounds Lg[w, i, g] = L[w, i]
            end
        end
        i += nth
    end
    tid <= W && (@inbounds SLg[tid, g] = SM[tid])
    @inbounds RNGg[tid, g] = rs
    if tid == Int32(1)
        @inbounds STg[1, g] = ws + ref + log2(Float32(C)); @inbounds STg[2, g] = exc
    end
    CUDA.atomic_add!(pointer(NACC, g), Int64(nacc))
    CUDA.atomic_add!(pointer(NPROP, g), Int64(nprop))
    return nothing
end

mutable struct BAnnealer
    net::Net
    nch::Int
    nth::Int
    shared::Bool
    shmem::Int
    L::CuArray{UInt64,3}
    CH::CuArray{Int32,3}
    SL::CuMatrix{UInt64}
    ST::CuMatrix{Float32}
    RNG::CuMatrix{UInt64}
    BESTE::CuVector{Float32}
    BESTCH::CuArray{Int32,3}
    BESTS::CuMatrix{UInt64}
    NACC::CuVector{Int64}
    NPROP::CuVector{Int64}
    cmask::CuMatrix{UInt64}
    cw::CuVector{Float32}
    stamp::Int32
    LBETA::CuVector{Float32}
    RU::CuArray{UInt64,3}
    RL::CuArray{UInt64,3}
    CNT::CuMatrix{Int32}
end

function shmem_bytes(net::Net, nth, shared)
    nn = 2net.nt - 1; W = net.W; K = net.K
    b = 8nth + 4nth + 8 + 8nth + 32 + 16W + 8W * K + 4K + 16
    b = (b + 7) & ~7
    b += 8nn + 4nn
    b = (b + 7) & ~7
    b += 8 * 256 + 4 * 256 + 4 * 64
    b = (b + 7) & ~7
    shared && (b += 8W * nn)
    b
end

"threads per chain: about one proposal per 8 internal nodes, a multiple of 32 in [32, 1024]"
default_threads(net::Net) = clamp(32 * cld(cld(net.nt - 1, 8), 32), 32, 1024) |> x -> prevpow(2, x)

function BAnnealer(net::Net, seeds::Vector; nch::Int = 0, nth::Int = default_threads(net), seed::Integer = 1,
                   shared::Union{Bool,Nothing} = nothing, dev = CUDA.device())
    nt = net.nt; nn = 2nt - 1; W = net.W
    maxsh = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    smsh = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
    nsm = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    thsm = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    sh = shared === nothing ? shmem_bytes(net, nth, true) <= maxsh : shared
    shb = shmem_bytes(net, nth, sh)
    @assert shb <= maxsh "instance too large for shared memory even without leg sets ($shb > $maxsh bytes)"
    if nch == 0   # fill the device: resident blocks per SM from shared memory and threads
        per = max(1, min(smsh ÷ (shb + 1024), thsm ÷ nth, 32))
        nch = nsm * per
    end
    Lh = Array{UInt64,3}(undef, W, nn, nch); CHh = Array{Int32,3}(undef, 2, nn, nch); SLh = zeros(UInt64, W, nch)
    cache = Dict{Int,Any}()
    for c in 1:nch
        k = mod1(c, length(seeds)); CHs, sl = seeds[k]
        Lh[:, :, c] .= get!(() -> legs(net, CHs), cache, k); CHh[:, :, c] .= CHs; SLh[:, c] .= slicemask(net, sl)
    end
    rng = Random.Xoshiro(seed)
    RNGh = [rand(rng, UInt64) | 1 for _ in 1:nth, _ in 1:nch]
    BAnnealer(net, nch, nth, sh, shb, CuArray(Lh), CuArray(CHh), CuArray(SLh), CUDA.zeros(Float32, 2, nch), CuArray(RNGh),
        CUDA.fill(Inf32, nch), CuArray(CHh[:, nt+1:nn, :]), CuArray(SLh), CUDA.zeros(Int64, nch), CUDA.zeros(Int64, nch),
        CuArray(net.cmask), CuArray(Float32.(net.cw)), Int32(0), CUDA.zeros(Float32, nch), CUDA.zeros(UInt64, W, 256, nch), CUDA.zeros(UInt64, W, 256, nch), CUDA.zeros(Int32, 64W, nch))
end

function step!(A::BAnnealer, nrounds, beta0, beta1; lam = 0.0, Wt = Inf, mix = 1.0, evalevery = 64, nslice = 0, kmax = 8, nreconf = 0, Wf = Wt, pheur = 0.0)
    net = A.net
    if beta0 isa AbstractVector     # per-chain temperatures, held fixed during the launch (tempering)
        copyto!(A.LBETA, Float32.(log2.(beta0))); lb = 0.0f0
    else
        fill!(A.LBETA, Float32(log2(beta0)))
        lb = Float32((log2(beta1) - log2(beta0)) / max(nrounds, 1))
    end
    k = @cuda launch = false block_kernel!(A.L, A.CH, A.SL, A.ST, A.RNG, A.BESTE, A.BESTCH, A.BESTS, A.NACC, A.NPROP,
        A.cmask, A.cw, A.LBETA, A.RU, A.RL, A.CNT, Val(A.shared), Int32(net.K), Int32(net.W), Int32(net.nt), Int32(2net.nt - 1), Int32(nrounds),
        0.0f0, lb, Float32(lam), Float32(Wt), Float32(mix), Int32(evalevery), Int32(nslice), A.stamp, Int32(min(kmax, 8)), Int32(nreconf), Float32(Wf), Float32(pheur))
    A.shmem > 48 * 1024 && (CUDA.attributes(k.fun)[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = A.shmem)
    k(A.L, A.CH, A.SL, A.ST, A.RNG, A.BESTE, A.BESTCH, A.BESTS, A.NACC, A.NPROP,
        A.cmask, A.cw, A.LBETA, A.RU, A.RL, A.CNT, Val(A.shared), Int32(net.K), Int32(net.W), Int32(net.nt), Int32(2net.nt - 1), Int32(nrounds),
        0.0f0, lb, Float32(lam), Float32(Wt), Float32(mix), Int32(evalevery), Int32(nslice), A.stamp, Int32(min(kmax, 8)), Int32(nreconf), Float32(Wf), Float32(pheur);
        threads = A.nth, blocks = A.nch, shmem = A.shmem)
    A.stamp += Int32(nrounds)   # OWN is re-zeroed per launch; stamps only need to be > 0
    A.stamp = Int32(0)
    nothing
end

function best(A::BAnnealer)
    BE = Array(A.BESTE); g = argmin(BE)
    nt = A.net.nt; nn = 2nt - 1
    CH = zeros(Int32, 2, nn); CH[:, nt+1:nn] .= Array(A.BESTCH[:, :, g])
    CH, sliceids(A.net, Array(A.BESTS[:, g])), Float64(BE[g]), g
end

"""
    temper!(A; time, betamin, betamax, rounds, ...)

Parallel tempering: chain temperatures form a geometric ladder betamin..betamax (each temperature held by
nch / nladder chains); after every launch, adjacent rungs try to swap temperatures with probability
min(1, 2^((b_i - b_j)(E_i - E_j))), E = current exact log2 C_S + lam * excess of the chain.
"""
function temper!(A::BAnnealer; time = 10.0, betamin = 2.0, betamax = 1000.0, rounds = 500, lam = 0.0, Wt = Inf,
                 mix = 1.0, evalevery = 64, nslice = 0, kmax = 8, nreconf = 0, trace = nothing, rng = Random.Xoshiro(11),
                 lam1 = lam, Wt0 = Wt, wfrac = 0.6, pheur = 0.0, resample = 0.0, resevery = 10)
    n = A.nch
    ladder = [betamin * (betamax / betamin)^((k - 1) / max(n - 1, 1)) for k in 1:n]
    perm = collect(1:n)            # perm[k] = chain holding rung k
    beta = zeros(n)
    CUDA.synchronize(); t0 = Base.time(); ep = 0
    while Base.time() - t0 < time
        for k in 1:n; beta[perm[k]] = ladder[k]; end
        f = (Base.time() - t0) / time
        lm = lam > 0 ? lam * (lam1 / lam)^f : 0.0
        we = isfinite(Wt0) ? Wt + (Wt0 - Wt) * max(0.0, 1 - f / wfrac) : Wt
        step!(A, rounds, beta, nothing; lam = lm, Wt = we, mix, evalevery, nslice, kmax, nreconf, Wf = Wt, pheur)
        CUDA.synchronize(); ep += 1
        ST = Array(A.ST)
        E = [ST[1, c] + lm * ST[2, c] for c in 1:n]
        if resample > 0 && ep % resevery == 0
            # population step: the worst `resample` fraction of chains (current energy) become copies of
            # chains drawn from the best fraction; they keep their temperatures
            o = sortperm(E); k = floor(Int, resample * n)
            for (d, s0) in zip(o[end-k+1:end], o[rand(rng, 1:max(k, 1), k)])
                A.L[:, :, d] .= view(A.L, :, :, s0); A.CH[:, :, d] .= view(A.CH, :, :, s0)
                A.SL[:, d] .= view(A.SL, :, s0); E[d] = E[s0]
            end
        end
        for k in (isodd(ep) ? (1:2:n-1) : (2:2:n-1))
            i = perm[k]; j = perm[k+1]
            if rand(rng) < exp2(min(0.0, (ladder[k] - ladder[k+1]) * (E[i] - E[j])))
                perm[k], perm[k+1] = j, i
            end
        end
        trace === nothing || trace(Base.time() - t0, A)
    end
    ep
end

"""run!(A; time, beta0, beta1, rounds (per launch), ...): anneal beta0 -> beta1 geometrically in wall time."""
function run!(A::BAnnealer; time = 10.0, beta0 = 1.0, beta1 = 100.0, rounds = 2000, lam = 0.0, Wt = Inf, mix = 1.0,
              evalevery = 64, nslice = 0, kmax = 8, nreconf = 0, trace = nothing, lam1 = lam)
    CUDA.synchronize()
    t0 = Base.time(); tep = 0.0; ep = 0
    while true
        t = Base.time() - t0
        t >= time && break
        f0 = t / time; f1 = min(1.0, (t + max(tep, 1e-3)) / time)
        b0 = beta0 * (beta1 / beta0)^f0; b1 = beta0 * (beta1 / beta0)^f1
        te = Base.time()
        lm = lam > 0 ? lam * (lam1 / lam)^f0 : 0.0
        step!(A, rounds, b0, b1; lam = lm, Wt, mix, evalevery, nslice, kmax, nreconf)
        CUDA.synchronize()
        tep = Base.time() - te; ep += 1
        trace === nothing || trace(Base.time() - t0, A)
    end
    ep
end

end # module
