module GPUAnneal

# Massively parallel simulated annealing of binary contraction trees on the GPU.
#
# State of one chain: a full binary tree over the nt input tensors (leaves 1..nt, internal nodes
# nt+1..2nt-1, root 2nt-1 fixed) given by its child array CH, the leg set L(v) of every node as a
# bitset of index ids, and a slice set S (bitset). L(v) = indices of subtree(v) that also occur
# outside subtree(v) or in the output; L(root) = output.
#
# Move 1 (tree rotation, as in TreeSA): internal node P with children X and Y = (B, C), Y internal,
# becomes P = (B, Y'), Y' = (X, C). Only Y changes its leg set:
#     L(Y') = (L(X) | L(C)) & (L(B) | L(P))
# and only the contractions at Y and P change their cost:
#     inv(Y') = L(X) | L(C),  inv(P') = L(B) | L(Y')
# so the exact change of the global objective is O(words) bit operations.
# Move 2 (slice move): add / remove / swap one sliced index; a full O(nodes * words) re-evaluation.
#
# Objective (log2 units): E = log2 C_S + lambda * sum_v max(0, w(L(v) \ S) - Wt)
#     C_S = 2^{w(S)} * sum_{internal v} 2^{w(inv(v) \ S)}       (cotengra contraction_cost with S sliced)
# optionally mixed with TreeSA's local criterion (mix > 0). w(set) = sum of log2(dim) of its indices;
# weights are grouped in K classes of equal dimension, so w(set) = sum_c w_c popcount(set & mask_c).
#
# Parallel layout: one chain per group of G lanes (G = power of two <= 32 chosen from the number of
# 64-bit words, so a group touches consecutive words of a leg set: coalesced). All lanes of a group run
# the same RNG stream, so control flow is uniform within the group; group sums use shfl_xor.
# Nothing here depends on the particular GPU: chain count and block size come from device attributes.

using CUDA
using Random
using Printf

export Net, readnet, Tree, tree_from_ssa, ssa_from_children, hostcost, Annealer, run_anneal!, write_tree, read_tree

include("hostio.jl")

# ------------------------------------------------------------------- device --

@inline function xs!(s::UInt64)
    s ⊻= s >> 12; s ⊻= s << 25; s ⊻= s >> 27
    s, s * 0x2545F4914F6CDD1D
end
@inline u01(x::UInt64) = Float32(x >> 40) * Float32(5.9604645f-8)       # [0,1)
@inline rint(x::UInt64, n::Integer) = Int32((UInt64(x >> 32) * UInt64(n)) >> 32)   # [0,n)

@inline function gsum(x, G::Int32, mask::UInt32)
    o = G >> Int32(1)
    while o > Int32(0)
        x += shfl_xor_sync(mask, x, o)
        o >>= Int32(1)
    end
    x
end
@inline function gmax(x, G::Int32, mask::UInt32)
    o = G >> Int32(1)
    while o > Int32(0)
        x = max(x, shfl_xor_sync(mask, x, o))
        o >>= Int32(1)
    end
    x
end

@inline function wpop(x::UInt64, w::Int32, cmask, cw, K::Int32)
    s = 0.0f0
    for c in Int32(1):K
        @inbounds s += cw[c] * Float32(count_ones(x & cmask[w, c]))
    end
    s
end

@inline function lse2(a::Float32, b::Float32)
    m = max(a, b)
    m + log2(exp2(a - m) + exp2(b - m))
end

# full evaluation of chain g with slice mask from array SM[:, g]; returns (mx, s, exc, wS)
@inline function fulleval(L, CH, SL, slot::Int32, g, cmask, cw, K, W, nt, nn, Wt, lane, G, mask)
    mx = -Inf32; s = 0.0; exc = 0.0f0
    v = Int32(nt + 1)
    while v <= nn
        @inbounds a = CH[1, v, g]; @inbounds b = CH[2, v, g]
        wi = 0.0f0; wl = 0.0f0
        w = lane + Int32(1)
        while w <= W
            @inbounds sm = ~SL[w, slot, g]
            @inbounds la = L[w, a, g]; @inbounds lb = L[w, b, g]; @inbounds lv = L[w, v, g]
            wi += wpop((la | lb) & sm, w, cmask, cw, K)
            wl += wpop(lv & sm, w, cmask, cw, K)
            w += G
        end
        wi = gsum(wi, G, mask); wl = gsum(wl, G, mask)
        exc += max(0.0f0, wl - Wt)
        if wi > mx
            s = s * Float64(exp2(mx - wi)) + 1.0; mx = wi
        else
            s += Float64(exp2(wi - mx))
        end
        v += Int32(1)
    end
    ws = 0.0f0
    w = lane + Int32(1)
    while w <= W
        @inbounds ws += wpop(SL[w, slot, g], w, cmask, cw, K)
        w += G
    end
    ws = gsum(ws, G, mask)
    mx, s, exc, ws
end

# pick a uniformly random set bit of bitset X (via function getword(w)); returns index (1-based) or 0
@inline function randbit(getword, W, lane, G, mask, r::UInt64)
    cnt = Int32(0)
    w = lane + Int32(1)
    while w <= W
        cnt += Int32(count_ones(getword(w)))
        w += G
    end
    tot = gsum(cnt, G, mask)
    tot == 0 && return Int32(0)
    k = rint(r, tot)               # rank among set bits, ordered by (lane, word)
    # exclusive prefix over lanes
    pre = cnt
    o = Int32(1)
    while o < G
        y = shfl_up_sync(mask, pre, o)
        (lane >= o) && (pre += y)
        o <<= Int32(1)
    end
    pre -= cnt
    res = Int32(0)
    if k >= pre && k < pre + cnt
        kk = k - pre
        w = lane + Int32(1)
        while w <= W
            x = getword(w)
            c = Int32(count_ones(x))
            if kk < c
                for _ in 1:kk
                    x &= x - UInt64(1)
                end
                res = (w - Int32(1)) * Int32(64) + Int32(trailing_zeros(x)) + Int32(1)
                break
            end
            kk -= c
            w += G
        end
    end
    gmax(res, G, mask)
end

function anneal_kernel!(L, CH, SL, ST, CREL, RNG, BESTE, BESTCH, BESTS, NACC, cmask, cw,
                        K::Int32, W::Int32, nt::Int32, nn::Int32, G::Int32, nch::Int32,
                        nsteps::Int32, beta0::Float32, lbstep::Float32, lam::Float32, Wt::Float32,
                        pslice::Float32, mix::Float32, sweep::Int32)
    tid = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x - 1)
    g = tid ÷ G + Int32(1)
    lane = tid % G
    g > nch && return nothing
    wl_ = Int32((threadIdx().x - 1) % 32)
    base = wl_ - lane
    mask = G == Int32(32) ? 0xffffffff : ((UInt32(1) << G) - UInt32(1)) << base
    ni = nn - nt
    @inbounds rs = RNG[g]
    @inbounds C = CREL[g]                 # relative total (per-slice) cost, times 2^ref
    @inbounds ref = ST[1, g]
    @inbounds exc = ST[2, g]
    @inbounds ws = ST[3, g]
    nacc = Int32(0)
    lbeta = log2(beta0)
    for step in Int32(1):nsteps
        beta = exp2(lbeta); lbeta += lbstep
        rs, r = xs!(rs)
        if pslice > 0.0f0 && u01(r) < pslice
            # ---------------------------------------------------------- slice move
            rs, r = xs!(rs)
            nS = Int32(0)
            w = lane + Int32(1)
            while w <= W
                @inbounds nS += Int32(count_ones(SL[w, 1, g]))
                w += G
            end
            nS = gsum(nS, G, mask)
            op = nS == 0 ? Int32(0) : rint(r, 3)        # 0 add, 1 remove, 2 swap
            # tentative mask into SL[:, 2, g]
            w = lane + Int32(1)
            while w <= W
                @inbounds SL[w, 2, g] = SL[w, 1, g]
                w += G
            end
            ok = true
            if op != Int32(0)
                rs, r = xs!(rs)
                i = randbit(ww -> (@inbounds SL[ww, 1, g]), W, lane, G, mask, r)
                if i > 0 && ((i - Int32(1)) >> 6) % G == lane
                    @inbounds SL[(i-Int32(1))>>6+Int32(1), 2, g] &= ~(UInt64(1) << ((i - Int32(1)) & Int32(63)))
                end
            end
            if op != Int32(1)
                # tournament: the largest of 4 random internal nodes, then a random unsliced leg of it
                best = Int32(0); bsz = -1.0f0
                for _ in 1:4
                    rs, r = xs!(rs)
                    v = nt + Int32(1) + rint(r, ni - Int32(1))   # never the root (its legs = output)
                    sz = 0.0f0
                    w = lane + Int32(1)
                    while w <= W
                        @inbounds sz += wpop(L[w, v, g] & ~SL[w, 2, g], w, cmask, cw, K)
                        w += G
                    end
                    sz = gsum(sz, G, mask)
                    if sz > bsz
                        bsz = sz; best = v
                    end
                end
                rs, r = xs!(rs)
                vb = best
                j = randbit(ww -> (@inbounds L[ww, vb, g] & ~SL[ww, 2, g]), W, lane, G, mask, r)
                if j > 0
                    if ((j - Int32(1)) >> 6) % G == lane
                        @inbounds SL[(j-Int32(1))>>6+Int32(1), 2, g] |= UInt64(1) << ((j - Int32(1)) & Int32(63))
                    end
                else
                    ok = false
                end
            end
            sync_warp(mask)
            if ok
                mx, s, excn, wsn = fulleval(L, CH, SL, Int32(2), g, cmask, cw, K, W, nt, nn, Wt, lane, G, mask)
                Eo = ws + ref + log2(Float32(C)) + lam * exc
                En = wsn + mx + log2(Float32(s)) + lam * excn
                dE = En - Eo
                rs, r = xs!(rs)
                if dE <= 0.0f0 || u01(r) < exp2(-beta * dE)
                    w = lane + Int32(1)
                    while w <= W
                        @inbounds SL[w, 1, g] = SL[w, 2, g]
                        w += G
                    end
                    C = s; ref = mx; exc = excn; ws = wsn
                    nacc += Int32(1)
                end
            end
            sync_warp(mask)
        else
            # ------------------------------------------------------- tree rotation
            rs, r = xs!(rs)
            P = nt + Int32(1) + rint(r, ni)
            sd = Int32(r & 1); t = Int32((r >> 1) & 1)
            @inbounds Y = CH[1+sd, P, g]; @inbounds X = CH[2-sd, P, g]
            if Y <= nt
                X, Y = Y, X
            end
            if Y > nt
                @inbounds B = CH[1+t, Y, g]; @inbounds Cc = CH[2-t, Y, g]
                ao = 0.0f0; bo = 0.0f0; an = 0.0f0; bn = 0.0f0; sy = 0.0f0; syn = 0.0f0
                w = lane + Int32(1)
                while w <= W
                    @inbounds sm = ~SL[w, 1, g]
                    @inbounds lx = L[w, X, g]; @inbounds lb = L[w, B, g]; @inbounds lc = L[w, Cc, g]
                    @inbounds lp = L[w, P, g]; @inbounds ly = L[w, Y, g]
                    ny = (lx | lc) & (lb | lp)
                    for c in Int32(1):K
                        @inbounds m = cmask[w, c] & sm
                        @inbounds wc = cw[c]
                        ao += wc * Float32(count_ones((lb | lc) & m))
                        bo += wc * Float32(count_ones((lx | ly) & m))
                        an += wc * Float32(count_ones((lx | lc) & m))
                        bn += wc * Float32(count_ones((lb | ny) & m))
                        sy += wc * Float32(count_ones(ly & m))
                        syn += wc * Float32(count_ones(ny & m))
                    end
                    w += G
                end
                ao = gsum(ao, G, mask); bo = gsum(bo, G, mask); an = gsum(an, G, mask)
                bn = gsum(bn, G, mask); sy = gsum(sy, G, mask); syn = gsum(syn, G, mask)
                dC = Float64(exp2(an - ref)) + Float64(exp2(bn - ref)) - Float64(exp2(ao - ref)) - Float64(exp2(bo - ref))
                Cn = C + dC
                dE = Cn > 0.0 ? log2(Float32(Cn / C)) : -Inf32
                if mix > 0.0f0
                    dE = (1.0f0 - mix) * dE + mix * (lse2(an, bn) - lse2(ao, bo))
                end
                dpen = max(0.0f0, syn - Wt) - max(0.0f0, sy - Wt)
                dE += lam * dpen
                rs, r = xs!(rs)
                if dE <= 0.0f0 || u01(r) < exp2(-beta * dE)
                    w = lane + Int32(1)
                    while w <= W
                        @inbounds lx = L[w, X, g]; @inbounds lb = L[w, B, g]; @inbounds lc = L[w, Cc, g]
                        @inbounds lp = L[w, P, g]
                        @inbounds L[w, Y, g] = (lx | lc) & (lb | lp)
                        w += G
                    end
                    if lane == Int32(0)
                        @inbounds CH[1, P, g] = B; @inbounds CH[2, P, g] = Y
                        @inbounds CH[1, Y, g] = X; @inbounds CH[2, Y, g] = Cc
                    end
                    nacc += Int32(1)
                    sync_warp(mask)
                    if Cn < C * 0.004 || !(Cn > 0.0)
                        # a dominant term was removed: the running sum lost its precision -> exact re-evaluation
                        mx, s, excn, wsn = fulleval(L, CH, SL, Int32(1), g, cmask, cw, K, W, nt, nn, Wt, lane, G, mask)
                        C = s; ref = mx; exc = excn; ws = wsn
                    else
                        C = Cn; exc += dpen
                    end
                    if C > 1.0e9 || C < 1.0e-9
                        k = floor(log2(C))
                        C = C * exp2(-k); ref += Float32(k)
                    end
                end
                sync_warp(mask)
            end
        end
        # ------------------------------------------------------------ best tracking
        if (step % sweep == Int32(0) || step == nsteps) && exc < 1.0f-3
            E = ws + ref + log2(Float32(C))
            @inbounds if E < BESTE[g]
                BESTE[g] = E
                v = nt + Int32(1) + lane
                while v <= nn
                    @inbounds BESTCH[1, v-nt, g] = CH[1, v, g]; @inbounds BESTCH[2, v-nt, g] = CH[2, v, g]
                    v += G
                end
                w = lane + Int32(1)
                while w <= W
                    @inbounds BESTS[w, g] = SL[w, 1, g]
                    w += G
                end
            end
            sync_warp(mask)
        end
    end
    if lane == Int32(0)
        @inbounds RNG[g] = rs; @inbounds CREL[g] = C
        @inbounds ST[1, g] = ref; @inbounds ST[2, g] = exc; @inbounds ST[3, g] = ws
        @inbounds NACC[g] += nacc
    end
    return nothing
end

# exact (re)initialisation of C, ref, exc, ws of every chain from its tree and S
function eval_kernel!(L, CH, SL, ST, CREL, cmask, cw, K::Int32, W::Int32, nt::Int32, nn::Int32, G::Int32,
                      nch::Int32, Wt::Float32)
    tid = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x - 1)
    g = tid ÷ G + Int32(1); lane = tid % G
    g > nch && return nothing
    wl_ = Int32((threadIdx().x - 1) % 32); base = wl_ - lane
    mask = G == Int32(32) ? 0xffffffff : ((UInt32(1) << G) - UInt32(1)) << base
    mx, s, exc, ws = fulleval(L, CH, SL, Int32(1), g, cmask, cw, K, W, nt, nn, Wt, lane, G, mask)
    if lane == Int32(0)
        @inbounds CREL[g] = s; @inbounds ST[1, g] = mx; @inbounds ST[2, g] = exc; @inbounds ST[3, g] = ws
    end
    return nothing
end

function clone_kernel!(L, CH, SL, ST, CREL, dst, src, nd::Int32, W::Int32, nn::Int32)
    i = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    k = Int32(blockIdx().y)
    k > nd && return nothing
    @inbounds d = dst[k]; @inbounds s = src[k]
    v = i
    while v <= nn
        @inbounds CH[1, v, d] = CH[1, v, s]; @inbounds CH[2, v, d] = CH[2, v, s]
        for w in Int32(1):W
            @inbounds L[w, v, d] = L[w, v, s]
        end
        v += Int32(blockDim().x * gridDim().x)
    end
    if i <= W
        @inbounds SL[i, 1, d] = SL[i, 1, s]
    end
    if i == Int32(1)
        @inbounds CREL[d] = CREL[s]
        for j in 1:3
            @inbounds ST[j, d] = ST[j, s]
        end
    end
    return nothing
end

# ------------------------------------------------------------------- driver --

mutable struct Annealer
    net::Net
    nch::Int
    G::Int32
    L::CuArray{UInt64,3}
    CH::CuArray{Int32,3}
    SL::CuArray{UInt64,3}
    ST::CuMatrix{Float32}
    CREL::CuVector{Float64}
    RNG::CuVector{UInt64}
    BESTE::CuVector{Float32}
    BESTCH::CuArray{Int32,3}
    BESTS::CuMatrix{UInt64}
    NACC::CuVector{Int64}
    cmask::CuMatrix{UInt64}
    cw::CuVector{Float32}
end

groupsize(W) = Int32(min(32, nextpow(2, W)))

"default number of chains: enough groups to fill every SM at full occupancy (device attributes only)"
function default_chains(net::Net; dev = CUDA.device())
    sm = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    th = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    Int(sm * th ÷ groupsize(net.W))
end

function chain_bytes(net::Net)
    nn = 2net.nt - 1
    nn * net.W * 8 + nn * 2 * 4 + (net.nt - 1) * 2 * 4 + 3 * net.W * 8 + 64
end

"""Annealer(net, seeds; nch, seed): chain c starts from seed tree seeds[mod1(c, end)] =
(children (2,nn), sliced ids)."""
function Annealer(net::Net, seeds::Vector; nch::Int = default_chains(net), seed::Integer = 1)
    nt = net.nt; nn = 2nt - 1; W = net.W
    Lh = Array{UInt64,3}(undef, W, nn, nch)
    CHh = Array{Int32,3}(undef, 2, nn, nch)
    SLh = zeros(UInt64, W, 2, nch)
    cache = Dict{Int,Any}()
    for c in 1:nch
        k = mod1(c, length(seeds))
        CHs, sl = seeds[k]
        Ls = get!(() -> legs(net, CHs), cache, k)
        Lh[:, :, c] .= Ls; CHh[:, :, c] .= CHs
        SLh[:, 1, c] .= slicemask(net, sl)
    end
    rng = Random.Xoshiro(seed)
    RNGh = [rand(rng, UInt64) | 1 for _ in 1:nch]
    A = Annealer(net, nch, groupsize(W), CuArray(Lh), CuArray(CHh), CuArray(SLh), CUDA.zeros(Float32, 3, nch),
        CUDA.zeros(Float64, nch), CuArray(RNGh), CUDA.fill(Inf32, nch), CuArray(CHh[:, nt+1:nn, :]),
        CuArray(SLh[:, 1, :]), CUDA.zeros(Int64, nch), CuArray(net.cmask), CuArray(Float32.(net.cw)))
    A
end

launchcfg(A::Annealer) = (threads = 128, blocks = cld(A.nch * Int(A.G), 128))

function evaluate!(A::Annealer; Wt = Inf32)
    net = A.net; c = launchcfg(A)
    @cuda threads = c.threads blocks = c.blocks eval_kernel!(A.L, A.CH, A.SL, A.ST, A.CREL, A.cmask, A.cw,
        Int32(net.K), Int32(net.W), Int32(net.nt), Int32(2net.nt - 1), A.G, Int32(A.nch), Float32(Wt))
    nothing
end

function step!(A::Annealer, nsteps, beta0, beta1; lam = 0.0, Wt = Inf, pslice = 0.0, mix = 0.0, sweep = A.net.nt)
    net = A.net; c = launchcfg(A)
    lb = Float32((log2(beta1) - log2(beta0)) / max(nsteps, 1))
    @cuda threads = c.threads blocks = c.blocks anneal_kernel!(A.L, A.CH, A.SL, A.ST, A.CREL, A.RNG, A.BESTE,
        A.BESTCH, A.BESTS, A.NACC, A.cmask, A.cw, Int32(net.K), Int32(net.W), Int32(net.nt), Int32(2net.nt - 1),
        A.G, Int32(A.nch), Int32(nsteps), Float32(beta0), lb, Float32(lam), Float32(Wt), Float32(pslice),
        Float32(mix), Int32(sweep))
    nothing
end

function clone!(A::Annealer, dst::Vector{Int}, src::Vector{Int})
    isempty(dst) && return
    nn = 2A.net.nt - 1
    @cuda threads = 256 blocks = (cld(nn, 256), length(dst)) clone_kernel!(A.L, A.CH, A.SL, A.ST, A.CREL,
        CuArray(Int32.(dst)), CuArray(Int32.(src)), Int32(length(dst)), Int32(A.net.W), Int32(nn))
    nothing
end

"current energies (log2 C_S + lam*excess) and excess, on host"
function energies(A::Annealer, lam)
    ST = Array(A.ST); C = Array(A.CREL)
    E = [ST[3, g] + ST[1, g] + log2(C[g]) + lam * ST[2, g] for g in 1:A.nch]
    E, ST[2, :]
end

"best feasible tree over all chains (host children (2,nn), slice ids, GPU energy)"
function best(A::Annealer)
    BE = Array(A.BESTE)
    g = argmin(BE)
    nt = A.net.nt; nn = 2nt - 1
    CH = zeros(Int32, 2, nn)
    CH[:, nt+1:nn] .= Array(A.BESTCH[:, :, g])
    S = Array(A.BESTS[:, g])
    CH, sliceids(A.net, S), Float64(BE[g]), g
end

"""
    run_anneal!(A; time, beta0, beta1, epoch, lam, Wt, pslice, mix, resample, trace)

Anneal every chain from beta0 to beta1 (geometric in wall time) for `time` seconds, in kernel launches
("epochs") of `epoch` steps per chain. If resample > 0, after every epoch the worst fraction `resample`
of the chains is replaced by copies of chains from the best fraction (population annealing style).
Calls trace(t, A) after each epoch.
"""
function run_anneal!(A::Annealer; time = 10.0, beta0 = 1.0, beta1 = 100.0, epoch = 2000, lam = 0.0, Wt = Inf,
                     pslice = 0.0, mix = 0.0, resample = 0.0, trace = nothing, rng = Random.Xoshiro(7), reeval = 4)
    evaluate!(A; Wt); CUDA.synchronize()
    t0 = Base.time(); ep = 0; tep = 0.0
    while true
        t = Base.time() - t0
        t >= time && break
        f0 = t / time
        f1 = min(1.0, (t + max(tep, 1e-3)) / time)
        b0 = beta0 * (beta1 / beta0)^f0; b1 = beta0 * (beta1 / beta0)^f1
        te = Base.time()
        ep % reeval == 0 && evaluate!(A; Wt)
        step!(A, epoch, b0, b1; lam, Wt, pslice, mix)
        CUDA.synchronize()
        tep = Base.time() - te
        ep += 1
        if resample > 0
            E, _ = energies(A, lam)
            o = sortperm(E)
            k = floor(Int, resample * A.nch)
            if k > 0
                dst = o[end-k+1:end]
                src = o[rand(rng, 1:k, k)]
                clone!(A, dst, src)
            end
        end
        trace === nothing || trace(Base.time() - t0, A)
    end
    ep
end

end # module
