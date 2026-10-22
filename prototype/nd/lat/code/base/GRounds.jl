# GPU rounds: the CPU builds the CSR graph, the GPU compresses it, computes the
# initial scores and runs the bulk-synchronous rounds (score, distance-2 Luby
# selection, elimination with Wing-Huang deficiency updates, supervariable
# detection); the CPU finishes a dense tail (if any) with the serial algorithm.
#
# Version 2 (Oct 19, see perf/PERF_NOTES.md): the round loop is device driven.
#   - All control decisions (thresholds, tail handoff, capacity checks) are
#     made on the device by the last block of a kernel; kernels read their
#     work counts from device memory and use grid-stride loops with a grid
#     sized from the device's occupancy (no host sizes, no per-round sync).
#   - Two rounds are captured into a CUDA graph that the host replays in
#     batches; the host only reads a control block between batches.
#   - Candidates come from an active list L = {live v : key(v) <= T}, kept
#     exact by re-keying the vertices that an elimination touches; the full
#     live list is scanned only when the candidate threshold passes T
#     (T is then reset from a key histogram). Per-round work is O(|L| + the
#     eliminations) instead of O(live vertices).
#   - Arena space is claimed with atomics (no prefix sums), and the
#     elimination is split into kernels whose clique rows are spread over all
#     blocks (large cliques no longer run on one block).
# Same selection rule and state evolution as version 1, so the elimination
# order has the same fill (checked against perf/GRounds_before.jl).
module GRoundsBase

using CUDA, CliqueTrees, Graphs
using ..Rounds
using ..Rounds: WSCALE, serialalg

export grounds, grounds_ens, GPUHybrid, grounds_small, grounds_batch, grounds_race

const ALIVE = 0x00
const ABSORBED = 0x01
const DEAD = 0x02
const FULL = 0xffffffff

# control block (Int64 slots)
const C_STATUS = 1   # 0 running, 1 done, 2 tail handoff, 3 needs host (capacity)
const C_NAL = 2      # live vertices
const C_NA = 3       # length of alist (superset of the live vertices)
const C_NA2 = 4
const C_CNT1 = 5     # length of list buffer 1
const C_CNT2 = 6     # length of list buffer 2
const C_NC = 7       # candidates
const C_NS = 8       # selected
const C_KTOP = 9     # clique arena
const C_BTOP = 10    # bit-matrix arena
const C_PB = 11      # pool bound of this round
const C_PTOP = 12    # pool top
const C_SMALL = 13
const C_NORD = 14
const C_NROUND = 15
const C_REB = 16     # rebuild in this round
const C_NREB = 17
const C_T = 18       # list threshold T (Float32 bits)
const C_THR = 19     # candidate threshold (Float32 bits)
const C_DONE = 20    # last-block counter
const C_CUR = 21     # buffer (1 or 2) that holds the current list
const C_KCAP = 22
const C_BCAP = 23
const C_PCAP = 24
const C_NEED = 25
const C_KTOT = 26
const C_BTOT = 27
const C_LMAX = 28    # largest list seen
const C_UTHR = 29    # candidate threshold of the untouched vertices (Float32 bits)
const C_NNZ = 30     # nnz(L) so far (counts of original vertices, diagonal included)
const C_FLOP = 31    # sum of the squared column counts so far
const C_TW = 32      # largest column count so far
const C_FH = 33      # fresh cap: hash threshold for untouched candidates (UInt32 range)
const C_FU = 34      # untouched candidates in this round
const C_OTOP = 35    # outside-mask arena top (dense mode), this round
const C_OCAP = 36    # outside-mask arena capacity
const C_OWANT = 37   # largest arena top asked for
const C_DACC = 38    # largest clique of this round (accumulator)
const C_DLAST = 39   # largest clique of the last round
const NCTL = 40

const DONE = 1
const TAIL = 2
const NEED = 3
const NBIN = 2048    # key histogram: top 11 bits of a nonnegative Float32
const TPB = 256      # threads per block of every kernel
const DGROUP = 64    # dense mode: grouped Wing-Huang updates for cliques of at least this many vertices
const CSHORT = 16    # k_cand!: lists up to this length are walked by one lane

@inline function hash32(v::Int32, seed::UInt32)
    x = reinterpret(UInt32, v) * 0x9e3779b1 + seed * 0x85ebca6b
    x ⊻= x >> 16; x *= 0x7feb352d; x ⊻= x >> 15; x *= 0x846ca68b; x ⊻= x >> 16
    return x
end

@inline function dmix(v::Int32)
    x = UInt64(reinterpret(UInt32, v)) * 0x9e3779b97f4a7c15
    x ⊻= x >> 32
    return x * 0xd6e8feb86659fd93
end

# is c in the sorted list pool[s:s+l-1]?
@inline function bsearch(pool, s::Int64, l::Int32, c::Int32)
    lo = s; hi = s + l - 1
    @inbounds while lo <= hi
        mid = (lo + hi) >> 1
        x = pool[mid]
        if x == c
            return true
        elseif x < c
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return false
end

@inline function warpsum(x::Int64)
    off = 16
    while off > 0
        x += shfl_down_sync(FULL, x, off)
        off >>= 1
    end
    return shfl_sync(FULL, x, 1)
end

@inline function warpsumu(x::UInt64)
    off = 16
    while off > 0
        x += shfl_down_sync(FULL, x, off)
        off >>= 1
    end
    return x      # valid in lane 0
end

@inline function warpmaxi(x::Int64)
    off = 16
    while off > 0
        x = max(x, shfl_down_sync(FULL, x, off))
        off >>= 1
    end
    return shfl_sync(FULL, x, 1)
end

@inline function warpmin(x::UInt32)
    off = 16
    while off > 0
        x = min(x, shfl_down_sync(FULL, x, off))
        off >>= 1
    end
    return x      # valid in lane 0
end

@inline laneid() = (threadIdx().x - 1) & 31
@inline gwarp() = ((blockIdx().x - 1) * (blockDim().x >> 5)) + ((threadIdx().x - 1) >> 5) + 1
@inline nwarps() = gridDim().x * (blockDim().x >> 5)
@inline gthread() = (blockIdx().x - 1) * blockDim().x + threadIdx().x
@inline nthreads() = gridDim().x * blockDim().x

# Execution context of the round kernels (SMALL_NOTES.md). NW = warps per block.
#   GridCx: the whole grid works on one graph (the launched kernels, and the
#           persistent cooperative kernel of GSmall.jl);
#   BlockCx: every block works on its own graph (GSmall.jl: one-block and
#           batched modes); the "grid" of the round code is then that block.
abstract type Cx{NW} end
struct GridCx{NW} <: Cx{NW} end
struct BlockCx{NW} <: Cx{NW} end
@inline gwarp(::GridCx) = gwarp()
@inline nwarps(::GridCx) = nwarps()
@inline gthread(::GridCx) = gthread()
@inline nthreads(::GridCx) = nthreads()
@inline vblock(::GridCx) = blockIdx().x
@inline vgrid(::GridCx) = gridDim().x
@inline gwarp(::BlockCx) = ((threadIdx().x - 1) >> 5) + 1
@inline nwarps(::BlockCx) = blockDim().x >> 5
@inline gthread(::BlockCx) = threadIdx().x
@inline nthreads(::BlockCx) = blockDim().x
@inline vblock(::BlockCx) = 1
@inline vgrid(::BlockCx) = 1
const GCX = GridCx{TPB ÷ 32}()     # context of the launched round kernels

@inline f2i(x::Float32) = Int64(reinterpret(UInt32, x))
@inline i2f(x::Int64) = reinterpret(Float32, x % UInt32)
@inline aload(a, i) = CUDA.atomic_add!(pointer(a, i), zero(eltype(a)))

@inline function vkey(def, wdeg, w, v, sc)
    @inbounds if sc == 0
        return Float32(wdeg[v])
    elseif sc == 1
        return Float32(def[v])
    else
        return Float32(def[v]) / Float32(w[v] + wdeg[v])
    end
end

@inline thrfun(kmin::Float32, relax::Float32, delta::Float32) =
    isinf(relax) ? Inf32 : max(kmin, kmin * (1f0 + relax) + delta)

# warp-aggregated append: 1-based slot for the lanes with pred (all lanes call)
@inline function wappend!(ctl, ci, pred::Bool)
    mask = vote_ballot_sync(FULL, pred)
    lane = laneid()
    base = Int64(0)
    if mask != 0
        leader = trailing_zeros(mask)
        if lane == leader
            base = CUDA.atomic_add!(pointer(ctl, ci), Int64(count_ones(mask)))
        end
        base = shfl_sync(FULL, base, leader + 1)
    end
    return base + count_ones(mask & ((UInt32(1) << lane) - UInt32(1))) + 1
end

# block minimum of x into the global UInt32 at kmin[i]
@inline function blockmin!(kmin, i, x::UInt32, sh)
    x = warpmin(x)
    laneid() == 0 && x != typemax(UInt32) && CUDA.atomic_min!(pointer(sh, 1), x)
    sync_threads()
    if threadIdx().x == 1
        sh[1] != typemax(UInt32) && CUDA.atomic_min!(pointer(kmin, i), sh[1])
    end
    return nothing
end

# true in exactly one block: the last one to get here (all earlier writes visible)
@inline lastblock(::GridCx, ctl, flag) = lastblock(ctl, flag)
@inline function lastblock(::BlockCx, ctl, flag)
    sync_threads()
    return true
end
@inline function lastblock(ctl, flag)
    threadfence()
    sync_threads()
    if threadIdx().x == 1
        t = CUDA.atomic_add!(pointer(ctl, C_DONE), Int64(1))
        flag[1] = t == gridDim().x - 1 ? Int32(1) : Int32(0)
    end
    sync_threads()
    r = flag[1] == Int32(1)
    if r
        threadfence()
    end
    return r
end

# ---------------------------------------------------------------- round kernels

# scan the list Lin into Lout (keeping the entries with key <= T) and take the
# minimum key; the last block decides: done, rebuild, or the candidate threshold
@inline function k_scan!(cx, Lout, keyout, Lin, cin, cout, inL, kv, ctl, kmin, relax, delta, tail, urelax, udelta)
    @inbounds ctl[C_STATUS] != 0 && return nothing
    sh = CuStaticSharedArray(UInt32, 1)
    flag = CuStaticSharedArray(Int32, 1)
    threadIdx().x == 1 && (@inbounds sh[1] = typemax(UInt32))
    sync_threads()
    @inbounds nl = ctl[cin]
    @inbounds T = i2f(ctl[C_T])
    kb = typemax(UInt32)
    lane = laneid()
    i0 = (gwarp(cx) - 1) * 32
    while i0 < nl
        i = i0 + lane + 1
        keep = false; k = 0f0; v = Int32(0)
        if i <= nl
            @inbounds v = Lin[i]
            @inbounds k = kv[v]
            if k != Inf32                       # live
                if k <= T
                    keep = true
                    kb = min(kb, reinterpret(UInt32, k))
                else
                    @inbounds inL[v] = UInt32(0)
                end
            end
        end
        j = wappend!(ctl, cout, keep)
        if keep
            @inbounds Lout[j] = v; @inbounds keyout[j] = k
        end
        i0 += nwarps(cx) * 32
    end
    blockmin!(kmin, 1, kb, sh)
    if lastblock(cx, ctl, flag) && threadIdx().x == 1
        @inbounds begin
            ctl[C_DONE] = 0
            ctl[C_CUR] = cout == C_CNT1 ? 1 : 2
            nal = ctl[C_NAL]
            ctl[C_NC] = 0; ctl[C_NS] = 0; ctl[C_KTOP] = 0; ctl[C_BTOP] = 0; ctl[C_PB] = 0
            if nal == 0 || nal <= tail
                ctl[C_STATUS] = DONE
            else
                km = aload(kmin, 1)
                kmin[1] = typemax(UInt32)
                cnt = aload(ctl, cout)
                thr = thrfun(reinterpret(Float32, km), relax, delta)
                if cnt == 0 || !(thr <= T)
                    ctl[C_REB] = 1
                    ctl[cout] = 0
                else
                    ctl[C_REB] = 0
                    ctl[C_THR] = f2i(thr)
                    ctl[C_UTHR] = f2i(min(thr, thrfun(reinterpret(Float32, km), urelax, udelta)))
                end
                ctl[C_LMAX] = max(ctl[C_LMAX], cnt)
            end
        end
    end
    return nothing
end

# rebuild 1: compact alist into alist2, global minimum and key histogram; the
# last block picks the new list threshold T from the histogram
@inline function k_reb1!(cx, alist2, alist, stt, kv, ctl, kmin, hist, relax, delta, ltarget, urelax, udelta)
    @inbounds (ctl[C_STATUS] != 0 || ctl[C_REB] == 0) && return nothing
    sh = CuStaticSharedArray(UInt32, 1)
    flag = CuStaticSharedArray(Int32, 1)
    hb = CuStaticSharedArray(UInt32, NBIN)
    ts = CuStaticSharedArray(Int64, 1024)
    tid = threadIdx().x; nth = blockDim().x
    tid == 1 && (@inbounds sh[1] = typemax(UInt32))
    b = tid
    while b <= NBIN
        @inbounds hb[b] = UInt32(0); b += nth
    end
    sync_threads()
    @inbounds na = ctl[C_NA]
    kb = typemax(UInt32)
    lane = laneid()
    i0 = (gwarp(cx) - 1) * 32
    while i0 < na
        i = i0 + lane + 1
        keep = false; v = Int32(0)
        if i <= na
            @inbounds v = alist[i]
            if (@inbounds stt[v]) == ALIVE
                keep = true
                kbits = reinterpret(UInt32, @inbounds kv[v])
                kb = min(kb, kbits)
                CUDA.atomic_add!(pointer(hb, Int(kbits >> 20) + 1), UInt32(1))
            end
        end
        j = wappend!(ctl, C_NA2, keep)
        keep && (@inbounds alist2[j] = v)
        i0 += nwarps(cx) * 32
    end
    sync_threads()
    b = tid
    while b <= NBIN
        c = @inbounds hb[b]
        c != 0 && CUDA.atomic_add!(pointer(hist, b), c)
        b += nth
    end
    blockmin!(kmin, 2, kb, sh)
    if lastblock(cx, ctl, flag)
        # read and clear the histogram; per-thread sums of consecutive bins
        per = cld(NBIN, nth)
        s = Int64(0)
        for q in 1:per
            b = (tid - 1) * per + q
            if b <= NBIN
                c = CUDA.atomic_xchg!(pointer(hist, b), UInt32(0))
                @inbounds hb[b] = c
                s += c
            end
        end
        @inbounds ts[tid] = s
        sync_threads()
        if tid == 1
            @inbounds begin
                ctl[C_DONE] = 0
                na2 = aload(ctl, C_NA2)
                ctl[C_NA] = na2; ctl[C_NA2] = 0
                km = aload(kmin, 2)
                kmin[2] = typemax(UInt32)
                thr = thrfun(reinterpret(Float32, km), relax, delta)
                target = max(ltarget, ctl[C_NAL] >> 5)
                T = Inf32
                acc = Int64(0)
                for t in 1:nth
                    if acc + ts[t] >= target
                        for q in 1:per
                            b = (t - 1) * per + q
                            b > NBIN && break
                            acc += hb[b]
                            if acc >= target
                                # top of the bin below the crossing one, so |L| <= target
                                T = b >= 2 ? reinterpret(Float32, (UInt32(b - 2) << 20) | 0x000fffff) : 0f0
                                break
                            end
                        end
                        break
                    end
                    acc += ts[t]
                end
                # at least the top of the bin of thr
                T = isinf(thr) ? Inf32 : max(T, reinterpret(Float32, reinterpret(UInt32, thr) | 0x000fffff))
                ctl[C_T] = f2i(T); ctl[C_THR] = f2i(thr)
                ctl[C_UTHR] = f2i(min(thr, thrfun(reinterpret(Float32, km), urelax, udelta)))
                ctl[C_NREB] += 1
            end
        end
    end
    return nothing
end

# rebuild 2: copy alist2 back to alist and refill Lout with the keys <= T
@inline function k_reb2!(cx, Lout, keyout, cout, alist, alist2, inL, kv, ctl)
    @inbounds (ctl[C_STATUS] != 0 || ctl[C_REB] == 0) && return nothing
    @inbounds na = ctl[C_NA]
    @inbounds T = i2f(ctl[C_T])
    lane = laneid()
    i0 = (gwarp(cx) - 1) * 32
    while i0 < na
        i = i0 + lane + 1
        keep = false; v = Int32(0); k = 0f0
        if i <= na
            @inbounds v = alist2[i]
            @inbounds alist[i] = v
            @inbounds k = kv[v]
            keep = k <= T
            @inbounds inL[v] = keep ? UInt32(1) : UInt32(0)
        end
        j = wappend!(ctl, cout, keep)
        if keep
            @inbounds Lout[j] = v; @inbounds keyout[j] = k
        end
        i0 += nwarps(cx) * 32
    end
    return nothing
end

# candidates (key <= thr) and the first Luby pass: best[x] = min priority near x
@inline function k_cand!(cx, cand, cpri, best, Lout, keyout, cout, ctl, pool, astart, alen, seed, tch, tidx, fcap, voff = Int32(0))
    @inbounds (ctl[C_STATUS] != 0) && return nothing
    @inbounds nl = ctl[cout]
    @inbounds thr = i2f(ctl[C_THR])
    @inbounds uthr = i2f(ctl[C_UTHR])
    @inbounds fh = ctl[C_FH]
    lane = laneid()
    i0 = (gwarp(cx) - 1) * 32
    while i0 < nl
        i = i0 + lane + 1
        v = Int32(0); p = UInt64(0); pred = false; isu = false
        if i <= nl
            @inbounds kk = keyout[i]
            if kk <= thr
                @inbounds v = Lout[i]
                # ties: a hash of (v, seed), or the vertex index (ascending); v - voff is
                # the vertex number in its own graph (batched graphs share one numbering)
                lo = tidx ? reinterpret(UInt32, v - voff) : hash32(v - voff, seed)
                if (@inbounds tch[v]) != 0x00
                    pred = true
                elseif kk <= uthr
                    # untouched vertices (never in a clique) only up to the stricter threshold,
                    # and with a fresh cap only those of hash <= fh (about fcap of them)
                    isu = true
                    pred = fcap == 0 || Int64(lo) <= fh
                end
                pred && (p = (UInt64(reinterpret(UInt32, kk)) << 32) | UInt64(lo))
            end
        end
        if fcap != 0
            nu = count_ones(vote_ballot_sync(FULL, isu))
            lane == 0 && nu != 0 && CUDA.atomic_add!(pointer(ctl, C_FU), Int64(nu))
        end
        j = wappend!(ctl, C_NC, pred)
        if pred
            @inbounds cand[j] = v; @inbounds cpri[j] = p
        end
        # best[x] = min priority over the candidates in N[x]: a lane per candidate
        # with a short list (4 entries in flight), the warp for each long one
        long = false
        if pred
            CUDA.atomic_min!(pointer(best, v), p)
            @inbounds s = astart[v]; @inbounds l = alen[v]
            if l <= CSHORT
                k = Int32(0)
                while k < l
                    x1 = @inbounds pool[s + k]
                    x2 = k + 1 < l ? (@inbounds pool[s + k + 1]) : Int32(0)
                    x3 = k + 2 < l ? (@inbounds pool[s + k + 2]) : Int32(0)
                    x4 = k + 3 < l ? (@inbounds pool[s + k + 3]) : Int32(0)
                    CUDA.atomic_min!(pointer(best, x1), p)
                    x2 != 0 && CUDA.atomic_min!(pointer(best, x2), p)
                    x3 != 0 && CUDA.atomic_min!(pointer(best, x3), p)
                    x4 != 0 && CUDA.atomic_min!(pointer(best, x4), p)
                    k += Int32(4)
                end
            else
                long = true
            end
        end
        mask = vote_ballot_sync(FULL, long)
        while mask != 0
            src = trailing_zeros(mask)
            mask &= mask - UInt32(1)
            vv = shfl_sync(FULL, v, src + 1)
            pp = shfl_sync(FULL, p, src + 1)
            @inbounds s = astart[vv]; @inbounds l = alen[vv]
            k = lane
            while k < l
                @inbounds x = pool[s + k]
                CUDA.atomic_min!(pointer(best, x), pp)
                k += 32
            end
        end
        i0 += nwarps(cx) * 32
    end
    return nothing
end

# second Luby pass: v is selected if it holds the minimum on all of N[v]
@inline function k_luby2!(cx, sel, best, cand, cpri, ctl, cin, pool, astart, alen)
    @inbounds (ctl[C_STATUS] != 0) && return nothing
    gthread(cx) == 1 && (@inbounds ctl[cin] = 0)
    @inbounds nc = ctl[C_NC]
    lane = laneid()
    wi = gwarp(cx)
    while wi <= nc
        @inbounds v = cand[wi]; @inbounds p = cpri[wi]
        @inbounds ok = best[v] == p
        @inbounds s = astart[v]; @inbounds l = alen[v]
        k = lane
        while k < l
            @inbounds x = pool[s + k]
            @inbounds ok &= best[x] == p
            k += 32
        end
        allok = vote_all_sync(FULL, ok)
        if allok && lane == 0
            j = CUDA.atomic_add!(pointer(ctl, C_NS), Int64(1)) + 1
            @inbounds sel[j] = v
        end
        wi += nwarps(cx)
    end
    return nothing
end

# reset best near the candidates; arena space for the selected; the last block
# is the gate (tail handoff, capacity checks)
@inline function k_sizes!(cx, kst, bst, best, cand, sel, ctl, pool, astart, alen, slog, tailsel, tailmax, logstats,
        kar, kblk, dsel, wks, kown, kidx, kstamp, stt, w, nm, cks, fcap, kwt, wun)
    @inbounds (ctl[C_STATUS] != 0) && return nothing
    flag = CuStaticSharedArray(Int32, 1)
    @inbounds nc = ctl[C_NC]; @inbounds ns = ctl[C_NS]
    @inbounds rid = (ctl[C_NROUND] + 1) % Int32
    @inbounds kcap = ctl[C_KCAP]
    lane = laneid()
    wi = gwarp(cx)
    while wi <= max(nc, ns)
        if wi <= nc
            @inbounds v = cand[wi]
            lane == 0 && (@inbounds best[v] = typemax(UInt64))
            @inbounds s = astart[v]; @inbounds l = alen[v]
            k = lane
            while k < l
                @inbounds best[pool[s + k]] = typemax(UInt64)
                k += 32
            end
        end
        if wi <= ns
            @inbounds v = sel[wi]; @inbounds s = astart[v]; @inbounds l = alen[v]
            acc = Int64(0)
            k = lane
            while k < l
                @inbounds x = pool[s + k]
                @inbounds acc += Int64(alen[x]) + Int64(l)
                k += 32
            end
            acc = warpsum(acc)
            ko = Int64(0)
            if lane == 0
                ko = CUDA.atomic_add!(pointer(ctl, C_KTOP), Int64(l))
                @inbounds kst[wi] = ko
                @inbounds bst[wi] = CUDA.atomic_add!(pointer(ctl, C_BTOP), Int64(l) * Int64(cld(l, 32)))
                CUDA.atomic_add!(pointer(ctl, C_PB), acc)
            end
            ko = shfl_sync(FULL, ko, 1)
            fits = ko + l <= kcap
            # K = live neighbours of v, in list order (sorted)
            d = Int32(0); WK = Int64(0); CK = Int64(0); WM = Int64(0)
            base = Int32(0)
            while base < l
                idx = base + Int32(lane)
                x = idx < l ? (@inbounds pool[s + idx]) : Int32(0)
                ok = idx < l && (@inbounds stt[x]) == ALIVE
                mask = vote_ballot_sync(FULL, ok)
                if ok
                    q = d + Int32(count_ones(mask & ((UInt32(1) << lane) - UInt32(1)))) + Int32(1)
                    @inbounds wx = w[x]
                    fits && (@inbounds kar[ko + q] = x; @inbounds kblk[ko + q] = wi % Int32; @inbounds kwt[ko + q] = wx)
                    @inbounds kown[x] = v; @inbounds kidx[x] = q; @inbounds kstamp[x] = rid
                    WK += wx; WM = max(WM, wx)
                    @inbounds CK += nm[x]
                end
                d += Int32(count_ones(mask))
                base += Int32(32)
            end
            WK = warpsum(WK); CK = warpsum(CK); WM = warpmaxi(WM)
            if fits
                q = d + Int32(lane) + Int32(1)
                while q <= l
                    @inbounds kblk[ko + q] = Int32(0)
                    q += Int32(32)
                end
            end
            if lane == 0
                CUDA.atomic_max!(pointer(ctl, C_DACC), Int64(d))
                @inbounds dsel[wi] = d; @inbounds wks[wi] = WK; @inbounds cks[wi] = CK
                # all clique weights equal (to WM): popcounts replace weighted bit sums in k_elim34!
                @inbounds wun[wi] = WK == Int64(d) * WM ? WM : Int64(0)
            end
        end
        wi += nwarps(cx)
    end
    if lastblock(cx, ctl, flag) && threadIdx().x == 1
        @inbounds begin
            ctl[C_DONE] = 0
            ctl[C_OWANT] = max(ctl[C_OWANT], ctl[C_OTOP]); ctl[C_OTOP] = 0
            ctl[C_DLAST] = aload(ctl, C_DACC); ctl[C_DACC] = 0
            ktot = aload(ctl, C_KTOP); btot = aload(ctl, C_BTOP); pb = aload(ctl, C_PB)
            fu = aload(ctl, C_FU); ctl[C_FU] = 0
            nal = ctl[C_NAL]
            small = (ns <= tailsel && ktot >= 32 * ns) ? ctl[C_SMALL] + 1 : Int64(0)
            if small >= 2 && nal <= tailmax
                ctl[C_SMALL] = small
                ctl[C_STATUS] = TAIL
            else
                need = Int64(0)
                ktot > ctl[C_KCAP] && (need |= 1)
                btot > ctl[C_BCAP] && (need |= 2)
                ctl[C_PTOP] + pb > ctl[C_PCAP] && (need |= 4)
                if need != 0
                    ctl[C_NEED] = need; ctl[C_KTOT] = ktot; ctl[C_BTOT] = btot
                    ctl[C_STATUS] = NEED
                else
                    ctl[C_SMALL] = small
                    # next round's fresh threshold: about fcap of the (now fu) untouched candidates
                    # (no limit after a round that selected nothing, so the loop always progresses)
                    if fcap != 0
                        ctl[C_FH] = (fu <= fcap || ns == 0) ? Int64(typemax(UInt32)) : (Int64(typemax(UInt32)) * fcap) ÷ fu
                    end
                    r = ctl[C_NROUND] + 1
                    ctl[C_NROUND] = r
                    if logstats && 3r <= length(slog)
                        slog[3r - 2] = nal % Int32; slog[3r - 1] = nc % Int32; slog[3r] = ns % Int32
                    end
                end
            end
        end
    end
    return nothing
end

# partner bits of clique row i in word g of its bit-matrix row (rb): the j > i,
# j <= d with {K[i], K[j]} not an edge (bit t of word g is j = 32(g-1) + t + 1)
# the same, with the row in shared memory (shr at offset sr) when inrow
@inline pmask2(inrow::Bool, shr, sr::Int64, bar, rb::Int64, g::Int32, i::Int32, d::Int32) =
    inrow ? pmask(shr, sr, g, i, d) : pmask(bar, rb, g, i, d)

@inline function pmask(bar, rb::Int64, g::Int32, i::Int32, d::Int32)
    lo = (g - Int32(1)) * Int32(32)
    lo + Int32(32) <= i && return UInt32(0)
    @inbounds bits = ~bar[rb + g]
    lo < i && (bits &= ~((UInt32(1) << (i - lo)) - UInt32(1)))
    lo + Int32(32) > d && (bits &= (UInt32(1) << (d - lo)) - UInt32(1))
    return bits
end

const OSHT = 2048                # outside neighbours staged per block (shared memory) in k_whg! (256 per warp at TPB)
const PSHT = 1024                # partner words listed per block (shared memory) in k_whg! (128 per warp at TPB)

# elimination 1+2 (warp per clique row i): row i of the adjacency bit matrix
# inside K, the K-neighbour weight of K[i], and the Wing-Huang updates over the
# missing pairs {K[i], K[j]}, j > i (cliques with d < dg; bigger cliques are
# grouped in k_ob! and k_whg!, dense mode)
@inline function k_elim12!(cx::Cx{NW}, bar, swar, def, kar, kblk, dsel, kown, kidx, sel, kst, bst, ctl, pool, astart, alen, stt, w,
        needdef, dg, lown, lid, wbuf = nothing) where {NW}
    @inbounds (ctl[C_STATUS] != 0) && return nothing
    # per warp, in shared memory: N(a) when it fits (membership tests; wbuf is a
    # buffer of OSHT Int32 of the caller, else one of our own), the bit-matrix row
    # when it has at most 32 words, and the list of missing pairs (up to 64)
    sha = wbuf === nothing ? CuStaticSharedArray(Int32, OSHT) : wbuf
    shr = CuStaticSharedArray(UInt32, 32 * NW)
    shp = CuStaticSharedArray(Int32, PSHT)
    cap = Int32(OSHT ÷ NW)
    pcap = Int32(min(64, PSHT ÷ NW))
    wl = Int64((threadIdx().x - 1) >> 5)
    so = wl * cap; sr = wl * 32; sp = wl * pcap
    @inbounds rid = ctl[C_NROUND]
    @inbounds ocap = ctl[C_OCAP]
    @inbounds ktot = ctl[C_KTOP]
    lane = laneid()
    lmask = (UInt32(1) << lane) - UInt32(1)
    s = Int64(gwarp(cx))
    while s <= ktot
        @inbounds bi = kblk[s]; @inbounds a = kar[s]
        if bi != 0
            @inbounds v = sel[bi]; @inbounds ko = kst[bi]; @inbounds d = dsel[bi]; @inbounds bo = bst[bi]
            @inbounds sa = astart[a]; @inbounds la = alen[a]
            wa = needdef ? (@inbounds w[a]) : Int64(0)
            i = Int32(s - ko)
            W = cld(d, Int32(32))
            rb = bo + Int64(i - 1) * W
            inrow = W <= Int32(32)
            sync_warp()       # the shared buffers of the last row are free
            if inrow
                lane < W && (@inbounds shr[sr + lane + 1] = UInt32(0))
            else
                kw = lane + 1
                while kw <= W
                    @inbounds bar[rb + kw] = UInt32(0)
                    kw += 32
                end
            end
            sync_warp()
            staged = la <= cap
            acc = Int64(0)
            k = lane
            while k < la      # 4 entries per lane in flight
                b1 = (@inbounds pool[sa + k])
                b2 = k + 32 < la ? (@inbounds pool[sa + k + 32]) : Int32(0)
                b3 = k + 64 < la ? (@inbounds pool[sa + k + 64]) : Int32(0)
                b4 = k + 96 < la ? (@inbounds pool[sa + k + 96]) : Int32(0)
                if staged
                    @inbounds sha[so + k + 1] = b1
                    k + 32 < la && (@inbounds sha[so + k + 33] = b2)
                    k + 64 < la && (@inbounds sha[so + k + 65] = b3)
                    k + 96 < la && (@inbounds sha[so + k + 97] = b4)
                end
                o1 = @inbounds kown[b1]
                o2 = b2 != 0 ? (@inbounds kown[b2]) : Int32(0)
                o3 = b3 != 0 ? (@inbounds kown[b3]) : Int32(0)
                o4 = b4 != 0 ? (@inbounds kown[b4]) : Int32(0)
                for (b, o) in ((b1, o1), (b2, o2), (b3, o3), (b4, o4))
                    if o == v
                        @inbounds j = kidx[b]
                        bit = UInt32(1) << ((j - 1) & 31)
                        if inrow
                            CUDA.atomic_or!(pointer(shr, sr + (j - 1) >> 5 + 1), bit)
                        else
                            CUDA.atomic_or!(pointer(bar, rb + (j - 1) >> 5 + 1), bit)
                        end
                        @inbounds acc += w[b]
                    end
                end
                k += 128
            end
            acc = warpsum(acc)
            lane == 0 && (@inbounds swar[s] = acc)
            threadfence_block()
            sync_warp()
            # the row: in shared memory (copied to bar for the dense kernels), or in bar
            inrow && lane < W && (@inbounds bar[rb + lane + 1] = shr[sr + lane + 1])
            if needdef && d >= dg
                # big clique (dense mode): its Wing-Huang updates are grouped in k_ob! and
                # k_whg!; here, if K[i] has partners, claim an outside-mask row (W words)
                # for each outside neighbour c not claimed yet in this round (one owner
                # per vertex and round; without a row, k_whg! uses binary searches for c)
                has = false
                g = (i >> 5) + Int32(1) + Int32(lane)
                while g <= W
                    has |= pmask2(inrow, shr, sr, bar, rb, g, i, d) != UInt32(0)
                    g += Int32(32)
                end
                if vote_any_sync(FULL, has)
                    key = (rid << 32) | Int64(bi)
                    k = lane
                    while k < la
                        @inbounds c = pool[sa + k]
                        if c != v && (@inbounds kown[c]) != v && (@inbounds stt[c]) == ALIVE
                            @inbounds old = lown[c]
                            if (old >> 32) != rid && CUDA.atomic_cas!(pointer(lown, c), old, key) == old
                                off = CUDA.atomic_add!(pointer(ctl, C_OTOP), Int64(W))
                                @inbounds lid[c] = off + W <= ocap ? off : Int64(-1)
                            end
                        end
                        k += 32
                    end
                end
            elseif needdef
                # missing pairs {a, K[j]}, j > i (fill edges): list them (lane per word),
                # load the data of up to pcap (<= 64) of them at once, then one pair at a time
                # the lanes walk N(b) for the common neighbours c of a and b:
                #   def[c] -= w(a) w(b);  def[a] -= w(b) w(outside c);  def[b] -= w(a) w(outside c)
                np = Int32(0)
                base = Int32(0)
                while base < W
                    kw = base + Int32(lane) + Int32(1)
                    bits = kw <= W ? pmask2(inrow, shr, sr, bar, rb, kw, i, d) : UInt32(0)
                    cb = Int32(count_ones(bits))
                    # exclusive scan of the counts over the lanes
                    x = cb
                    off = 1
                    while off < 32
                        y = shfl_up_sync(FULL, x, off)
                        lane >= off && (x += y)
                        off <<= 1
                    end
                    q = np + x - cb
                    lo = (kw - Int32(1)) * Int32(32)
                    while bits != 0 && q < pcap
                        t = trailing_zeros(bits)
                        bits &= bits - UInt32(1)
                        @inbounds shp[sp + q + 1] = lo + Int32(t) + Int32(1)
                        q += Int32(1)
                    end
                    np += shfl_sync(FULL, x, 32)
                    base += Int32(32)
                end
                sync_warp()
                if np <= pcap
                    # pair data: lane t holds pairs t+1 and t+33
                    b1 = Int32(0); wb1 = Int64(0); sb1 = Int64(0); lb1 = Int32(0)
                    b2 = Int32(0); wb2 = Int64(0); sb2 = Int64(0); lb2 = Int32(0)
                    if lane < np
                        @inbounds b1 = kar[ko + shp[sp + lane + 1]]
                        @inbounds wb1 = w[b1]; @inbounds sb1 = astart[b1]; @inbounds lb1 = alen[b1]
                    end
                    if lane + 32 < np
                        @inbounds b2 = kar[ko + shp[sp + lane + 33]]
                        @inbounds wb2 = w[b2]; @inbounds sb2 = astart[b2]; @inbounds lb2 = alen[b2]
                    end
                    for t in Int32(0):(np - Int32(1))
                        src = (t & Int32(31)) + Int32(1)
                        hi = t >= Int32(32)
                        b = shfl_sync(FULL, hi ? b2 : b1, src)
                        wb = shfl_sync(FULL, hi ? wb2 : wb1, src)
                        sb = shfl_sync(FULL, hi ? sb2 : sb1, src)
                        lb = shfl_sync(FULL, hi ? lb2 : lb1, src)
                        wab = wa * wb
                        wout = Int64(0)
                        k = lane
                        while k < lb      # 2 entries per lane in flight
                            @inbounds c1 = pool[sb + k]
                            c2 = k + 32 < lb ? (@inbounds pool[sb + k + 32]) : Int32(0)
                            s1 = @inbounds stt[c1]; o1 = @inbounds kown[c1]; w1 = @inbounds w[c1]
                            s2 = DEAD; o2 = Int32(0); w2 = Int64(0)
                            if c2 != 0
                                @inbounds s2 = stt[c2]; @inbounds o2 = kown[c2]; @inbounds w2 = w[c2]
                            end
                            if c1 != v && s1 == ALIVE && (staged ? bsearch(sha, so + 1, la, c1) : bsearch(pool, sa, la, c1))
                                CUDA.atomic_sub!(pointer(def, c1), wab)
                                o1 != v && (wout += w1)
                            end
                            if c2 != v && s2 == ALIVE && (staged ? bsearch(sha, so + 1, la, c2) : bsearch(pool, sa, la, c2))
                                CUDA.atomic_sub!(pointer(def, c2), wab)
                                o2 != v && (wout += w2)
                            end
                            k += 64
                        end
                        wout = warpsum(wout)
                        if lane == 0 && wout != 0
                            CUDA.atomic_sub!(pointer(def, a), wb * wout)
                            CUDA.atomic_sub!(pointer(def, b), wa * wout)
                        end
                    end
                else
                    for kw in Int32(1):W
                        bits = pmask2(inrow, shr, sr, bar, rb, kw, i, d)
                        lo = (kw - Int32(1)) * Int32(32)          # bit t is j = lo + t + 1
                        while bits != 0
                            t = trailing_zeros(bits)
                            bits &= bits - UInt32(1)
                            j = lo + t + 1
                            @inbounds b = kar[ko + j]
                            @inbounds wb = w[b]
                            wab = wa * wb
                            @inbounds sb = astart[b]; @inbounds lb = alen[b]
                            wout = Int64(0)
                            k = lane
                            while k < lb
                                @inbounds c = pool[sb + k]
                                sc = @inbounds stt[c]; oc = @inbounds kown[c]; wc = @inbounds w[c]
                                if c != v && sc == ALIVE && (staged ? bsearch(sha, so + 1, la, c) : bsearch(pool, sa, la, c))
                                    CUDA.atomic_sub!(pointer(def, c), wab)
                                    oc != v && (wout += wc)
                                end
                                k += 32
                            end
                            wout = warpsum(wout)
                            if lane == 0 && wout != 0
                                CUDA.atomic_sub!(pointer(def, a), wb * wout)
                                CUDA.atomic_sub!(pointer(def, b), wa * wout)
                            end
                        end
                    end
                end
            end
        end
        s += nwarps(cx)
    end
    return nothing
end

# number of K entries (kar[ko+1:ko+d], sorted) that are < e
@inline function krank(kar, ko::Int64, d::Int32, e::Int32)
    lo = Int32(0); hi = d
    @inbounds while lo < hi
        mid = (lo + hi) >> 1
        if kar[ko + mid + 1] < e
            lo = mid + Int32(1)
        else
            hi = mid
        end
    end
    return lo
end

# elimination 3+4 (warp per clique entry x): removal of v, and the new sorted
# adjacency list O ∪ (K \ x) with O = N(x) \ K \ v (live), written by a
# warp-parallel merge (each O entry knows its rank in K \ x, and writes the K
# entries between its predecessor in O and itself); hash of O for detection
@inline function k_elim34!(cx::Cx{NW}, pool, astart, alen, wdeg, def, hsh, cnt, kar, swar, kblk, dsel, wks, kown, sel, kst, ctl, stt, w,
        needdef, kv, inL, Lout, cout, kstamp, sc, ob, wbuf = nothing) where {NW}
    @inbounds (ctl[C_STATUS] != 0) && return nothing
    # K staged per warp in shared memory (when it fits) for the ranks of the merge
    shk = wbuf === nothing ? CuStaticSharedArray(Int32, OSHT) : wbuf
    cap = Int32(OSHT ÷ NW)
    so = Int64(((threadIdx().x - 1) >> 5) * cap)
    # clear the outside-mask rows used in this round (dense mode)
    @inbounds otop = min(ctl[C_OTOP], ctl[C_OCAP])
    q = Int64(gthread(cx))
    while q <= otop
        @inbounds ob[q] = UInt32(0)
        q += nthreads(cx)
    end
    @inbounds ktot = ctl[C_KTOP]
    @inbounds rid = ctl[C_NROUND] % Int32
    @inbounds T = i2f(ctl[C_T])
    lane = laneid()
    lmask = (UInt32(1) << lane) - UInt32(1)
    s = Int64(gwarp(cx))
    while s <= ktot
        @inbounds bi = kblk[s]
        if bi != 0
            @inbounds v = sel[bi]; @inbounds ko = kst[bi]; @inbounds d = dsel[bi]
            i = Int32(s - ko)                   # x = K[i]
            @inbounds x = kar[s]
            if lane == 0
                @inbounds swx = swar[s]
                @inbounds wx = w[x]; @inbounds wv = w[v]
                Wn = (@inbounds wks[bi]) - wx - swx
                @inbounds Wo = wdeg[x] - wv - swx
                needdef && CUDA.atomic_add!(pointer(def, x), Wo * (Wn - wv))
                @inbounds wdeg[x] += Wn - wv
            end
            @inbounds sx = astart[x]; @inbounds lx = alen[x]
            staged = d <= cap
            if staged
                sync_warp()
                q = lane + 1
                while q <= d
                    @inbounds shk[so + q] = kar[ko + q]
                    q += 32
                end
                sync_warp()
            end
            # pass 1: |O|
            c = Int32(0)
            base = Int32(0)
            while base < lx
                k = base + Int32(lane)
                ok = false
                if k < lx
                    @inbounds e = pool[sx + k]
                    ok = e != v && (@inbounds stt[e]) == ALIVE && (@inbounds kown[e]) != v
                end
                c += Int32(count_ones(vote_ballot_sync(FULL, ok)))
                base += Int32(32)
            end
            newlen = c + d - Int32(1)
            pos = Int64(0)
            lane == 0 && (pos = CUDA.atomic_add!(pointer(ctl, C_PTOP), Int64(newlen)) + 1)
            pos = shfl_sync(FULL, pos, 1)
            # pass 2: O entry number t (0-based) with r = rank in K \ x goes to
            # pos + t + r; the K \ x entries r' in [rprev, r) go to pos + t + r'
            t0 = Int32(0); rcarry = Int32(0)
            h = UInt64(0)
            base = Int32(0)
            while base < lx
                k = base + Int32(lane)
                ok = false; e = Int32(0)
                if k < lx
                    @inbounds e = pool[sx + k]
                    ok = e != v && (@inbounds stt[e]) == ALIVE && (@inbounds kown[e]) != v
                end
                mask = vote_ballot_sync(FULL, ok)
                r = Int32(-1)
                if ok
                    r = staged ? krank(shk, so, d, e) : krank(kar, ko, d, e)
                    r -= (x < e) ? Int32(1) : Int32(0)
                end
                # exclusive max-scan of r over the lanes (r is nondecreasing over the O entries)
                m = r
                off = 1
                while off < 32
                    y = shfl_up_sync(FULL, m, off)
                    lane >= off && (m = max(m, y))
                    off <<= 1
                end
                rprev = shfl_up_sync(FULL, m, 1)
                lane == 0 && (rprev = Int32(-1))
                rprev = max(rprev, rcarry)
                if ok
                    t = t0 + Int32(count_ones(mask & lmask))
                    @inbounds pool[pos + t + r] = e
                    h += dmix(e)
                    # e is outside every clique of this round: its key is final now
                    if (@inbounds kstamp[e]) != rid
                        ke = vkey(def, wdeg, w, e, sc)
                        @inbounds kv[e] = ke
                        if ke <= T && (@inbounds inL[e]) == UInt32(0)
                            if CUDA.atomic_xchg!(pointer(inL, e), UInt32(1)) == UInt32(0)
                                jj = CUDA.atomic_add!(pointer(ctl, cout), Int64(1)) + 1
                                @inbounds Lout[jj] = e
                            end
                        end
                    end
                    for rr in rprev:(r - Int32(1))
                        j = rr + (rr >= i - Int32(1) ? Int32(1) : Int32(0))     # K index (0-based) of entry rr of K \ x
                        @inbounds pool[pos + t + rr] = staged ? shk[so + j + 1] : kar[ko + j + 1]
                    end
                end
                rcarry = max(rcarry, shfl_sync(FULL, m, 32))
                t0 += Int32(count_ones(mask))
                base += Int32(32)
            end
            # the K \ x entries after the last O entry
            rr = rcarry + Int32(lane)
            while rr < d - Int32(1)
                j = rr + (rr >= i - Int32(1) ? Int32(1) : Int32(0))
                @inbounds pool[pos + c + rr] = staged ? shk[so + j + 1] : kar[ko + j + 1]
                rr += Int32(32)
            end
            h = warpsumu(h)
            if lane == 0
                @inbounds astart[x] = pos
                @inbounds alen[x] = newlen
                @inbounds hsh[s] = h; @inbounds cnt[s] = c
            end
        end
        s += nwarps(cx)
    end
    return nothing
end

# ---- dense mode: grouped Wing-Huang updates of the big cliques (d >= dg) ----
# Port of CliqueTrees' grouped deficiency update (mfupd_def, Oct 19 2026): the
# clique members are taken in groups of 32 (one word of the bit matrix), and each
# vertex c adjacent to K carries a mask per group of the members it is adjacent
# to. Inside K the masks are the rows of the bit matrix; outside K they are the
# rows claimed in k_elim12! and filled here. k_whg! then scans the neighbourhood
# of each member a once per group of its fill partners, instead of scanning
# N(b) once per fill edge {a, b} (k_elim12!, small cliques). The deficiency
# changes are the same integers, so the elimination is the same.

# warp per row a = K[i] of a big clique: set bit i of a's group in the mask rows
# of a's outside neighbours that this clique owns
@inline function k_ob!(cx, ob, lown, lid, kar, kblk, dsel, kown, sel, kst, ctl, pool, astart, alen, stt, dg, needdef)
    @inbounds (ctl[C_STATUS] != 0) && return nothing
    needdef || return nothing
    @inbounds ktot = ctl[C_KTOP]
    @inbounds rid = ctl[C_NROUND]
    lane = laneid()
    s = Int64(gwarp(cx))
    while s <= ktot
        @inbounds bi = kblk[s]
        if bi != 0 && (@inbounds dsel[bi]) >= dg
            @inbounds v = sel[bi]; @inbounds ko = kst[bi]
            i = Int32(s - ko)
            g = (i - Int32(1)) >> 5 + Int32(1)
            bit = UInt32(1) << ((i - Int32(1)) & Int32(31))
            key = (rid << 32) | Int64(bi)
            @inbounds a = kar[s]
            @inbounds sa = astart[a]; @inbounds la = alen[a]
            k = lane
            while k < la
                @inbounds c = pool[sa + k]
                if (@inbounds lown[c]) == key && c != v && (@inbounds kown[c]) != v
                    @inbounds L = lid[c]
                    L >= 0 && CUDA.atomic_or!(pointer(ob, L + g), bit)
                end
                k += 32
            end
        end
        s += nwarps(cx)
    end
    return nothing
end

# weight of the members of group g (0-based offset lo = 32(g-1)) in mask m
@inline function gsum(m::UInt32, wu::Int64, kwt, ko::Int64, lo::Int32)
    wu != 0 && return Int64(count_ones(m)) * wu
    acc = Int64(0)
    while m != UInt32(0)
        t = trailing_zeros(m)
        m &= m - UInt32(1)
        @inbounds acc += kwt[ko + lo + t + 1]
    end
    return acc
end

# warp per row a = K[i] of a big clique, with partners P = {K[j] : j > i, not adjacent}:
#   outside c ~ a:  m = mask of c in group g, restricted to P;
#                   def[c] -= w(a) w(m);  def[a] -= w(c) w(m);  def[b] -= w(a) w(c) for b in m
#   inside c ~ a:   the same first update, with the bit-matrix row of c as the mask
# Only the groups that hold partners are visited (their list is staged in shared
# memory when it is short), so a nearly complete clique costs little.
@inline function k_whg!(cx::Cx{NW}, def, bar, ob, lown, lid, kar, kwt, wun, kblk, dsel, kown, sel, kst, bst, ctl, pool, astart, alen, stt, w, dg, needdef, wbuf = nothing) where {NW}
    @inbounds (ctl[C_STATUS] != 0) && return nothing
    needdef || return nothing
    # per-warp staging: OSH and PSH entries per warp at 8 warps per block, the
    # same totals (so the same shared memory) at other block sizes
    OSH = OSHT ÷ NW; PSH = PSHT ÷ NW
    shO = wbuf === nothing ? CuStaticSharedArray(Int32, OSHT) : wbuf
    shG = CuStaticSharedArray(Int32, PSHT)
    sacc = CuStaticSharedArray(Int64, 32 * NW)
    @inbounds ktot = ctl[C_KTOP]
    @inbounds rid = ctl[C_NROUND]
    lane = laneid()
    lmask = (UInt32(1) << lane) - UInt32(1)
    wl = Int32((threadIdx().x - 1) >> 5)
    so = wl * Int32(OSH); s0 = wl * Int32(32); sg = wl * Int32(PSH)
    s = Int64(gwarp(cx))
    while s <= ktot
        @inbounds bi = kblk[s]
        if bi != 0 && (@inbounds dsel[bi]) >= dg
            @inbounds v = sel[bi]; @inbounds ko = kst[bi]; @inbounds d = dsel[bi]; @inbounds bo = bst[bi]
            i = Int32(s - ko)
            W = cld(d, Int32(32))
            rb = bo + Int64(i - 1) * W
            gs = (i >> 5) + Int32(1)
            # the words with partners: listed if at most PSH, else scanned
            np = Int32(0)
            gb = gs
            while gb <= W
                g = gb + Int32(lane)
                ok = g <= W && pmask(bar, rb, g, i, d) != UInt32(0)
                mask = vote_ballot_sync(FULL, ok)
                r = np + Int32(count_ones(mask & lmask))
                ok && r < PSH && (@inbounds shG[sg + r + 1] = g)
                np += Int32(count_ones(mask))
                gb += Int32(32)
            end
            sync_warp()
            listed = np <= PSH
            ng = listed ? np : W - gs + Int32(1)
            if np > 0
                @inbounds a = kar[s]; @inbounds wa = w[a]; @inbounds wu = wun[bi]
                key = (rid << 32) | Int64(bi)
                @inbounds sa = astart[a]; @inbounds la = alen[a]
                ta = Int64(0)
                pos = Int32(0)
                while pos < la
                    # stage the next outside neighbours (whole 32-chunks, at most OSH)
                    no = Int32(0)
                    while pos < la
                        kk = pos + Int32(lane)
                        ok = false; e = Int32(0)
                        if kk < la
                            @inbounds e = pool[sa + kk]
                            ok = e != v && (@inbounds kown[e]) != v && (@inbounds stt[e]) == ALIVE
                        end
                        mask = vote_ballot_sync(FULL, ok)
                        nm = Int32(count_ones(mask))
                        no + nm > OSH && break
                        ok && (@inbounds shO[so + no + Int32(count_ones(mask & lmask)) + 1] = e)
                        no += nm
                        pos += Int32(32)
                    end
                    sync_warp()
                    if no > 0
                        for gi in Int32(1):ng
                            g2 = listed ? (@inbounds shG[sg + gi]) : gs + gi - Int32(1)
                            pm = pmask(bar, rb, g2, i, d)
                            pm == UInt32(0) && continue
                            lo = (g2 - Int32(1)) * Int32(32)
                            @inbounds sacc[s0 + lane + 1] = Int64(0)
                            sync_warp()
                            q = Int32(lane)
                            while q < no
                                @inbounds c = shO[so + q + 1]
                                m = UInt32(0)
                                L = Int64(-1)
                                (@inbounds lown[c]) == key && (@inbounds L = lid[c])
                                if L >= 0
                                    @inbounds m = ob[L + g2] & pm
                                else
                                    bb = pm
                                    while bb != UInt32(0)
                                        t = trailing_zeros(bb)
                                        bb &= bb - UInt32(1)
                                        @inbounds b = kar[ko + lo + t + 1]
                                        @inbounds bsearch(pool, astart[b], alen[b], c) && (m |= UInt32(1) << t)
                                    end
                                end
                                if m != UInt32(0)
                                    sq = gsum(m, wu, kwt, ko, lo)
                                    CUDA.atomic_sub!(pointer(def, c), wa * sq)
                                    @inbounds wc = w[c]
                                    ta += wc * sq
                                    mm = m
                                    while mm != UInt32(0)
                                        t = trailing_zeros(mm)
                                        mm &= mm - UInt32(1)
                                        CUDA.atomic_add!(pointer(sacc, s0 + t + 1), wc)
                                    end
                                end
                                q += Int32(32)
                            end
                            sync_warp()
                            @inbounds xs = sacc[s0 + lane + 1]
                            if xs != 0
                                @inbounds b = kar[ko + lo + lane + 1]
                                CUDA.atomic_sub!(pointer(def, b), wa * xs)
                            end
                            sync_warp()
                        end
                    end
                    sync_warp()
                end
                ta = warpsum(ta)
                lane == 0 && ta != 0 && CUDA.atomic_sub!(pointer(def, a), ta)
                # inside K: lane per K-neighbour c = K[jc] of a (bit `lane` of each word of
                # row i), over the partner words; one update per (a, c)
                for kw in Int32(1):W
                    @inbounds sbits = bar[rb + kw]
                    if (sbits >> lane) & UInt32(1) != UInt32(0)
                        jc = (kw - Int32(1)) * Int32(32) + Int32(lane) + Int32(1)
                        rc = bo + Int64(jc - 1) * W
                        acc = Int64(0)
                        for gi in Int32(1):ng
                            g2 = listed ? (@inbounds shG[sg + gi]) : gs + gi - Int32(1)
                            pm = pmask(bar, rb, g2, i, d)
                            if pm != UInt32(0)
                                m = (@inbounds bar[rc + g2]) & pm
                                m != UInt32(0) && (acc += gsum(m, wu, kwt, ko, (g2 - Int32(1)) * Int32(32)))
                            end
                        end
                        acc != 0 && CUDA.atomic_sub!(pointer(def, @inbounds kar[ko + jc]), wa * acc)
                    end
                end
            end
        end
        s += nwarps(cx)
    end
    return nothing
end

# block per selected vertex: K vertices with equal neighbourhoods outside K.
# For K[i], the candidate representative is the smallest j < i with the same
# (hash, count) of its outside list (a quadratic scan for small K, a hash
# table for large K); a warp then verifies that the outside lists are equal.
function k_detect!(cx, par, hsh, cnt, htab, pool, astart, alen, stt, kown, kar, sel, kst, dsel, ctl)
    @inbounds (ctl[C_STATUS] != 0) && return nothing
    @inbounds ns = ctl[C_NS]
    bi = Int64(vblock(cx))
    while bi <= ns
        detect_one!(par, hsh, cnt, htab, pool, astart, alen, stt, kown, kar, sel, kst, dsel, bi)
        bi += vgrid(cx)
    end
    return nothing
end

# detection for selected vertex number bi, by the whole block
@inline function detect_one!(par, hsh, cnt, htab, pool, astart, alen, stt, kown, kar, sel, kst, dsel, bi)
    tid = threadIdx().x; nth = blockDim().x; lane = laneid()
    nw = nth >> 5; wl = (tid - 1) >> 5 + 1
    @inbounds v = sel[bi]; @inbounds ko = kst[bi]; @inbounds d = dsel[bi]
    if d <= 64
        i = tid
        while i <= d
            @inbounds h = hsh[ko + i]; @inbounds c = cnt[ko + i]
            r = Int32(0)
            for j in 1:(i - 1)
                if (@inbounds hsh[ko + j]) == h && (@inbounds cnt[ko + j]) == c
                    r = Int32(j); break
                end
            end
            @inbounds par[ko + i] = r
            i += nth
        end
    else
        tsz = Int32(1) << (32 - leading_zeros(UInt32(2d - 1)))     # power of two >= 2d
        mask = tsz - Int32(1)
        to = 4 * ko
        q = tid
        while q <= tsz
            @inbounds htab[to + q] = Int32(0); q += nth
        end
        sync_threads()
        i = tid
        while i <= d
            @inbounds h = hsh[ko + i]; @inbounds c = cnt[ko + i]
            slot = Int32((h ⊻ (UInt64(c) * 0x9e3779b97f4a7c15)) >> 40) & mask
            while true
                cur = CUDA.atomic_cas!(pointer(htab, to + slot + 1), Int32(0), Int32(i))
                cur == 0 && break
                if (@inbounds hsh[ko + cur]) == h && (@inbounds cnt[ko + cur]) == c
                    CUDA.atomic_min!(pointer(htab, to + slot + 1), Int32(i))
                    break
                end
                slot = (slot + Int32(1)) & mask
            end
            i += nth
        end
        sync_threads()
        i = tid
        while i <= d
            @inbounds h = hsh[ko + i]; @inbounds c = cnt[ko + i]
            slot = Int32((h ⊻ (UInt64(c) * 0x9e3779b97f4a7c15)) >> 40) & mask
            r = Int32(0)
            while true
                cur = @inbounds htab[to + slot + 1]
                if (@inbounds hsh[ko + cur]) == h && (@inbounds cnt[ko + cur]) == c
                    r = cur < i ? cur : Int32(0)
                    break
                end
                slot = (slot + Int32(1)) & mask
            end
            @inbounds par[ko + i] = r
            i += nth
        end
    end
    sync_threads()
    # verification, warp per candidate pair
    i = wl
    while i <= d
        @inbounds r = par[ko + i]
        if r != 0
            @inbounds x = kar[ko + i]; @inbounds y = kar[ko + r]
            @inbounds sx = astart[x]; @inbounds lx = alen[x]
            @inbounds sy = astart[y]; @inbounds ly = alen[y]
            bad = false
            k = lane
            while k < lx
                @inbounds o = pool[sx + k]
                if (@inbounds kown[o]) != v && (@inbounds stt[o]) == ALIVE && !bsearch(pool, sy, ly, o)
                    bad = true
                end
                k += 32
            end
            anybad = vote_any_sync(FULL, bad)
            lane == 0 && (@inbounds par[ko + i] = anybad ? Int32(0) : y)
        end
        i += nw
    end
    sync_threads()
    return nothing
end

# block per selected vertex: absorb the detected twins, retire v, and put the
# touched vertices (K and its neighbours) whose key is <= T on the list
function k_apply!(cx, Lout, cout, inL, order, parent, par, stt, w, wdeg, def, kar, sel, kst, dsel, ctl, kv, sc, compress, tch, nm, cks)
    @inbounds (ctl[C_STATUS] != 0) && return nothing
    sh_n = CuStaticSharedArray(Int64, 1)
    @inbounds ns = ctl[C_NS]
    @inbounds T = i2f(ctl[C_T])
    bi = Int64(vblock(cx))
    while bi <= ns
        apply_one!(Lout, cout, inL, order, parent, par, stt, w, wdeg, def, kar, sel, kst, dsel, ctl, kv, sc, compress, tch, nm, cks,
            bi, T, sh_n)
        bi += vgrid(cx)
    end
    return nothing
end

# absorption, retirement and re-keying for selected vertex number bi, by the whole block
@inline function apply_one!(Lout, cout, inL, order, parent, par, stt, w, wdeg, def, kar, sel, kst, dsel, ctl, kv, sc, compress, tch,
        nm, cks, bi, T, sh_n)
    tid = threadIdx().x; nth = blockDim().x
    @inbounds v = sel[bi]; @inbounds ko = kst[bi]; @inbounds d = dsel[bi]
    tid == 1 && (@inbounds sh_n[1] = 1)
    sync_threads()
    if compress
        i = tid
        while i <= d
            @inbounds r = par[ko + i]
            if r != 0
                @inbounds x = kar[ko + i]
                @inbounds stt[x] = ABSORBED
                @inbounds kv[x] = Inf32
                @inbounds parent[x] = r
                @inbounds wx = w[x]
                CUDA.atomic_add!(pointer(w, r), wx)
                CUDA.atomic_sub!(pointer(wdeg, r), wx)
                CUDA.atomic_add!(pointer(nm, r), @inbounds nm[x])
                CUDA.atomic_add!(pointer(sh_n, 1), Int64(1))
            end
            i += nth
        end
    end
    sync_threads()
    if tid == 1
        @inbounds stt[v] = DEAD
        @inbounds kv[v] = Inf32
        pos = CUDA.atomic_add!(pointer(ctl, C_NORD), Int64(1)) + 1
        @inbounds order[pos] = v
        # fill of the supervariable v (q members, clique count c): columns c+q, ..., c+1
        q = @inbounds nm[v]; c = @inbounds cks[bi]
        CUDA.atomic_add!(pointer(ctl, C_NNZ), q * c + (q * (q + 1)) >> 1)
        CUDA.atomic_add!(pointer(ctl, C_FLOP), q * c * c + c * q * (q + 1) + (q * (q + 1) * (2q + 1)) ÷ 6)
        CUDA.atomic_max!(pointer(ctl, C_TW), c + q)
        CUDA.atomic_sub!(pointer(ctl, C_NAL), @inbounds sh_n[1])
    end
    # re-key the clique (its outside neighbours were re-keyed in k_elim34!)
    i = tid
    while i <= d
        @inbounds x = kar[ko + i]
        @inbounds tch[x] = 0x01
        if (@inbounds stt[x]) == ALIVE
            k = vkey(def, wdeg, w, x, sc)
            @inbounds kv[x] = k
            if k <= T && (@inbounds inL[x]) == UInt32(0)
                if CUDA.atomic_xchg!(pointer(inL, x), UInt32(1)) == UInt32(0)
                    j = CUDA.atomic_add!(pointer(ctl, cout), Int64(1)) + 1
                    @inbounds Lout[j] = x
                end
            end
        end
        i += nth
    end
    sync_threads()
    return nothing
end

# copy the live lists to a fresh pool
function k_repack!(newpool, newstart, pool, astart, alen, alist, stt, nal, off)
    wi = gwarp(); lane = laneid()
    while wi <= nal
        @inbounds v = alist[wi]
        if (@inbounds stt[v]) == ALIVE
            @inbounds s = astart[v]; @inbounds l = alen[v]
            @inbounds o = off[wi]
            k = lane
            while k < l
                @inbounds newpool[o + k + 1] = pool[s + k]
                k += 32
            end
            lane == 0 && (@inbounds newstart[v] = o + 1)
        end
        wi += nwarps()
    end
    return nothing
end

function k_livelen!(lv, alist, stt, alen, na)
    i = gthread()
    while i <= na
        @inbounds v = alist[i]
        @inbounds lv[i] = (@inbounds stt[v]) == ALIVE ? Int64(alen[v]) : Int64(0)
        i += nthreads()
    end
    return nothing
end

# ---------------------------------------------------------------- initial state

function k_hash!(hk, pool, astart, alen, n)
    v = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if v <= n
        @inbounds s = astart[v]; @inbounds l = alen[v]
        h = dmix(Int32(v))
        for k in 0:(l - 1)
            @inbounds h += dmix(pool[s + k])
        end
        @inbounds hk[v] = h
    end
    return nothing
end

function k_runhead!(hd, perm, hk, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds hd[i] = (i == 1 || hk[perm[i]] != hk[perm[i - 1]]) ? Int32(i) : Int32(0)
    end
    return nothing
end

# x joins the run head if their closed neighbourhoods are equal
function k_twins!(par, perm, hd, pool, astart, alen, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds x = perm[i]; @inbounds y = perm[hd[i]]
        r = Int32(0)
        if x != y && (@inbounds alen[x]) == (@inbounds alen[y])
            @inbounds sx = astart[x]; @inbounds lx = alen[x]
            @inbounds sy = astart[y]; @inbounds ly = alen[y]
            ok = bsearch(pool, sx, lx, y)
            k = 0
            while ok && k < lx
                @inbounds u = pool[sx + k]
                ok = u == y || bsearch(pool, sy, ly, u)
                k += 1
            end
            ok && (r = y)
        end
        @inbounds par[x] = r
    end
    return nothing
end

function k_absorb0!(parent, stt, w, par, n, nm)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if x <= n
        @inbounds r = par[x]
        if r != 0
            @inbounds stt[x] = ABSORBED
            @inbounds parent[x] = r
            CUDA.atomic_add!(pointer(w, r), @inbounds w[x])
            CUDA.atomic_add!(pointer(nm, r), Int64(1))
        end
    end
    return nothing
end

function k_wdeg!(wdeg, pool, astart, alen, stt, w, n)
    v = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if v <= n
        @inbounds s = astart[v]; @inbounds l = alen[v]
        acc = Int64(0)
        for k in 0:(l - 1)
            @inbounds u = pool[s + k]
            (@inbounds stt[u]) == ALIVE && (@inbounds acc += w[u])
        end
        @inbounds wdeg[v] = acc
    end
    return nothing
end

# warp per live vertex: def = ((sum w)^2 - sum w^2)/2 - weight of the edges inside N(v)
function k_initdef!(def, alist, nal, pool, astart, alen, stt, w)
    wi = gwarp(); lane = laneid()
    if wi <= nal
        @inbounds v = alist[wi]; @inbounds sv = astart[v]; @inbounds lv = alen[v]
        W = Int64(0); W2 = Int64(0); tri = Int64(0)
        k = lane
        while k < lv
            @inbounds a = pool[sv + k]
            if (@inbounds stt[a]) == ALIVE
                @inbounds wa = w[a]
                W += wa; W2 += wa * wa
                @inbounds sa = astart[a]; @inbounds la = alen[a]
                for q in 0:(la - 1)
                    @inbounds b = pool[sa + q]
                    if (@inbounds stt[b]) == ALIVE && bsearch(pool, sv, lv, b)
                        @inbounds tri += wa * w[b]
                    end
                end
            end
            k += 32
        end
        W = warpsum(W); W2 = warpsum(W2); tri = warpsum(tri)
        lane == 0 && (@inbounds def[v] = ((W * W - W2) >> 1) - (tri >> 1))
    end
    return nothing
end

function k_initkey!(kv, stt, def, wdeg, w, sc, n)
    v = gthread()
    if v <= n
        @inbounds kv[v] = (@inbounds stt[v]) == ALIVE ? vkey(def, wdeg, w, v, sc) : Inf32
    end
    return nothing
end

function k_alive!(alist, acount, stt, n)
    v = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if v <= n && (@inbounds stt[v]) == ALIVE
        j = CUDA.atomic_add!(pointer(acount, 1), Int32(1)) + Int32(1)
        @inbounds alist[j] = Int32(v)
    end
    return nothing
end

# tail extraction: local ids, then the live neighbours of each tail vertex
function k_tailcnt!(tc, tv, loc, pool, astart, alen, nt)
    i = gthread()
    while i <= nt
        @inbounds v = tv[i]; @inbounds s = astart[v]; @inbounds l = alen[v]
        c = Int64(0)
        for k in 0:(l - 1)
            (@inbounds loc[pool[s + k]]) > 0 && (c += 1)
        end
        @inbounds tc[i] = c
        i += nthreads()
    end
    return nothing
end

function k_tailfill!(tadj, toff, tv, loc, pool, astart, alen, nt)
    i = gthread()
    while i <= nt
        @inbounds v = tv[i]; @inbounds s = astart[v]; @inbounds l = alen[v]
        o = @inbounds toff[i] + 1
        for k in 0:(l - 1)
            j = @inbounds loc[pool[s + k]]
            if j > 0
                @inbounds tadj[o] = j; o += 1
            end
        end
        i += nthreads()
    end
    return nothing
end

function k_loc!(loc, tv, nt)
    i = gthread()
    while i <= nt
        @inbounds loc[tv[i]] = Int32(i)
        i += nthreads()
    end
    return nothing
end

# ---------------------------------------------------------------- driver


nblk(n, t = TPB) = max(1, cld(n, t))

"""
    grounds(weights, graph, alg::RoundsAlg; stats=nothing, timing=nothing)

GPU version of `Rounds.rounds` (selection is always the one-pass Luby).
"""
function grounds(weights::AbstractVector, graph::AbstractGraph, alg::RoundsAlg; kw...)
    T0 = time()
    ptr, tgt = csr(graph)
    return grounds_csr(weights, ptr, tgt, alg; tcsr = time() - T0, kw...)
end

# sorted CSR without self loops (Int32 targets)
function csr(graph::SimpleGraph)
    # fast path: SimpleGraph keeps its adjacency lists sorted
    fl = graph.fadjlist
    n = length(fl)
    ptr = Vector{Int64}(undef, n + 1); ptr[1] = 1
    @inbounds for v in 1:n
        ptr[v + 1] = ptr[v] + length(fl[v])
    end
    tgt = Vector{Int32}(undef, ptr[n + 1] - 1)
    k = 1; loops = false
    @inbounds for v in 1:n
        lst = fl[v]
        ptr[v] = k
        for u in lst
            tgt[k] = u % Int32
            k += 1
        end
        loops |= insorted(v, lst)
    end
    ptr[n + 1] = k
    loops && return csr_generic(graph)
    return ptr, tgt
end

csr(graph::AbstractGraph) = csr_generic(graph)

function csr_generic(graph::AbstractGraph)
    n = nv(graph)
    ptr = Vector{Int64}(undef, n + 1); ptr[1] = 1
    @inbounds for v in 1:n
        ptr[v + 1] = ptr[v] + length(neighbors(graph, v))
    end
    tgt = Vector{Int32}(undef, ptr[n + 1] - 1)
    k = 1; loops = false
    @inbounds for v in 1:n
        k0 = k
        for u in neighbors(graph, v)
            if u == v
                loops = true
            else
                tgt[k] = u; k += 1
            end
        end
        ptr[v] = k0
        seg = view(tgt, k0:(k - 1))
        issorted(seg) || sort!(seg)
    end
    ptr[n + 1] = k
    if loops
        resize!(tgt, k - 1)
    end
    return ptr, tgt
end

# grid sizes from the device: resident blocks per SM times the SM count
struct LaunchCfg
    nsm::Int
    wave::Int
end
function LaunchCfg()
    d = CUDA.device()
    nsm = CUDA.attribute(d, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    tps = CUDA.attribute(d, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    return LaunchCfg(nsm, nsm * tps)
end
grid(L::LaunchCfg, k, cap) = max(1, min(L.nsm * CUDA.active_blocks(k.fun, TPB), cap))

mutable struct Dev
    n::Int
    pool::CuVector{Int32}; astart::CuVector{Int64}; alen::CuVector{Int32}
    stt::CuVector{UInt8}; w::CuVector{Int64}; wdeg::CuVector{Int64}; def::CuVector{Int64}
    parent::CuVector{Int32}; order::CuVector{Int32}
    alist::CuVector{Int32}; alist2::CuVector{Int32}
    L1::CuVector{Int32}; L2::CuVector{Int32}; key1::CuVector{Float32}; key2::CuVector{Float32}
    inL::CuVector{UInt32}
    kown::CuVector{Int32}; kidx::CuVector{Int32}; kstamp::CuVector{Int32}; kv::CuVector{Float32}; best::CuVector{UInt64}
    cand::CuVector{Int32}; cpri::CuVector{UInt64}; sel::CuVector{Int32}
    kst::CuVector{Int64}; bst::CuVector{Int64}; dsel::CuVector{Int32}; wks::CuVector{Int64}
    kar::CuVector{Int32}; kblk::CuVector{Int32}; swar::CuVector{Int64}; hsh::CuVector{UInt64}
    cnt::CuVector{Int32}; par::CuVector{Int32}; htab::CuVector{Int32}; bar::CuVector{UInt32}
    ctl::CuVector{Int64}; kmin::CuVector{UInt32}; hist::CuVector{UInt32}; slog::CuVector{Int32}
    tch::CuVector{UInt8}; nm::CuVector{Int64}; cks::CuVector{Int64}
    kwt::CuVector{Int64}; wun::CuVector{Int64}
    lown::CuVector{Int64}; lid::CuVector{Int64}; ob::CuVector{UInt32}
end

function growk!(D::Dev, m)
    for f in (:kar, :kblk, :swar, :hsh, :cnt, :par, :kwt)
        a = getfield(D, f)
        CUDA.unsafe_free!(a)
        setfield!(D, f, similar(a, m))
    end
    CUDA.unsafe_free!(D.htab)
    D.htab = similar(D.htab, 4m)
end

# kernels of one round (list Lin -> Lout) and their arguments, in launch order
function round_args(D, p, Lin, cin, Lout, kout, cout)
    r = (
        scan = (k_scan!, (GCX, Lout, kout, Lin, cin, cout, D.inL, D.kv, D.ctl, D.kmin, p.relax, p.delta, p.tail, p.urelax, p.udelta)),
        reb1 = (k_reb1!, (GCX, D.alist2, D.alist, D.stt, D.kv, D.ctl, D.kmin, D.hist, p.relax, p.delta, p.ltarget, p.urelax, p.udelta)),
        reb2 = (k_reb2!, (GCX, Lout, kout, cout, D.alist, D.alist2, D.inL, D.kv, D.ctl)),
        cand = (k_cand!, (GCX, D.cand, D.cpri, D.best, Lout, kout, cout, D.ctl, D.pool, D.astart, D.alen, p.seed, D.tch, p.tidx, p.fcap)),
        luby2 = (k_luby2!, (GCX, D.sel, D.best, D.cand, D.cpri, D.ctl, cin, D.pool, D.astart, D.alen)),
        sizes = (k_sizes!, (GCX, D.kst, D.bst, D.best, D.cand, D.sel, D.ctl, D.pool, D.astart, D.alen, D.slog, p.tailsel, p.tailmax,
            p.logstats, D.kar, D.kblk, D.dsel, D.wks, D.kown, D.kidx, D.kstamp, D.stt, D.w, D.nm, D.cks, p.fcap, D.kwt, D.wun)),
        elim12 = (k_elim12!, (GCX, D.bar, D.swar, D.def, D.kar, D.kblk, D.dsel, D.kown, D.kidx, D.sel, D.kst, D.bst, D.ctl, D.pool,
            D.astart, D.alen, D.stt, D.w, p.needdef, p.dg, D.lown, D.lid)),
        ob = (k_ob!, (GCX, D.ob, D.lown, D.lid, D.kar, D.kblk, D.dsel, D.kown, D.sel, D.kst, D.ctl, D.pool, D.astart, D.alen, D.stt,
            p.dg, p.needdef)),
        whg = (k_whg!, (GCX, D.def, D.bar, D.ob, D.lown, D.lid, D.kar, D.kwt, D.wun, D.kblk, D.dsel, D.kown, D.sel, D.kst, D.bst,
            D.ctl, D.pool, D.astart, D.alen, D.stt, D.w, p.dg, p.needdef)),
        elim34 = (k_elim34!, (GCX, D.pool, D.astart, D.alen, D.wdeg, D.def, D.hsh, D.cnt, D.kar, D.swar, D.kblk, D.dsel, D.wks, D.kown,
            D.sel, D.kst, D.ctl, D.stt, D.w, p.needdef, D.kv, D.inL, Lout, cout, D.kstamp, p.sc, D.ob)),
        detect = (k_detect!, (GCX, D.par, D.hsh, D.cnt, D.htab, D.pool, D.astart, D.alen, D.stt, D.kown, D.kar, D.sel, D.kst, D.dsel, D.ctl)),
        apply = (k_apply!, (GCX, Lout, cout, D.inL, D.order, D.parent, D.par, D.stt, D.w, D.wdeg, D.def, D.kar, D.sel, D.kst, D.dsel,
            D.ctl, D.kv, p.sc, p.compress, D.tch, D.nm, D.cks)),
    )
    return p.compress ? r : Base.structdiff(r, NamedTuple{(:detect,)})
end

function compile_kernels(D::Dev, p)
    ra = round_args(D, p, D.L1, C_CNT1, D.L2, D.key2, C_CNT2)
    return map(fa -> cufunction(fa[1], Tuple{map(a -> typeof(cudaconvert(a)), fa[2])...}), ra)
end

# captures two rounds (list buffer `cur` -> other -> cur) into an executable graph
function capture_rounds(D::Dev, K, G, cur::Int, p)
    LA, kA, cA, LB, kB, cB = cur == 1 ? (D.L1, D.key1, C_CNT1, D.L2, D.key2, C_CNT2) :
        (D.L2, D.key2, C_CNT2, D.L1, D.key1, C_CNT1)
    function round!(Lin, cin, Lout, kout, cout)
        ra = round_args(D, p, Lin, cin, Lout, kout, cout)
        for nm in keys(ra)
            # the grouped kernels run in dense mode only (dg < typemax)
            (nm === :ob || nm === :whg) && p.dg == typemax(Int32) && continue
            K[nm](ra[nm][2]...; threads = TPB, blocks = G[nm])
        end
    end
    g = CUDA.capture() do
        round!(LA, cA, LB, kB, cB)
        round!(LB, cB, LA, kA, cA)
    end
    return CUDA.instantiate(g)
end

"""
    grounds_csr(weights, ptr, tgt, alg; ...)

GPU rounds on a graph given as sorted CSR adjacency (1-based, no self loops).
"""
function grounds_csr(weights::AbstractVector, ptr::AbstractVector, tgt::AbstractVector, alg::RoundsAlg;
        stats = nothing, timing = nothing, poolfactor = 2.0, tcsr = 0.0, maxbatch::Int = 16,
        gputail::Bool = alg.tailalg === nothing, tailrelax::Real = 0.0, cpualpha::Real = 1.1e-9,
        urelax::Real = NaN, udelta::Real = NaN, fcap::Integer = 0, dgroup::Integer = DGROUP, ocap::Integer = 1 << 20,
        mode::Symbol = :replay, G::Integer = 0, nw::Integer = 0, prof::Bool = false)
    # persistent modes (GSmall.jl): the whole ordering in one kernel launch
    if mode !== :replay
        pc = mode === :auto ? pick_mode(length(ptr) - 1) : PCfg(mode, Int(G), Int(nw))
        if pc.kind !== :replay
            stats === nothing || @warn "round statistics are not logged in the persistent modes"
            return grounds_pcsr(weights, ptr, tgt, alg, pc; timing, tcsr, poolfactor, gputail, tailrelax, cpualpha, urelax, udelta,
                fcap, dgroup, ocap, prof)
        end
    end
    T0 = time()
    n = length(ptr) - 1
    n == 0 && return Int[], Int[]
    sc = alg.score === :md ? 0 : alg.score === :mf ? 1 : 2
    needdef = sc != 0
    tot = Int64(ptr[n + 1] - 1)
    # pool: poolfactor × the input, and up to 8× for small inputs (fewer repacks,
    # each of which costs a host sync and a graph re-capture)
    cap = max(Int64(1 << 16), round(Int64, poolfactor * tot), min(8tot, Int64(1) << 26))
    pool = CuVector{Int32}(undef, cap)
    copyto!(pool, 1, tgt isa Vector{Int32} ? tgt : convert(Vector{Int32}, tgt), 1, tot)
    astart = CuVector{Int64}(ptr[1:n])
    alen = CuVector{Int32}(undef, n)
    alenH = Vector{Int32}(undef, n)
    @inbounds for v in 1:n
        alenH[v] = ptr[v + 1] - ptr[v]
    end
    copyto!(alen, alenH)
    stt = CUDA.fill(ALIVE, n)
    w = CuVector{Int64}([max(1, round(Int, WSCALE * weights[v])) for v in 1:n])
    parent = CUDA.zeros(Int32, n)
    nm = CUDA.ones(Int64, n)        # original vertices per supervariable (for the fill counters)
    if alg.compress
        hk = CuVector{UInt64}(undef, n)
        @cuda threads = TPB blocks = nblk(n) k_hash!(hk, pool, astart, alen, Int32(n))
        perm = Int32.(sortperm(hk))
        hd = CuVector{Int32}(undef, n)
        @cuda threads = TPB blocks = nblk(n) k_runhead!(hd, perm, hk, Int32(n))
        accumulate!(max, hd, hd)
        par0 = CuVector{Int32}(undef, n)
        @cuda threads = TPB blocks = nblk(n) k_twins!(par0, perm, hd, pool, astart, alen, Int32(n))
        @cuda threads = TPB blocks = nblk(n) k_absorb0!(parent, stt, w, par0, Int32(n), nm)
        foreach(CUDA.unsafe_free!, (hk, hd, par0, perm))
    end
    wdeg = CuVector{Int64}(undef, n)
    @cuda threads = TPB blocks = nblk(n) k_wdeg!(wdeg, pool, astart, alen, stt, w, Int32(n))
    alist = CuVector{Int32}(undef, n)
    acount = CUDA.zeros(Int32, 1)
    @cuda threads = TPB blocks = nblk(n) k_alive!(alist, acount, stt, Int32(n))
    nal = Int(Array(acount)[1])
    def = CUDA.zeros(Int64, n)
    needdef && @cuda threads = TPB blocks = nblk(32nal) k_initdef!(def, alist, Int32(nal), pool, astart, alen, stt, w)
    kv = CuVector{Float32}(undef, n)
    @cuda threads = TPB blocks = nblk(n) k_initkey!(kv, stt, def, wdeg, w, Int32(sc), Int32(n))
    CUDA.synchronize()
    T1 = time()

    LC = LaunchCfg()
    kcap0 = max(1 << 16, 2 * Int(maximum(alenH; init = 0)))
    ctlH = zeros(Int64, NCTL)
    ctlH[C_NAL] = nal; ctlH[C_NA] = nal; ctlH[C_PTOP] = tot; ctlH[C_PCAP] = cap
    ctlH[C_KCAP] = kcap0; ctlH[C_BCAP] = kcap0
    ocap0 = Int(ocap)
    ctlH[C_OCAP] = ocap0
    ctlH[C_FH] = fcap == 0 ? Int64(typemax(UInt32)) : (Int64(typemax(UInt32)) * fcap) ÷ max(fcap, nal)
    logstats = stats !== nothing
    D = Dev(n, pool, astart, alen, stt, w, wdeg, def, parent, CuVector{Int32}(undef, n),
        alist, CuVector{Int32}(undef, n),
        CuVector{Int32}(undef, n), CuVector{Int32}(undef, n), CuVector{Float32}(undef, n), CuVector{Float32}(undef, n),
        CUDA.zeros(UInt32, n),
        CUDA.zeros(Int32, n), CUDA.zeros(Int32, n), CUDA.zeros(Int32, n), kv, CUDA.fill(typemax(UInt64), n),
        CuVector{Int32}(undef, n), CuVector{UInt64}(undef, n), CuVector{Int32}(undef, n),
        CuVector{Int64}(undef, n), CuVector{Int64}(undef, n), CuVector{Int32}(undef, n), CuVector{Int64}(undef, n),
        CuVector{Int32}(undef, kcap0), CuVector{Int32}(undef, kcap0), CuVector{Int64}(undef, kcap0),
        CuVector{UInt64}(undef, kcap0), CuVector{Int32}(undef, kcap0), CuVector{Int32}(undef, kcap0),
        CuVector{Int32}(undef, 4kcap0), CuVector{UInt32}(undef, kcap0),
        CuVector{Int64}(ctlH), CUDA.fill(typemax(UInt32), 2), CUDA.zeros(UInt32, NBIN),
        logstats ? CuVector{Int32}(undef, 3n) : CuVector{Int32}(undef, 3),
        CUDA.zeros(UInt8, n), nm, CuVector{Int64}(undef, n),
        CuVector{Int64}(undef, kcap0), CuVector{Int64}(undef, n),
        CUDA.zeros(Int64, n), CuVector{Int64}(undef, n), CUDA.zeros(UInt32, ocap0))
    p = (sc = Int32(sc), relax = Float32(alg.relax), delta = Float32(alg.delta * Rounds.scoreunit(alg.score)),
        tail = Int64(alg.tail), tailsel = Int64(alg.tailsel), tailmax = Int64(alg.tailmax),
        seed = UInt32(alg.seed % UInt32), ltarget = Int64(LC.wave ÷ 4), needdef = needdef,
        compress = alg.compress, logstats = logstats, tidx = alg.ties === :index, fcap = Int64(fcap),
        urelax = Float32(isnan(urelax) ? alg.relax : urelax),
        udelta = Float32((isnan(udelta) ? alg.delta : udelta) * Rounds.scoreunit(alg.score)),
        dg = typemax(Int32))
    K = compile_kernels(D, p)
    G = (scan = grid(LC, K.scan, nblk(32n)), reb1 = grid(LC, K.reb1, nblk(32n)), reb2 = grid(LC, K.reb2, nblk(32n)),
        cand = grid(LC, K.cand, nblk(32n)), luby2 = grid(LC, K.luby2, nblk(32n)), sizes = grid(LC, K.sizes, nblk(32n)),
        elim12 = grid(LC, K.elim12, nblk(32n)), elim34 = grid(LC, K.elim34, nblk(32n)),
        ob = grid(LC, K.ob, nblk(32n)), whg = grid(LC, K.whg, nblk(32n)),
        detect = alg.compress ? grid(LC, K.detect, n) : 1, apply = grid(LC, K.apply, n))
    cur = 1
    exec = capture_rounds(D, K, G, cur, p)
    nrepack = 0; nsync = 0; ncap = 1; ntail = 0; rdense = -1
    # grow the outside-mask arena of dense mode to what the last rounds asked for
    function growob!()
        ctlH[C_OWANT] <= ctlH[C_OCAP] && return false
        CUDA.unsafe_free!(D.ob)
        D.ob = CUDA.zeros(UInt32, 2 * ctlH[C_OWANT])
        ctlH[C_OCAP] = length(D.ob)
        return true
    end
    batch = 1
    T2a = time()
    tlast = T2a; rlast = 0; tround = 0.0
    while true
        for _ in 1:batch
            CUDA.launch(exec)
        end
        copyto!(ctlH, D.ctl)    # sync
        nsync += 1
        st = ctlH[C_STATUS]
        tnow = time(); nr = ctlH[C_NROUND]
        if nr >= rlast + 16      # round time over the last >= 16 rounds
            tround = (tnow - tlast) / (nr - rlast)
            tlast = tnow; rlast = nr
            if st == 0 && ntail > 0
                # in the GPU tail: hand the rest to the CPU once the serial dense
                # algorithm (~cpualpha * t^3 s on one core) is cheaper than
                # half of t more rounds at the measured round time
                tnal = ctlH[C_NAL]
                cpualpha * Float64(tnal)^3 < 0.5 * tnal * tround && break
            end
        end
        if st == 0
            # dense mode (grouped Wing-Huang updates for cliques with d >= dgroup) from
            # the first batch whose last round had such a clique; it stays on, since
            # cliques grow towards the end. Same elimination either way.
            dense = needdef && p.dg == typemax(Int32) && dgroup < typemax(Int32) && ctlH[C_DLAST] >= dgroup
            if dense
                p = merge(p, (dg = Int32(dgroup),)); rdense = Int(nr)
            end
            if growob!() || dense
                copyto!(D.ctl, ctlH)
                cur = Int(ctlH[C_CUR])
                exec = capture_rounds(D, K, G, cur, p)
                ncap += 1; batch = 1
                tlast = time(); rlast = ctlH[C_NROUND]
                continue
            end
            batch = min(2batch, maxbatch)
            continue
        end
        # dense end: stay on the GPU with the serial rule (candidates = the minimum)
        if st == TAIL && gputail
            p = merge(p, (relax = Float32(tailrelax), tailsel = Int64(0), urelax = min(p.urelax, Float32(tailrelax))))
            ntail = Int(ctlH[C_NAL])
            ctlH[C_STATUS] = 0
            copyto!(D.ctl, ctlH)
            cur = Int(ctlH[C_CUR])
            exec = capture_rounds(D, K, G, cur, p)
            ncap += 1; batch = 1
            tlast = time(); rlast = ctlH[C_NROUND]
            continue
        end
        st == NEED || break
        need = ctlH[C_NEED]
        if need & 1 != 0
            growk!(D, max(ctlH[C_KTOT], 2 * length(D.kar)))
            ctlH[C_KCAP] = length(D.kar)
        end
        if need & 2 != 0
            CUDA.unsafe_free!(D.bar)
            D.bar = CuVector{UInt32}(undef, max(ctlH[C_BTOT], 2 * length(D.bar)))
            ctlH[C_BCAP] = length(D.bar)
        end
        if need & 4 != 0
            # repack the live lists, growing the pool if they fill most of it
            na = ctlH[C_NA]; pb = ctlH[C_PB]
            lv = CuVector{Int64}(undef, na)
            @cuda threads = TPB blocks = nblk(na) k_livelen!(lv, D.alist, D.stt, D.alen, Int32(na))
            off = similar(lv)
            accumulate!(+, off, lv)
            used = na > 0 ? Int64(Array(view(off, na:na))[1]) : Int64(0)
            off .-= lv
            newcap = max(length(D.pool), round(Int64, 2 * (used + pb)), min(2 * length(D.pool), Int64(1) << 26))
            newpool = CuVector{Int32}(undef, newcap)
            @cuda threads = TPB blocks = nblk(32na) k_repack!(newpool, D.astart, D.pool, D.astart, D.alen, D.alist, D.stt, Int32(na), off)
            CUDA.unsafe_free!(D.pool); CUDA.unsafe_free!(lv); CUDA.unsafe_free!(off)
            D.pool = newpool
            ctlH[C_PTOP] = used; ctlH[C_PCAP] = newcap
            nrepack += 1
        end
        growob!()
        ctlH[C_STATUS] = 0; ctlH[C_NEED] = 0
        copyto!(D.ctl, ctlH)
        cur = Int(ctlH[C_CUR])
        K = compile_kernels(D, p)
        exec = capture_rounds(D, K, G, cur, p)
        ncap += 1
        batch = 1
        tlast = time(); rlast = ctlH[C_NROUND]
    end
    T2 = time()
    nal = Int(ctlH[C_NAL]); nord = Int(ctlH[C_NORD])
    fnnz = ctlH[C_NNZ]; fflop = ctlH[C_FLOP]; ftw = ctlH[C_TW]
    # back to the CPU: order so far, absorption parents, remaining graph
    final = Array(view(D.order, 1:nord))
    parentH = Array(D.parent)
    if nal > 0
        sttH = Array(D.stt)
        tvH = Int32[v for v in 1:n if sttH[v] == ALIVE]
        @assert length(tvH) == nal
        tv = CuVector{Int32}(tvH)
        loc = CUDA.zeros(Int32, n)
        @cuda threads = TPB blocks = nblk(nal) k_loc!(loc, tv, Int32(nal))
        tc = CuVector{Int64}(undef, nal)
        @cuda threads = TPB blocks = nblk(nal) k_tailcnt!(tc, tv, loc, D.pool, D.astart, D.alen, Int32(nal))
        toff = similar(tc)
        accumulate!(+, toff, tc)
        tcH = Array(tc); m2 = sum(tcH)
        toff .-= tc
        tadj = CuVector{Int32}(undef, max(1, m2))
        @cuda threads = TPB blocks = nblk(nal) k_tailfill!(tadj, toff, tv, loc, D.pool, D.astart, D.alen, Int32(nal))
        tadjH = Array(tadj); toffH = Array(toff); wH = Array(D.w)
        fadj = Vector{Vector{Int}}(undef, nal)
        for i in 1:nal
            seg = tadjH[(toffH[i] + 1):(toffH[i] + tcH[i])]
            fadj[i] = sort!(Vector{Int}(seg))
        end
        sub = SimpleGraph(m2 ÷ 2, fadj)
        wsub = [wH[v] / WSCALE for v in tvH]
        so, _ = permutation(wsub, sub, serialalg(alg))
        for i in so
            push!(final, tvH[i])
        end
        # fill of the CPU tail
        nmH = Array(D.nm)
        qs = [nmH[v] for v in tvH]
        cw = tailcw(qs, sub, so)
        for (k, i) in enumerate(so)
            q = qs[i]; c = cw[k] - q
            fnnz += q * c + (q * (q + 1)) >> 1
            fflop += q * c * c + c * q * (q + 1) + (q * (q + 1) * (2q + 1)) ÷ 6
            ftw = max(ftw, c + q)
        end
        foreach(CUDA.unsafe_free!, (tv, loc, tc, toff, tadj))
    end
    if logstats
        nr = Int(ctlH[C_NROUND])
        sl = Array(view(D.slog, 1:min(3nr, length(D.slog))))
        for r in 1:(length(sl) ÷ 3)
            push!(stats, (Int(sl[3r - 2]), Int(sl[3r - 1]), Int(sl[3r])))
        end
    end
    out = expand(final, parentH, n)
    for f in fieldnames(Dev)
        a = getfield(D, f)
        a isa CuArray && CUDA.unsafe_free!(a)
    end
    T3 = time()
    if timing !== nothing
        timing[:pre] = T1 - T0; timing[:csr] = tcsr; timing[:gpu] = T2 - T1; timing[:cpu_post] = T3 - T2
        timing[:setup] = T2a - T1
        timing[:rounds] = Int(ctlH[C_NROUND]); timing[:repack] = nrepack; timing[:tail] = nal
        timing[:rebuild] = Int(ctlH[C_NREB]); timing[:syncs] = nsync; timing[:captures] = ncap
        timing[:lmax] = Int(ctlH[C_LMAX]); timing[:gtail] = ntail; timing[:dense] = rdense
        timing[:nnz] = fnnz; timing[:flops] = fflop; timing[:tw] = ftw - 1
        timing[:mode] = "replay"
    end
    return out, invperm(out)
end

# weighted column counts (own weight + higher filled neighbours), by elimination position
function tailcw(weights::AbstractVector{W}, graph0, order::AbstractVector) where {W}
    CT = CliqueTrees
    graph = CT.BipartiteGraph(graph0)
    V = eltype(graph); E = CT.etype(graph); n = nv(graph); m = CT.half(CT.de(graph))
    index = invperm(order)
    lower, upper, tree, sets, wwork0, wwork1, vwork0, vwork1, vwork2, vwork3, vwork4, vwork5 = CT.bestwork(W, V, E, n, m)
    for v in 1:n
        wwork1[index[v]] = weights[v]
    end
    CT.sympermute!_impl!(upper, graph, convert(Vector{V}, index), CT.Forward)
    CT.reverse!_impl!(lower, upper)
    CT.etree_impl!(tree, vwork0, upper)
    CT.supcnt_impl!(wwork0, vwork0, vwork1, vwork2, vwork3, vwork4, vwork5, sets, wwork1, lower, tree)
    return W[wwork0[i] for i in 1:n]
end

# each vertex is followed by the vertices absorbed into it
function expand(final::Vector{Int32}, parentH::Vector{Int32}, n::Int)
    head = zeros(Int32, n); nxt = zeros(Int32, n)
    @inbounds for x in 1:n
        r = parentH[x]
        if r != 0
            nxt[x] = head[r]; head[r] = x
        end
    end
    out = Vector{Int}(undef, n); no = 0
    stack = Vector{Int32}(undef, n); sp = 0
    @inbounds for v in final
        stack[sp += 1] = v
        while sp > 0
            x = stack[sp]; sp -= 1
            out[no += 1] = x
            c = head[x]
            while c != 0
                stack[sp += 1] = c; c = nxt[c]
            end
        end
    end
    @assert no == n "emitted $no of $n"
    return out
end

"""
    grounds_ens(weights, graph, members; metric=:nnz, conc=length(members), timing=nothing)

Ensemble of independent GPU-round runs. `members` is a vector of `(alg, kw)` pairs
(`alg::RoundsAlg`, `kw` a NamedTuple of `grounds_csr` keywords such as `urelax`).
At most `conc` members run at a time, each in its own task (and so its own CUDA
stream), so their round loops overlap on the device. Every member counts its
fill on the device; the order with the smallest `metric` (`:nnz`, `:flops` or
`:tw`; ties to the earlier member) is returned as `(order, index, info)`, where
`info` holds the per-member metrics and the index of the winner.
"""
function grounds_ens(weights::AbstractVector, graph::AbstractGraph, members::AbstractVector;
        metric::Symbol = :nnz, conc::Int = length(members), timing = nothing)
    T0 = time()
    ptr, tgt = csr(graph)
    tcsr = time() - T0
    res = Vector{Any}(undef, length(members))
    sem = Base.Semaphore(max(1, conc))
    @sync for (i, (alg, kw)) in enumerate(members)
        @async Base.acquire(sem) do
            tm = Dict{Symbol, Any}()
            o, _ = grounds_csr(weights, ptr, tgt, alg; timing = tm, kw...)
            res[i] = (o, tm)
        end
    end
    best = argmin(i -> (res[i][2][metric], i), eachindex(res))
    info = (best = best, nnz = [r[2][:nnz] for r in res], flops = [r[2][:flops] for r in res],
        tw = [r[2][:tw] for r in res], rounds = [r[2][:rounds] for r in res])
    if timing !== nothing
        timing[:total] = time() - T0; timing[:csr] = tcsr
    end
    o = res[best][1]
    return o, invperm(o), info
end

"""
    GPUHybrid(alg::RoundsAlg, small; nmin=20000)

GPU rounds for graphs with at least `nmin` vertices, the elimination
algorithm `small` (on the CPU) for the others; for use inside ND or BestWidth.
"""
struct GPUHybrid{A <: CliqueTrees.EliminationAlgorithm} <: CliqueTrees.EliminationAlgorithm
    alg::RoundsAlg
    small::A
    nmin::Int
end

GPUHybrid(alg::RoundsAlg, small; nmin::Integer = 20000) = GPUHybrid(alg, small, Int(nmin))

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::GPUHybrid; verbose::Bool = false) where {V}
    if nv(graph) < alg.nmin
        return permutation(weights, graph, alg.small)
    end
    order, index = grounds(weights, graph, alg.alg)
    return convert(Vector{V}, order), convert(Vector{V}, index)
end

include("GSmall.jl")

end # module
