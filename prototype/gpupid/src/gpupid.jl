# GPU PID-BT: level-synchronous positive-instance-driven Bouchitté–Todinca on bitsets.
# Same rules and rounds as src/pidref.jl (the CPU reference); see that file for the algorithm.
# Every kernel is a grid-stride loop with a launch size from CUDA's occupancy API, so nothing is
# tuned to a particular card.

using CUDA, Graphs, CliqueTrees
include("pidref.jl")       # bitsets, PGraph, CPU reference (used for checks and extraction)
include("devfuns.jl")

# ------------------------------------------------------------------ launch helpers
function launch!(f, n::Integer, args...; shmem = 0)
    n <= 0 && return
    k = @cuda launch = false f(args...)
    cfg = launch_configuration(k.fun; shmem)
    threads = cfg.threads
    blocks = min(cld(n, threads), cfg.blocks * 4)
    if DEBUG_SYNC[]
        t0 = time(); LASTK[] = string(f)
        k(args...; threads, blocks, shmem)
        CUDA.synchronize()
    else
        k(args...; threads, blocks, shmem)
    end
    return
end
const DEBUG_SYNC = Ref(get(ENV, "GPUPID_SYNC", "0") == "1")
const LASTK = Ref("")
@inline gtid() = (blockIdx().x - 1) * blockDim().x + threadIdx().x
@inline gstride() = Int(gridDim().x) * Int(blockDim().x)

# positive Float64 <-> ordered UInt64, for atomic minimum of rejected weights
@inline f2u(x::Float64) = reinterpret(UInt64, x)
@inline u2f(x::UInt64) = reinterpret(Float64, x)

@inline function push!_(buf, ctr, cap, val)
    i = CUDA.atomic_add!(pointer(ctr), Int64(1)) + Int64(1)
    if i <= cap
        @inbounds buf[i] = val
    end
    return i
end

# ------------------------------------------------------------------ device hash set
# keys: device vector of sets; tab: Int32 slots, 0 = empty, s > 0 = keys[s], s < 0 = candidate -s
mutable struct DSet{W}
    keys::CuVector{BS{W}}
    n::Int
    tab::CuVector{Int32}
end
const INITCAP = Ref(parse(Int, get(ENV, "GPUPID_INITCAP", "1024")))
DSet{W}(cap::Int = INITCAP[]) where {W} = DSet{W}(CuVector{BS{W}}(undef, cap), 0, CUDA.zeros(Int32, nextpow(2, 4 * cap)))

function k_hins!(tab, keys, cand, m, isnew, slot)
    mask = UInt64(length(tab) - 1)
    i = gtid()
    while i <= m
        @inbounds key = cand[i]
        h = bs_hash(key) & mask
        new = Int32(0); sl = Int64(0)
        while true
            p = pointer(tab, Int(h) + 1)
            old = CUDA.atomic_cas!(p, Int32(0), Int32(-i))
            if old == Int32(0)
                new = Int32(1); sl = Int64(h + 1); break
            end
            other = old > 0 ? (@inbounds keys[old]) : (@inbounds cand[-old])
            other === key && break
            h = (h + one(UInt64)) & mask
        end
        @inbounds isnew[i] = new; @inbounds slot[i] = sl
        i += gstride()
    end
    return
end
function k_hput!(tab, keys, cand, m, isnew, rank, slot, base, pos)
    i = gtid()
    while i <= m
        @inbounds if isnew[i] == 1
            j = base + rank[i]
            keys[j] = cand[i]; tab[slot[i]] = Int32(j); pos[i] = Int32(j)
        else
            pos[i] = Int32(0)
        end
        i += gstride()
    end
    return
end
function k_rehash!(tab, keys, n)
    mask = UInt64(length(tab) - 1)
    i = gtid()
    while i <= n
        @inbounds h = bs_hash(keys[i]) & mask
        while CUDA.atomic_cas!(pointer(tab, Int(h) + 1), Int32(0), Int32(i)) != Int32(0)
            h = (h + one(UInt64)) & mask
        end
        i += gstride()
    end
    return
end
function reserve!(s::DSet{W}, extra::Int) where {W}
    need = s.n + extra
    if need > length(s.keys)
        cap = max(2 * length(s.keys), need)
        nk = CuVector{BS{W}}(undef, cap)
        s.n > 0 && copyto!(nk, 1, s.keys, 1, s.n)
        s.keys = nk
    end
    if 2 * need > length(s.tab)
        s.tab = CUDA.zeros(Int32, nextpow(2, 4 * need))
        launch!(k_rehash!, s.n, s.tab, s.keys, s.n)
    end
    return
end
function clear!(s::DSet)
    s.n = 0; fill!(s.tab, Int32(0)); return s
end
# Insert m candidates; returns (pos, nnew): pos[i] = new key index if candidate i was new, else 0.
function dinsert!(s::DSet{W}, cand::CuVector{BS{W}}, m::Int) where {W}
    m == 0 && return CUDA.zeros(Int32, 0), 0
    reserve!(s, m)
    isnew = CuVector{Int32}(undef, m); slot = CuVector{Int64}(undef, m)
    launch!(k_hins!, m, s.tab, s.keys, cand, m, isnew, slot)
    rank = cumsum(isnew) .- isnew            # exclusive prefix
    nnew = Int(sum(isnew))
    pos = CuVector{Int32}(undef, m)
    launch!(k_hput!, m, s.tab, s.keys, cand, m, isnew, rank, slot, Int32(s.n + 1), pos)
    s.n += nnew
    return pos, nnew
end
function k_hfind(tab, keys, key)
    mask = UInt64(length(tab) - 1)
    h = bs_hash(key) & mask
    while true
        @inbounds s = tab[Int(h) + 1]
        s == Int32(0) && return false
        s > 0 && (@inbounds keys[s]) === key && return true
        h = (h + one(UInt64)) & mask
    end
end

# ------------------------------------------------------------------ append buffer
mutable struct DBuf{T}
    data::CuVector{T}
    ctr::CuVector{Int64}
end
DBuf{T}(cap::Int = INITCAP[]) where {T} = DBuf{T}(CuVector{T}(undef, cap), CUDA.zeros(Int64, 1))
reset!(b::DBuf) = (fill!(b.ctr, 0); b)
bcount(b::DBuf) = Int(Array(b.ctr)[1])
# Run f(data, ctr, cap) until the output fits.
function fill_buf!(f, b::DBuf{T}) where {T}
    c, _ = fill_buf_cap!(f, b, typemax(Int))
    return c
end
# The same, but the buffer grows to at most maxcap entries: returns (count, fits).
function fill_buf_cap!(f, b::DBuf{T}, maxcap::Int) where {T}
    while true
        reset!(b)
        f(b.data, b.ctr, length(b.data))
        c = bcount(b)
        c <= length(b.data) && return c, true
        c > maxcap && return c, false
        b.data = CuVector{T}(undef, min(maxcap, nextpow(2, c + c ÷ 2)))
    end
end

# ------------------------------------------------------------------ kernels
@inline okw(x, K, tol) = x <= K + tol
@inline function rejmin!(rej, x)
    CUDA.atomic_min!(pointer(rej), f2u(x)); return
end

# R0: X = N[v]
function k_r0!(out, ctr, cap, adj, n, masks, wcls, ncls, K, tol, rej)
    v = gtid()
    while v <= n
        @inbounds X = bs_or(adj[v], bs_bit(bsval(adj[v]), v))
        x = bs_weight(X, masks, wcls, ncls)
        okw(x, K, tol) ? push!_(out, ctr, cap, X) : rejmin!(rej, x)
        v += gstride()
    end
    return
end

# Classify unique X candidates: 1 -> O-block candidate (outbound full comp A, weight test),
# 2 -> PMC candidate. pmc_only: never an O-block (rule R3, R0).
function k_classify!(oA, oN, octr, ocap, pout, pctr, pcap, big, bctr, X, m, adj, all, masks, wcls, ncls, K, tol, rej, scr, pmc_only)
    i = gtid(); st = gstride(); sbase = gtid()
    while i <= m
        @inbounds x = X[i]
        kind, A = dv_classify(adj, all, x, scr, sbase, st, pmc_only)
        if kind == Int32(1)
            wv = bs_weight(x, masks, wcls, ncls) + bs_wmin(A, masks, wcls, ncls)
            if okw(wv, K, tol)
                j = push!_(oA, octr, ocap, A)
                j <= ocap && (@inbounds oN[j] = x)
            else
                rejmin!(rej, wv)
            end
        elseif kind == Int32(2)
            push!_(pout, pctr, pcap, x)
        elseif kind == Int32(3)
            push!_(big, bctr, length(big), x)
        end
        i += st
    end
    return
end

# R1: for each new I-block (C, N(C)): the outbound full component of N(C).
function k_r1!(oA, oN, octr, ocap, IN, lo, hi, adj, all, masks, wcls, ncls, K, tol, rej)
    i = gtid() + lo - 1
    while i <= hi
        @inbounds S = IN[i]
        A = dv_full(adj, all, S)
        if !bs_isempty(A)
            wv = bs_weight(S, masks, wcls, ncls) + bs_wmin(A, masks, wcls, ncls)
            if okw(wv, K, tol)
                j = push!_(oA, octr, ocap, A)
                j <= ocap && (@inbounds oN[j] = S)
            else
                rejmin!(rej, wv)
            end
        end
        i += gstride()
    end
    return
end

# R2 join: I-blocks ilo:ihi × O-blocks olo:ohi. Each thread holds one O-block; the I-blocks
# stream through shared memory in tiles. Pair passes iff C ⊆ A and w(N(A)) + w(N(C) ∩ A) <= K.
function k_join!(out, ctr, cap, IC, IN, ilo, ihi, OA, ON, Ow, olo, ohi, masks, wcls, ncls, K, tol, rej, ::Val{T}) where {T}
    BT = eltype(IC)
    sC = CuStaticSharedArray(BT, T)
    sN = CuStaticSharedArray(BT, T)
    tid = threadIdx().x
    b = (blockIdx().x - 1) * blockDim().x + tid + olo - 1
    valid = b <= ohi
    A = valid ? (@inbounds OA[b]) : IC[ilo]
    NA = valid ? (@inbounds ON[b]) : IC[ilo]
    wA = valid ? (@inbounds Ow[b]) : 0.0
    best = Inf
    t0 = ilo + (blockIdx().y - 1) * T
    while t0 <= ihi
        nt = min(T, ihi - t0 + 1)
        sync_threads()
        if tid <= nt
            @inbounds sC[tid] = IC[t0 + tid - 1]; @inbounds sN[tid] = IN[t0 + tid - 1]
        end
        sync_threads()
        if valid
            for j in 1:nt
                @inbounds C = sC[j]
                if bs_subset(C, A)
                    @inbounds NC = sN[j]
                    x = wA + bs_weight(bs_and(NC, A), masks, wcls, ncls)
                    if okw(x, K, tol)
                        push!_(out, ctr, cap, bs_or(NA, NC))
                    else
                        best = min(best, x)
                    end
                end
            end
        end
        t0 += T * gridDim().y
    end
    best < Inf && rejmin!(rej, best)
    return
end

# R3: pairs (O-block b, v ∈ N(A_b)), flattened by an exclusive prefix count off[b].
function k_r3!(out, ctr, cap, OA, ON, lo, hi, off, total, adj, masks, wcls, ncls, K, tol, rej)
    t = gtid()
    while t <= total
        # binary search for the O-block of pair t
        l = lo; h = hi
        while l < h
            mid = (l + h + 1) >> 1
            @inbounds off[mid - lo + 1] < t ? (l = mid) : (h = mid - 1)
        end
        b = l
        @inbounds S = ON[b]; @inbounds A = OA[b]
        r = t - (@inbounds off[b - lo + 1])           # r-th vertex of S (1-based)
        v = 0; cnt = 0
        for i in 1:length(S)
            c = count_ones(S[i])
            if cnt + c >= r
                word = S[i]
                for _ in 1:(r - cnt - 1)
                    word &= word - one(UInt64)
                end
                v = 64 * (i - 1) + trailing_zeros(word) + 1
                break
            end
            cnt += c
        end
        X = bs_or(S, bs_and(@inbounds(adj[v]), A))
        x = bs_weight(X, masks, wcls, ncls)
        okw(x, K, tol) ? push!_(out, ctr, cap, X) : rejmin!(rej, x)
        t += gstride()
    end
    return
end

# R4: pending PMCs. The first missing support component (blocker) is checked first.
function k_r4!(icC, icN, icP, ictr, icap, still, sctr, root, pend, np, P, blk, adj, all, Itab, Ikeys)
    i = gtid()
    while i <= np
        @inbounds p = pend[i]
        @inbounds Ω = P[p]
        @inbounds bk = blk[p]
        go = bs_isempty(bk) || k_hfind(Itab, Ikeys, bk)
        if go
            outlet, has = dv_outlet(adj, all, Ω)
            rest = bs_andnot(all, Ω); crib = bs_andnot(Ω, outlet); miss = false
            while !bs_isempty(rest)
                v = bs_min(rest)
                D = bs_comp(adj, rest, v)
                rest = bs_andnot(rest, D)
                N = bs_open(adj, D)
                if !has || !bs_subset(N, outlet)
                    if !k_hfind(Itab, Ikeys, D)
                        miss = true; @inbounds blk[p] = D; break
                    end
                    crib = bs_or(crib, D)
                end
            end
            if miss
                go = false
            elseif !has
                CUDA.atomic_cas!(pointer(root), Int32(0), Int32(p))
            else
                j = push!_(icC, ictr, icap, crib)
                if j <= icap
                    @inbounds icN[j] = outlet; @inbounds icP[j] = Int32(p)
                end
            end
        end
        go || push!_(still, sctr, length(still), p)
        i += gstride()
    end
    return
end

function k_weights!(w, S, lo, hi, masks, wcls, ncls)
    i = gtid() + lo - 1
    while i <= hi
        @inbounds w[i] = bs_weight(S[i], masks, wcls, ncls)
        i += gstride()
    end
    return
end
function k_gather!(dst, src, idx, m)
    i = gtid()
    while i <= m
        @inbounds j = idx[i]
        j > 0 && (@inbounds dst[j] = src[i])
        i += gstride()
    end
    return
end
function k_newlist!(out, pos, m, ctr)   # compact the new positions
    i = gtid()
    while i <= m
        @inbounds p = pos[i]
        if p > 0
            k = CUDA.atomic_add!(pointer(ctr), Int32(1)) + Int32(1)
            @inbounds out[k] = p
        end
        i += gstride()
    end
    return
end
function k_popc!(off, ON, lo, hi)
    i = gtid() + lo - 1
    while i <= hi
        @inbounds off[i - lo + 1] = Int32(bs_count(ON[i]))
        i += gstride()
    end
    return
end

include("sortjoin.jl")

function launch_join!(f, jargs, ws, nti, nyo)
    k = @cuda launch = false f(jargs...)
    cfg = launch_configuration(k.fun)
    thr = max(ws, (min(cfg.threads, 256) ÷ ws) * ws); nwb = thr ÷ ws
    gx = cld(nti, nwb); gy = clamp(cld(cfg.blocks * 8, gx), 1, max(1, nyo))
    k(jargs...; threads = thr, blocks = (gx, gy))
    return
end

# ------------------------------------------------------------------ growable vectors
mutable struct DVec{T}
    v::CuVector{T}
    n::Int
end
DVec{T}(cap = INITCAP[]) where {T} = DVec{T}(CuVector{T}(undef, cap), 0)
function ensure!(d::DVec{T}, cap::Int) where {T}
    if cap > length(d.v)
        nv = CuVector{T}(undef, max(cap, 2 * length(d.v)))
        d.n > 0 && copyto!(nv, 1, d.v, 1, d.n)
        d.v = nv
    end
    return d
end

# ------------------------------------------------------------------ the decision
mutable struct GPUGraph{W}
    pg::PGraph{W}
    adj::CuVector{BS{W}}
    masks::CuVector{BS{W}}
    wcls::CuVector{Float64}
    ncls::Int32
end
GPUGraph(pg::PGraph{W}) where {W} = GPUGraph{W}(pg, CuVector(pg.adj), CuVector(pg.masks), CuVector(pg.wcls), Int32(length(pg.wcls)))

struct GPUResult{W}
    yes::Bool
    nextK::Float64
    root::Int
    I::Vector{BS{W}}; Ibp::Vector{Int}
    P::Vector{BS{W}}
    stats::Dict{Symbol, Any}
end

function gpu_decide(gg::GPUGraph{W}, K::Float64; tol = 1e-9, verbose = false, tile = Val(64), deadline = Inf, dump = false, join = :pruned3, TO = 8, xdedup = :round) where {W}
    pg = gg.pg; n = pg.n; adj = gg.adj; all = pg.all
    masks = gg.masks; wcls = gg.wcls; ncls = gg.ncls
    rej = CuVector{UInt64}([f2u(Inf)])
    Iset = DSet{W}(); IN = DVec{BS{W}}(); Ibp = DVec{Int32}()
    Oset = DSet{W}(); ON = DVec{BS{W}}(); Ow = DVec{Float64}()
    Pset = DSet{W}(); blk = DVec{BS{W}}()
    X2 = DSet{W}(); X3 = DSet{W}()
    bufX = DBuf{BS{W}}(); bufOA = DBuf{BS{W}}(); bufON = CuVector{BS{W}}(undef, INITCAP[])
    bufP = DBuf{BS{W}}(); bufBig = DBuf{BS{W}}(64)
    bufIC = DBuf{BS{W}}(); bufIN = CuVector{BS{W}}(undef, INITCAP[]); bufIP = CuVector{Int32}(undef, INITCAP[])
    pend = DVec{Int32}(); still = DBuf{Int32}()
    root = CUDA.zeros(Int32, 1)
    ws = CUDA.warpsize(CUDA.device())
    ItC = DVec{BS{W}}(); ItN = DVec{BS{W}}()
    OtAor = DVec{BS{W}}(); OtAand = DVec{BS{W}}(); OtNAor = DVec{BS{W}}(); OtW = DVec{Float64}()
    function isum!(oldn)
        nt = cld(Iset.n, ws); nt == 0 && return
        ensure!(ItC, nt); ensure!(ItN, nt); tlo = oldn ÷ ws + 1
        launch!(k_isum!, nt - tlo + 1, ItC.v, ItN.v, Iset.keys, IN.v, tlo, nt, Iset.n, ws)
        ItC.n = ItN.n = nt
    end
    function osum!(oldn)
        nt = cld(Oset.n, TO); nt == 0 && return
        ensure!(OtAor, nt); ensure!(OtAand, nt); ensure!(OtNAor, nt); ensure!(OtW, nt); tlo = oldn ÷ TO + 1
        launch!(k_osum!, nt - tlo + 1, OtAor.v, OtAand.v, OtNAor.v, OtW.v, Oset.keys, ON.v, Ow.v, tlo, nt, Oset.n, TO)
        OtAor.n = OtAand.n = OtNAor.n = OtW.n = nt
    end
    st = Dict{Symbol, Any}(:rounds => 0, :pairs => 0, :xcand2 => 0, :xcand3 => 0, :pmc_tests => 0, :t => Dict{Symbol, Float64}())
    tt = st[:t]
    tick(f, key) = (t0 = time(); r = f(); CUDA.synchronize(); tt[key] = get(tt, key, 0.0) + time() - t0; r)
    scr = CuVector{UInt64}(undef, 1)
    # the largest candidate buffer: a share of the free device memory (no device-specific constant)
    maxbuf = haskey(ENV, "GPUPID_MAXBUF") ? parse(Int, ENV["GPUPID_MAXBUF"]) : max(1 << 16, Int(floor(0.06 * CUDA.total_memory() / sizeof(BS{W}))))

    # classify a set of unique X's; O-block and PMC candidates go to bufOA/bufON and bufP (appended)
    function classify!(Xkeys, lo, m, pmc_only)
        m == 0 && return
        Xv = view(Xkeys, lo:(lo + m - 1))
        while true
            oc = bcount(bufOA); pc = bcount(bufP)
            # scratch for a full grid-stride launch
            k = @cuda launch = false k_classify!(bufOA.data, bufON, bufOA.ctr, length(bufOA.data), bufP.data, bufP.ctr, length(bufP.data),
                bufBig.data, bufBig.ctr, Xv, m, adj, all, masks, wcls, ncls, K, tol, rej, scr, pmc_only)
            cfg = launch_configuration(k.fun)
            blocks = min(cld(m, cfg.threads), cfg.blocks * 4); T = blocks * cfg.threads
            length(scr) < 256 * T && (scr = CuVector{UInt64}(undef, 256 * T))
            k(bufOA.data, bufON, bufOA.ctr, length(bufOA.data), bufP.data, bufP.ctr, length(bufP.data),
                bufBig.data, bufBig.ctr, Xv, m, adj, all, masks, wcls, ncls, K, tol, rej, scr, pmc_only; threads = cfg.threads, blocks)
            oc2 = bcount(bufOA); pc2 = bcount(bufP)
            if oc2 <= length(bufOA.data) && pc2 <= length(bufP.data)
                bcount(bufBig) > 0 && error("bag of more than 128 vertices: not supported")
                st[:pmc_tests] += m
                return
            end
            # overflow: grow and redo this batch from the counts before it
            growto = nextpow(2, 2 * max(oc2, pc2, 1024))
            nOA = CuVector{BS{W}}(undef, max(growto, length(bufOA.data))); copyto!(nOA, 1, bufOA.data, 1, oc); bufOA.data = nOA
            nON = CuVector{BS{W}}(undef, length(nOA)); copyto!(nON, 1, bufON, 1, oc); bufON = nON
            nP = CuVector{BS{W}}(undef, max(growto, length(bufP.data))); copyto!(nP, 1, bufP.data, 1, pc); bufP.data = nP
            copyto!(bufOA.ctr, Int64[oc]); copyto!(bufP.ctr, Int64[pc])
        end
    end
    # insert the PMC candidates in bufP; new PMCs join the pending list
    function flushP!()
        m = bcount(bufP); m == 0 && return 0
        pos, nnew = dinsert!(Pset, bufP.data, m)
        ensure!(blk, Pset.n); fill!(view(blk.v, (blk.n + 1):Pset.n), bs_zero(Val(W))); blk.n = Pset.n
        ensure!(pend, pend.n + nnew)
        c = CuVector{Int32}([pend.n]); launch!(k_newlist!, m, view(pend.v, 1:length(pend.v)), pos, m, c)
        # k_newlist! writes at pend.n + k
        pend.n += nnew; reset!(bufP)
        return nnew
    end
    function flushO!()
        m = bcount(bufOA); m == 0 && return 0
        idx = sortidx(bufON, m)
        cA = permute(bufOA.data, idx, m); cN = permute(bufON, idx, m)
        oldn = Oset.n
        pos, nnew = dinsert!(Oset, cA, m)
        ensure!(ON, Oset.n); ensure!(Ow, Oset.n)
        launch!(k_gather!, m, ON.v, cN, pos, m)
        lo = ON.n + 1; ON.n = Oset.n; Ow.n = Oset.n
        launch!(k_weights!, ON.n - lo + 1, Ow.v, ON.v, lo, ON.n, masks, wcls, ncls)
        osum!(oldn)
        reset!(bufOA)
        return nnew
    end
    function dedupX!(S::DSet, cnt)   # insert bufX into S, return range of new keys
        lo = S.n + 1
        dinsert!(S, bufX.data, cnt)
        return lo, S.n - lo + 1
    end

    # R0
    c = fill_buf!(bufX) do d, ctr, cap
        launch!(k_r0!, n, d, ctr, cap, adj, n, masks, wcls, ncls, K, tol, rej)
    end
    lo3, m3 = dedupX!(X3, c); classify!(X3.keys, lo3, m3, true); flushP!()

    nIold = 0; nOold = 0; rootp = 0
    while true
        time() > deadline && (st[:timeout] = true; break)
        # R4
        tick(:r4) do
            np = pend.n
            if np > 0
                still.data = length(still.data) < np ? CuVector{Int32}(undef, np) : still.data
                c = fill_buf!(bufIC) do d, ctr, cap
                    length(bufIN) < length(d) && (bufIN = CuVector{BS{W}}(undef, length(d)); bufIP = CuVector{Int32}(undef, length(d)))
                    reset!(still)
                    launch!(k_r4!, np, d, bufIN, bufIP, ctr, cap, still.data, still.ctr, root, pend.v, np, Pset.keys, blk.v, adj, all, Iset.tab, Iset.keys)
                end
                ns = bcount(still)
                copyto!(pend.v, 1, still.data, 1, ns); pend.n = ns
                rootp = Int(Array(root)[1])
                if rootp == 0 && c > 0
                    idx = sortidx(bufIC.data, c)
                    cC = permute(bufIC.data, idx, c); cN = permute(bufIN, idx, c); cP = permute(bufIP, idx, c)
                    oldn = Iset.n
                    pos, nnew = dinsert!(Iset, cC, c)
                    ensure!(IN, Iset.n); ensure!(Ibp, Iset.n)
                    launch!(k_gather!, c, IN.v, cN, pos, c); launch!(k_gather!, c, Ibp.v, cP, pos, c)
                    IN.n = Iset.n; Ibp.n = Iset.n
                    isum!(oldn)
                end
            end
        end
        rootp > 0 && break
        dIlo = nIold + 1; dIhi = Iset.n; dOlo = nOold + 1; dOhi = Oset.n
        (dIhi < dIlo && dOhi < dOlo) && break
        st[:rounds] += 1
        # R1
        tick(:r1) do
            if dIhi >= dIlo
                fill_buf!(bufOA) do d, ctr, cap
                    length(bufON) < length(d) && (bufON = CuVector{BS{W}}(undef, length(d)))
                    launch!(k_r1!, dIhi - dIlo + 1, d, bufON, ctr, cap, IN.v, dIlo, dIhi, adj, all, masks, wcls, ncls, K, tol, rej)
                end
            end
        end
        xdedup == :round && (clear!(X2); clear!(X3))
        # R2 join, in chunks whose candidates fit the buffer; each chunk is deduplicated and classified
        # at once (the O-block candidates of R1 stay in bufOA; classify appends to them)
        stack = [(ilo, ihi, olo, ohi) for (ilo, ihi, olo, ohi) in ((dIlo, dIhi, 1, dOhi), (1, nIold, dOlo, dOhi)) if ihi >= ilo && ohi >= olo]
        timedout = false
        while !isempty(stack)
            (ilo, ihi, olo, ohi) = pop!(stack)
            cX, fits = tick(:join) do
                fill_buf_cap!(bufX, maxbuf) do d, ctr, cap
                    jargs = (d, ctr, cap, Iset.keys, IN.v, ilo, ihi, ItC.v, ItN.v, Oset.keys, ON.v, Ow.v, olo, ohi,
                        OtAor.v, OtAand.v, OtNAor.v, OtW.v, masks, wcls, ncls, K, tol, rej, TO)
                    nti = (ihi - 1) ÷ ws - (ilo - 1) ÷ ws + 1; nto = (ohi - 1) ÷ TO - (olo - 1) ÷ TO + 1
                    launch_join!(k_join3!, jargs, ws, nti, cld(nto, ws))
                end
            end
            if !fits
                # split the larger side
                if ihi - ilo >= ohi - olo && ihi > ilo
                    mid = (ilo + ihi) ÷ 2; push!(stack, (ilo, mid, olo, ohi), (mid + 1, ihi, olo, ohi))
                elseif ohi > olo
                    mid = (olo + ohi) ÷ 2; push!(stack, (ilo, ihi, olo, mid), (ilo, ihi, mid + 1, ohi))
                else
                    error("a single pair overflows the candidate buffer")
                end
                continue
            end
            st[:pairs] += (ihi - ilo + 1) * (ohi - olo + 1); st[:xcand2] += cX
            tick(:classify2) do
                lo2, m2 = dedupX!(X2, cX)
                classify!(X2.keys, lo2, m2, false)
            end
            time() > deadline && (timedout = true; break)
        end
        timedout && (st[:timeout] = true; break)
        # R3 on ΔO, in chunks of O-blocks whose pairs (A, v ∈ N(A)) fit the buffer
        tick(:r3) do
            if dOhi >= dOlo
                nb = dOhi - dOlo + 1
                off = CuVector{Int32}(undef, nb)
                launch!(k_popc!, nb, off, ON.v, dOlo, dOhi)
                cnt = Array(off)
                lo = 1
                while lo <= nb
                    hi = lo; tot = Int(cnt[lo])
                    while hi < nb && tot + cnt[hi + 1] <= maxbuf
                        hi += 1; tot += cnt[hi]
                    end
                    c64 = CuVector{Int64}(Int64.(cnt[lo:hi])); offx = cumsum(c64) .- c64
                    blo = dOlo + lo - 1; bhi = dOlo + hi - 1
                    c3 = fill_buf!(bufX) do d, ctr, cap
                        launch!(k_r3!, tot, d, ctr, cap, Oset.keys, ON.v, blo, bhi, offx, tot, adj, masks, wcls, ncls, K, tol, rej)
                    end
                    st[:xcand3] += c3
                    lo3, m3 = dedupX!(X3, c3)
                    classify!(X3.keys, lo3, m3, true)
                    lo = hi + 1
                end
            end
        end
        nIold = dIhi; nOold = dOhi
        tick(:flush) do
            flushO!(); flushP!()
        end
        verbose && println("round $(st[:rounds]): |I|=$(Iset.n) |O|=$(Oset.n) |P|=$(Pset.n) pending=$(pend.n) pairs=$(st[:pairs])")
    end
    st[:I] = Iset.n; st[:O] = Oset.n; st[:P] = Pset.n
    nextK = u2f(Array(rej)[1])
    if dump
        st[:dump] = (; I = Array(view(Iset.keys, 1:Iset.n)), IN = Array(view(IN.v, 1:IN.n)), O = Array(view(Oset.keys, 1:Oset.n)), ON = Array(view(ON.v, 1:ON.n)), Ow = Array(view(Ow.v, 1:Ow.n)))
    end
    if rootp > 0
        Ih = Array(view(Iset.keys, 1:Iset.n)); Ibh = Int.(Array(view(Ibp.v, 1:Iset.n))); Ph = Array(view(Pset.keys, 1:Pset.n))
        return GPUResult{W}(true, nextK, rootp, Ih, Ibh, Ph, st)
    end
    return GPUResult{W}(false, nextK, 0, BS{W}[], Int[], BS{W}[], st)
end

function gpu_order(pg::PGraph{W}, r::GPUResult{W}) where {W}
    res = PIDResult{W}(true, r.nextK, r.root, r.I, r.Ibp, BS{W}[], r.P, Dict{Symbol, Int}())
    return pid_order(pg, res)
end

# Exact weighted treewidth. lb is a certified lower bound (start: lo); best is the width of the
# best ordering found (start: the width of `init`, or Inf). While best is infinite, K goes up from
# lb in steps of at least minstep (minstep = 0: K = lb, the classic upward search, where every NO
# certifies all widths below its nextK); once best is finite, K bisects [lb, best]. A YES lowers best
# to the width of the decomposition found, a NO raises lb to nextK(K). search = :bisect starts from
# `init` (required). Returns (width, lb, order, runs, stats, status); :found means width == lb.
function gpu_tw(pg::PGraph, lo::Float64; hi = Inf, verbose = false, tlimit = Inf, search = :up, init = nothing, graph = nothing, minstep = 0.0)
    gg = GPUGraph(pg); t0 = time(); runs = 0; allst = Any[]; dl = t0 + tlimit
    best = Inf; bestorder = Int[]
    if graph === nothing
        graph = SimpleGraph(pg.n)
        for v in 1:pg.n, u in 1:pg.n
            u > v && bs_has(pg.adj[v], u) && add_edge!(graph, v, u)
        end
    end
    if init !== nothing
        best = treewidth(pg.weights, graph, init); bestorder = collect(init)
    end
    lb = lo; Kprev = -Inf
    while true
        lb >= best - 1e-9 && return (; width = best, lb = best, order = bestorder, runs, stats = allst, status = :found)
        lb > hi && return (; width = best, lb, order = bestorder, runs, stats = allst, status = :above)
        K = isfinite(best) ? (lb + best) / 2 : max(lb, Kprev + minstep)
        K >= best - 1e-9 && (K = lb)
        runs += 1
        r = gpu_decide(gg, K; verbose, deadline = dl)
        push!(allst, (K, r.yes, r.stats))
        verbose && println("K=$K yes=$(r.yes) next=$(r.nextK) lb=$lb best=$best")
        get(r.stats, :timeout, false) && return (; width = best, lb, order = bestorder, runs, stats = allst, status = :timeout)
        if r.yes
            o = gpu_order(pg, r); wo = treewidth(pg.weights, graph, o)
            wo < best && (best = wo; bestorder = o)
        else
            isfinite(r.nextK) || return (; width = best, lb = Inf, order = bestorder, runs, stats = allst, status = :none)
            lb = max(lb, r.nextK)
        end
        Kprev = K
    end
end
# A width value to decide: here simply x (the decision is exact for any real K; NO runs return
# the exact next candidate).
floor_width(pg, x, lb) = x

# Certify: decide "width < ub" by one run at K just below ub. NO -> ub is optimal (lb = nextK >= ub);
# YES -> a decomposition of width < ub.
function gpu_certify(pg::PGraph, ub::Float64; tlimit = Inf, verbose = false, eps = 1e-6)
    gg = GPUGraph(pg); t0 = time()
    r = gpu_decide(gg, ub - eps; verbose, deadline = t0 + tlimit)
    get(r.stats, :timeout, false) && return (; status = :timeout, lb = NaN, order = Int[], stats = r.stats)
    r.yes && return (; status = :improved, lb = NaN, order = gpu_order(pg, r), stats = r.stats)
    return (; status = :optimal, lb = r.nextK, order = Int[], stats = r.stats)
end
