# GPU nested dissection with batched leaves (ND_NOTES.md).
#
# gnd_perm(ptr, adj; opts, leafsolve) orders a graph given as a sorted CSR
# without self loops (1-based ptr/adj on the host):
#   1. upload; compress closed twins on the device (hash + verify), giving a
#      weighted quotient graph (one vertex per twin class);
#   2. nested dissection on the quotient, one ND level at a time. All parts of
#      a level form one disjoint-union graph whose parts are contiguous vertex
#      ranges, so every step below is a few launches for all parts together:
#        - coarsening: locally dominant heavy-edge matching (edge priority =
#          (weight, symmetric hash)), contraction by member lists with per-row
#          sort/merge; a part stops coarsening at C0 vertices;
#        - initial separator on the coarsest graph: one warp per part, one
#          trial per lane (BFS-grown half, frontier = separator, then serial
#          node-FM passes with rollback, METIS style); best lane wins;
#        - uncoarsening: projection, then parallel separator refinement:
#          separator vertices move into one side (per part one direction per
#          step, so simultaneous moves never create an A-B edge) when the move
#          shrinks the separator; their neighbours on the other side join it;
#          a per-part atomic budget keeps the balance;
#        - split: separators get the last positions of their part's range;
#          children at most Lb (original vertices) become leaves, larger ones
#          form the next level's union graph (renumbered contiguously).
#   3. the leaves are cut out of the original graph on the device (one
#      union CSR), downloaded once, and ordered by `leafsolve` (e.g. one
#      grounds_batch call, MF1); their orders fill each leaf's range.
# Final positions: part ranges top-down (A, then B, then the separator).
module GND

using CUDA

export gnd_perm, GNDOpts

const TPB = 256
const FULL = 0xffffffff
const SA = 0x00
const SB = 0x01
const SS = 0x02
const SQ = 0x03     # initial separator: queued in the BFS
const LOCK = 0x04

@inline nblk(n) = max(cld(Int(n), TPB), 1)
@inline gtid() = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

Base.@kwdef struct GNDOpts
    Lb::Int = 1024           # leaf bound in original vertices
    C0::Int = 100            # coarsening stops at this many vertices per part
    eps::Float64 = 0.2       # balance: each side at most (1+eps)/2 of the part weight
    npass::Int = 0           # node-FM passes per lane trial in the initial separator (0: BFS trials only; the warp FM follows)
    rpass::Int = 8           # refinement passes per level
    nmatch::Int = 4          # matching sweeps per coarsening level
    stall::Float64 = 0.95    # stop coarsening when a level keeps this fraction of the vertices
    sepmax::Float64 = 0.5    # a part whose separator weighs more than this fraction is made a leaf
    maxlevel::Int = 48
    compress::Bool = true
    seed::UInt32 = 0x00000001
    lf::Int = 3              # FM log entries per vertex (per trial)
    ref::Symbol = :jet       # refinement: :jet (hill climbing with afterburner) or :greedy (positive gains only)
    jiter::Int = 12          # jet iterations per level
    jneg::Int = 1            # jet candidates: gain >= -jneg * w(v)
    jfilt::Int = 2           # jet afterburner rounds
    tfm::Int = 1 << 30       # warp node-FM at levels whose largest part has at most this many vertices (jet above)
    fmpass::Int = 10         # node-FM passes per level (METIS niter)
    fmlimit::Int = 150       # node-FM: limit = min(3|S|, fmlimit) (METIS: 300); a pass stops after 3 limit moves without improvement
    rmax::Int = 8            # independent separator trials (replicas) per part
    rlev::Int = 4            # replicas start after this many coarsening levels (METIS: 4) ...
    rmin::Int = 30           # ... or once the largest part has at most rmin * C0 vertices
    hop2::Bool = true        # 2-hop matching of unmatched vertices that share a neighbour
    hop2r::Int = 3           # 2-hop rounds (each pairs the two smallest candidates of every hub)
    fminc::Bool = true       # incremental gains in the node FM
    fmfine::Int = 2          # levels above the fmfine finest ones ...
    fmskip::Int = 2          # ... skip refinement for fmskip levels out of every fmfine + fmskip
    pregreedy::Int = 3       # parallel positive-gain passes before the node FM ...
    pgmin::Int = 20000       # ... at levels whose largest part has at least this many vertices
    pfmk::Int = 0            # parallel FM: warps per part (0/1: off) ...
    pfmmin::Int = 20000      # ... at levels whose largest part has at least this many vertices ...
    pfmmax::Int = 2048       # ... while parts x warps <= pfmmax
    qnd::Int = 0             # the first qnd ND levels refine like METIS (FM limit qlim at every coarse level)
    qlim::Int = 300
    cthr::Int = 256          # node FM: chunked argmax when the separator list is longer than this
    fm1::Bool = true         # 1-sided node FM (METIS ND default) instead of 2-sided
    mdeg::Bool = false       # matching: prefer edges with a small degree sum (METIS visits low degrees first)
    ptarget::Int = 0         # replicas while parts x replicas < ptarget (0: 4 x SM count)
    # CPU+GPU hybrid (hybfm.jl): 0 off; 1 cost model picks the uncoarsening level from which the CPU threads refine;
    # 2 always switch at the first refined level (testing)
    hyb::Int = 0
    cfmlimit::Int = 300      # CPU node FM limit (METIS); the CPU refines every level below the switch
    cfmpass::Int = 10
    hybmin::Int = 0          # hybrid only at ND levels whose largest part has at least this many vertices (0: any)
    # top-k variant (kway.jl): the first kway ND levels from one hierarchy (0: off)
    kway::Int = 0
    kwc0::Int = 100          # coarsest graph ~ kwc0 * 2^kway vertices
    kwtrials::Int = 8        # initial separator trials per tree node
end
GNDOpts(o::GNDOpts; kw...) = GNDOpts(; (f => getfield(o, f) for f in fieldnames(GNDOpts))..., kw...)

# ---------------------------------------------------------------- hashing
@inline function mix32(x::UInt32)
    x ⊻= x >> 16; x *= 0x7feb352d; x ⊻= x >> 15; x *= 0x846ca68b; x ⊻= x >> 16
    return x
end
@inline function ehash(u::Int32, v::Int32, seed::UInt32)
    a = reinterpret(UInt32, min(u, v)); b = reinterpret(UInt32, max(u, v))
    return mix32(mix32(a * 0x9e3779b1 + seed) ⊻ (b * 0x85ebca6b))
end
@inline function vmix(v::Int32)
    x = UInt64(reinterpret(UInt32, v)) * 0x9e3779b97f4a7c15
    x ⊻= x >> 29; x *= 0xbf58476d1ce4e5b9; x ⊻= x >> 32
    return x
end

@inline function warpsum32(x::Int32)
    x += shfl_xor_sync(FULL, x, 16); x += shfl_xor_sync(FULL, x, 8); x += shfl_xor_sync(FULL, x, 4)
    x += shfl_xor_sync(FULL, x, 2); x += shfl_xor_sync(FULL, x, 1)
    return x
end

# in-thread shell sort of keys ka[s:e] (with values kw if given)
@inline function shsort!(ka, kw, s::Int32, e::Int32)
    len = e - s + Int32(1)
    len <= Int32(1) && return
    gap = Int32(1)
    while gap < len ÷ Int32(3)
        gap = Int32(3) * gap + Int32(1)
    end
    @inbounds while gap >= Int32(1)
        i = s + gap
        while i <= e
            ki = ka[i]; wi = kw[i]; j = i
            while j - gap >= s && ka[j - gap] > ki
                ka[j] = ka[j - gap]; kw[j] = kw[j - gap]; j -= gap
            end
            ka[j] = ki; kw[j] = wi
            i += Int32(1)
        end
        gap = (gap - Int32(1)) ÷ Int32(3)
    end
    return
end
@inline function shsort1!(ka, s::Int32, e::Int32)
    len = e - s + Int32(1)
    len <= Int32(1) && return
    gap = Int32(1)
    while gap < len ÷ Int32(3)
        gap = Int32(3) * gap + Int32(1)
    end
    @inbounds while gap >= Int32(1)
        i = s + gap
        while i <= e
            ki = ka[i]; j = i
            while j - gap >= s && ka[j - gap] > ki
                ka[j] = ka[j - gap]; j -= gap
            end
            ka[j] = ki
            i += Int32(1)
        end
        gap = (gap - Int32(1)) ÷ Int32(3)
    end
    return
end

# ---------------------------------------------------------------- device graph
mutable struct DG
    n::Int
    m::Int                         # adjacency entries
    ptr::CuVector{Int32}           # n+1, ptr[1] = 1
    adj::CuVector{Int32}
    ew::CuVector{Int32}
    vw::CuVector{Int32}
    pid::CuVector{Int32}           # part of each vertex
end

function dfree!(g::DG)
    foreach(CUDA.unsafe_free!, (g.ptr, g.adj, g.ew, g.vw, g.pid))
end

# inclusive scan into a new array (one block, one launch, for arrays up to SCAN1 entries; else CUDA.jl's scan)
const SCAN1 = 1 << 17
function k_scan1!(y, x, n::Int32, off)
    t = threadIdx().x; nt = blockDim().x
    sh = CuStaticSharedArray(eltype(y), 1024)
    ch = (n + Int32(nt) - Int32(1)) ÷ Int32(nt)
    a = (Int32(t) - Int32(1)) * ch + Int32(1); b = min(a + ch - Int32(1), n)
    s = zero(eltype(y))
    @inbounds for i in a:b
        s += x[i]
    end
    @inbounds sh[t] = s
    sync_threads()
    d = 1
    while d < nt
        v = t > d ? @inbounds(sh[t - d]) : zero(eltype(y))
        sync_threads()
        @inbounds sh[t] += v
        sync_threads()
        d <<= 1
    end
    acc = (t == 1 ? zero(eltype(y)) : @inbounds(sh[t - 1])) + off
    @inbounds for i in a:b
        acc += x[i]; y[i] = acc
    end
    return
end
function iscan(x::CuVector{T}) where {T}
    y = similar(x)
    n = length(x)
    if 0 < n <= SCAN1
        @cuda threads = 1024 blocks = 1 k_scan1!(y, x, Int32(n), zero(T))
    elseif n > 0
        accumulate!(+, y, x)
    end
    return y
end
lastval(x::CuVector) = length(x) == 0 ? zero(eltype(x)) : Array(view(x, length(x):length(x)))[1]

# CSR pointer (1-based, n+1) from row counts
function k_one!(p)
    @inbounds p[1] = Int32(1)
    return
end
function ptr_from_counts(cnt::CuVector{Int32})
    n = length(cnt)
    p = CuVector{Int32}(undef, n + 1)
    if 0 < n <= SCAN1
        @cuda threads = 1024 blocks = 1 k_scan1!(view(p, 2:(n + 1)), cnt, Int32(n), Int32(1))
        @cuda threads = 1 blocks = 1 k_one!(p)
    else
        p[1:1] .= Int32(1)
        if n > 0
            s = iscan(cnt)
            p[2:end] .= s .+ Int32(1)
            CUDA.unsafe_free!(s)
        end
    end
    return p
end

# ---------------------------------------------------------------- twins
function k_thash!(h, ptr, adj, n::Int32)
    v = gtid(); v > n && return
    @inbounds begin
        s = vmix(v)
        p0 = ptr[v]; p1 = ptr[v + Int32(1)] - Int32(1)
        for q in p0:p1
            s += vmix(adj[q])
        end
        s += UInt64(p1 - p0 + Int32(1)) * 0x632be59bd9b4e019
        h[v] = s == 0 ? UInt64(1) : s
    end
    return
end

function k_tinsert!(keys, vals, h, n::Int32, mask::UInt64)
    v = gtid(); v > n && return
    @inbounds begin
        k = h[v]
        s = Int64(k & mask) + 1
        while true
            old = CUDA.atomic_cas!(pointer(keys, s), UInt64(0), k)
            if old == UInt64(0) || old == k
                CUDA.atomic_min!(pointer(vals, s), v)
                break
            end
            s = s == Int64(mask) + 1 ? Int64(1) : s + 1
        end
    end
    return
end

function k_tlookup!(rep, keys, vals, h, ptr, adj, n::Int32, mask::UInt64)
    v = gtid(); v > n && return
    @inbounds begin
        k = h[v]
        s = Int64(k & mask) + 1
        while keys[s] != k
            s = s == Int64(mask) + 1 ? Int64(1) : s + 1
        end
        r = vals[s]
        ok = r == v
        if !ok
            # closed twins: N(v) - r == N(r) - v (sorted lists)
            i = ptr[v]; ie = ptr[v + Int32(1)]; j = ptr[r]; je = ptr[r + Int32(1)]
            ok = (ie - i) == (je - j)
            hasr = false
            while ok
                while i < ie && adj[i] == r
                    i += Int32(1); hasr = true
                end
                while j < je && adj[j] == v
                    j += Int32(1)
                end
                if i >= ie || j >= je
                    ok = (i >= ie) && (j >= je)
                    break
                end
                if adj[i] != adj[j]
                    ok = false
                    break
                end
                i += Int32(1); j += Int32(1)
            end
            ok = ok && hasr
        end
        rep[v] = ok ? r : v
    end
    return
end

function k_isrep!(flag, rep, n::Int32)
    v = gtid(); v > n && return
    @inbounds flag[v] = rep[v] == v ? Int32(1) : Int32(0)
    return
end

function k_cmaprep!(cmap, cidx, rep, n::Int32)
    v = gtid(); v > n && return
    @inbounds cmap[v] = cidx[rep[v]]
    return
end

function k_csize!(csz, cmap, n::Int32)
    v = gtid(); v > n && return
    @inbounds CUDA.atomic_add!(pointer(csz, cmap[v]), Int32(1))
    return
end

function k_memscatter!(mem, slot, memptr, cmap, n::Int32)
    v = gtid(); v > n && return
    @inbounds begin
        c = cmap[v]
        k = CUDA.atomic_add!(pointer(slot, c), Int32(1))
        mem[memptr[c] + k] = v
    end
    return
end

function k_memsort!(mem, memptr, crank, nc::Int32)
    c = gtid(); c > nc && return
    @inbounds begin
        s = memptr[c]; e = memptr[c + Int32(1)] - Int32(1)
        shsort1!(mem, s, e)
        for i in s:e
            crank[mem[i]] = i - s
        end
    end
    return
end

# ---------------------------------------------------------------- contraction
function k_ccount!(cnt, memptr, mem, ptr, nc::Int32)
    c = gtid(); c > nc && return
    @inbounds begin
        t = Int32(0)
        for i in memptr[c]:(memptr[c + Int32(1)] - Int32(1))
            x = mem[i]
            t += ptr[x + Int32(1)] - ptr[x]
        end
        cnt[c] = t
    end
    return
end

function k_cfill!(tadj, tew, ndeg, tptr, memptr, mem, ptr, adj, ew, cmap, nc::Int32)
    c = gtid(); c > nc && return
    @inbounds begin
        s = tptr[c]; k = s
        for i in memptr[c]:(memptr[c + Int32(1)] - Int32(1))
            x = mem[i]
            for q in ptr[x]:(ptr[x + Int32(1)] - Int32(1))
                cu = cmap[adj[q]]
                if cu != c
                    tadj[k] = cu; tew[k] = ew[q]; k += Int32(1)
                end
            end
        end
        e = k - Int32(1)
        shsort!(tadj, tew, s, e)
        o = s - Int32(1)
        for i in s:e
            if o >= s && tadj[o] == tadj[i]
                tew[o] += tew[i]
            else
                o += Int32(1); tadj[o] = tadj[i]; tew[o] = tew[i]
            end
        end
        ndeg[c] = o - s + Int32(1)
    end
    return
end

function k_ccompact!(cadj, cew, cptr, tadj, tew, tptr, nc::Int32)
    c = gtid(); c > nc && return
    @inbounds begin
        s = tptr[c]; d = cptr[c]; l = cptr[c + Int32(1)] - d
        for i in Int32(0):(l - Int32(1))
            cadj[d + i] = tadj[s + i]; cew[d + i] = tew[s + i]
        end
    end
    return
end

function k_cvw!(cvw, cpid, memptr, mem, vw, pid, nc::Int32)
    c = gtid(); c > nc && return
    @inbounds begin
        t = Int32(0)
        s = memptr[c]
        for i in s:(memptr[c + Int32(1)] - Int32(1))
            t += vw[mem[i]]
        end
        cvw[c] = t; cpid[c] = pid[mem[s]]
    end
    return
end

# coarse graph from a vertex map (cmap: 1..nc) and member lists
# [cptr[end], largest part (vertices) of the coarse graph] in one download (saves a sync per coarsening step)
function k_mp2!(out, cptr, cpoff, np::Int32)
    t = threadIdx().x
    sh = CuStaticSharedArray(Int32, 256)
    m = Int32(0); p = Int32(t)
    @inbounds while p <= np
        m = max(m, cpoff[p + Int32(1)] - cpoff[p]); p += Int32(256)
    end
    @inbounds sh[t] = m
    sync_threads()
    d = 128
    while d >= 1
        t <= d && @inbounds(sh[t] = max(sh[t], sh[t + d]))
        sync_threads()
        d >>= 1
    end
    if t == 1
        @inbounds out[1] = cptr[length(cptr)]; @inbounds out[2] = sh[1]
    end
    return
end

function contract(g::DG, cmap::CuVector{Int32}, nc::Int, memptr::CuVector{Int32}, mem::CuVector{Int32}; cpoff = nothing, mpref = nothing)
    nc32 = Int32(nc)
    cnt = CuVector{Int32}(undef, nc)
    @cuda threads = TPB blocks = nblk(nc) k_ccount!(cnt, memptr, mem, g.ptr, nc32)
    tptr = ptr_from_counts(cnt)
    T = g.m                     # every vertex is a member of exactly one coarse vertex
    tadj = CuVector{Int32}(undef, max(T, 1)); tew = CuVector{Int32}(undef, max(T, 1))
    ndeg = cnt     # reuse
    @cuda threads = TPB blocks = nblk(nc) k_cfill!(tadj, tew, ndeg, tptr, memptr, mem, g.ptr, g.adj, g.ew, cmap, nc32)
    cptr = ptr_from_counts(ndeg)
    if cpoff === nothing
        M = Int(lastval(cptr)) - 1
    else
        o2 = CuVector{Int32}(undef, 2)
        @cuda threads = 256 blocks = 1 k_mp2!(o2, cptr, cpoff, Int32(length(cpoff) - 1))
        h2 = Array(o2); CUDA.unsafe_free!(o2)
        M = Int(h2[1]) - 1; mpref[] = Int(h2[2])
    end
    cadj = CuVector{Int32}(undef, max(M, 1)); cew = CuVector{Int32}(undef, max(M, 1))
    @cuda threads = TPB blocks = nblk(nc) k_ccompact!(cadj, cew, cptr, tadj, tew, tptr, nc32)
    cvw = CuVector{Int32}(undef, nc); cpid = CuVector{Int32}(undef, nc)
    @cuda threads = TPB blocks = nblk(nc) k_cvw!(cvw, cpid, memptr, mem, g.vw, g.pid, nc32)
    foreach(CUDA.unsafe_free!, (cnt, tptr, tadj, tew))
    return DG(nc, M, cptr, cadj, cew, cvw, cpid)
end

# ---------------------------------------------------------------- matching
function k_propose!(prop, mate, ptr, adj, ew, vw, pid, pmaxvw, n::Int32, seed::UInt32, mdeg::Bool)
    v = gtid(); v > n && return
    @inbounds begin
        if mate[v] != Int32(0)
            prop[v] = Int32(0)
            return
        end
        mx = pmaxvw[pid[v]]
        best = Int32(0); bk = UInt64(0)
        if mx > Int32(0)
            wv = vw[v]
            dv = ptr[v + Int32(1)] - ptr[v]
            for q in ptr[v]:(ptr[v + Int32(1)] - Int32(1))
                u = adj[q]
                mate[u] == Int32(0) || continue
                wv + vw[u] > mx && continue
                # edge priority: weight, then (mdeg) small degree sum, then a symmetric hash
                d = mdeg ? UInt64(0xffff - min(dv + ptr[u + Int32(1)] - ptr[u], Int32(0xffff))) : UInt64(0)
                k = (UInt64(reinterpret(UInt32, ew[q])) << 40) | (d << 24) | UInt64(ehash(u, v, seed) >> 8)
                if k > bk
                    best = u; bk = k
                end
            end
        end
        prop[v] = best
    end
    return
end

function k_accept!(mate, prop, n::Int32)
    v = gtid(); v > n && return
    @inbounds if mate[v] == Int32(0)
        u = prop[v]
        if u != Int32(0) && prop[u] == v
            mate[v] = u
        end
    end
    return
end


# 2-hop matching (METIS Match_2Hop): unmatched vertices that share their heaviest neighbour are paired
# (consecutive in (hub, vertex) order), so stars and pendant vertices keep coarsening.
function k_hub!(key, mate, ptr, adj, ew, vw, pid, pmaxvw, n::Int32, seed::UInt32)
    v = gtid(); v > n && return
    @inbounds begin
        key[v] = typemax(UInt64)
        (mate[v] == Int32(0) && pmaxvw[pid[v]] > Int32(0)) || return
        h = Int32(0); bk = UInt64(0)
        for q in ptr[v]:(ptr[v + Int32(1)] - Int32(1))
            u = adj[q]
            k = (UInt64(reinterpret(UInt32, ew[q])) << 32) | UInt64(ehash(u, v, seed))
            if k > bk
                bk = k; h = u
            end
        end
        h == Int32(0) && return
        key[v] = (UInt64(reinterpret(UInt32, h)) << 32) | UInt64(reinterpret(UInt32, v))
    end
    return
end

function k_runhead2!(hd, key, perm, n::Int32)
    i = gtid(); i > n && return
    @inbounds begin
        k = key[perm[i]]
        hd[i] = (i == Int32(1) || (key[perm[i - Int32(1)]] >> 32) != (k >> 32)) ? i : Int32(0)
    end
    return
end

function k_pair2!(mate, key, perm, rs, vw, pid, pmaxvw, n::Int32)
    i = gtid(); i >= n && return
    @inbounds begin
        v = perm[i]; k = key[v]
        k == typemax(UInt64) && return
        ((i - rs[i]) & Int32(1)) == Int32(0) || return
        u = perm[i + Int32(1)]; ku = key[u]
        (ku >> 32) == (k >> 32) || return
        vw[v] + vw[u] <= pmaxvw[pid[v]] || return
        mate[v] = u; mate[u] = v
    end
    return
end


# deterministic 2-hop pairing without a sort: per hub h, the two smallest unmatched vertices that chose h
# (atomicMin twice) are paired; repeated hop2r rounds.
function k_hubpick!(hub, mate, ptr, adj, ew, vw, pid, pmaxvw, n::Int32, seed::UInt32)
    v = gtid(); v > n && return
    @inbounds begin
        hub[v] = Int32(0)
        (mate[v] == Int32(0) && pmaxvw[pid[v]] > Int32(0)) || return
        h = Int32(0); bk = UInt64(0)
        for q in ptr[v]:(ptr[v + Int32(1)] - Int32(1))
            u = adj[q]
            k = (UInt64(reinterpret(UInt32, ew[q])) << 32) | UInt64(ehash(u, v, seed))
            if k > bk
                bk = k; h = u
            end
        end
        hub[v] = h
    end
    return
end
function k_hubmin1!(first, hub, mate, n::Int32)
    v = gtid(); v > n && return
    @inbounds begin
        h = hub[v]
        (h != Int32(0) && mate[v] == Int32(0)) && CUDA.atomic_min!(pointer(first, h), v)
    end
    return
end
function k_hubmin2!(second, first, hub, mate, n::Int32)
    v = gtid(); v > n && return
    @inbounds begin
        h = hub[v]
        (h != Int32(0) && mate[v] == Int32(0) && first[h] != v) && CUDA.atomic_min!(pointer(second, h), v)
    end
    return
end
function k_hubpair!(mate, first, second, hub, vw, pid, pmaxvw, n::Int32)
    v = gtid(); v > n && return
    @inbounds begin
        h = hub[v]
        (h != Int32(0) && mate[v] == Int32(0) && second[h] == v) || return
        f = first[h]
        if vw[v] + vw[f] <= pmaxvw[pid[v]]
            mate[v] = f; mate[f] = v
        end
        hub[v] = Int32(0); hub[f] = Int32(0)    # done with this pair (paired or not)
    end
    return
end

function k_finmatch!(mate, flag, n::Int32)
    v = gtid(); v > n && return
    @inbounds begin
        m = mate[v]
        if m == Int32(0)
            m = v; mate[v] = v
        end
        flag[v] = m >= v ? Int32(1) : Int32(0)
    end
    return
end

function k_pairmap!(cmap, csz, cidx, mate, n::Int32)
    v = gtid(); v > n && return
    @inbounds begin
        m = mate[v]
        r = min(v, m)
        c = cidx[r]
        cmap[v] = c
        r == v && (csz[c] = m == v ? Int32(1) : Int32(2))
    end
    return
end

function k_pairmem!(mem, memptr, cidx, mate, n::Int32)
    v = gtid(); v > n && return
    @inbounds begin
        m = mate[v]
        if m >= v
            p = memptr[cidx[v]]
            mem[p] = v
            m != v && (mem[p + Int32(1)] = m)
        end
    end
    return
end

function k_pmaxvw!(pmaxvw, poff, Wp, C0::Int32, np::Int32)
    p = gtid(); p > np && return
    @inbounds begin
        cnt = poff[p + Int32(1)] - poff[p]
        pmaxvw[p] = cnt <= C0 ? Int32(0) : max(Int32(2), Int32(cld(3 * Int64(Wp[p]), 2 * Int64(C0))))
    end
    return
end

function k_cpoff!(cpoff, cmap, poff, np::Int32, nc::Int32)
    p = gtid(); p > np + Int32(1) && return
    @inbounds cpoff[p] = p == np + Int32(1) ? nc + Int32(1) : cmap[poff[p]]
    return
end

# ---------------------------------------------------------------- initial separator
# one warp per part, one trial per lane; wh/lg are lane-interleaved scratch
@inline whi(v::Int32, lane::Int32) = (v - Int32(1)) * Int32(32) + lane + Int32(1)

function k_initsep!(label, ptr, adj, vw, poff, maxside, np::Int32, wh, lg, LF::Int32, npass::Int32, seed::UInt32)
    wid = (gtid() - Int32(1)) >> 5 + Int32(1)
    lane = (threadIdx().x - Int32(1)) & Int32(31)
    wid > np && return
    @inbounds begin
        a = poff[wid]; b = poff[wid + Int32(1)] - Int32(1); m = b - a + Int32(1)
        ms = maxside[wid]
        lbase = (a - Int32(1)) * LF
        cap = LF * m
        W = Int32(0)
        for v in a:b
            W += vw[v]; wh[whi(v, lane)] = SB
        end
        target = W ÷ Int32(2)
        # BFS-grown half A
        s0 = a + Int32(mix32(seed ⊻ (reinterpret(UInt32, wid) * 0x9e3779b1) ⊻ (reinterpret(UInt32, lane) * 0x85ebca6b + 0x632be5ab)) % reinterpret(UInt32, m))
        qh = Int32(1); qt = Int32(1)
        lg[whi(lbase + qt, lane)] = s0; wh[whi(s0, lane)] = SQ
        wA = Int32(0); nxt = a
        while true
            if qh > qt
                while nxt <= b && wh[whi(nxt, lane)] != SB
                    nxt += Int32(1)
                end
                nxt > b && break
                qt += Int32(1); lg[whi(lbase + qt, lane)] = nxt; wh[whi(nxt, lane)] = SQ
            end
            v = lg[whi(lbase + qh, lane)]; qh += Int32(1)
            wv = vw[v]
            (wA > Int32(0) && wA + wv > target) && break
            wh[whi(v, lane)] = SA; wA += wv
            for q in ptr[v]:(ptr[v + Int32(1)] - Int32(1))
                u = adj[q]
                if wh[whi(u, lane)] == SB
                    qt += Int32(1); lg[whi(lbase + qt, lane)] = u; wh[whi(u, lane)] = SQ
                end
            end
        end
        for v in a:b
            wh[whi(v, lane)] == SQ && (wh[whi(v, lane)] = SB)
        end
        for v in a:b
            if wh[whi(v, lane)] == SA
                for q in ptr[v]:(ptr[v + Int32(1)] - Int32(1))
                    u = adj[q]
                    wh[whi(u, lane)] == SB && (wh[whi(u, lane)] = SS)
                end
            end
        end
        wB = Int32(0); wS = Int32(0)
        for v in a:b
            t = wh[whi(v, lane)]
            t == SB && (wB += vw[v]); t == SS && (wS += vw[v])
        end
        limit = max(Int32(32), m ÷ Int32(8))
        # node FM passes (2-sided), rollback to the best prefix
        for pass in Int32(1):npass
            nlog = Int32(0); bestS = wS; bestD = abs(wA - wB); bestlog = Int32(0); since = Int32(0)
            startS = bestS; startD = bestD
            while true
                bv = Int32(0); bx = SA; bp = typemin(Int64)
                lightA = wA <= wB
                for v in a:b
                    wh[whi(v, lane)] == SS || continue
                    sA = Int32(0); sB = Int32(0)
                    for q in ptr[v]:(ptr[v + Int32(1)] - Int32(1))
                        u = adj[q]
                        t = wh[whi(u, lane)] & 0x03
                        t == SA && (sA += vw[u]); t == SB && (sB += vw[u])
                    end
                    wv = vw[v]
                    if wA + wv <= ms
                        p = Int64(wv - sB) * 2 + (lightA ? 1 : 0)
                        if p > bp
                            bp = p; bv = v; bx = SA
                        end
                    end
                    if wB + wv <= ms
                        p = Int64(wv - sA) * 2 + (lightA ? 0 : 1)
                        if p > bp
                            bp = p; bv = v; bx = SB
                        end
                    end
                end
                bv == Int32(0) && break
                nlog + Int32(1) + (ptr[bv + Int32(1)] - ptr[bv]) > cap && break
                Y = bx == SA ? SB : SA
                nlog += Int32(1); lg[whi(lbase + nlog, lane)] = ((bv - a) << 3) | Int32(SS)
                wh[whi(bv, lane)] = bx | LOCK
                wbv = vw[bv]
                wS -= wbv
                bx == SA ? (wA += wbv) : (wB += wbv)
                for q in ptr[bv]:(ptr[bv + Int32(1)] - Int32(1))
                    u = adj[q]
                    t = wh[whi(u, lane)]
                    if t & 0x03 == Y
                        nlog += Int32(1); lg[whi(lbase + nlog, lane)] = ((u - a) << 3) | Int32(t)
                        wh[whi(u, lane)] = SS
                        wu = vw[u]
                        wS += wu
                        Y == SA ? (wA -= wu) : (wB -= wu)
                    end
                end
                D = abs(wA - wB)
                if wS < bestS || (wS == bestS && D < bestD)
                    bestS = wS; bestD = D; bestlog = nlog; since = Int32(0)
                else
                    since += Int32(1)
                    since > limit && break
                end
            end
            i = nlog
            while i > bestlog
                e = lg[whi(lbase + i, lane)]
                wh[whi(a + (e >> 3), lane)] = UInt8(e & Int32(7))
                i -= Int32(1)
            end
            wA = Int32(0); wB = Int32(0); wS = Int32(0)
            for v in a:b
                t = wh[whi(v, lane)] & 0x03
                wh[whi(v, lane)] = t
                t == SA && (wA += vw[v]); t == SB && (wB += vw[v]); t == SS && (wS += vw[v])
            end
            (bestS < startS || (bestS == startS && bestD < startD)) || break
        end
        bad = max(wA, wB) > ms
        key = (UInt64(bad) << 63) | (UInt64(reinterpret(UInt32, wS)) << 31) | UInt64(reinterpret(UInt32, abs(wA - wB)))
    end
    k = key
    k = min(k, shfl_xor_sync(FULL, k, 16)); k = min(k, shfl_xor_sync(FULL, k, 8)); k = min(k, shfl_xor_sync(FULL, k, 4))
    k = min(k, shfl_xor_sync(FULL, k, 2)); k = min(k, shfl_xor_sync(FULL, k, 1))
    mask = vote_ballot_sync(FULL, key == k)
    win = Int32(trailing_zeros(mask))
    @inbounds begin
        v = a + lane
        while v <= b
            label[v] = wh[whi(v, win)] & 0x03
            v += Int32(32)
        end
    end
    return
end

# ---------------------------------------------------------------- refinement
function k_project!(lf, lc, cmap, n::Int32)
    v = gtid(); v > n && return
    @inbounds lf[v] = lc[cmap[v]]
    return
end

# per-part weights (and counts) of A, B, S: pw[6(p-1) + 1:3] weights, 4:6 counts
function k_pw!(pw, label, vw, pid, n::Int32)
    v = gtid()
    p = Int32(0); lab = 0x00; w = Int32(0)
    if v <= n
        @inbounds p = pid[v]; @inbounds lab = label[v]; @inbounds w = vw[v]
    end
    p0 = shfl_sync(FULL, p, 1)
    if vote_all_sync(FULL, p == p0)
        if p0 != Int32(0)
            for t in 0x00:0x02
                s = warpsum32(lab == t ? w : Int32(0))
                c = warpsum32(lab == t ? Int32(1) : Int32(0))
                if (threadIdx().x - Int32(1)) & Int32(31) == Int32(0)
                    c > Int32(0) && CUDA.atomic_add!(pointer(pw, Int32(6) * (p0 - Int32(1)) + Int32(t) + Int32(1)), s)
                    c > Int32(0) && CUDA.atomic_add!(pointer(pw, Int32(6) * (p0 - Int32(1)) + Int32(t) + Int32(4)), c)
                end
            end
        end
    elseif p != Int32(0)
        CUDA.atomic_add!(pointer(pw, Int32(6) * (p - Int32(1)) + Int32(lab) + Int32(1)), w)
        CUDA.atomic_add!(pointer(pw, Int32(6) * (p - Int32(1)) + Int32(lab) + Int32(4)), Int32(1))
    end
    return
end

# dir 0: separator vertices move into the lighter side of their part, dir 1: into the heavier side
function k_select!(mv, label, ptr, adj, vw, pid, pw, used, maxside, n::Int32, dir::Int32, nmv)
    v = gtid(); v > n && return
    @inbounds begin
        mv[v] = 0x00
        label[v] == SS || return
        p = pid[v]
        wA = pw[Int32(6) * (p - Int32(1)) + Int32(1)]; wB = pw[Int32(6) * (p - Int32(1)) + Int32(2)]
        lighter = wA <= wB ? SA : SB
        X = dir == Int32(0) ? lighter : (lighter == SA ? SB : SA)
        Y = X == SA ? SB : SA
        sY = Int32(0)
        for q in ptr[v]:(ptr[v + Int32(1)] - Int32(1))
            u = adj[q]
            label[u] == Y && (sY += vw[u])
        end
        wv = vw[v]
        wv - sY > Int32(0) || return
        wX = X == SA ? wA : wB
        old = CUDA.atomic_add!(pointer(used, p), wv)
        wX + old + wv <= maxside[p] || return
        mv[v] = X + 0x01
        CUDA.atomic_add!(pointer(nmv, 1), Int32(1))
    end
    return
end

function k_apply!(label, mv, ptr, adj, n::Int32)
    v = gtid(); v > n && return
    @inbounds begin
        m = mv[v]
        m == 0x00 && return
        X = m - 0x01; Y = X == SA ? SB : SA
        label[v] = X
        for q in ptr[v]:(ptr[v + Int32(1)] - Int32(1))
            u = adj[q]
            label[u] == Y && (label[u] = SS)
        end
    end
    return
end


# ---------------------------------------------------------------- jet-style separator refinement
# Iteration it moves separator vertices of each part into one side X (dirmode 0: lighter, 1: heavier).
# Candidates may have negative gain (>= -jneg w(v)); the afterburner charges each vertex u of the other
# side Y that a move would pull into the separator to its highest-priority candidate neighbour and drops
# candidates whose charged gain is negative (jfilt rounds). The best labelling per part is kept.
@inline jprio(g::Int32, v::Int32, it::Int32) = (UInt64(reinterpret(UInt32, g + Int32(0x40000000))) << 32) |
    UInt64(mix32(reinterpret(UInt32, v) * 0x9e3779b1 + reinterpret(UInt32, it) * 0x85ebca6b) | 0x00000001)

function k_jcand!(prio, mv, label, ptr, adj, vw, pid, pw, maxside, tabu, n::Int32, dirmode::Int32, it::Int32, jneg::Int32)
    v = gtid(); v > n && return
    @inbounds begin
        prio[v] = UInt64(0); mv[v] = 0x00
        label[v] == SS || return
        tabu[v] == it - Int32(1) && return
        p = pid[v]
        wA = pw[Int32(6) * (p - Int32(1)) + Int32(1)]; wB = pw[Int32(6) * (p - Int32(1)) + Int32(2)]
        lighter = wA <= wB ? SA : SB
        X = dirmode == Int32(0) ? lighter : (lighter == SA ? SB : SA)
        Y = X == SA ? SB : SA
        sY = Int32(0)
        for q in ptr[v]:(ptr[v + Int32(1)] - Int32(1))
            u = adj[q]
            label[u] == Y && (sY += vw[u])
        end
        wv = vw[v]
        g = wv - sY
        g >= -jneg * wv || return
        (X == SA ? wA : wB) + wv <= maxside[p] || return
        prio[v] = jprio(g, v, it); mv[v] = X + 0x01
    end
    return
end

function k_jpay!(maxp, prio, mv, label, ptr, adj, n::Int32)
    v = gtid(); v > n && return
    @inbounds begin
        pr = prio[v]
        pr == UInt64(0) && return
        Y = mv[v] == 0x01 ? SB : SA
        for q in ptr[v]:(ptr[v + Int32(1)] - Int32(1))
            u = adj[q]
            label[u] == Y && CUDA.atomic_max!(pointer(maxp, u), pr)
        end
    end
    return
end

function k_jfilt!(prio, mv, maxp, label, ptr, adj, vw, n::Int32)
    v = gtid(); v > n && return
    @inbounds begin
        pr = prio[v]
        pr == UInt64(0) && return
        Y = mv[v] == 0x01 ? SB : SA
        cost = Int32(0)
        for q in ptr[v]:(ptr[v + Int32(1)] - Int32(1))
            u = adj[q]
            (label[u] == Y && maxp[u] == pr) && (cost += vw[u])
        end
        if vw[v] - cost < Int32(0)
            prio[v] = UInt64(0); mv[v] = 0x00
        end
    end
    return
end

function k_jbudget!(mv, prio, vw, pid, pw, used, maxside, n::Int32)
    v = gtid(); v > n && return
    @inbounds begin
        prio[v] == UInt64(0) && return
        p = pid[v]; X = mv[v] - 0x01
        wX = pw[Int32(6) * (p - Int32(1)) + Int32(X) + Int32(1)]
        old = CUDA.atomic_add!(pointer(used, p), vw[v])
        wX + old + vw[v] <= maxside[p] || (mv[v] = 0x00)
    end
    return
end

function k_japply!(label, mv, tabu, ptr, adj, n::Int32, it::Int32, cnt)
    v = gtid(); v > n && return
    @inbounds begin
        m = mv[v]
        m == 0x00 && return
        X = m - 0x01; Y = X == SA ? SB : SA
        label[v] = X; tabu[v] = it
        for q in ptr[v]:(ptr[v + Int32(1)] - Int32(1))
            u = adj[q]
            if label[u] == Y
                label[u] = SS; tabu[u] = it
            end
        end
        CUDA.atomic_add!(pointer(cnt, 1), Int32(1))
    end
    return
end

function k_jbest!(bestk, improved, pw, np::Int32)
    p = gtid(); p > np && return
    @inbounds begin
        o = Int32(6) * (p - Int32(1))
        k = (UInt64(reinterpret(UInt32, pw[o + Int32(3)])) << 32) | UInt64(reinterpret(UInt32, abs(pw[o + Int32(1)] - pw[o + Int32(2)])))
        if k < bestk[p]
            bestk[p] = k; improved[p] = 0x01
        else
            improved[p] = 0x00
        end
    end
    return
end

function k_jcopy!(blab, label, pid, improved, n::Int32)
    v = gtid(); v > n && return
    @inbounds improved[pid[v]] == 0x01 && (blab[v] = label[v])
    return
end

function jet_refine!(label::CuVector{UInt8}, g::DG, np::Int, maxside::CuVector{Int32}, o::GNDOpts, W)
    n = g.n; n32 = Int32(n); np32 = Int32(np); nb = nblk(n)
    pw, used, cnt, mv, prio, maxp, tabu, blab, bestk, improved = W.pw, W.used, W.cnt, W.mv, W.prio, W.maxp, W.tabu, W.blab, W.bestk, W.improved
    fill!(view(tabu, 1:n), Int32(-5)); fill!(bestk, typemax(UInt64))
    fill!(pw, Int32(0))
    @cuda threads = TPB blocks = nb k_pw!(pw, label, g.vw, g.pid, n32)
    @cuda threads = TPB blocks = nblk(np) k_jbest!(bestk, improved, pw, np32)
    copyto!(view(blab, 1:n), label)
    idle = 0
    for it in Int32(1):Int32(o.jiter)
        fill!(cnt, Int32(0))
        @cuda threads = TPB blocks = nb k_jcand!(prio, mv, label, g.ptr, g.adj, g.vw, g.pid, pw, maxside, tabu, n32, (it - Int32(1)) % Int32(2),
            it, Int32(o.jneg))
        for f in 1:o.jfilt
            fill!(view(maxp, 1:n), UInt64(0))
            @cuda threads = TPB blocks = nb k_jpay!(maxp, prio, mv, label, g.ptr, g.adj, n32)
            @cuda threads = TPB blocks = nb k_jfilt!(prio, mv, maxp, label, g.ptr, g.adj, g.vw, n32)
        end
        fill!(used, Int32(0))
        @cuda threads = TPB blocks = nb k_jbudget!(mv, prio, g.vw, g.pid, pw, used, maxside, n32)
        @cuda threads = TPB blocks = nb k_japply!(label, mv, tabu, g.ptr, g.adj, n32, it, cnt)
        fill!(pw, Int32(0))
        @cuda threads = TPB blocks = nb k_pw!(pw, label, g.vw, g.pid, n32)
        @cuda threads = TPB blocks = nblk(np) k_jbest!(bestk, improved, pw, np32)
        @cuda threads = TPB blocks = nb k_jcopy!(blab, label, g.pid, improved, n32)
        if Array(cnt)[1] == 0
            idle += 1
            idle >= 2 && break
        else
            idle = 0
        end
    end
    copyto!(label, view(blab, 1:n))
    return
end


# ---------------------------------------------------------------- warp node-FM refinement (one warp per part)
# METIS-style 2-sided node FM on the current labels: per step the warp scans the separator list for the
# best move (gain w(v) - w(N(v) in other side), ties to the lighter side), applies it (moved vertex locked,
# its neighbours on the other side join the separator), and the pass rolls back to its best prefix.
@inline lanemask_lt(lane::Int32) = (UInt32(1) << lane) - UInt32(1)

function k_wfm!(label, ptr, adj, vw, poff, maxside, np::Int32, slist, spos, lg, LF::Int32, npass::Int32, lim0::Int32)
    wid = (gtid() - Int32(1)) >> 5 + Int32(1)
    lane = (threadIdx().x - Int32(1)) & Int32(31)
    wid > np && return
    @inbounds begin
        a = poff[wid]; b = poff[wid + Int32(1)] - Int32(1); m = b - a + Int32(1)
        ms = maxside[wid]
        sb = a - Int32(1)                  # slist region: slist[sb + 1 : sb + m]
        lb = (a - Int32(1)) * LF           # log region: lg[lb + 1 : lb + LF m]
        cap = LF * m
        for pass in Int32(1):npass
            # separator list and side weights
            ns = Int32(0); wA = Int32(0); wB = Int32(0); wS = Int32(0)
            base = a
            while base <= b
                v = base + lane
                t = v <= b ? (label[v] & 0x03) : 0x03
                w = v <= b ? vw[v] : Int32(0)
                isS = t == SS
                mask = vote_ballot_sync(FULL, isS)
                if isS
                    pos = ns + Int32(count_ones(mask & lanemask_lt(lane)))
                    slist[sb + pos + Int32(1)] = v; spos[v] = pos
                end
                ns += Int32(count_ones(mask))
                wA += warpsum32(t == SA ? w : Int32(0)); wB += warpsum32(t == SB ? w : Int32(0)); wS += warpsum32(isS ? w : Int32(0))
                base += Int32(32)
            end
            sync_warp()
            # METIS FM_2WayNodeRefine2Sided rules
            limit = min(Int32(2) * ns, lim0)
            nlog = Int32(0); bestS = wS; bestD = abs(wA - wB); bestlog = Int32(0); nsw = Int32(0); bestsw = Int32(0)
            startS = bestS; startD = bestD
            while true
                # best move into A and into B (unlocked separator vertices)
                kA = UInt64(0); kB = UInt64(0)
                i = lane
                while i < ns
                    v = slist[sb + i + Int32(1)]
                    if label[v] & LOCK == 0x00
                        sA = Int32(0); sB = Int32(0)
                        for q in ptr[v]:(ptr[v + Int32(1)] - Int32(1))
                            u = adj[q]
                            t = label[u] & 0x03
                            t == SA && (sA += vw[u]); t == SB && (sB += vw[u])
                        end
                        wv = vw[v]
                        kv = UInt64(reinterpret(UInt32, v))
                        kA = max(kA, (UInt64(reinterpret(UInt32, (wv - sB) + typemin(Int32))) << 32) | kv)
                        kB = max(kB, (UInt64(reinterpret(UInt32, (wv - sA) + typemin(Int32))) << 32) | kv)
                    end
                    i += Int32(32)
                end
                kA = max(kA, shfl_xor_sync(FULL, kA, 16)); kA = max(kA, shfl_xor_sync(FULL, kA, 8))
                kA = max(kA, shfl_xor_sync(FULL, kA, 4)); kA = max(kA, shfl_xor_sync(FULL, kA, 2))
                kA = max(kA, shfl_xor_sync(FULL, kA, 1))
                kB = max(kB, shfl_xor_sync(FULL, kB, 16)); kB = max(kB, shfl_xor_sync(FULL, kB, 8))
                kB = max(kB, shfl_xor_sync(FULL, kB, 4)); kB = max(kB, shfl_xor_sync(FULL, kB, 2))
                kB = max(kB, shfl_xor_sync(FULL, kB, 1))
                (kA == UInt64(0) && kB == UInt64(0)) && break
                vA = reinterpret(Int32, UInt32(kA & 0xffffffff)); vB = reinterpret(Int32, UInt32(kB & 0xffffffff))
                okA = kA != UInt64(0) && wA + vw[vA] <= ms
                okB = kB != UInt64(0) && wB + vw[vB] <= ms
                to = 0x03
                if kA != UInt64(0) && kB != UInt64(0)
                    gA = (kA >> 32); gB = (kB >> 32)
                    to = gA > gB ? SA : gA < gB ? SB : ((pass & Int32(1)) == Int32(0) ? SA : SB)
                    if to == SA && !okA
                        to = SB
                    elseif to == SB && !okB
                        to = SA
                    end
                    (to == SA && !okA) && (to = 0x03)
                    (to == SB && !okB) && (to = 0x03)
                elseif okA
                    to = SA
                elseif okB
                    to = SB
                end
                to == 0x03 && break
                bv = to == SA ? vA : vB
                Y = to == SA ? SB : SA
                p0 = ptr[bv]; p1 = ptr[bv + Int32(1)]
                nlog + Int32(1) + (p1 - p0) > cap && break
                wbv = vw[bv]
                # weight pulled into the separator, and the stop rule
                sYp = Int32(0)
                q = p0 + lane
                while q < p1
                    u = adj[q]
                    (label[u] & 0x03) == Y && (sYp += vw[u])
                    q += Int32(32)
                end
                sY = warpsum32(sYp)
                newS = wS - wbv + sY
                nsw += Int32(1)
                if !(newS < bestS || (newS == bestS && abs((to == SA ? wA : wB) + wbv - ((to == SA ? wB : wA) - sY)) < bestD))
                    (nsw - bestsw > Int32(2) * limit || (nsw - bestsw > limit && Int64(newS) * 10 > Int64(bestS) * 11)) && break
                end
                if lane == Int32(0)
                    nlog1 = nlog + Int32(1)
                    lg[lb + nlog1] = ((bv - a) << 3) | Int32(SS)
                    label[bv] = to | LOCK
                    pb = spos[bv]; last = slist[sb + ns]
                    slist[sb + pb + Int32(1)] = last; spos[last] = pb
                end
                sync_warp()
                ns -= Int32(1); nlog += Int32(1)
                wS -= wbv
                to == SA ? (wA += wbv) : (wB += wbv)
                q = p0
                while q < p1
                    qq = q + lane
                    u = qq < p1 ? adj[qq] : Int32(0)
                    t = qq < p1 ? label[u] : 0x00
                    pull = qq < p1 && (t & 0x03) == Y
                    mask = vote_ballot_sync(FULL, pull)
                    wu = Int32(0)
                    if pull
                        pos = Int32(count_ones(mask & lanemask_lt(lane)))
                        slist[sb + ns + pos + Int32(1)] = u; spos[u] = ns + pos
                        lg[lb + nlog + pos + Int32(1)] = ((u - a) << 3) | Int32(t)
                        label[u] = SS | (t & LOCK)
                        wu = vw[u]
                    end
                    c = Int32(count_ones(mask))
                    ns += c; nlog += c
                    sw = warpsum32(wu)
                    wS += sw
                    Y == SA ? (wA -= sw) : (wB -= sw)
                    q += Int32(32)
                end
                sync_warp()
                D = abs(wA - wB)
                if wS < bestS || (wS == bestS && D < bestD)
                    bestS = wS; bestD = D; bestlog = nlog; bestsw = nsw
                end
            end
            # roll back to the best prefix (serial, reverse order), clear locks
            if lane == Int32(0)
                i = nlog
                while i > bestlog
                    e = lg[lb + i]
                    label[a + (e >> 3)] = UInt8(e & Int32(7))
                    i -= Int32(1)
                end
            end
            sync_warp()
            v = a + lane
            while v <= b
                label[v] = label[v] & 0x03
                v += Int32(32)
            end
            sync_warp()
            (bestS < startS || (bestS == startS && bestD < startD)) || break
        end
    end
    return
end


# 1-sided node FM (METIS FM_2WayNodeRefine1Sided): 2 npass passes alternating the target side; within a pass
# every move goes into the target side, so no locks are needed; ties broken by a hash (METIS: random order).
function k_wfm1!(label, ptr, adj, vw, poff, maxside, np::Int32, slist, spos, lg, LF::Int32, npass::Int32, lim0::Int32, seed::UInt32)
    wid = (gtid() - Int32(1)) >> 5 + Int32(1)
    lane = (threadIdx().x - Int32(1)) & Int32(31)
    wid > np && return
    @inbounds begin
        a = poff[wid]; b = poff[wid + Int32(1)] - Int32(1); m = b - a + Int32(1)
        ms = maxside[wid]
        sb = a - Int32(1)
        lb = (a - Int32(1)) * LF
        cap = LF * m
        to = 0x00
        for pass in Int32(0):(Int32(2) * npass - Int32(1))
            ns = Int32(0); wA = Int32(0); wB = Int32(0); wS = Int32(0)
            base = a
            while base <= b
                v = base + lane
                t = v <= b ? label[v] : 0x03
                w = v <= b ? vw[v] : Int32(0)
                isS = t == SS
                mask = vote_ballot_sync(FULL, isS)
                if isS
                    pos = ns + Int32(count_ones(mask & lanemask_lt(lane)))
                    slist[sb + pos + Int32(1)] = v; spos[v] = pos
                end
                ns += Int32(count_ones(mask))
                wA += warpsum32(t == SA ? w : Int32(0)); wB += warpsum32(t == SB ? w : Int32(0)); wS += warpsum32(isS ? w : Int32(0))
                base += Int32(32)
            end
            sync_warp()
            if pass == Int32(0)
                to = wA < wB ? SA : SB      # first pass moves into the lighter side
            else
                to = to == SA ? SB : SA
            end
            other = to == SA ? SB : SA
            hs = mix32(seed + reinterpret(UInt32, pass) * 0x9e3779b1 + reinterpret(UInt32, wid) * 0x85ebca6b)
            limit = min(Int32(3) * ns, lim0)
            nlog = Int32(0); initS = wS; bestS = wS; bestD = abs(wA - wB); bestlog = Int32(0); nsw = Int32(0); bestsw = Int32(-1)
            while true
                key = UInt64(0); bvl = Int32(0)
                i = lane
                while i < ns
                    v = slist[sb + i + Int32(1)]
                    sO = Int32(0)
                    for q in ptr[v]:(ptr[v + Int32(1)] - Int32(1))
                        u = adj[q]
                        label[u] == other && (sO += vw[u])
                    end
                    k = (UInt64(reinterpret(UInt32, (vw[v] - sO) + typemin(Int32))) << 32) |
                        UInt64(mix32(reinterpret(UInt32, v) ⊻ hs) & 0xffffffe0) | UInt64(lane)
                    if k > key
                        key = k; bvl = v
                    end
                    i += Int32(32)
                end
                kk = key
                kk = max(kk, shfl_xor_sync(FULL, kk, 16)); kk = max(kk, shfl_xor_sync(FULL, kk, 8))
                kk = max(kk, shfl_xor_sync(FULL, kk, 4)); kk = max(kk, shfl_xor_sync(FULL, kk, 2))
                kk = max(kk, shfl_xor_sync(FULL, kk, 1))
                kk == UInt64(0) && break
                wl = Int32(kk & UInt64(31))
                bv = shfl_sync(FULL, bvl, wl + Int32(1))
                g = Int32(reinterpret(Int32, UInt32(kk >> 32)) - typemin(Int32))
                wbv = vw[bv]
                (to == SA ? wA : wB) + wbv > ms && break
                p0 = ptr[bv]; p1 = ptr[bv + Int32(1)]
                nlog + Int32(1) + (p1 - p0) > cap && break
                newS = wS - g
                pulled = wbv - g
                newD = abs((to == SA ? wA : wB) + wbv - ((to == SA ? wB : wA) - pulled))
                if !(newS < bestS || (newS == bestS && newD < bestD))
                    (nsw - bestsw > Int32(3) * limit || (nsw - bestsw > limit && Int64(newS) * 10 > Int64(bestS) * 11)) && break
                end
                if lane == Int32(0)
                    lg[lb + nlog + Int32(1)] = ((bv - a) << 3) | Int32(SS)
                    label[bv] = to
                    pb = spos[bv]; last = slist[sb + ns]
                    slist[sb + pb + Int32(1)] = last; spos[last] = pb
                end
                sync_warp()
                ns -= Int32(1); nlog += Int32(1)
                wS -= wbv
                to == SA ? (wA += wbv) : (wB += wbv)
                q = p0
                while q < p1
                    qq = q + lane
                    u = qq < p1 ? adj[qq] : Int32(0)
                    pull = qq < p1 && label[u] == other
                    mask = vote_ballot_sync(FULL, pull)
                    wu = Int32(0)
                    if pull
                        pos = Int32(count_ones(mask & lanemask_lt(lane)))
                        slist[sb + ns + pos + Int32(1)] = u; spos[u] = ns + pos
                        lg[lb + nlog + pos + Int32(1)] = ((u - a) << 3) | Int32(other)
                        label[u] = SS
                        wu = vw[u]
                    end
                    c = Int32(count_ones(mask))
                    ns += c; nlog += c
                    sw = warpsum32(wu)
                    wS += sw
                    other == SA ? (wA -= sw) : (wB -= sw)
                    q += Int32(32)
                end
                sync_warp()
                D = abs(wA - wB)
                if wS < bestS || (wS == bestS && D < bestD)
                    bestS = wS; bestD = D; bestlog = nlog; bestsw = nsw
                end
                nsw += Int32(1)
            end
            if lane == Int32(0)
                i = nlog
                while i > bestlog
                    e = lg[lb + i]
                    label[a + (e >> 3)] = UInt8(e & Int32(7))
                    i -= Int32(1)
                end
            end
            sync_warp()
            ((pass & Int32(1)) == Int32(1) && (bestlog == Int32(0) || bestS >= initS)) && break
        end
    end
    return
end


# 1-sided node FM with incremental gains. The separator list of each part (slist[a .. a+ns-1], built by
# k_sinit!) and the gain of every listed vertex (gl, by list position) are kept across moves and passes:
# moving v into `to` changes no other gain; pulling u from `other` raises the gain of its separator
# neighbours by w(u). Per move the warp scans the gains (coalesced), not the neighbourhoods.
function k_sinit!(slist, spos, nsp, label, pid, poff, cs, n::Int32)
    v = gtid(); v > n && return
    @inbounds begin
        p = pid[v]; a = poff[p]
        base = a == Int32(1) ? Int32(0) : cs[a - Int32(1)]
        if label[v] == SS
            pos = cs[v] - Int32(1) - base
            slist[a + pos] = v; spos[v] = pos
        end
        if v == poff[p + Int32(1)] - Int32(1)
            nsp[p] = cs[v] - base
        end
    end
    return
end

@inline function gainof(v::Int32, other::UInt8, ptr, adj, vw, label)
    sO = Int32(0)
    @inbounds for q in ptr[v]:(ptr[v + Int32(1)] - Int32(1))
        u = adj[q]
        label[u] == other && (sO += vw[u])
    end
    return @inbounds vw[v] - sO
end

@inline ekey(g::Int32, v::Int32, i::Int32, hs::UInt32) =
    (UInt64(reinterpret(UInt32, g + typemin(Int32))) << 32) | UInt64(mix32(reinterpret(UInt32, v) ⊻ hs) & 0xffffffe0) | UInt64(i & Int32(31))

function k_wfm1b!(label, ptr, adj, vw, poff, maxside, pw, nsp, np::Int32, slist, spos, gl, lg, LF::Int32, npass::Int32, lim0::Int32, seed::UInt32, fst, cmax, cthr::Int32, smax)
    wid = (gtid() - Int32(1)) >> 5 + Int32(1)
    lane = (threadIdx().x - Int32(1)) & Int32(31)
    wib = (threadIdx().x - Int32(1)) >> 5 + Int32(1)       # warp in block (4 per block)
    dl = CuStaticSharedArray(Int32, (64, 4)); dn = CuStaticSharedArray(Int32, 4)
    wid > np && return
    @inbounds begin
        a = poff[wid]; b = poff[wid + Int32(1)] - Int32(1); m = b - a + Int32(1)
        ms = maxside[wid]
        sb = a - Int32(1)                 # list entry i (0-based) at sb + i + 1
        lb = (a - Int32(1)) * LF
        cap = LF * m
        o6 = Int32(6) * (wid - Int32(1))
        wA = pw[o6 + Int32(1)]; wB = pw[o6 + Int32(2)]; wS = pw[o6 + Int32(3)]
        ns = nsp[wid]
        totmv = Int32(0); totscan = Int32(0)
        ph1 = Int64(0); ph2 = Int64(0); ph3 = Int64(0); ph4 = Int64(0); ph5 = Int64(0)
        to = wA < wB ? SB : SA            # flipped at the start of every pass: pass 0 moves into the lighter side
        for pass in Int32(0):(Int32(2) * npass - Int32(1))
            to = to == SA ? SB : SA
            other = to == SA ? SB : SA
            # gains into `to`
            i = lane
            while i < ns
                gl[sb + i + Int32(1)] = gainof(slist[sb + i + Int32(1)], other, ptr, adj, vw, label)
                i += Int32(32)
            end
            sync_warp()
            hs = mix32(seed + reinterpret(UInt32, pass) * 0x9e3779b1 + reinterpret(UInt32, wid) * 0x85ebca6b)
            cm = ns > cthr                 # chunked argmax (max key of every 32 list entries) for long lists
            if cm
                c = lane
                while c < (ns + Int32(31)) >> 5
                    km = UInt64(0)
                    for j in Int32(0):Int32(31)
                        i = (c << 5) + j
                        i < ns && (km = max(km, ekey(gl[sb + i + Int32(1)], slist[sb + i + Int32(1)], i, hs)))
                    end
                    cmax[sb + c + Int32(1)] = km
                    c += Int32(32)
                end
                sync_warp()
            end
            cm2 = false    # second level (max of every 32 chunk maxima): disabled, see ND_NOTES (inconsistent lists in tests)
            if cm2
                nch = (ns + Int32(31)) >> 5
                c = lane
                while c < (nch + Int32(31)) >> 5
                    km = UInt64(0)
                    for j in Int32(0):Int32(31)
                        cc = (c << 5) + j
                        cc < nch && (km = max(km, cmax[sb + cc + Int32(1)]))
                    end
                    smax[sb + c + Int32(1)] = km
                    c += Int32(32)
                end
                sync_warp()
            end
            limit = min(Int32(3) * ns, lim0)
            nlog = Int32(0); initS = wS; bestS = wS; bestD = abs(wA - wB); bestA = wA; bestB = wB
            bestlog = Int32(0); nsw = Int32(0); bestsw = Int32(-1)
            while true
                tc0 = reinterpret(Int64, clock(UInt64))
                key = UInt64(0); pbl = Int32(-1)
                if cm2
                    nch = (ns + Int32(31)) >> 5
                    c = lane; bsc = Int32(-1)
                    while c < (nch + Int32(31)) >> 5
                        k = smax[sb + c + Int32(1)]
                        if k > key
                            key = k; bsc = c
                        end
                        c += Int32(32)
                    end
                    k2 = key
                    k2 = max(k2, shfl_xor_sync(FULL, k2, 16)); k2 = max(k2, shfl_xor_sync(FULL, k2, 8))
                    k2 = max(k2, shfl_xor_sync(FULL, k2, 4)); k2 = max(k2, shfl_xor_sync(FULL, k2, 2))
                    k2 = max(k2, shfl_xor_sync(FULL, k2, 1))
                    kl = key
                    key = UInt64(0); pbl = Int32(-1)
                    if k2 != UInt64(0)
                        w2 = Int32(trailing_zeros(vote_ballot_sync(FULL, kl == k2)))
                        sc = shfl_sync(FULL, bsc, w2 + Int32(1))
                        cc = (sc << 5) + lane
                        k = cc < nch ? cmax[sb + cc + Int32(1)] : UInt64(0)
                        if k == k2
                            key = k; pbl = (cc << 5) | Int32(k & UInt64(31))
                        end
                    end
                elseif cm
                    c = lane
                    while c < (ns + Int32(31)) >> 5
                        k = cmax[sb + c + Int32(1)]
                        if k > key
                            key = k; pbl = (c << 5) | Int32(k & UInt64(31))
                        end
                        c += Int32(32)
                    end
                else
                    i = lane
                    while i < ns
                        k = ekey(gl[sb + i + Int32(1)], slist[sb + i + Int32(1)], i, hs)
                        if k > key
                            key = k; pbl = i
                        end
                        i += Int32(32)
                    end
                end
                kk = key
                kk = max(kk, shfl_xor_sync(FULL, kk, 16)); kk = max(kk, shfl_xor_sync(FULL, kk, 8))
                kk = max(kk, shfl_xor_sync(FULL, kk, 4)); kk = max(kk, shfl_xor_sync(FULL, kk, 2))
                kk = max(kk, shfl_xor_sync(FULL, kk, 1))
                kk == UInt64(0) && break
                wl = Int32(trailing_zeros(vote_ballot_sync(FULL, key == kk)))
                pb = shfl_sync(FULL, pbl, wl + Int32(1))
                if pb < Int32(0)
                    lane == Int32(0) && CUDA.@cuprintln("wfm1b: winner not found part $wid ns $ns")
                    break
                end
                bv = slist[sb + pb + Int32(1)]
                g = reinterpret(Int32, UInt32(kk >> 32)) - typemin(Int32)
                wbv = vw[bv]
                (to == SA ? wA : wB) + wbv > ms && break
                p0 = ptr[bv]; p1 = ptr[bv + Int32(1)]
                nlog + Int32(1) + (p1 - p0) > cap && break
                newS = wS - g
                pulledw = wbv - g
                newD = abs((to == SA ? wA : wB) + wbv - ((to == SA ? wB : wA) - pulledw))
                if !(newS < bestS || (newS == bestS && newD < bestD))
                    (nsw - bestsw > Int32(3) * limit || (nsw - bestsw > limit && Int64(newS) * 10 > Int64(bestS) * 11)) && break
                end
                tc1 = reinterpret(Int64, clock(UInt64))
                # move bv: log, label, swap-remove from the list
                if lane == Int32(0)
                    lg[lb + nlog + Int32(1)] = ((bv - a) << 3) | Int32(SS)
                    label[bv] = to
                    last = slist[sb + ns]
                    slist[sb + pb + Int32(1)] = last; gl[sb + pb + Int32(1)] = gl[sb + ns]; spos[last] = pb
                    dl[1, wib] = pb >> 5; dl[2, wib] = (ns - Int32(1)) >> 5; dn[wib] = Int32(2)
                end
                sync_warp()
                ns -= Int32(1); nlog += Int32(1)
                ns0 = ns
                wS -= wbv
                to == SA ? (wA += wbv) : (wB += wbv)
                tc2 = reinterpret(Int64, clock(UInt64))
                # pull the `other` neighbours into the separator; a pulling lane keeps u, w(u) and u's list
                # bounds in registers (used below when bv has at most 32 neighbours)
                q = p0
                one = (p1 - p0) <= Int32(32)
                pm = UInt32(0); myu = Int32(0); mywu = Int32(0); myq0 = Int32(0); myq1 = Int32(0); mypos = Int32(0)
                while q < p1
                    qq = q + lane
                    u = qq < p1 ? adj[qq] : Int32(0)
                    pull = qq < p1 && label[u] == other
                    mask = vote_ballot_sync(FULL, pull)
                    wu = Int32(0)
                    if pull
                        pos = ns + Int32(count_ones(mask & lanemask_lt(lane)))
                        slist[sb + pos + Int32(1)] = u; spos[u] = pos
                        lg[lb + nlog + Int32(count_ones(mask & lanemask_lt(lane))) + Int32(1)] = ((u - a) << 3) | Int32(other)
                        label[u] = SS
                        wu = vw[u]
                        myu = u; mywu = wu; myq0 = ptr[u]; myq1 = ptr[u + Int32(1)]; mypos = pos
                    end
                    pm = mask
                    c = Int32(count_ones(mask))
                    ns += c; nlog += c
                    sw = warpsum32(wu)
                    wS += sw
                    other == SA ? (wA -= sw) : (wB -= sw)
                    q += Int32(32)
                end
                sync_warp()
                # gains: new separator vertices from scratch; old ones gain w(u) per pulled neighbour u
                tc3 = reinterpret(Int64, clock(UInt64))
                # (no atomics on gl: they bypass L1, and later plain loads could read stale lines)
                # per pulled u, the warp walks N(u) once: separator neighbours already listed gain w(u),
                # and the `other` neighbours give u's own gain
                j = ns0
                rem = pm
                while j < ns
                    if one
                        l = Int32(trailing_zeros(rem)); rem &= rem - UInt32(1)
                        u = shfl_sync(FULL, myu, l + Int32(1)); wu = shfl_sync(FULL, mywu, l + Int32(1))
                        qx = shfl_sync(FULL, myq0, l + Int32(1)) + lane; qe = shfl_sync(FULL, myq1, l + Int32(1))
                        jp = shfl_sync(FULL, mypos, l + Int32(1))
                    else
                        jp = j
                        u = slist[sb + j + Int32(1)]; wu = vw[u]
                        qx = ptr[u] + lane; qe = ptr[u + Int32(1)]
                    end
                    so = Int32(0)
                    while qx < qe
                        x = adj[qx]
                        lx = label[x]; px = spos[x]; wx = vw[x]
                        if lx == SS
                            if px < ns0
                                gl[sb + px + Int32(1)] += wu
                                if cm
                                    d = CUDA.atomic_add!(pointer(dn, wib), Int32(1)) + Int32(1)
                                    d <= Int32(64) && (dl[d, wib] = px >> 5)
                                end
                            end
                        elseif lx == other
                            so += wx
                        end
                        qx += Int32(32)
                    end
                    so = warpsum32(so)
                    lane == Int32(0) && (gl[sb + jp + Int32(1)] = wu - so)
                    sync_warp()
                    j += Int32(1)
                end
                sync_warp()
                tc4 = reinterpret(Int64, clock(UInt64))
                if cm
                    nd = dn[wib]
                    c0 = ns0 >> 5; c1 = (ns - Int32(1)) >> 5      # chunks of the new entries
                    nrec = (nd > Int32(64) ? Int32(0) : nd) + (ns > ns0 ? c1 - c0 + Int32(1) : Int32(0))
                    if nd > Int32(64)                              # too many: rebuild every chunk
                        c = lane
                        while c < (ns + Int32(32)) >> 5
                            km = UInt64(0)
                            for j in Int32(0):Int32(31)
                                ii = (c << 5) + j
                                ii < ns && (km = max(km, ekey(gl[sb + ii + Int32(1)], slist[sb + ii + Int32(1)], ii, hs)))
                            end
                            cmax[sb + c + Int32(1)] = km
                            c += Int32(32)
                        end
                        if cm2
                            sync_warp()
                            nch = (ns + Int32(31)) >> 5
                            c = lane
                            while c < (nch + Int32(31)) >> 5
                                km = UInt64(0)
                                for j in Int32(0):Int32(31)
                                    cc = (c << 5) + j
                                    cc < nch && (km = max(km, cmax[sb + cc + Int32(1)]))
                                end
                                smax[sb + c + Int32(1)] = km
                                c += Int32(32)
                            end
                        end
                    end
                    for r in Int32(1):nrec
                        c = r <= (nd > Int32(64) ? Int32(0) : nd) ? dl[r, wib] : c0 + (r - (nd > Int32(64) ? Int32(0) : nd) - Int32(1))
                        ii = (c << 5) + lane
                        km = ii < ns ? ekey(gl[sb + ii + Int32(1)], slist[sb + ii + Int32(1)], ii, hs) : UInt64(0)
                        km = max(km, shfl_xor_sync(FULL, km, 16)); km = max(km, shfl_xor_sync(FULL, km, 8))
                        km = max(km, shfl_xor_sync(FULL, km, 4)); km = max(km, shfl_xor_sync(FULL, km, 2))
                        km = max(km, shfl_xor_sync(FULL, km, 1))
                        lane == Int32(0) && (cmax[sb + c + Int32(1)] = km)
                        if cm2
                            sync_warp()
                            nch = (ns + Int32(31)) >> 5
                            sc = c >> 5; cc = (sc << 5) + lane
                            km = cc < nch ? cmax[sb + cc + Int32(1)] : UInt64(0)
                            km = max(km, shfl_xor_sync(FULL, km, 16)); km = max(km, shfl_xor_sync(FULL, km, 8))
                            km = max(km, shfl_xor_sync(FULL, km, 4)); km = max(km, shfl_xor_sync(FULL, km, 2))
                            km = max(km, shfl_xor_sync(FULL, km, 1))
                            lane == Int32(0) && (smax[sb + sc + Int32(1)] = km)
                            sync_warp()
                        end
                    end
                    sync_warp()
                end
                tc5 = reinterpret(Int64, clock(UInt64))
                if lane == Int32(0)
                    ph1 += tc1 - tc0; ph2 += tc2 - tc1; ph3 += tc3 - tc2; ph4 += tc4 - tc3; ph5 += tc5 - tc4
                end
                D = abs(wA - wB)
                if wS < bestS || (wS == bestS && D < bestD)
                    bestS = wS; bestD = D; bestA = wA; bestB = wB; bestlog = nlog; bestsw = nsw
                end
                nsw += Int32(1)
            end
            # roll back to the best prefix: restore labels (serial, reverse), then rebuild the list:
            # survivors of the current list, then the moved vertices that return to the separator
            if lane == Int32(0)
                i = nlog
                while i > bestlog
                    e = lg[lb + i]
                    label[a + (e >> 3)] = UInt8(e & Int32(7))
                    i -= Int32(1)
                end
            end
            sync_warp()
            w = Int32(0); base = Int32(0)
            while base < ns
                i = base + lane
                v = i < ns ? slist[sb + i + Int32(1)] : Int32(0)
                keep = i < ns && label[v] == SS
                mask = vote_ballot_sync(FULL, keep)
                sync_warp()
                if keep
                    pos = w + Int32(count_ones(mask & lanemask_lt(lane)))
                    slist[sb + pos + Int32(1)] = v; spos[v] = pos
                end
                w += Int32(count_ones(mask))
                base += Int32(32)
            end
            base = bestlog
            while base < nlog
                i = base + lane
                e = i < nlog ? lg[lb + i + Int32(1)] : Int32(0)
                # a moved vertex returns to the separator unless it had been pulled in during this pass
                back = i < nlog && (e & Int32(7)) == Int32(SS) && label[a + (e >> 3)] == SS
                mask = vote_ballot_sync(FULL, back)
                if back
                    v = a + (e >> 3)
                    pos = w + Int32(count_ones(mask & lanemask_lt(lane)))
                    slist[sb + pos + Int32(1)] = v; spos[v] = pos
                end
                w += Int32(count_ones(mask))
                base += Int32(32)
            end
            sync_warp()
            ns = w; wA = bestA; wB = bestB; wS = bestS
            totmv += nsw; totscan += bestsw + Int32(1)
            ((pass & Int32(1)) == Int32(1) && (bestlog == Int32(0) || bestS >= initS)) && break
        end
        if lane == Int32(0)
            CUDA.atomic_add!(pointer(fst, 1), Int64(totmv)); CUDA.atomic_max!(pointer(fst, 2), Int64(totmv))
            CUDA.atomic_add!(pointer(fst, 3), Int64(totscan)); CUDA.atomic_max!(pointer(fst, 4), Int64(totscan))
            CUDA.atomic_add!(pointer(fst, 5), ph1); CUDA.atomic_add!(pointer(fst, 6), ph2); CUDA.atomic_add!(pointer(fst, 7), ph3)
            CUDA.atomic_add!(pointer(fst, 8), ph4); CUDA.atomic_add!(pointer(fst, 9), ph5)
        end
    end
    return
end

function k_sflag!(fl, label, n::Int32)
    v = gtid(); v > n && return
    @inbounds fl[v] = label[v] == SS ? Int32(1) : Int32(0)
    return
end


# ---------------------------------------------------------------- parallel node FM (one block of K warps per part)
# For parts with long separators (the top ND levels): the separator list (vertex order, so segments are
# roughly contiguous in space) is split into K segments, one per warp; every warp runs the 1-sided FM of
# k_wfm1b! on its own segment at the same time. All moves of a pass go into the same side, so concurrent
# moves keep a valid separator; pulls claim a vertex with a CAS on its label word. After the pass each
# warp rolls back to its own best prefix; a vertex pulled by a rolled-back move that still touches a
# kept move is put back into the separator (repair). A pass that ends worse than it started is undone
# completely and ends the refinement of this level.
@inline function labcas!(label, u::Int32, old::UInt8, new::UInt8)
    # CAS on the 32-bit word holding label[u]; true if label[u] was `old` and is now `new`
    wp = reinterpret(Core.LLVMPtr{UInt32, CUDA.AS.Global}, pointer(label, ((u - Int32(1)) & ~Int32(3)) + Int32(1)))
    sh = UInt32(((u - Int32(1)) & Int32(3)) * Int32(8))
    while true
        w = unsafe_load(wp)
        UInt8((w >> sh) & 0xff) == old || return false
        nw = (w & ~(UInt32(0xff) << sh)) | (UInt32(new) << sh)
        CUDA.atomic_cas!(wp, w, nw) == w && return true
    end
end

function k_pfm!(label, ptr, adj, vw, poff, maxside, pw, nsp, sl0, slist, spos, gl, lg, LF::Int32, npass::Int32, lim0::Int32, seed::UInt32,
        K::Int32, fst)
    p = blockIdx().x
    w = (threadIdx().x - Int32(1)) >> 5                  # warp 0..K-1
    lane = (threadIdx().x - Int32(1)) & Int32(31)
    sh = CuStaticSharedArray(Int32, 8)                   # 1 wA, 2 wB, 3 wS, 4 improved flag, 5 worse flag
    swcnt = CuStaticSharedArray(Int32, 32)
    @inbounds begin
        a = poff[p]; b = poff[p + Int32(1)] - Int32(1); m = b - a + Int32(1)
        ms = maxside[p]
        R = m ÷ K                                         # list capacity per warp
        sb = a - Int32(1) + w * R                         # this warp's list: slist[sb + 1 : sb + R]
        cap = LF * R
        lb = (a - Int32(1)) * LF + w * cap
        o6 = Int32(6) * (p - Int32(1))
        if threadIdx().x == Int32(1)
            sh[1] = pw[o6 + Int32(1)]; sh[2] = pw[o6 + Int32(2)]; sh[3] = pw[o6 + Int32(3)]
        end
        ns0all = nsp[p]
        seg = (ns0all + K - Int32(1)) ÷ K
        f0 = min(w * seg, ns0all); f1 = min(f0 + seg, ns0all)
        ns = f1 - f0
        if ns > R
            ns = Int32(0)                                 # cannot hold the segment (tiny part): this warp idles
        end
        i = lane
        while i < ns
            v = sl0[a + f0 + i]
            slist[sb + i + Int32(1)] = v; spos[v] = w * R + i
            i += Int32(32)
        end
        sync_threads()
        wA0 = sh[1]; wB0 = sh[2]
        to = wA0 < wB0 ? SB : SA
        totmv = Int32(0)
        for pass in Int32(0):(Int32(2) * npass - Int32(1))
            to = to == SA ? SB : SA
            other = to == SA ? SB : SA
            sync_threads()
            startS = sh[3]; startD = abs(sh[1] - sh[2])
            sync_threads()
            i = lane
            while i < ns
                gl[sb + i + Int32(1)] = gainof(slist[sb + i + Int32(1)], other, ptr, adj, vw, label)
                i += Int32(32)
            end
            sync_warp()
            hs = mix32(seed + reinterpret(UInt32, pass) * 0x9e3779b1 + reinterpret(UInt32, p) * 0x85ebca6b + reinterpret(UInt32, w) * 0x27d4eb2f)
            limit = min(Int32(3) * ns, lim0)
            nlog = Int32(0); dS = Int32(0); bestdS = Int32(0); bestlog = Int32(0); nsw = Int32(0); bestsw = Int32(-1)
            full = false
            while ns > Int32(0)
                key = UInt64(0); pbl = Int32(-1)
                i = lane
                while i < ns
                    k = ekey(gl[sb + i + Int32(1)], slist[sb + i + Int32(1)], i, hs)
                    if k > key
                        key = k; pbl = i
                    end
                    i += Int32(32)
                end
                kk = key
                kk = max(kk, shfl_xor_sync(FULL, kk, 16)); kk = max(kk, shfl_xor_sync(FULL, kk, 8))
                kk = max(kk, shfl_xor_sync(FULL, kk, 4)); kk = max(kk, shfl_xor_sync(FULL, kk, 2))
                kk = max(kk, shfl_xor_sync(FULL, kk, 1))
                kk == UInt64(0) && break
                wl = Int32(trailing_zeros(vote_ballot_sync(FULL, key == kk)))
                pb = shfl_sync(FULL, pbl, wl + Int32(1))
                bv = slist[sb + pb + Int32(1)]
                label[bv] == SS || break                  # (defensive: the list must hold separator vertices)
                g = reinterpret(Int32, UInt32(kk >> 32)) - typemin(Int32)
                wbv = vw[bv]
                (to == SA ? sh[1] : sh[2]) + wbv > ms && break
                p0 = ptr[bv]; p1 = ptr[bv + Int32(1)]
                (nlog + Int32(1) + (p1 - p0) > cap || ns + (p1 - p0) > R) && (full = true; break)
                if !(dS - g < bestdS)
                    (nsw - bestsw > Int32(3) * limit || (nsw - bestsw > limit && Int64(startS + dS - g) * 10 > Int64(startS + bestdS) * 11)) && break
                end
                if lane == Int32(0)
                    lg[lb + nlog + Int32(1)] = ((bv - a) << 3) | Int32(SS)
                    label[bv] = to
                    last = slist[sb + ns]
                    slist[sb + pb + Int32(1)] = last; gl[sb + pb + Int32(1)] = gl[sb + ns]; spos[last] = w * R + pb
                    CUDA.atomic_add!(pointer(sh, Int32(3)), -wbv)
                    CUDA.atomic_add!(pointer(sh, to == SA ? Int32(1) : Int32(2)), wbv)
                end
                sync_warp()
                ns -= Int32(1); nlog += Int32(1)
                ns0 = ns
                dS -= wbv
                q = p0
                while q < p1
                    qq = q + lane
                    u = qq < p1 ? adj[qq] : Int32(0)
                    pull = qq < p1 && label[u] == other && labcas!(label, u, other, SS)
                    mask = vote_ballot_sync(FULL, pull)
                    wu = Int32(0)
                    if pull
                        k = Int32(count_ones(mask & lanemask_lt(lane)))
                        slist[sb + ns + k + Int32(1)] = u; spos[u] = w * R + ns + k
                        lg[lb + nlog + k + Int32(1)] = ((u - a) << 3) | Int32(other)
                        wu = vw[u]
                    end
                    c = Int32(count_ones(mask))
                    ns += c; nlog += c
                    sw = warpsum32(wu)
                    dS += sw
                    if lane == Int32(0) && sw != Int32(0)
                        CUDA.atomic_add!(pointer(sh, Int32(3)), sw)
                        CUDA.atomic_add!(pointer(sh, other == SA ? Int32(1) : Int32(2)), -sw)
                    end
                    q += Int32(32)
                end
                sync_warp()
                j = ns0
                while j < ns
                    u = slist[sb + j + Int32(1)]; wu = vw[u]
                    qx = ptr[u] + lane; qe = ptr[u + Int32(1)]
                    so = Int32(0)
                    while qx < qe
                        x = adj[qx]
                        lx = label[x]; px = spos[x]; wx = vw[x]
                        if lx == SS
                            (px >= w * R && px < w * R + ns0) && (gl[sb + (px - w * R) + Int32(1)] += wu)
                        elseif lx == other
                            so += wx
                        end
                        qx += Int32(32)
                    end
                    so = warpsum32(so)
                    lane == Int32(0) && (gl[sb + j + Int32(1)] = wu - so)
                    sync_warp()
                    j += Int32(1)
                end
                if dS < bestdS
                    bestdS = dS; bestlog = nlog; bestsw = nsw
                end
                nsw += Int32(1)
            end
            totmv += nsw
            # own rollback to the best prefix
            if lane == Int32(0)
                i = nlog
                while i > bestlog
                    e = lg[lb + i]
                    v = a + (e >> 3); old = UInt8(e & Int32(7)); cur = label[v]
                    wv = vw[v]
                    cur == SS && CUDA.atomic_add!(pointer(sh, Int32(3)), -wv)
                    cur == SA && CUDA.atomic_add!(pointer(sh, Int32(1)), -wv)
                    cur == SB && CUDA.atomic_add!(pointer(sh, Int32(2)), -wv)
                    old == SS && CUDA.atomic_add!(pointer(sh, Int32(3)), wv)
                    old == SA && CUDA.atomic_add!(pointer(sh, Int32(1)), wv)
                    old == SB && CUDA.atomic_add!(pointer(sh, Int32(2)), wv)
                    label[v] = old
                    i -= Int32(1)
                end
            end
            threadfence_block()
            sync_threads()
            # repair: a rolled-back pull that touches a kept move goes back into the separator
            nrep = Int32(0)
            base = bestlog
            while base < nlog
                i = base + lane
                e = i < nlog ? lg[lb + i + Int32(1)] : Int32(0)
                rep = false
                if i < nlog && (e & Int32(7)) == Int32(other)
                    v = a + (e >> 3)
                    if label[v] == other
                        for qx in ptr[v]:(ptr[v + Int32(1)] - Int32(1))
                            label[adj[qx]] == to && (rep = true)
                        end
                    end
                end
                mask = vote_ballot_sync(FULL, rep)
                if rep
                    v = a + (e >> 3)
                    label[v] = SS
                    lg[lb + nlog + nrep + Int32(count_ones(mask & lanemask_lt(lane))) + Int32(1)] = e     # remembered for an undo
                    wv = vw[v]
                    CUDA.atomic_add!(pointer(sh, Int32(3)), wv)
                    CUDA.atomic_add!(pointer(sh, other == SA ? Int32(1) : Int32(2)), -wv)
                end
                nrep += Int32(count_ones(mask))
                base += Int32(32)
            end
            threadfence_block()
            sync_threads()
            endS = sh[3]; endD = abs(sh[1] - sh[2])
            worse = endS > startS || (endS == startS && endD > startD) || max(sh[1], sh[2]) > ms
            better = endS < startS || (endS == startS && endD < startD)
            sync_threads()
            if worse
                # undo the whole pass: repairs, then the kept prefix in reverse
                if lane == Int32(0)
                    i = nlog + nrep
                    while i > Int32(0)
                        if i > nlog || i <= bestlog
                            e = lg[lb + i]
                            v = a + (e >> 3)
                            label[v] = i > nlog ? UInt8(e & Int32(7)) : UInt8(e & Int32(7))
                        end
                        i -= Int32(1)
                    end
                end
                break
            end
            # rebuild this warp's list: survivors, moved vertices that came back, repaired vertices
            wcount = Int32(0); base = Int32(0)
            while base < ns
                i = base + lane
                v = i < ns ? slist[sb + i + Int32(1)] : Int32(0)
                keep = i < ns && label[v] == SS
                mask = vote_ballot_sync(FULL, keep)
                sync_warp()
                if keep
                    pos = wcount + Int32(count_ones(mask & lanemask_lt(lane)))
                    slist[sb + pos + Int32(1)] = v; spos[v] = w * R + pos
                end
                wcount += Int32(count_ones(mask))
                base += Int32(32)
            end
            base = bestlog
            while base < nlog + nrep
                i = base + lane
                e = i < nlog + nrep ? lg[lb + i + Int32(1)] : Int32(0)
                v = a + (e >> 3)
                back = i < nlog + nrep && ((i < nlog && (e & Int32(7)) == Int32(SS)) || i >= nlog) && label[v] == SS && wcount < R - Int32(32)
                mask = vote_ballot_sync(FULL, back)
                if back
                    pos = wcount + Int32(count_ones(mask & lanemask_lt(lane)))
                    slist[sb + pos + Int32(1)] = v; spos[v] = w * R + pos
                end
                wcount += Int32(count_ones(mask))
                base += Int32(32)
            end
            sync_warp()
            ns = wcount
            sync_threads()
            ((pass & Int32(1)) == Int32(1) && !better) && break
        end
        if lane == Int32(0)
            CUDA.atomic_add!(pointer(fst, 1), Int64(totmv)); CUDA.atomic_max!(pointer(fst, 2), Int64(totmv))
        end
    end
    return
end

const FMDIAG = Ref(false)
const FMLOG = Any[]
const FST = Ref{CuVector{Int64}}()     # FM counters: total moves, max moves of a part (per launch), scan steps
fmstats_reset!() = (isassigned(FST) || (FST[] = CUDA.zeros(Int64, 9)); fill!(FST[], 0))
fmstats() = Array(FST[])

function wfm_refine!(label::CuVector{UInt8}, g::DG, poff::CuVector{Int32}, np::Int, maxside::CuVector{Int32}, o::GNDOpts)
    n = g.n
    slist = CuVector{Int32}(undef, n); spos = CuVector{Int32}(undef, n); lg = CuVector{Int32}(undef, o.lf * n)
    if o.fm1 && o.fminc
        n32 = Int32(n)
        fl = CuVector{Int32}(undef, n)
        @cuda threads = TPB blocks = nblk(n) k_sflag!(fl, label, n32)
        cs = iscan(fl)
        nsp = CuVector{Int32}(undef, np)
        @cuda threads = TPB blocks = nblk(n) k_sinit!(slist, spos, nsp, label, g.pid, poff, cs, n32)
        pw = CUDA.zeros(Int32, 6np)
        @cuda threads = TPB blocks = nblk(n) k_pw!(pw, label, g.vw, g.pid, n32)
        gl = fl
        if o.pfmk > 1 && np * o.pfmk <= o.pfmmax && maxpartsize(poff) >= o.pfmmin
            K = o.pfmk
            sl0 = slist; slw = CuVector{Int32}(undef, n)
            @cuda threads = 32K blocks = np k_pfm!(label, g.ptr, g.adj, g.vw, poff, maxside, pw, nsp, sl0, slw, spos, gl, lg, Int32(o.lf),
                Int32(o.fmpass), Int32(o.fmlimit), mix32(o.seed * 0x2545f491 + UInt32(g.n)), Int32(K), FST[])
            CUDA.unsafe_free!(slw)
            foreach(CUDA.unsafe_free!, (fl, cs, nsp, pw, slist, spos, lg))
            if FMDIAG[]
                CUDA.synchronize(); t = time()
                x = Array(FST[]); push!(FMLOG, (n, np, x[2], t)); FST[][2:2] .= 0
            end
            return
        end
        cmax = CuVector{UInt64}(undef, n); smax = CuVector{UInt64}(undef, n)
        @cuda threads = 128 blocks = cld(np, 4) k_wfm1b!(label, g.ptr, g.adj, g.vw, poff, maxside, pw, nsp, Int32(np), slist, spos, gl, lg,
            Int32(o.lf), Int32(o.fmpass), Int32(o.fmlimit), mix32(o.seed * 0x2545f491 + UInt32(g.n)), FST[], cmax, Int32(o.cthr), smax)
        foreach(CUDA.unsafe_free!, (fl, cs, nsp, pw, cmax, smax))
        if FMDIAG[]
            CUDA.synchronize(); t = time()
            x = Array(FST[]); push!(FMLOG, (n, np, x[2], t)); FST[][2:2] .= 0
        end
    elseif o.fm1
        @cuda threads = 128 blocks = cld(np, 4) k_wfm1!(label, g.ptr, g.adj, g.vw, poff, maxside, Int32(np), slist, spos, lg, Int32(o.lf),
            Int32(o.fmpass), Int32(o.fmlimit), mix32(o.seed * 0x2545f491 + UInt32(g.n)))
    else
        @cuda threads = 128 blocks = cld(np, 4) k_wfm!(label, g.ptr, g.adj, g.vw, poff, maxside, Int32(np), slist, spos, lg, Int32(o.lf),
            Int32(o.fmpass), Int32(o.fmlimit))
    end
    foreach(CUDA.unsafe_free!, (slist, spos, lg))
    return
end

# ---------------------------------------------------------------- split
@inline lo32(x::Int64) = Int32(x & 0xffffffff)
@inline hi32(x::Int64) = Int32(x >> 32)

function k_sflags!(s1, s2, s3, label, vw, n::Int32)
    v = gtid(); v > n && return
    @inbounds begin
        l = label[v]; w = Int64(vw[v])
        s1[v] = l == SA ? Int64(1) : l == SB ? Int64(1) << 32 : Int64(0)
        s2[v] = l == SA ? w : l == SB ? w << 32 : Int64(0)
        s3[v] = l == SS ? Int32(w) : Int32(0)
    end
    return
end

# exclusive (A count, B count, A weight, B weight, S weight) at vertex i
@inline function exvals(i::Int32, s1, s2, s3, label, vw)
    @inbounds begin
        l = label[i]; w = vw[i]
        x1 = s1[i]; x2 = s2[i]; x3 = s3[i]
        cA = lo32(x1) - (l == SA ? Int32(1) : Int32(0)); cB = hi32(x1) - (l == SB ? Int32(1) : Int32(0))
        wA = lo32(x2) - (l == SA ? w : Int32(0)); wB = hi32(x2) - (l == SB ? w : Int32(0))
        wS = x3 - (l == SS ? w : Int32(0))
    end
    return cA, cB, wA, wB, wS
end

# pk: per part (kind, kA, idA, kB, idB, sepstart, WA, WB); kind 1 = whole part is leaf idA
function k_split!(newidx, ngid, npid, nvw, dest, cpos, label, vw, pid, gid, poff, s1, s2, s3, pk, noff, n::Int32)
    v = gtid(); v > n && return
    @inbounds begin
        p = pid[v]; a = poff[p]; l = label[v]; g = gid[v]
        cA, cB, wA, wB, wS = exvals(v, s1, s2, s3, label, vw)
        aA, aB, awA, awB, awS = exvals(a, s1, s2, s3, label, vw)
        o = Int32(8) * (p - Int32(1))
        newidx[v] = Int32(0)
        if pk[o + Int32(1)] == Int32(1)
            dest[g] = pk[o + Int32(3)]
            cpos[g] = l == SA ? wA - awA : l == SB ? pk[o + Int32(7)] + wB - awB : pk[o + Int32(7)] + pk[o + Int32(8)] + wS - awS
        elseif l == SS
            dest[g] = Int32(0)
            cpos[g] = pk[o + Int32(6)] + wS - awS
        else
            k = l == SA ? pk[o + Int32(2)] : pk[o + Int32(4)]
            id = l == SA ? pk[o + Int32(3)] : pk[o + Int32(5)]
            if k == Int32(1)
                dest[g] = id
                cpos[g] = l == SA ? wA - awA : wB - awB
            else
                nv = noff[id] + (l == SA ? cA - aA : cB - aB)
                newidx[v] = nv; ngid[nv] = g; npid[nv] = id; nvw[nv] = vw[v]
            end
        end
    end
    return
end

function k_ecount!(ndeg, newidx, label, ptr, adj, n::Int32)
    v = gtid(); v > n && return
    @inbounds begin
        nv = newidx[v]
        nv == Int32(0) && return
        l = label[v]; c = Int32(0)
        for q in ptr[v]:(ptr[v + Int32(1)] - Int32(1))
            u = adj[q]
            (newidx[u] != Int32(0) && label[u] == l) && (c += Int32(1))
        end
        ndeg[nv] = c
    end
    return
end

function k_efill!(nadj, new_ew, nptr, newidx, label, ptr, adj, ew, n::Int32)
    v = gtid(); v > n && return
    @inbounds begin
        nv = newidx[v]
        nv == Int32(0) && return
        l = label[v]; k = nptr[nv]
        for q in ptr[v]:(ptr[v + Int32(1)] - Int32(1))
            u = adj[q]
            nu = newidx[u]
            if nu != Int32(0) && label[u] == l
                nadj[k] = nu; new_ew[k] = ew[q]; k += Int32(1)
            end
        end
    end
    return
end

# ---------------------------------------------------------------- leaves and final permutation
function k_leafmap!(uvert, lloc, lid, perm, ccls, crank, dest, cpos, lunion, N::Int32)
    v = gtid(); v > N && return
    @inbounds begin
        c = ccls[v]; r = crank[v]; l = dest[c]
        if l == Int32(0)
            perm[cpos[c] + r] = v; lid[v] = Int32(0)
        else
            loc = cpos[c] + r + Int32(1)
            lloc[v] = loc; lid[v] = l
            uvert[lunion[l] + loc - Int32(1)] = v
        end
    end
    return
end

function k_lcount!(udeg, uvert, lid, ptr, adj, NU::Int32)
    i = gtid(); i > NU && return
    @inbounds begin
        v = uvert[i]; l = lid[v]; c = Int32(0)
        for q in ptr[v]:(ptr[v + Int32(1)] - Int32(1))
            lid[adj[q]] == l && (c += Int32(1))
        end
        udeg[i] = c
    end
    return
end

function k_lfill!(uadj, uptr, uvert, lid, lloc, ptr, adj, NU::Int32)
    i = gtid(); i > NU && return
    @inbounds begin
        v = uvert[i]; l = lid[v]; k = uptr[i]
        for q in ptr[v]:(ptr[v + Int32(1)] - Int32(1))
            u = adj[q]
            if lid[u] == l
                uadj[k] = lloc[u]; k += Int32(1)
            end
        end
        shsort1!(uadj, uptr[i], k - Int32(1))
    end
    return
end

function k_permleaf!(perm, ocat, uvert, uleaf, lunion, lpos, NU::Int32)
    i = gtid(); i > NU && return
    @inbounds begin
        l = uleaf[i]; j = i - lunion[l]
        perm[lpos[l] + j] = uvert[lunion[l] + ocat[i] - Int32(1)]
    end
    return
end

function k_uleaf!(uleaf, lunion, nl::Int32)
    l = gtid(); l > nl && return
    @inbounds for i in lunion[l]:(lunion[l + Int32(1)] - Int32(1))
        uleaf[i] = l
    end
    return
end

# ---------------------------------------------------------------- twins (host driver)
function compress_twins(ptr0::CuVector{Int32}, adj0::CuVector{Int32}, N::Int)
    N32 = Int32(N)
    h = CuVector{UInt64}(undef, N)
    @cuda threads = TPB blocks = nblk(N) k_thash!(h, ptr0, adj0, N32)
    tsz = nextpow(2, 2N)
    keys = CUDA.zeros(UInt64, tsz); vals = CUDA.fill(typemax(Int32), tsz)
    mask = UInt64(tsz - 1)
    @cuda threads = TPB blocks = nblk(N) k_tinsert!(keys, vals, h, N32, mask)
    rep = CuVector{Int32}(undef, N)
    @cuda threads = TPB blocks = nblk(N) k_tlookup!(rep, keys, vals, h, ptr0, adj0, N32, mask)
    foreach(CUDA.unsafe_free!, (h, keys, vals))
    flag = CuVector{Int32}(undef, N)
    @cuda threads = TPB blocks = nblk(N) k_isrep!(flag, rep, N32)
    cidx = iscan(flag)
    nc = Int(lastval(cidx))
    CUDA.unsafe_free!(flag)
    if nc == N
        foreach(CUDA.unsafe_free!, (cidx, rep))
        return nothing
    end
    cmap = CuVector{Int32}(undef, N)
    @cuda threads = TPB blocks = nblk(N) k_cmaprep!(cmap, cidx, rep, N32)
    csz = CUDA.zeros(Int32, nc)
    @cuda threads = TPB blocks = nblk(N) k_csize!(csz, cmap, N32)
    memptr = ptr_from_counts(csz)
    slot = CUDA.zeros(Int32, nc); mem = CuVector{Int32}(undef, N)
    @cuda threads = TPB blocks = nblk(N) k_memscatter!(mem, slot, memptr, cmap, N32)
    crank = CuVector{Int32}(undef, N)
    @cuda threads = TPB blocks = nblk(nc) k_memsort!(mem, memptr, crank, Int32(nc))
    foreach(CUDA.unsafe_free!, (cidx, rep, csz, slot))
    return cmap, crank, nc, memptr, mem
end

# ---------------------------------------------------------------- multilevel separator of all parts
struct Level
    g::DG
    poff::CuVector{Int32}
    cmap::Union{Nothing, CuVector{Int32}}    # to the next coarser level
    maxpart::Int                             # largest part (vertices)
end
maxpartsize(poff::CuVector{Int32}) = (h = Array(poff); maximum(diff(h)))

function coarsen_step(g::DG, poff::CuVector{Int32}, np::Int, Wp::CuVector{Int32}, o::GNDOpts, salt::UInt32)
    n = g.n; n32 = Int32(n)
    pmaxvw = CuVector{Int32}(undef, np)
    @cuda threads = TPB blocks = nblk(np) k_pmaxvw!(pmaxvw, poff, Wp, Int32(o.C0), Int32(np))
    mate = CUDA.zeros(Int32, n); prop = CuVector{Int32}(undef, n)
    for it in 1:o.nmatch
        s = mix32(o.seed + salt * 0x9e3779b1 + UInt32(it) * 0x85ebca6b)
        @cuda threads = TPB blocks = nblk(n) k_propose!(prop, mate, g.ptr, g.adj, g.ew, g.vw, g.pid, pmaxvw, n32, s, o.mdeg)
        @cuda threads = TPB blocks = nblk(n) k_accept!(mate, prop, n32)
    end
    if o.hop2
        hub = CuVector{Int32}(undef, n); first = CuVector{Int32}(undef, n); second = CuVector{Int32}(undef, n)
        @cuda threads = TPB blocks = nblk(n) k_hubpick!(hub, mate, g.ptr, g.adj, g.ew, g.vw, g.pid, pmaxvw, n32, mix32(o.seed + salt))
        for r in 1:o.hop2r
            fill!(first, typemax(Int32)); fill!(second, typemax(Int32))
            @cuda threads = TPB blocks = nblk(n) k_hubmin1!(first, hub, mate, n32)
            @cuda threads = TPB blocks = nblk(n) k_hubmin2!(second, first, hub, mate, n32)
            @cuda threads = TPB blocks = nblk(n) k_hubpair!(mate, first, second, hub, g.vw, g.pid, pmaxvw, n32)
        end
        foreach(CUDA.unsafe_free!, (hub, first, second))
    end
    flag = prop
    @cuda threads = TPB blocks = nblk(n) k_finmatch!(mate, flag, n32)
    cidx = iscan(flag)
    nc = Int(lastval(cidx))
    if nc > o.stall * n
        foreach(CUDA.unsafe_free!, (pmaxvw, mate, prop, cidx))
        return nothing
    end
    cmap = CuVector{Int32}(undef, n); csz = CuVector{Int32}(undef, nc)
    @cuda threads = TPB blocks = nblk(n) k_pairmap!(cmap, csz, cidx, mate, n32)
    memptr = ptr_from_counts(csz)
    mem = CuVector{Int32}(undef, n)
    @cuda threads = TPB blocks = nblk(n) k_pairmem!(mem, memptr, cidx, mate, n32)
    cpoff = CuVector{Int32}(undef, np + 1)
    @cuda threads = TPB blocks = nblk(np + 1) k_cpoff!(cpoff, cmap, poff, Int32(np), Int32(nc))
    mpr = Ref(0)
    gc = contract(g, cmap, nc, memptr, mem; cpoff, mpref = mpr)
    foreach(CUDA.unsafe_free!, (pmaxvw, mate, prop, cidx, csz, memptr, mem))
    return gc, cpoff, cmap, mpr[]
end

function refine!(label::CuVector{UInt8}, g::DG, np::Int, maxside::CuVector{Int32}, o::GNDOpts, pw, used, nmv, mv)
    n = g.n; n32 = Int32(n)
    for it in 1:o.rpass
        fill!(nmv, Int32(0))
        for dir in Int32(0):Int32(1)
            fill!(pw, Int32(0)); fill!(used, Int32(0))
            @cuda threads = TPB blocks = nblk(n) k_pw!(pw, label, g.vw, g.pid, n32)
            @cuda threads = TPB blocks = nblk(n) k_select!(mv, label, g.ptr, g.adj, g.vw, g.pid, pw, used, maxside, n32, dir, nmv)
            @cuda threads = TPB blocks = nblk(n) k_apply!(label, mv, g.ptr, g.adj, n32)
        end
        Array(nmv)[1] == 0 && break
    end
    return
end

# separator labels (A/B/S) for every part of g
function coarsen_levels(g::DG, poff::CuVector{Int32}, np::Int, Wp::CuVector{Int32}, o::GNDOpts, lvl::Int, st, maxsteps::Int, minpart::Int)
    levels = Level[]
    cur = g; cpo = poff
    k = 0
    t0 = time()
    mp = maxpartsize(cpo)
    while true
        k += 1
        r = (k > maxsteps || mp <= minpart) ? nothing : coarsen_step(cur, cpo, np, Wp, o, UInt32(lvl * 64 + k + 32 * (maxsteps < typemax(Int))))
        if r === nothing
            push!(levels, Level(cur, cpo, nothing, mp))
            break
        end
        gc, cpoff, cmap, mpc = r
        push!(levels, Level(cur, cpo, cmap, mp))
        cur = gc; cpo = cpoff; mp = mpc
    end
    st[:coarsen] = get(st, :coarsen, 0.0) + time() - t0
    st[:clevels] = get(st, :clevels, 0) + length(levels) - 1
    return levels
end

function refine_level!(lab, gg::DG, gpoff, maxpart::Int, np::Int, maxside, o::GNDOpts, st, li::Int = 1)
    if li > o.fmfine && li <= o.fmfine + o.fmskip
        # skip the refinement of this level (it is refined again one level finer)
        return
    end
    if haskey(st, :cpufm)
        h = Array(lab)
        st[:cpufm](h, Array(gg.ptr), Array(gg.adj), Array(gg.vw), Array(gpoff), Array(maxside))
        copyto!(lab, h)
    elseif maxpart <= o.tfm
        if o.pregreedy > 0 && maxpart >= o.pgmin
            # parallel positive-gain moves first (thins a projected separator in a few launches), then FM
            n = gg.n
            pw = CUDA.zeros(Int32, 6np); used = CUDA.zeros(Int32, np); nmv = CUDA.zeros(Int32, 1); mv = CuVector{UInt8}(undef, n)
            refine!(lab, gg, np, maxside, GNDOpts(o; rpass = o.pregreedy), pw, used, nmv, mv)
            foreach(CUDA.unsafe_free!, (pw, used, nmv, mv))
        end
        wfm_refine!(lab, gg, gpoff, np, maxside, o)
    else
        n = gg.n
        pw = CUDA.zeros(Int32, 6np); used = CUDA.zeros(Int32, np); nmv = CUDA.zeros(Int32, 1); mv = CuVector{UInt8}(undef, n)
        if o.ref === :jet
            W = (pw = pw, used = used, cnt = nmv, mv = mv, prio = CuVector{UInt64}(undef, n), maxp = CuVector{UInt64}(undef, n),
                tabu = CuVector{Int32}(undef, n), blab = CuVector{UInt8}(undef, n), bestk = CuVector{UInt64}(undef, np),
                improved = CuVector{UInt8}(undef, np))
            jet_refine!(lab, gg, np, maxside, o, W)
            foreach(CUDA.unsafe_free!, (W.prio, W.maxp, W.tabu, W.blab, W.bestk, W.improved))
        else
            refine!(lab, gg, np, maxside, o, pw, used, nmv, mv)
        end
        foreach(CUDA.unsafe_free!, (pw, used, nmv, mv))
    end
end

function sepweight(lab, gg::DG, np::Int)
    pw = CUDA.zeros(Int32, 6np)
    @cuda threads = TPB blocks = nblk(gg.n) k_pw!(pw, lab, gg.vw, gg.pid, Int32(gg.n))
    h = Array(pw); CUDA.unsafe_free!(pw)
    return sum(h[6p - 3] for p in 1:np)
end

# project the labels of levels[end] down to levels[1], refining at every level; frees levels[2:end]
function uncoarsen!(lab, levels::Vector{Level}, np::Int, maxside, o::GNDOpts, lvl::Int, st)
    trace = get(st, :trace, false)
    tr = trace ? get!(st, :tr, Any[]) : nothing
    hy = o.hyb > 0 && levels[1].maxpart >= o.hybmin
    for i in (length(levels) - 1):-1:1
        L = levels[i]
        lf = CuVector{UInt8}(undef, L.g.n)
        @cuda threads = TPB blocks = nblk(L.g.n) k_project!(lf, lab, L.cmap, Int32(L.g.n))
        CUDA.unsafe_free!(lab); lab = lf
        spg = -1
        if hy && !(i > o.fmfine && i <= o.fmfine + o.fmskip)
            gocpu, spg = hyb_go_cpu(lab, levels, i, np, o, st)
            # CPU threads refine levels i .. 1 (projection on the host), one upload at the end
            gocpu && return hyb_finish!(lab, levels, i, np, maxside, o, lvl, st)
        end
        if spg >= 0
            # GPU level under the hybrid: time it with events (read at the end of the ND level) to update the model
            e0 = CuEvent(); CUDA.record(e0)
            refine_level!(lab, L.g, L.poff, L.maxpart, np, maxside, o, st, i)
            e1 = CuEvent(); CUDA.record(e1)
            push!(HPEND, (e0, e1, spg))
        elseif HCALIB[] && !(i > o.fmfine && i <= o.fmfine + o.fmskip)
            pwc = CUDA.zeros(Int32, 6np)
            @cuda threads = TPB blocks = nblk(L.g.n) k_pw!(pwc, lab, L.g.vw, L.g.pid, Int32(L.g.n))
            h = Array(pwc); CUDA.unsafe_free!(pwc)
            spm = maximum(Int(h[6p]) for p in 1:np)
            tc0 = time()
            refine_level!(lab, L.g, L.poff, L.maxpart, np, maxside, o, st, i)
            CUDA.synchronize()
            push!(HCALREC, (spm, np, time() - tc0))
        else
            refine_level!(lab, L.g, L.poff, L.maxpart, np, maxside, o, st, i)
        end
        trace && push!(tr, (lvl, i, L.g.n, :ref, sepweight(lab, L.g, np)))
        dfree!(levels[i + 1].g); CUDA.unsafe_free!(levels[i + 1].poff)
        CUDA.unsafe_free!(L.cmap)
    end
    return lab
end

function partweights(lab, g::DG, np::Int)
    pw = CUDA.zeros(Int32, 6np)
    @cuda threads = TPB blocks = nblk(g.n) k_pw!(pw, lab, g.vw, g.pid, Int32(g.n))
    pwH = Array(pw); CUDA.unsafe_free!(pw)
    return pwH
end

# separator labels (A/B/S) for every part of g: full multilevel
function separate(g::DG, poff::CuVector{Int32}, np::Int, Wp::CuVector{Int32}, maxside::CuVector{Int32}, o::GNDOpts, lvl::Int, st)
    levels = coarsen_levels(g, poff, np, Wp, o, lvl, st, typemax(Int), 0)
    t1 = time()
    top = levels[end]
    nt = top.g.n
    get(st, :trace, false) && push!(get!(st, :coarsest, Any[]), (lvl, np, nt, top.maxpart))
    LF = o.lf
    wh = CuVector{UInt8}(undef, 32 * nt)
    lg = CuVector{Int32}(undef, 32 * LF * nt)
    lab = CuVector{UInt8}(undef, nt)
    s = mix32(o.seed ⊻ UInt32(lvl) * 0x27d4eb2f)
    @cuda threads = 128 blocks = cld(np, 4) k_initsep!(lab, top.g.ptr, top.g.adj, top.g.vw, top.poff, maxside, Int32(np), wh, lg,
        Int32(LF), Int32(o.npass), s)
    CUDA.unsafe_free!(wh); CUDA.unsafe_free!(lg)
    CUDA.synchronize()
    t2 = time()
    st[:initsep] = get(st, :initsep, 0.0) + t2 - t1
    trace = get(st, :trace, false)
    trace && push!(get!(st, :tr, Any[]), (lvl, length(levels), nt, :init, sepweight(lab, top.g, np)))
    refine_level!(lab, top.g, top.poff, top.maxpart, np, maxside, o, st)
    lab = uncoarsen!(lab, levels, np, maxside, o, lvl, st)
    pwH = partweights(lab, g, np)
    st[:refine] = get(st, :refine, 0.0) + time() - t2
    return lab, pwH
end

# ---------------------------------------------------------------- replicas: R independent trials per part
function k_repv!(ptr2, vw2, pid2, ptr, vw, pid, n::Int32, m::Int32, np::Int32, R::Int32)
    v = gtid(); v > R * n && return
    @inbounds begin
        r = (v - Int32(1)) ÷ n; v0 = v - r * n
        ptr2[v] = r * m + ptr[v0]; vw2[v] = vw[v0]; pid2[v] = r * np + pid[v0]
        v == R * n && (ptr2[v + Int32(1)] = R * m + Int32(1))
    end
    return
end
function k_repe!(adj2, ew2, adj, ew, n::Int32, m::Int32, R::Int32)
    e = gtid(); e > R * m && return
    @inbounds begin
        r = (e - Int32(1)) ÷ m; e0 = e - r * m
        adj2[e] = adj[e0] + r * n; ew2[e] = ew[e0]
    end
    return
end
function k_repgather!(lab, labr, pid, rbest, n::Int32)
    v = gtid(); v > n && return
    @inbounds lab[v] = labr[rbest[pid[v]] * n + v]
    return
end
function replicate(g::DG, poffH::Vector{Int32}, np::Int, R::Int)
    n = g.n; m = g.m
    ptr2 = CuVector{Int32}(undef, R * n + 1); vw2 = CuVector{Int32}(undef, R * n); pid2 = CuVector{Int32}(undef, R * n)
    adj2 = CuVector{Int32}(undef, max(R * m, 1)); ew2 = CuVector{Int32}(undef, max(R * m, 1))
    @cuda threads = TPB blocks = nblk(R * n) k_repv!(ptr2, vw2, pid2, g.ptr, g.vw, g.pid, Int32(n), Int32(m), Int32(np), Int32(R))
    R * m > 0 && @cuda threads = TPB blocks = nblk(R * m) k_repe!(adj2, ew2, g.adj, g.ew, Int32(n), Int32(m), Int32(R))
    poff2 = Vector{Int32}(undef, R * np + 1)
    for r in 0:(R - 1), p in 1:np
        poff2[r * np + p] = r * n + poffH[p]
    end
    poff2[end] = R * n + 1
    return DG(R * n, R * m, ptr2, adj2, ew2, vw2, pid2), poff2
end

include(joinpath(@__DIR__, "hybfm.jl"))
include(joinpath(@__DIR__, "kway.jl"))

# ---------------------------------------------------------------- driver
"""
    gnd_perm(ptr, adj; opts=GNDOpts(), leafsolve, timing=nothing) -> perm

`perm[i]` = the vertex eliminated i-th. `ptr`/`adj`: sorted CSR, no self loops.
`leafsolve(csrs)` gets a vector of `(ptr::Vector{Int64}, tgt::AbstractVector{Int32})`
leaf graphs (local 1-based ids) and returns their elimination orders (local ids).
"""
function gnd_perm(ptrH::AbstractVector, adjH::AbstractVector; opts::GNDOpts = GNDOpts(), leafsolve = nothing, leafdev = nothing, timing = nothing,
        abort = nothing)
    o = opts; o0 = opts
    st = timing === nothing ? Dict{Symbol, Any}() : timing
    T0 = time()
    gc0 = Base.gc_time_ns()
    fmstats_reset!()
    N = length(ptrH) - 1
    N32 = Int32(N)
    ptr0 = CuVector{Int32}(ptrH); adj0 = CuVector{Int32}(adjH)
    M0 = length(adjH)
    CUDA.synchronize()
    T1 = time()
    st[:upload] = T1 - T0
    # 1. twins
    tw = o.compress ? compress_twins(ptr0, adj0, N) : nothing
    g0 = DG(N, M0, ptr0, adj0, CUDA.ones(Int32, max(M0, 1)), CUDA.ones(Int32, N), CUDA.ones(Int32, N))
    if tw === nothing
        ccls = CuVector{Int32}(1:N); crank = CUDA.zeros(Int32, N)
        NC = N
        G = DG(N, M0, ptr0, adj0, g0.ew, g0.vw, g0.pid)
        ownG = false
    else
        cmap, crank, NC, memptr, mem = tw
        ccls = cmap
        G = contract(g0, cmap, NC, memptr, mem)
        CUDA.unsafe_free!(memptr); CUDA.unsafe_free!(mem)
        CUDA.unsafe_free!(g0.ew); CUDA.unsafe_free!(g0.vw); CUDA.unsafe_free!(g0.pid)
        ownG = true
    end
    CUDA.synchronize()
    T2 = time()
    st[:twins] = T2 - T1; st[:nc] = NC
    # 2. nested dissection
    dest = CUDA.zeros(Int32, NC); cpos = CUDA.zeros(Int32, NC)
    gid = CuVector{Int32}(1:NC)
    np = 1
    poffH = Int32[1, NC + 1]; WpH = Int32[N]; pstartH = Int32[1]
    lposH = Int32[]; lsizeH = Int32[]
    lvl = 0
    nsep = 0
    seplog = Tuple{Int, Int, Int, Int}[]      # (level, parts, separator weight, part weight) per level
    lvlt = Float64[]
    codesD = nothing; pnodeH = Int32[1]        # top-k variant: tree codes, tree node of every part
    while np > 0
        # race (policy.jl): give up at an ND level boundary once the caller says so
        if abort !== nothing && abort[]
            hyb_prefetch_wait!()
            CUDA.synchronize()
            return nothing
        end
        tl0 = time()
        lvl += 1
        o = lvl <= o0.qnd ? GNDOpts(o0; fmlimit = o0.qlim, fmfine = 1 << 30) : o0
        poff = CuVector{Int32}(poffH); Wp = CuVector{Int32}(WpH)
        maxside = CuVector{Int32}([Int32(floor((1 + o.eps) / 2 * w)) for w in WpH])
        n = G.n; n32 = Int32(n)
        get(st, :debug, false) && (println("level $lvl: parts=$np n=$n m=$(G.m) maxW=$(maximum(WpH))"); flush(stdout))
        forceleaf = lvl > o.maxlevel
        kwl = o.kway > 0 && lvl <= o.kway && !forceleaf
        if o.hyb > 0 && !forceleaf && !kwl && maximum(diff(poffH)) >= o.hybmin && np <= Threads.nthreads() * HCAL.gsep / HCAL.csep
            # the CPU may refine this level's finest graph: stage it to the host while the GPU coarsens
            arena_reserve!(ceil(Int, 2.5 * (3n + G.m + np + 1)))
            hyb_prefetch!(G, poff)
        end
        if kwl
            lvl == 1 && (codesD = kway_codes(G, o, st))
            lab = CuVector{UInt8}(undef, n); pnD = CuVector{Int32}(pnodeH)
            @cuda threads = TPB blocks = nblk(n) k_kwlab!(lab, codesD, gid, G.pid, pnD, n32)
            CUDA.unsafe_free!(pnD)
            pwH = partweights(lab, G, np)
        elseif forceleaf
            lab = CUDA.fill(SA, n)
            pw = CUDA.zeros(Int32, 6np)
            @cuda threads = TPB blocks = nblk(n) k_pw!(pw, lab, G.vw, G.pid, n32)
            pwH = Array(pw)
        else
            ptarget = o.ptarget > 0 ? o.ptarget : 4 * CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            R = clamp(cld(ptarget, np), 1, o.rmax)
            if R == 1
                lab, pwH = separate(G, poff, np, Wp, maxside, o, lvl, st)
            else
                # METIS MlevelNodeBisectionL2: coarsen a few levels once, run the R trials on that graph
                levels = coarsen_levels(G, poff, np, Wp, o, lvl, st, o.rlev, o.C0 * o.rmin)
                top = levels[end]
                tpoffH = Array(top.poff)
                Gr, poffR = replicate(top.g, tpoffH, np, R)
                poffRD = CuVector{Int32}(poffR)
                WpR = CuVector{Int32}(repeat(WpH, R)); msR = CuVector{Int32}(repeat(Array(maxside), R))
                labR, pwR = separate(Gr, poffRD, R * np, WpR, msR, o, lvl, st)
                rbest = zeros(Int32, np)
                for p in 1:np
                    bk = (typemax(Int64), typemax(Int64)); br = 0
                    for r in 0:(R - 1)
                        q = 6 * (r * np + p - 1)
                        k = (Int64(pwR[q + 3]), Int64(abs(pwR[q + 1] - pwR[q + 2])))
                        k < bk && (bk = k; br = r)
                    end
                    rbest[p] = br
                end
                lab = CuVector{UInt8}(undef, top.g.n)
                rb = CuVector{Int32}(rbest)
                @cuda threads = TPB blocks = nblk(top.g.n) k_repgather!(lab, labR, top.g.pid, rb, Int32(top.g.n))
                foreach(CUDA.unsafe_free!, (labR, rb, poffRD, WpR, msR)); dfree!(Gr)
                t2 = time()
                lab = uncoarsen!(lab, levels, np, maxside, o, lvl, st)
                pwH = partweights(lab, G, np)
                st[:refine] = get(st, :refine, 0.0) + time() - t2
            end
            st[:reps] = push!(get(st, :reps, Int[]), R)
        end
        hyb_prefetch_wait!()
        hyb_gpu_update!()
        # decisions per part
        pk = zeros(Int32, 8np)
        noffH = Int32[1]; nWp = Int32[]; npst = Int32[]; nnode = Int32[]
        sw = 0
        for p in 1:np
            wA, wB, wS = pwH[6p - 5], pwH[6p - 4], pwH[6p - 3]
            cA, cB = pwH[6p - 2], pwH[6p - 1]
            W = wA + wB + wS
            ps = pstartH[p]
            q = 8(p - 1)
            pk[q + 7] = wA; pk[q + 8] = wB
            if forceleaf || (wS > o.sepmax * W) || max(wA, wB) == W || (wA == 0 && wB == 0)
                pk[q + 1] = 1
                push!(lposH, ps); push!(lsizeH, W)
                pk[q + 3] = length(lposH)
                continue
            end
            sw += wS
            pk[q + 6] = ps + wA + wB
            for (side, w, c, start) in ((0, wA, cA, ps), (1, wB, cB, ps + wA))
                kk = q + 2 + 2side
                if w == 0
                    pk[kk] = 0
                elseif w <= o.Lb
                    pk[kk] = 1
                    push!(lposH, start); push!(lsizeH, w)
                    pk[kk + 1] = length(lposH)
                else
                    pk[kk] = 2
                    push!(nWp, w); push!(npst, start); push!(noffH, noffH[end] + c)
                    push!(nnode, Int32(2) * pnodeH[p] + Int32(side))
                    pk[kk + 1] = length(nWp)
                end
            end
        end
        push!(seplog, (lvl, np, sw, sum(WpH)))
        nsep += sw
        nn = Int(noffH[end]) - 1
        s1 = CuVector{Int64}(undef, n); s2 = CuVector{Int64}(undef, n); s3 = CuVector{Int32}(undef, n)
        @cuda threads = TPB blocks = nblk(n) k_sflags!(s1, s2, s3, lab, G.vw, n32)
        t1 = iscan(s1); t2 = iscan(s2); t3 = iscan(s3)
        foreach(CUDA.unsafe_free!, (s1, s2, s3))
        newidx = CuVector{Int32}(undef, n)
        ngid = CuVector{Int32}(undef, max(nn, 1)); npid = CuVector{Int32}(undef, max(nn, 1)); nvw = CuVector{Int32}(undef, max(nn, 1))
        pkD = CuVector{Int32}(pk); noffD = CuVector{Int32}(noffH)
        @cuda threads = TPB blocks = nblk(n) k_split!(newidx, ngid, npid, nvw, dest, cpos, lab, G.vw, G.pid, gid, poff, t1, t2, t3,
            pkD, noffD, n32)
        foreach(CUDA.unsafe_free!, (t1, t2, t3, pkD, noffD, poff, Wp, maxside))
        if nn > 0
            ndeg = CUDA.zeros(Int32, nn)
            @cuda threads = TPB blocks = nblk(n) k_ecount!(ndeg, newidx, lab, G.ptr, G.adj, n32)
            nptr = ptr_from_counts(ndeg)
            nm = Int(lastval(nptr)) - 1
            nadj = CuVector{Int32}(undef, max(nm, 1)); new_ew = CuVector{Int32}(undef, max(nm, 1))
            @cuda threads = TPB blocks = nblk(n) k_efill!(nadj, new_ew, nptr, newidx, lab, G.ptr, G.adj, G.ew, n32)
            CUDA.unsafe_free!(ndeg)
            NG = DG(nn, nm, nptr, nadj, new_ew, nvw, npid)
        else
            NG = nothing
            CUDA.unsafe_free!(nvw); CUDA.unsafe_free!(npid)
        end
        CUDA.unsafe_free!(newidx); CUDA.unsafe_free!(lab); CUDA.unsafe_free!(gid)
        if ownG
            dfree!(G)
        end
        ownG = true
        np = length(nWp)
        push!(lvlt, time() - tl0)
        np == 0 && break
        G = NG; gid = ngid
        poffH = noffH; WpH = nWp; pstartH = npst; pnodeH = nnode
    end
    codesD === nothing || CUDA.unsafe_free!(codesD)
    CUDA.synchronize()
    abort !== nothing && abort[] && return nothing
    T3 = time()
    fs = fmstats(); st[:fm_moves] = fs[1]; st[:fm_maxmoves_sum] = fs[2]; st[:fm_scan] = fs[3]
    st[:fm_cyc] = fs[5:9]     # clock cycles (sum over parts): argmax, move checks + swap, pulls, gain updates, chunk maxima
    st[:lvlt] = lvlt
    st[:nd] = T3 - T2; st[:levels] = lvl; st[:nsep] = nsep; st[:seplog] = seplog
    # 3. leaves
    nl = length(lposH)
    lunionH = Vector{Int32}(undef, nl + 1); lunionH[1] = 1
    for l in 1:nl
        lunionH[l + 1] = lunionH[l] + lsizeH[l]
    end
    NU = Int(lunionH[end]) - 1
    lunion = CuVector{Int32}(lunionH); lpos = CuVector{Int32}(lposH)
    perm = CUDA.zeros(Int32, N)
    uvert = CuVector{Int32}(undef, max(NU, 1)); lloc = CuVector{Int32}(undef, N); lid = CuVector{Int32}(undef, N)
    @cuda threads = TPB blocks = nblk(N) k_leafmap!(uvert, lloc, lid, perm, ccls, crank, dest, cpos, lunion, N32)
    udeg = CuVector{Int32}(undef, NU)
    NU32 = Int32(NU)
    NU > 0 && @cuda threads = TPB blocks = nblk(NU) k_lcount!(udeg, uvert, lid, ptr0, adj0, NU32)
    uptr = ptr_from_counts(udeg)
    MU = Int(lastval(uptr)) - 1
    uadj = CuVector{Int32}(undef, max(MU, 1))
    NU > 0 && @cuda threads = TPB blocks = nblk(NU) k_lfill!(uadj, uptr, uvert, lid, lloc, ptr0, adj0, NU32)
    if leafdev !== nothing
        # device leaves: leafdev(uptr, uadj, lunionH) -> device orders (local ids) in each leaf's union range
        CUDA.synchronize()
        T4 = time()
        st[:leafcut] = T4 - T3; st[:nleaves] = nl; st[:nleafv] = NU
        ocD = leafdev(uptr, uadj, lunionH)
        T6 = time()
        uleaf = CuVector{Int32}(undef, max(NU, 1))
        nl > 0 && @cuda threads = TPB blocks = nblk(nl) k_uleaf!(uleaf, lunion, Int32(nl))
        NU > 0 && @cuda threads = TPB blocks = nblk(NU) k_permleaf!(perm, ocD, uvert, uleaf, lunion, lpos, NU32)
        permH = Array(perm)
        T7 = time()
        foreach(CUDA.unsafe_free!, (ocD, uleaf, perm, uvert, lloc, lid, udeg, uptr, uadj, lunion, lpos, dest, cpos, ccls, crank, ptr0, adj0))
        st[:leafprep] = 0.0; st[:leaves] = T6 - T4; st[:assemble] = T7 - T6; st[:total] = T7 - T0
        st[:gc] = (Base.gc_time_ns() - gc0) / 1e9
        return permH
    end
    uptrH = Array(uptr); uadjH = Array(uadj)
    T4 = time()
    st[:leafcut] = T4 - T3; st[:nleaves] = nl; st[:nleafv] = NU
    csrs = Vector{Tuple{Vector{Int64}, SubArray{Int32, 1, Vector{Int32}, Tuple{UnitRange{Int64}}, true}}}(undef, nl)
    for l in 1:nl
        a = Int(lunionH[l]); b = Int(lunionH[l + 1])
        p0 = Int64(uptrH[a])
        pl = Vector{Int64}(undef, b - a + 1)
        @inbounds for i in a:b
            pl[i - a + 1] = Int64(uptrH[i]) - p0 + 1
        end
        csrs[l] = (pl, view(uadjH, Int(p0):Int(uptrH[b]) - 1))
    end
    T5 = time()
    orders = leafsolve(csrs)
    T6 = time()
    ocat = Vector{Int32}(undef, NU)
    for l in 1:nl
        a = Int(lunionH[l]); ol = orders[l]
        @inbounds for i in eachindex(ol)
            ocat[a + i - 1] = ol[i]
        end
    end
    ocD = CuVector{Int32}(ocat)
    uleaf = CuVector{Int32}(undef, max(NU, 1))
    nl > 0 && @cuda threads = TPB blocks = nblk(nl) k_uleaf!(uleaf, lunion, Int32(nl))
    NU > 0 && @cuda threads = TPB blocks = nblk(NU) k_permleaf!(perm, ocD, uvert, uleaf, lunion, lpos, NU32)
    permH = Array(perm)
    T7 = time()
    foreach(CUDA.unsafe_free!, (ocD, uleaf, perm, uvert, lloc, lid, udeg, uptr, uadj, lunion, lpos, dest, cpos, ccls, crank, ptr0, adj0))
    st[:leafprep] = T5 - T4; st[:leaves] = T6 - T5; st[:assemble] = T7 - T6; st[:total] = T7 - T0
    st[:gc] = (Base.gc_time_ns() - gc0) / 1e9
    return permH
end

end # module
