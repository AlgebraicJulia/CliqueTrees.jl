# Host side of the CPU+GPU hybrid refinement (ND_NOTES.md, "Closing the AMD gap"); included in module GND.
#
# The GPU coarsens and projects; the finest uncoarsening levels of an ND level (few, large parts, where the
# one-warp-per-part GPU FM is a chain of ~2 us dependent-load moves) are refined on the CPU threads Julia
# was started with: labels are downloaded once at the switch level, every finer level is projected and
# refined on the host (parts in parallel, largest first), and the final labels are uploaded once.
# The CPU FM is METIS FM_2WayNodeRefine1Sided (same rules as k_wfm1b!) with a binary max-heap of gains.
# The switch level is chosen per level from a cost model whose constants are measured at run time
# (GPU and CPU time per FM move, host<->device bandwidth, CPU time per vertex for projection/setup).

# ---------------------------------------------------------------- calibration state (measured, generic)
# Rates are per separator vertex (the FM work of a part grows with its separator; the move count per
# separator vertex is absorbed in the rate, so GPU and CPU limits/pre-passes need no separate model).
mutable struct HybCal
    gsep::Float64       # GPU s per separator vertex of the largest part, one refined level (pre-passes + FM)
    glaunch::Float64    # GPU fixed s per refined level (launches, syncs)
    csep::Float64       # CPU s per separator vertex, one thread, one level (FM)
    cvert::Float64      # CPU s per vertex per level, one thread (projection, part scan)
    bw::Float64         # device -> host bytes/s (into the pinned staging arena)
    lat::Float64        # s per transfer call (latency)
    cfix::Float64       # s per switch (synchronize, thread start, label download/upload)
    calibrated::Bool
end
const HCAL = HybCal(2e-5, 3e-4, 2e-6, 3e-9, 5e9, 2e-5, 3e-4, false)
ema!(old::Float64, new::Real, a = 0.3) = isfinite(new) && new > 0 ? (1 - a) * old + a * new : old

# per-thread workspace (heap); vertex-indexed arrays are shared (parts are disjoint vertex ranges)
mutable struct FMWork
    hk::Vector{Int64}
    hv::Vector{Int32}
    lg::Vector{Int32}
    sl::Vector{Int32}
end
FMWork() = FMWork(Int64[], Int32[], Int32[], Int32[])

mutable struct HostWS
    gain::Vector{Int32}
    hpos::Vector{Int32}
    works::Vector{FMWork}
end
const HWS = HostWS(Int32[], Int32[], FMWork[])
function hws(n::Int)
    w = HWS
    if length(w.gain) < n
        w.gain = Vector{Int32}(undef, n); w.hpos = zeros(Int32, n)
    end
    nt = Threads.maxthreadid()
    while length(w.works) < nt
        push!(w.works, FMWork())
    end
    return w
end

# ---------------------------------------------------------------- binary max-heap with positions
@inline hkey(g::Int32, v::Int32, hs::UInt32) = (Int64(g) << 32) | Int64(mix32(reinterpret(UInt32, v) ⊻ hs))

@inline function hup!(hk, hv, hpos, i::Int)
    @inbounds begin
        k = hk[i]; v = hv[i]
        while i > 1
            p = i >> 1
            hk[p] >= k && break
            hk[i] = hk[p]; hv[i] = hv[p]; hpos[hv[i]] = i
            i = p
        end
        hk[i] = k; hv[i] = v; hpos[v] = i
    end
    return
end
@inline function hdown!(hk, hv, hpos, i::Int, n::Int)
    @inbounds begin
        k = hk[i]; v = hv[i]
        while true
            c = i << 1
            c > n && break
            (c < n && hk[c + 1] > hk[c]) && (c += 1)
            hk[c] <= k && break
            hk[i] = hk[c]; hv[i] = hv[c]; hpos[hv[i]] = i
            i = c
        end
        hk[i] = k; hv[i] = v; hpos[v] = i
    end
    return
end

@inline function cgain(v::Int32, other::UInt8, ptr, adj, vw, lab)
    s = Int32(0)
    @inbounds for q in ptr[v]:(ptr[v + 1] - 1)
        u = adj[q]
        lab[u] == other && (s += vw[u])
    end
    return @inbounds vw[v] - s
end

# 1-sided node FM on part [a, b] (METIS FM_2WayNodeRefine1Sided); returns (moves, initial separator size)
# verts: the part's vertices (a range, or a list: then every other vertex has a label other than SA/SB/SS)
function cpu_fm1_part!(lab::Vector{UInt8}, ptr::Vector{Int32}, adj::Vector{Int32}, vw::Vector{Int32}, verts::AbstractVector, ms::Int32,
        npass::Int, lim0::Int, seed::UInt32, ws::HostWS, wk::FMWork)
    gain = ws.gain; hpos = ws.hpos
    sl = wk.sl; empty!(sl)
    wA = Int32(0); wB = Int32(0); wS = Int32(0)
    isempty(verts) && return 0, 0
    a = Int(first(verts)); nv = length(verts)
    @inbounds for v in verts
        l = lab[v]; w = vw[v]
        if l == SA
            wA += w
        elseif l == SB
            wB += w
        else
            wS += w; push!(sl, Int32(v))
        end
    end
    sp0 = length(sl)
    sp0 == 0 && return 0, 0
    hk = wk.hk; hv = wk.hv; lg = wk.lg
    length(hk) < nv && (resize!(hk, nv); resize!(hv, nv))
    totmv = 0
    to = wA < wB ? SB : SA
    for pass in 0:(2npass - 1)
        to = to == SA ? SB : SA
        other = to == SA ? SB : SA
        hs = mix32(seed + UInt32(pass) * 0x9e3779b1 + UInt32(a) * 0x85ebca6b)
        nh = 0
        @inbounds for v in sl
            g = cgain(v, other, ptr, adj, vw, lab)
            gain[v] = g
            nh += 1; hk[nh] = hkey(g, v, hs); hv[nh] = v; hpos[v] = nh
        end
        @inbounds for i in (nh >> 1):-1:1
            hdown!(hk, hv, hpos, i, nh)
        end
        ns = length(sl)
        limit = min(3ns, lim0)
        empty!(lg)
        initS = wS; bestS = wS; bestD = abs(wA - wB); bestA = wA; bestB = wB; bestlog = 0; nsw = 0; bestsw = -1
        @inbounds while nh > 0
            v = hv[1]; g = gain[v]
            wv = vw[v]
            (to == SA ? wA : wB) + wv > ms && break
            newS = wS - g
            pulledw = wv - g
            newD = abs((to == SA ? wA : wB) + wv - ((to == SA ? wB : wA) - pulledw))
            if !(newS < bestS || (newS == bestS && newD < bestD))
                (nsw - bestsw > 3limit || (nsw - bestsw > limit && Int64(newS) * 10 > Int64(bestS) * 11)) && break
            end
            # pop v
            hpos[v] = 0
            if nh > 1
                hk[1] = hk[nh]; hv[1] = hv[nh]; hpos[hv[1]] = 1
                nh -= 1
                hdown!(hk, hv, hpos, 1, nh)
            else
                nh = 0
            end
            push!(lg, (v << 3) | Int32(SS))
            lab[v] = to
            wS -= wv
            to == SA ? (wA += wv) : (wB += wv)
            for q in ptr[v]:(ptr[v + 1] - 1)
                u = adj[q]
                lab[u] == other || continue
                push!(lg, (u << 3) | Int32(other))
                lab[u] = SS
                wu = vw[u]
                wS += wu
                other == SA ? (wA -= wu) : (wB -= wu)
                push!(sl, u)
                s = Int32(0)
                for qq in ptr[u]:(ptr[u + 1] - 1)
                    x = adj[qq]
                    lx = lab[x]
                    if lx == other
                        s += vw[x]
                    elseif lx == SS
                        hp = hpos[x]
                        if hp != 0
                            gx = gain[x] + wu
                            gain[x] = gx
                            hk[hp] = hkey(gx, x, hs)
                            hup!(hk, hv, hpos, Int(hp))
                        end
                    end
                end
                gu = wu - s
                gain[u] = gu
                nh += 1; hk[nh] = hkey(gu, u, hs); hv[nh] = u
                hup!(hk, hv, hpos, nh)
            end
            D = abs(wA - wB)
            if wS < bestS || (wS == bestS && D < bestD)
                bestS = wS; bestD = D; bestA = wA; bestB = wB; bestlog = length(lg); bestsw = nsw
            end
            nsw += 1
        end
        @inbounds for i in 1:nh
            hpos[hv[i]] = 0
        end
        @inbounds for i in length(lg):-1:(bestlog + 1)
            e = lg[i]
            lab[e >> 3] = UInt8(e & Int32(7))
        end
        # new separator list: listed vertices (initial + pulled) that are in S after the rollback
        k = 0
        @inbounds for v in sl
            if lab[v] == SS
                k += 1; sl[k] = v
            end
        end
        resize!(sl, k)
        wA = bestA; wB = bestB; wS = bestS
        totmv += nsw
        (isodd(pass) && (bestlog == 0 || bestS >= initS)) && break
    end
    return totmv, sp0
end

# refine all parts (vertex ranges poff[p] .. poff[p+1]-1) on the CPU threads, largest parts first.
# Returns (moves, max moves of a part, sum of part times, sum of initial separator sizes, wall time).
function cpu_refine_parts!(lab::Vector{UInt8}, ptr::Vector{Int32}, adj::Vector{Int32}, vw::Vector{Int32}, poff::Vector{Int32},
        maxside::Vector{Int32}, npass::Int, lim0::Int, seed::UInt32)
    t0 = time()
    np = length(poff) - 1
    ws = hws(length(vw))
    order = sortperm(diff(poff); rev = true)
    nt = min(Threads.nthreads(), np)
    next = Threads.Atomic{Int}(1)
    K = Threads.maxthreadid()
    mv = zeros(Int, K); mmax = zeros(Int, K); tp = zeros(K); sps = zeros(Int, K)
    work = function (tid)
        wk = ws.works[tid]
        while true
            i = Threads.atomic_add!(next, 1)
            i > np && break
            p = order[i]
            t1 = time()
            m, sp = cpu_fm1_part!(lab, ptr, adj, vw, Int(poff[p]):(Int(poff[p + 1]) - 1), maxside[p], npass, lim0, seed, ws, wk)
            tp[tid] += time() - t1
            mv[tid] += m; mmax[tid] = max(mmax[tid], m); sps[tid] += sp
        end
    end
    if nt <= 1
        work(1)
    else
        @sync for k in 1:nt          # workspace per task index (tasks may migrate between threads)
            Threads.@spawn work(k)
        end
    end
    return sum(mv), maximum(mmax), sum(tp), sum(sps), time() - t0
end

# threaded host projection labf[v] = labc[cmap[v]]
function hproject!(labf::Vector{UInt8}, labc::Vector{UInt8}, cmap::Vector{Int32})
    n = length(labf)
    nt = Threads.nthreads()
    if nt == 1 || n < 50_000
        @inbounds for v in 1:n
            labf[v] = labc[cmap[v]]
        end
    else
        ch = cld(n, nt)
        @sync for t in 1:nt
            Threads.@spawn @inbounds for v in ((t - 1) * ch + 1):min(t * ch, n)
                labf[v] = labc[cmap[v]]
            end
        end
    end
    return labf
end


# ---------------------------------------------------------------- pinned staging arena (host copies of level graphs)
# One persistent page-locked Int32 buffer (grown by doubling, pinned with CUDA.pin); per ND level a bump
# allocator hands out unsafe_wrap'ed vectors. Requests that do not fit get ordinary (pageable) vectors, so
# the arena never moves while wrapped vectors are alive.
mutable struct PinArena
    buf::Vector{Int32}
    top::Int
    pinned::Bool
end
const ARENA = PinArena(Int32[], 0, false)
function arena_reserve!(n::Int)
    if length(ARENA.buf) < n
        ARENA.buf = Vector{Int32}(undef, max(n, 2 * length(ARENA.buf)))
        ARENA.pinned = try
            CUDA.pin(ARENA.buf); true
        catch
            false
        end
    end
    ARENA.top = 0
    return ARENA
end
function arena_take(n::Int)
    a = ARENA
    if a.top + n <= length(a.buf)
        v = unsafe_wrap(Array, pointer(a.buf, a.top + 1), n)
        a.top += n
        return v
    end
    return Vector{Int32}(undef, n)
end
dl!(x::CuVector{Int32}) = (h = arena_take(length(x)); length(x) > 0 && copyto!(h, x); h)

# host copy of a level graph: (ptr, adj, vw, poff, cmap)
dl_level(L, withcmap::Bool) = (dl!(L.g.ptr), dl!(L.g.adj), dl!(L.g.vw), dl!(L.poff), withcmap ? dl!(L.cmap) : Int32[])

# prefetch of the finest graph of an ND level on a helper task (own stream), started before coarsening:
# PF[] = (g, task) ; the task waits for an event recorded after g was built
const PF = Ref{Any}(nothing)
function hyb_prefetch!(G, poff::CuVector{Int32})
    ev = CuEvent(); CUDA.record(ev)
    t = Threads.@spawn begin
        CUDA.synchronize(ev)
        (dl!(G.ptr), dl!(G.adj), dl!(G.vw), dl!(poff))
    end
    PF[] = (G, t)
    return
end
# wait for a pending prefetch (before its graph is freed)
hyb_prefetch_wait!() = (PF[] === nothing || wait(PF[][2]); PF[] = nothing)

# ---------------------------------------------------------------- cost model and device glue
# Decide at uncoarsening level i (labels just projected on the device) whether the CPU refines levels i .. 1.
# Predicted remaining time (separator counts of finer levels grow ~ sqrt of the vertex ratio):
#   GPU: sum over levels the GPU would refine of  glaunch + gsep * spmax
#   CPU: cfix + sum over all levels of  csep * max(spmax, spsum / T) + cvert * n / T + 5 lat,
#        plus bytes / bw for the level graphs not yet on the host
function hyb_go_cpu(lab, levels, i::Int, np::Int, o, st)
    o.hyb == 2 && return true, -1
    L = levels[i]
    pw = CUDA.zeros(Int32, 6np)
    @cuda threads = TPB blocks = nblk(L.g.n) k_pw!(pw, lab, L.g.vw, L.g.pid, Int32(L.g.n))
    h = Array(pw); CUDA.unsafe_free!(pw)
    spmax = 0; spsum = 0
    @inbounds for p in 1:np
        c = Int(h[6p]); spmax = max(spmax, c); spsum += c
    end
    T = Threads.nthreads()
    c = HCAL
    pf = PF[] !== nothing && PF[][1] === levels[1].g
    tg = 0.0; tc = c.cfix; bytes = Float64(L.g.n + levels[1].g.n)
    for j in i:-1:1
        Lj = levels[j]
        f = sqrt(Lj.g.n / L.g.n)
        !(j > o.fmfine && j <= o.fmfine + o.fmskip) && (tg += c.glaunch + c.gsep * spmax * f)
        tc += c.csep * max(spmax * f, spsum * f / T) + c.cvert * Lj.g.n / T + 5 * c.lat
        (j == 1 && pf) || (bytes += 4.0 * (2 * Lj.g.n + Lj.g.m + 1) + (j < i ? 4.0 * Lj.g.n : 0.0))
    end
    tc += bytes / c.bw
    get(st, :trace, false) && push!(get!(st, :hybdec, Any[]), (i, np, spmax, spsum, tg, tc))
    return tc < tg, spmax
end

# GPU levels refined under the hybrid: (start event, end event, largest part separator); folded into the
# model at the end of each ND level (events are complete by then)
const HPEND = Tuple{CuEvent, CuEvent, Int}[]
function hyb_gpu_update!()
    isempty(HPEND) && return
    c = HCAL
    for (e0, e1, sp) in HPEND
        t = Float64(CUDA.elapsed(e0, e1))
        if sp >= 64
            c.gsep = ema!(c.gsep, max(t - c.glaunch, 0.1 * t) / sp, 0.2)
        elseif sp <= 16
            c.glaunch = ema!(c.glaunch, t, 0.2)
        end
    end
    empty!(HPEND)
    return
end

# CPU refinement of levels i .. 1; lab = device labels of level i (projected, unrefined). Frees what uncoarsen! would.
function hyb_finish!(lab, levels, i::Int, np::Int, maxside, o, lvl::Int, st)
    t0 = time()
    CUDA.synchronize()
    pf = PF[] !== nothing && PF[][1] === levels[1].g ? PF[][2] : nothing
    # downloads in refinement order on one helper task (own stream), overlapping the CPU work on coarser levels
    ch = Channel{NTuple{5, Vector{Int32}}}(i)
    dlt = Threads.@spawn begin
        for j in i:-1:1
            if j == 1 && pf !== nothing
                p, a, w, po = fetch(pf)
                put!(ch, (p, a, w, po, j < i ? dl!(levels[1].cmap) : Int32[]))
            else
                put!(ch, dl_level(levels[j], j < i))
            end
        end
    end
    labH = Array(lab); CUDA.unsafe_free!(lab)
    msH = Array(maxside)
    t1 = time()
    T = Threads.nthreads()
    seed = mix32(o.seed * 0x2545f491 + UInt32(levels[1].g.n))
    tproj = 0.0; tref = 0.0; twait = 0.0; mvs = 0; tpart = 0.0; sps = 0; nproj = 0
    for j in i:-1:1
        tw = time()
        ptrH, adjH, vwH, poffH, cmH = take!(ch)
        tp = time(); twait += tp - tw
        if j < i
            lf = Vector{UInt8}(undef, length(vwH)); hproject!(lf, labH, cmH); labH = lf
            nproj += length(vwH)
        end
        tr = time(); tproj += tr - tp
        m, mm, tps, sp, wall = cpu_refine_parts!(labH, ptrH, adjH, vwH, poffH, msH, o.cfmpass, o.cfmlimit, seed)
        tref += wall; mvs += m; tpart += tps; sps += sp
    end
    wait(dlt)
    pf !== nothing && (PF[] = nothing)
    t2 = time()
    labD = CuVector{UInt8}(labH)
    for j in 1:i
        dfree!(levels[j + 1].g); CUDA.unsafe_free!(levels[j + 1].poff); CUDA.unsafe_free!(levels[j].cmap)
    end
    t3 = time()
    # online CPU rates (single-thread seconds per separator vertex / per projected vertex)
    sps > 0 && (HCAL.csep = ema!(HCAL.csep, tpart / sps, 0.2))
    st[:hyb_n] = get(st, :hyb_n, 0) + 1
    st[:hyb_dl] = get(st, :hyb_dl, 0.0) + (t1 - t0) + twait
    st[:hyb_proj] = get(st, :hyb_proj, 0.0) + tproj
    st[:hyb_ref] = get(st, :hyb_ref, 0.0) + tref
    st[:hyb_up] = get(st, :hyb_up, 0.0) + (t3 - t2)
    st[:hyb_moves] = get(st, :hyb_moves, 0) + mvs
    get(st, :trace, false) && push!(get!(st, :hyblog, Any[]), (lvl, i, np, levels[1].g.n, t1 - t0, twait, tproj, tref, t3 - t2, mvs, sps, tpart))
    return labD
end

# ---------------------------------------------------------------- calibration (measured on this machine; no device constants)
const HCALIB = Ref(false)
const HCALREC = Tuple{Int, Int, Float64}[]     # (spmax, np, seconds) per GPU-refined level

# 2D/3D grid graph as a sorted 1-based CSR (calibration input)
function grid_csr(d::NTuple{N, Int}) where {N}
    n = prod(d)
    li = LinearIndices(d); ci = CartesianIndices(d)
    ptr = Vector{Int64}(undef, n + 1); adj = Int32[]
    ptr[1] = 1
    for v in 1:n
        c = ci[v]; nb = Int32[]
        for k in 1:N, s in (-1, 1)
            t = Tuple(c); x = t[k] + s
            1 <= x <= d[k] || continue
            push!(nb, li[CartesianIndex(ntuple(j -> j == k ? x : t[j], N))])
        end
        sort!(nb); append!(adj, nb); ptr[v + 1] = length(adj) + 1
    end
    return ptr, adj
end

medianf(x) = isempty(x) ? NaN : (s = sort(x); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

"""
    hyb_calibrate!(; graphs) -> HCAL

Measures the cost-model rates on this machine (not part of any timed region): GPU time per refined level
as glaunch + gsep * (largest part's separator) from synchronized per-level timings with every level on the
device (robust: medians), CPU seconds per separator vertex with every level on the host threads, the
device->host bandwidth into the pinned arena, the per-transfer latency and the fixed cost of a switch.
"""
function hyb_calibrate!(; graphs = ((300, 300), (30, 30, 30), (500, 500)), Lb = 1024)
    nul = csrs -> [Int32.(1:(length(p) - 1)) for (p, _) in csrs]
    recs = Tuple{Int, Int, Float64}[]; csum = 0.0; ssum = 0; fix = Float64[]
    for d in graphs
        ptr, adj = grid_csr(d)
        for rep in 1:2
            empty!(HCALREC)
            HCALIB[] = true
            try
                gnd_perm(ptr, adj; opts = GNDOpts(; Lb = Lb, hyb = 0), leafsolve = nul)
            finally
                HCALIB[] = false
            end
            rep == 2 && append!(recs, HCALREC)
            st = Dict{Symbol, Any}(:trace => true)
            gnd_perm(ptr, adj; opts = GNDOpts(; Lb = Lb, hyb = 2), leafsolve = nul, timing = st)
            if rep == 2
                for r in get(st, :hyblog, [])
                    csum += r[12]; ssum += r[11]
                    push!(fix, r[5] + r[9])
                end
            end
        end
    end
    sp = [Float64(r[1]) for r in recs]; ts = [r[3] for r in recs]
    if !isempty(sp)
        q = sort(sp)[max(1, length(sp) ÷ 4)]
        gl = medianf([t for (x, t) in zip(sp, ts) if x <= q])
        big = [(t - gl) / x for (x, t) in zip(sp, ts) if x >= medianf(sp) && x > 0]
        isfinite(gl) && (HCAL.glaunch = gl)
        g = medianf(big); isfinite(g) && g > 0 && (HCAL.gsep = g)
    end
    ssum > 0 && (HCAL.csep = csum / ssum)
    isempty(fix) || (HCAL.cfix = medianf(fix))
    x = CUDA.zeros(Int32, 1 << 24)
    arena_reserve!(1 << 24)
    h = arena_take(1 << 24); copyto!(h, x)
    t = minimum(@elapsed(copyto!(h, x)) for _ in 1:3)
    HCAL.bw = 4 * (1 << 24) / t
    y = CUDA.zeros(Int32, 16); hy = zeros(Int32, 16); copyto!(hy, y)
    HCAL.lat = minimum(@elapsed(copyto!(hy, y)) for _ in 1:20)
    CUDA.unsafe_free!(x); CUDA.unsafe_free!(y)
    HCAL.calibrated = true
    return HCAL
end
