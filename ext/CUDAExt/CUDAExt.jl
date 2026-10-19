#
# The rounds of the exact searches on the GPU (see bt/backend.jl).
#
#   - SepEnumGPU and SepDPGPU are sepenum and sepdp: the same lists of
#     separators in the same order, the same blocks, values, and caps,
#     and the same orderings. A set is a word (n <= 64) or a tuple of
#     W words (n <= 64W), up to NMAX vertices.
#   - PIDGPU decides the rounds by a positive-instance driven search
#     (Tamaki 2017), downward from a greedy ordering (see pidround!);
#     its orderings are not those of sepdp.
#
# A search with the backend GPUSeptw runs its rounds on the CPU until
# it has spent its budget (bt_esc, bt_round!). The round that runs out
# of budget, and every round after it, go to the GPU: a round whose
# separators the CPU has listed goes on with GPU sepdp on them. A round
# that fails on the GPU (out of memory, a stack or a bag too large for
# the device code) is run on the CPU, and so are the rounds after it.
# The GPU runs the rounds of one search at a time; a round that finds
# it busy runs on the CPU.
#
# The budget is delta times the cost of a GPU round on a small graph,
# measured once on the device at hand (cost!): a search that the CPU
# finishes in less pays nothing for the GPU, and one that it does not
# finish pays at most delta times its GPU cost for the CPU.
#
module CUDAExt

using CUDA
using CliqueTrees
using CliqueTrees: GPUSeptw, BTStatus, BT_FOUND, BT_ABOVE, BT_TIMEOUT, bt_int, bt_next, sepenum, sepdp
using Graphs

include("bits.jl")
include("sepenum.jl")
include("sepdp.jl")
include("pid/pid.jl")

using .GPUBits: Bits
using .SepEnumGPU: SepEnumGPU
using .SepDPGPU: SepDPGPU
using .PIDGPU: PIDGPU

# the largest graph of SepEnumGPU and SepDPGPU
const NMAX = 512

# the GPU runs one search at a time
const LOCK = ReentrantLock()

# the cost of a GPU round on a small graph, for each engine (see cost!),
# and whether the device has failed for good
const COST = Dict{Symbol, Float64}()
const BROKEN = Ref(false)

# counts of the rounds: on the CPU, escalated, on the GPU (found, above,
# timeout), failed on the GPU, and busy; and the seconds of the rounds on
# the CPU and on the GPU
const STATS = Dict{Symbol, Float64}()
const STATSLOCK = ReentrantLock()

function count!(key::Symbol, x::Real = 1)
    lock(STATSLOCK) do
        STATS[key] = get(STATS, key, 0.0) + x
    end

    return
end

function resetstats!()
    lock(STATSLOCK) do
        empty!(STATS)
    end

    return
end

usable() = !BROKEN[] && CUDA.functional()

# The state of the rounds of a search: the backend, the end of the CPU
# budget, whether the rounds are on the GPU or (after a failure) on the
# CPU for good, the times of the last two CPU rounds, and the graph of
# PIDGPU (made at the first PID round).
mutable struct Esc
    backend::GPUSeptw
    cdl::Float64
    gpu::Bool
    off::Bool
    t1::Float64
    t2::Float64
    pid::Any
end

function CliqueTrees.bt_esc(backend::GPUSeptw, weights::AbstractVector, graph::AbstractGraph, dl::Real)
    n = nv(graph)
    n < max(backend.nmin, 2) && return nothing
    backend.engine === :septw && n > NMAX && return nothing
    usable() || return nothing

    if isnan(backend.budget)
        c = cost!(backend.engine)
        isnan(c) && return nothing
        budget = backend.delta * c
    else
        budget = backend.budget
    end

    return Esc(backend, time() + budget, false, false, 0.0, 0.0, nothing)
end

function CliqueTrees.bt_round!(esc::Esc, weights::AbstractVector{W}, graph::AbstractGraph, k::W, dl::Real, nt::Integer) where {W}
    sep = nothing

    if !esc.gpu && !esc.off
        ts = time()

        # the next round, judged by the growth of the last two
        est = ispositive(esc.t1) ? max(2.0, ispositive(esc.t2) ? esc.t1 / esc.t2 : 2.0) * esc.t1 : 0.0

        if ts + est <= esc.cdl
            cdl = min(dl, esc.cdl)
            list, status = sepenum(weights, graph, k, cdl, nt)

            if status != BT_TIMEOUT
                perm, _, mink, status = sepdp(weights, graph, list, k, cdl, nt)

                if status != BT_TIMEOUT
                    esc.t2 = esc.t1; esc.t1 = time() - ts; count!(:cpu); count!(:tcpu, esc.t1)
                    status == BT_ABOVE && return perm, bt_next(weights, graph, k, mink), BT_ABOVE
                    return perm, mink, status
                end

                sep = list
            end
        end

        count!(:tcpu, time() - ts)
        time() > dl && return Int[], k, BT_TIMEOUT
        esc.gpu = true; count!(:escalated)
    end

    if !esc.off
        result = gpuround!(esc, weights, graph, k, dl, sep)
        result isa Tuple && return result
        result === :failed && (esc.off = true)
    end

    ts = time(); result = CliqueTrees.bt_round!(nothing, weights, graph, k, dl, nt)
    count!(:cpu); count!(:tcpu, time() - ts)
    return result
end

ispositive(x::Real) = x > zero(x)

# A round on the GPU: (perm, next, status), :busy, or :failed.
function gpuround!(esc::Esc, weights::AbstractVector{W}, graph::AbstractGraph, k::W, dl::Real, sep) where {W}
    trylock(LOCK) || (count!(:busy); return :busy)
    ts = time()

    try
        if esc.backend.engine === :pid
            result = pidround!(esc, weights, graph, k, dl)
        else
            result = septwround(weights, graph, k, dl, sep)
        end

        if result isa Tuple
            status = result[3]
            count!(status == BT_FOUND ? :gpufound : status == BT_ABOVE ? :gpuabove : :gputimeout)
        else
            count!(:gpufailed)
        end

        return result
    catch err
        err isa InterruptException && rethrow()
        failed!(err)
        return :failed
    finally
        count!(:tgpu, time() - ts)
        unlock(LOCK)
    end
end

# After a failure of the device: out of memory is not fatal, but an error
# of a kernel spoils the context, and the GPU is not used again.
function failed!(err)
    count!(:gpufailed)

    if err isa CUDA.OutOfGPUMemoryError || err isa ErrorException
        @debug "septw: GPU round failed" exception = err
    else
        BROKEN[] = true
        @warn "septw: the GPU failed; the exact searches stay on the CPU" exception = err maxlog = 1
    end

    try
        GC.gc(false); CUDA.reclaim()
    catch
        BROKEN[] = true
    end

    return
end

# the type of a set of n vertices on the device
settype(n::Integer) = n <= 64 ? UInt64 : Bits{cld(n, 64) <= 2 ? 2 : cld(n, 64) <= 4 ? 4 : 8}

# A round of sepenum and sepdp on the GPU, or sepdp alone on the
# separators sep that the CPU has listed.
function septwround(weights::AbstractVector{W}, graph::AbstractGraph, k::W, dl::Real, sep) where {W}
    n = nv(graph); T = settype(n); wgt = Float64[weights[v] for v in 1:n]; dl = Float64(dl)

    if isnothing(sep)
        batch = SepEnumGPU.prepare(T, [graph], [wgt], [Float64(k)])
        lists, ovf, _, info = SepEnumGPU.gpu_sepenum(batch; split = 3, dl)
        info.aborted && return Int[], k, BT_TIMEOUT
        ovf[1] && return :failed
        sep = SepEnumGPU.sepgraph(lists[1], n)
    end

    time() > dl && return Int[], k, BT_TIMEOUT
    res, info = SepDPGPU.gpu_sepdp(T, [wgt], [graph], [sep], [Float64(k)]; dl)
    info.aborted && return Int[], k, BT_TIMEOUT
    info.novf > 0 && return :failed
    perm, _, mink, found = res[1]
    mink = towidth(W, mink)
    found === :found && return perm, mink, BT_FOUND
    return perm, bt_next(weights, graph, k, mink), BT_ABOVE
end

towidth(::Type{W}, x::Float64) where {W} = isfinite(x) ? convert(W, x) : typemax(W)
towidth(::Type{W}, x::Float64) where {W <: AbstractFloat} = convert(W, x)

# A round decided by PIDGPU, on the graph relabeled in reverse BFS order
# (the order of the search; see PIDGPU.vorder). The search goes down from
# an upper bound ub, the width of a greedy ordering (MF) at first: it asks
# for a decomposition lighter than ub, which, if there is one, lowers ub to
# its width. A positive instance is cheap for PIDGPU and a negative one is
# not, and only the last question is negative: it proves that ub is the
# width. The round at k is then answered from ub: found if ub <= k, above
# (with the next width ub) if the width ub is proved and above k. The
# bound and the ordering are kept for the next rounds of the search.
function pidround!(esc::Esc, weights::AbstractVector{W}, graph::AbstractGraph, k::W, dl::Real) where {W}
    n = nv(graph)

    if isnothing(esc.pid)
        adj = [Int[x for x in neighbors(graph, v) if x != v] for v in 1:n]
        wgt = Float64[weights[v] for v in 1:n]
        perm = PIDGPU.vorder(adj, :rbfs); pg = PIDGPU.relabel(adj, wgt, perm)
        order, _ = permutation(weights, graph, MF())
        esc.pid = (; perm, pg, gg = PIDGPU.GPUGraph(pg), order = Ref(collect(order)),
            ub = Ref(convert(W, treewidth(weights, graph, order))), proved = Ref(false))
    end

    (; perm, pg, gg, order, ub, proved) = esc.pid

    while true
        ub[] <= k && return order[], ub[], BT_FOUND
        proved[] && return Int[], ub[], BT_ABOVE

        r = PIDGPU.gpu_decide(gg, Float64(CliqueTrees.rtw_below(ub[])); tol = 0.0, deadline = Float64(dl))
        get(r.stats, :timeout, false) && return Int[], k, BT_TIMEOUT

        if r.yes
            new = perm[PIDGPU.gpu_order(pg, r)]; width = convert(W, treewidth(weights, graph, new))

            if width < ub[]
                order[] = new; ub[] = width
            else
                proved[] = true
            end
        else
            proved[] = true
        end
    end
end

# The cost of a GPU round on a small graph (a grid of 30 vertices, unit
# weights, at the width 5): compiled first, then the least of a few runs.
function cost!(engine::Symbol)
    haskey(COST, engine) && return COST[engine]

    lock(LOCK) do
        haskey(COST, engine) && return COST[engine]
        c = NaN

        try
            # (a new state for each run: the PID rounds keep their bounds)
            g = BipartiteGraph(Graphs.grid([5, 6])); w = ones(Float64, nv(g))
            esc() = Esc(GPUSeptw(; engine), Inf, true, false, 0.0, 0.0, nothing)
            once() = engine === :pid ? pidround!(esc(), w, g, 5.0, Inf) : septwround(w, g, 5.0, Inf, nothing)
            once(); c = minimum(@elapsed(once()) for _ in 1:5)
        catch err
            err isa InterruptException && rethrow()
            failed!(err)
        end

        COST[engine] = c
        return c
    end
end

# Compile the device code for every set type up to nmax vertices, and
# measure the cost of a round.
function CliqueTrees.septwwarmup!(backend::GPUSeptw; nmax::Integer = NMAX)
    usable() || return backend
    cost!(backend.engine)

    lock(LOCK) do
        nwrd = backend.engine === :pid ? (1:cld(nmax, 64)) : unique(nwords(settype(64w)) for w in 1:cld(min(nmax, NMAX), 64))

        for w in nwrd
            # a cycle with a chord, with two weights
            n = 64(w - 1) + 10; g = cycle_graph(n); add_edge!(g, 1, n ÷ 2); g = BipartiteGraph(g)
            wgt = Float64[isodd(v) ? 1.0 : 2.0 for v in 1:n]

            try
                if backend.engine === :pid
                    esc = Esc(backend, Inf, true, false, 0.0, 0.0, nothing); pidround!(esc, wgt, g, 3.0, Inf)
                else
                    septwround(wgt, g, 3.0, Inf, nothing)
                end
            catch err
                err isa InterruptException && rethrow()
                failed!(err)
            end
        end
    end

    return backend
end

nwords(::Type{UInt64}) = 1
nwords(::Type{Bits{W}}) where {W} = W

end
