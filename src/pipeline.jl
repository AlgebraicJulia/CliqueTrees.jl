# ---------------------------------------------------------------------------
# A hybrid CPU+GPU pipeline for tree decompositions
#
# The pipeline orders a graph by stages, each run where it is fastest:
#
#   1. the front end, on the CPU: the safe reduction rules (with the true
#      twins compressed between rounds), then the safe separators, which
#      split the reduced graph into the leaves of its safe completion (the
#      atoms). The width of the graph is the largest width of an atom.
#   2. the atoms, widest first, each by the cheapest stage that can still
#      lower the width of the graph:
#        - certified: the lower bound (lb) meets the best greedy ordering
#          (MF, MF with strategy 1); nothing to do;
#        - exact (at most nexact vertices): a lower bound (MTW), septw from
#          the largest lower bound for a short share of the time, then the
#          contraction-recursive search RTW, which certifies its answer.
#          The rounds of septw (also those inside RTW) are raced between
#          the CPU and the GPU by the backend GPUSeptw (see bt/backend.jl):
#          each round starts on the CPU, goes to the GPU too when the GPU
#          is expected to be faster, and is answered by the first to
#          finish;
#        - the rest: the anytime search HBT, whose exact steps run on the
#          backend `escalate`, in turns, until the atom is no longer the
#          widest;
#   3. stop when certified: an atom stops when its width meets the largest
#      lower bound of any atom (no ordering of the graph is lighter), and
#      the pipeline stops when the widest atom does.
#   4. huge graphs (more than nhuge vertices): no safe separators; nested
#      dissection (METIS, if loaded) on the top levels, the algorithm leaf
#      (MF with strategy 1, or its GPU rounds) on the leaves, then HBT where
#      the time allows (see hy_huge).
#
# Every raced atom is recorded (its size and density, its time, the rounds
# that the GPU won) in a routing model (HyModel), learned online: an atom
# whose neighbours in size and density never had a round won by the GPU
# skips the race and runs on the CPU alone. Nothing depends on the device
# at hand except through these measurements and those of the backend.
# ---------------------------------------------------------------------------

"""
    HybridTW{L, M, H, B, A, D, P, R} <: EliminationAlgorithm

    HybridTW(; time=10.0, threads=Threads.nthreads(), gpu=true, engine=:auto, nexact=600, kwargs...)

A hybrid CPU+GPU pipeline for small-width tree decompositions: safe reduction rules and safe
separators split the graph into atoms; each atom is certified by its bounds, solved exactly
(its rounds raced between the CPU and the GPU), or improved by the anytime search
[`HBT`](@ref), widest first, until the width of the graph is certified or the time runs out.
See [`hybridtw`](@ref) for the result with its certificate.

The GPU stages need the package extension of CUDA.jl and a working device; without them, the
pipeline runs on the CPU and gives the same kind of answer.

### Parameters

  - `time`: time budget in seconds
  - `threads`: number of tasks of the CPU searches
  - `seed`: random seed of the anytime search
  - `lb`: lower bound algorithm of the atoms
  - `rules`: level of the safe reduction rules
  - `mins`: minimal triangulations of the safe separators
  - `maxmissing`: largest number of missing edges of a safe separator given to the slow tests
  - `nexact`: atoms with at most this many vertices are solved exactly first
  - `probe`: share of the time left given to the exact search of an atom by septw (and half
    of it to its lower bound [`MTW`](@ref), if `mtw`)
  - `rfrac`: share of the time left given next to [`RTW`](@ref), if septw did not solve the atom
  - `mtw`: compute the lower bound MTW of the atoms that are searched exactly
  - `gpu`: race the rounds of the exact searches on the GPU (with `GPUSeptw`, which needs an
    interactive thread for its queue: `julia -t n,2`)
  - `engine`: the GPU engine of the races (`:auto`, `:pid`, or `:septw`, see [`GPUSeptw`](@ref))
  - `escalate`: the backend of the exact steps of HBT ([`CPUSeptw`](@ref), [`GPUSeptw`](@ref)), or
    `nothing`: `GPUSeptw()` if the GPU is used, else the backend in place
  - `hbt`: parameters of the anytime search (its `time` and `threads` are not used)
  - `nhuge`: graphs with more vertices take the path of huge graphs
  - `ndlevel`: levels of nested dissection of a huge graph
  - `leaf`: the elimination algorithm of the leaves of that dissection (for instance the GPU
    rounds of MF with strategy 1, which are a prototype for now)
  - `dissection`: the separator algorithm of a nested dissection that seeds HBT on an atom
    that is most of the graph (the safe separators found no small ones, as in expander-like
    networks): `nothing` for [`KaHyParND`](@ref) if KaHyPar.jl is loaded, else
    [`METISND`](@ref) if Metis.jl is, else none; or `false` for none
  - `portfolio`: the fast orderings that [`hybridcost`](@ref) compares with that of the pipeline
  - `refine`: a cost-based refinement for [`hybridcost`](@ref): a function
    `(weights, hypergraph, point, parent) -> parent` of a contraction tree, or `nothing`
  - `model`: a file in which the routing model is kept between runs, or `nothing`
  - `verbose`: print progress
"""
@kwdef struct HybridTW{L <: WidthOrAlgorithm, M <: Tuple, H <: HBT, B, A <: EliminationAlgorithm, D, P <: Tuple, R} <: EliminationAlgorithm
    time::Float64 = 10.0
    threads::Int = Threads.nthreads()
    seed::Int = 1
    lb::L = DEFAULT_LOWER_BOUND_ALGORITHM
    rules::Int = 5
    mins::M = (MinimalChordal(MF()), MinimalChordal(MF(; strategy = 1)), MinimalChordal(AMF()), MinimalChordal(MMD()))
    maxmissing::Int = 50
    nexact::Int = 600
    probe::Float64 = 0.1
    rfrac::Float64 = 0.1
    mtw::Bool = true
    gpu::Bool = true
    engine::Symbol = :auto
    escalate::B = nothing
    hbt::H = HBT()
    nhuge::Int = 200_000
    ndlevel::Int = 12
    leaf::A = MF(; strategy = 1)
    dissection::D = nothing
    portfolio::P = (MF(), MF(; strategy = 1), AMF(), MMD())
    refine::R = nothing
    model::Union{Nothing, String} = nothing
    verbose::Bool = false
end

isconcurrent(::HybridTW) = false

"""
    HybridResult{V, W}

The result of [`hybridtw`](@ref).

  - `order`: an elimination ordering of the graph
  - `width`: its width (the largest weight of a bag)
  - `lower`: a lower bound on the width of every ordering
  - `certified`: whether `width == lower`, so that the ordering is optimal
  - `trace`: the pairs (seconds, width) at which the width of the graph fell
  - `atoms`: a record of each atom (vertices, edges, route, status, seconds, winner)
  - `time`: seconds spent
"""
struct HybridResult{V, W}
    order::Vector{V}
    width::W
    lower::W
    certified::Bool
    trace::Vector{Tuple{Float64, W}}
    atoms::Vector{Any}
    time::Float64
end

function Base.show(io::IO, r::HybridResult)
    print(io, "HybridResult(width = $(r.width), lower = $(r.lower), certified = $(r.certified), atoms = $(length(r.atoms)), time = $(round(r.time; digits = 3)))")
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::HybridTW; verbose::Bool = false)
    order = hybridtw(weights, graph, alg).order
    return order, invperm(order)
end

"""
    hybridtw(weights, graph, alg::HybridTW=HybridTW())

    hybridtw(graph, alg::HybridTW=HybridTW())

Run the pipeline [`HybridTW`](@ref) on a graph with vertex weights. Returns a
[`HybridResult`](@ref): an ordering, its width, a lower bound, and whether they meet.
Widths are those of [`treewidth`](@ref): with unit weights, the treewidth plus one.
"""
function hybridtw(graph, alg::HybridTW = HybridTW())
    g = BipartiteGraph(graph)
    return hybridtw(ones(Float64, nv(g)), g, alg)
end

function hybridtw(weights::AbstractVector, graph, alg::HybridTW = HybridTW())
    g = BipartiteGraph(graph)
    t0 = time(); dl = t0 + alg.time
    gpu = alg.gpu && hy_gpu!(alg.engine)
    model = hy_model!(alg.model)
    escalate = isnothing(alg.escalate) ? (gpu ? GPUSeptw() : septwbackend()) : alg.escalate
    old = septwbackend!(escalate)

    try
        if nv(g) > alg.nhuge
            return hy_huge(weights, g, alg, t0, dl, gpu, model)
        else
            return hy_pipe(weights, g, alg, t0, dl, gpu, model)
        end
    finally
        septwbackend!(old)
        hy_save!(model, alg.model)
    end
end

# The state of a run: the start, the trace of widths, and the record of the
# atoms.
struct HyRun{W}
    t0::Float64
    trace::Vector{Tuple{Float64, W}}
    atoms::Vector{Any}
    verbose::Bool
end

function hy_trace!(run::HyRun{W}, width) where {W}
    width = convert(W, width)

    if isempty(run.trace) || width < last(run.trace)[2]
        t = time() - run.t0; push!(run.trace, (t, width))
        run.verbose && println("hybrid: width $(hbt_str(width)) at $(round(t; digits = 3)) s")
    end

    return
end

# Stages 1-3.
function hy_pipe(weights::AbstractVector, graph::AbstractGraph{V}, alg::HybridTW, t0::Float64, dl::Float64, gpu::Bool, model) where {V}
    W = promote_type(eltype(weights), typeof(lowerbound(weights, graph, alg.lb)))
    weights = convert(Vector{W}, collect(view(weights, oneto(nv(graph)))))
    run = HyRun{W}(t0, Tuple{Float64, W}[], Any[], alg.verbose)

    # the front end: rules, then safe separators
    lb0 = convert(W, lowerbound(weights, graph, alg.lb))
    weights10, graph10, rules = compressreduce(alg.rules, weights, graph, lb0; minfrac = 1.0e-3)
    lb = max(lb0, convert(W, rules.width))

    if nv(graph10) > 1
        weights30, graphs30, safe = safesolve(weights10, graph10, alg.mins, alg.maxmissing)
        lb = max(lb, convert(W, safe.width))
        alg.verbose && println("hybrid: $(nv(graph)) vertices, $(nv(graph10)) after the rules, $(length(graphs30)) atoms (the largest with $(maximum(nv, graphs30; init = 0)) vertices) in $(round(time() - t0; digits = 3)) s")
        orders, lb = hy_atoms!(run, weights30, graphs30, lb, alg, dl, gpu, model, nv(graph10))
        order10 = safe_pull(safe, graphs30, orders)
    else
        order10 = collect(V, oneto(nv(graph10)))
    end

    order = convert(Vector{V}, rule_pull(rules, order10))
    width = treewidth(weights, graph, order)
    hy_trace!(run, width)
    return hy_result(order, width, lb, run, t0)
end

# The atoms: returns their orderings and a lower bound on the width of the
# graph, at least lb.
function hy_atoms!(run::HyRun{W}, weights::AbstractVector, graphs::AbstractVector{<:AbstractGraph{V}}, lb::W, alg::HybridTW, dl::Float64, gpu::Bool, model, nall::Integer = sum(nv, graphs; init = 0)) where {W, V}
    natom = length(graphs)
    items = map(i -> hbt_graph(weights[i], graphs[i]), oneto(natom))
    orders = Vector{Vector{V}}(undef, natom)
    widths = Vector{W}(undef, natom)
    lowers = Vector{W}(undef, natom)
    routes = fill(:certified, natom)
    visits = zeros(Int, natom)      # the exact visits of each atom
    texact = zeros(natom)           # the seconds of its last exact visit
    exact = falses(natom)           # whether its last turn was exact
    xgo = trues(natom)              # whether its last exact visit made progress
    searches = Vector{Any}(nothing, natom)

    # the bounds of each atom, and its best greedy ordering
    for i in oneto(natom)
        bgraph, wgt = items[i]
        lowers[i] = convert(W, lowerbound(wgt, bgraph, alg.lb))
        widths[i] = typemax(W)

        for greedy in (MF(), MF(; strategy = 1))
            widths[i] <= lowers[i] && break
            order, _ = permutation(wgt, bgraph, greedy)
            width = convert(W, treewidth(wgt, bgraph, order))

            if width < widths[i]
                orders[i] = convert(Vector{V}, order); widths[i] = width
            end
        end
    end

    lb = max(lb, maximum(lowers; init = lb))
    width() = max(lb, maximum(widths; init = lb))
    hy_trace!(run, width())

    while ispositive(natom)
        k, i = findmax(widths)
        k <= lb && break
        time() >= dl && break
        bgraph, wgt = items[i]; n = nv(bgraph)

        # the width of the next widest atom
        ub = nothing

        for j in oneto(natom)
            j == i || (ub = isnothing(ub) ? widths[j] : max(ub, widths[j]))
        end

        if n <= alg.nexact && xgo[i] && !exact[i]
            # the exact stages, from the largest lower bound: an ordering no
            # wider than lb is as good as an optimal one. A visit takes the
            # shares probe and rfrac of the time left, or twice the last
            # visit if longer; the atom then gets as long a turn of HBT, and
            # so on, as long as each visit raises the lower bound or lowers
            # the width
            visits[i] += 1; exact[i] = true; ts = time(); lower0 = lowers[i]; width0 = widths[i]
            order, kx, status, rec = hy_exact(wgt, bgraph, lowers[i], lb, widths[i], dl, 2texact[i], isone(visits[i]), alg, gpu, model)
            texact[i] = time() - ts; routes[i] = rec.route

            if kx.width < widths[i]
                orders[i] = convert(Vector{V}, order); widths[i] = kx.width
            end

            lowers[i] = max(lowers[i], kx.lower); lb = max(lb, lowers[i])
            xgo[i] = lowers[i] > lower0 || widths[i] < width0
            push!(run.atoms, (; n, m = ne(bgraph), rec.route, status = Symbol(status), lo = lowers[i], width = widths[i], rec.winner, t = time() - ts, rec.tcpu, rec.tgpu, rec.won, rec.served))
            alg.verbose && println("hybrid: atom $i (n = $n) $(rec.route), $(status), width $(hbt_str(widths[i])), lb $(hbt_str(lb)), winner $(rec.winner), $(round(time() - ts; digits = 3)) s")
        else
            # the anytime search, in short turns (for the trace), until the
            # atom falls below the next widest or meets lb
            # (an atom searched exactly gets as long a turn as its last
            # exact visit, at least 0.25 seconds)
            routes[i] = :hbt; ts = time(); exact[i] = false
            tdl = n <= alg.nexact && xgo[i] ? min(dl, ts + max(0.25, texact[i])) : dl

            if isnothing(searches[i])
                # an atom that is most of the graph: a nested dissection may
                # start HBT better than the greedy orderings
                if 2n >= nall
                    nd = hy_dissect(wgt, bgraph, alg)

                    if !isnothing(nd)
                        knd = convert(W, treewidth(wgt, bgraph, nd))
                        alg.verbose && println("hybrid: atom $i (n = $n): nested dissection width $(hbt_str(knd)) (greedy $(hbt_str(widths[i]))) in $(round(time() - ts; digits = 3)) s")

                        if knd < widths[i]
                            orders[i] = convert(Vector{V}, nd); widths[i] = knd; hy_trace!(run, width())
                        end
                    end
                end

                searches[i] = hbt_new(wgt, bgraph, alg.hbt; seed = alg.seed, lb = lowers[i], init = orders[i], deadline = dl)
            end

            while time() < tdl
                mt = min(tdl - time(), max(0.25, (time() - run.t0) / 4))
                order, _, kh = hbtrun!(searches[i]; mt, nt = alg.threads, lb, ub)

                if kh < widths[i]
                    orders[i] = convert(Vector{V}, order); widths[i] = kh
                    hy_trace!(run, width())
                end

                (kh <= lb || (!isnothing(ub) && kh < ub)) && break
            end

            push!(run.atoms, (; n, m = ne(bgraph), route = :hbt, status = :anytime, lo = lowers[i], width = widths[i], winner = :cpu, t = time() - ts, tcpu = time() - ts, tgpu = NaN, won = 0, served = 0))
        end

        hy_trace!(run, width())
    end

    for i in oneto(natom)
        if routes[i] === :certified
            n = nv(items[i][1])
            push!(run.atoms, (; n, m = ne(items[i][1]), route = widths[i] <= lowers[i] ? :certified : :untouched, status = :bounds, lo = lowers[i], width = widths[i], winner = :none, t = 0.0, tcpu = 0.0, tgpu = NaN, won = 0, served = 0))
        end
    end

    return orders, lb
end

# ---------------------------------------------------------------------------
# The exact searches
#
# The rounds of the exact stages of an atom run on the backend of its
# route:
#
#   - :race: GPUSeptw (the GPU rounds of bt/backend.jl and the extension of
#     CUDA.jl). Each round starts on the CPU at once and asks for the GPU
#     when its expected CPU time exceeds its expected GPU time, both learned
#     from the measured rounds; the first of the two to finish answers, and
#     the other is stopped;
#   - :cpu: CPUSeptw, when the routing model says that the GPU does not pay
#     for atoms of this size and density (no race at all).
#
# The atoms are searched one at a time, so the backend is set for each.
# ---------------------------------------------------------------------------

# The backend of the races, with the given engine (see GPUSeptw).
function hy_backend(engine::Symbol)
    return GPUSeptw(; engine)
end

# Whether the GPU is usable: the extension is loaded and the backend gives
# a GPU state for a small graph.
function hy_gpu!(engine::Symbol)
    isnothing(Base.get_extension(CliqueTrees, :CUDAExt)) && return false
    g = hy_grid(5, 6)
    return !isnothing(bt_esc(hy_backend(engine), ones(Float64, nv(g)), g, Inf))
end

"""
    hybridwarmup!(; engine=:auto, nmax=600)

Compile the GPU rounds of [`HybridTW`](@ref) for atoms of up to `nmax` vertices, and measure
their cost (see [`septwwarmup!`](@ref)), which the first races would otherwise do. Returns
whether the GPU is usable.
"""
function hybridwarmup!(; engine::Symbol = :auto, nmax::Integer = 600)
    hy_gpu!(engine) || return false
    septwwarmup!(hy_backend(engine); nmax)
    return true
end

function hy_grid(r::Integer, c::Integer)
    n = r * c; ptr = [1]; tgt = Int[]

    for v in 1:n
        i, j = divrem(v - 1, c)
        i > 0 && push!(tgt, v - c)
        j > 0 && push!(tgt, v - 1)
        j < c - 1 && push!(tgt, v + 1)
        i < r - 1 && push!(tgt, v + c)
        push!(ptr, length(tgt) + 1)
    end

    return BipartiteGraph{Int, Int}(n, n, length(tgt), ptr, tgt)
end

# The counts of the GPU rounds (the statistics of the extension, if it is
# loaded): rounds answered by the GPU, rounds the GPU served, and seconds
# on the GPU.
function hy_counts()
    ext = Base.get_extension(CliqueTrees, :CUDAExt)
    (isnothing(ext) || !isdefined(ext, :STATS)) && return (0.0, 0.0, 0.0)
    stats = ext.STATS

    lock(ext.STATSLOCK) do
        return (get(stats, :gpuwon, 0.0), get(stats, :escalated, 0.0), get(stats, :tgpu, 0.0))
    end
end

# The exact stages of an atom with the lower bound lower and an ordering
# of width upper, in a batch with the lower bound lb, by the deadline dl:
#
#   1. on the first visit, a better lower bound, MTW, for at most half the
#      time of 2;
#   2. septw from the bound, for the share probe of the time left, or
#      tmin / 2 if longer (as the probes of RTW: a round is not started
#      that would not end in time);
#   3. RTW, from the bound, for the share rfrac of the time left, or
#      tmin / 2 if longer: the contraction-recursive search, anytime, which
#      certifies its answer.
#
# Every stage stops once the atom meets lb. The rounds of septw (in 2 and
# in the exact steps of 3) run on the backend of the route (see above).
# Returns (order, (; width, lower), status, record): the lightest ordering
# found (or nothing), its width, and a lower bound on the width of the
# atom.
function hy_exact(weights::AbstractVector{W}, graph::AbstractGraph, lower::W, lb::W, upper::W, dl::Float64, tmin::Float64, first::Bool, alg::HybridTW, gpu::Bool, model) where {W}
    t0 = time(); n = nv(graph); dens = hy_density(graph)
    route = gpu ? hy_route(model, n, dens) : :cpu
    old = septwbackend!(route === :race ? hy_backend(alg.engine) : CPUSeptw())
    won0, served0, tgpu0 = hy_counts()
    order = nothing; width = upper; status = BT_TIMEOUT; stage = :mtw

    try
        # 1. the lower bound
        pdl = min(dl, t0 + max(alg.probe * (dl - t0), tmin / 2))
        first && alg.mtw && (lower = max(lower, convert(W, lowerbound(weights, graph, rtw_lb(MTW(), (t0 + pdl) / 2)))))

        if max(lower, lb) >= upper
            status = BT_FOUND
        else
            # 2. septw
            stage = :septw; lo = max(lower, lb)
            perm, _, k, status = rtw_probe(weights, graph, lo, pdl, alg.threads)

            if status == BT_FOUND
                order = collect(perm); width = k
                k > lo && (lower = k)       # every round below k failed
            else
                status == BT_TIMEOUT && (lower = max(lower, k))
                lo = max(lower, lb)

                # 3. RTW
                if alg.rfrac > 0 && time() < dl && lo < upper
                    stage = :rtw; now = time(); rdl = min(dl, now + max(alg.rfrac * (dl - now), tmin / 2))
                    hbt = HBT(; time = min(2.0, (rdl - now) / 4), threads = alg.threads, seed = alg.seed)
                    perm, _, k, status, _ = rtw(weights, graph, lo, typemax(W), rdl, alg.threads, (; alg = MF(; strategy = 1), hbt, ticks = 100, verbose = false))

                    if k < width
                        order = collect(perm); width = k
                    end

                    status == BT_FOUND && k > lo && (lower = k)
                end
            end
        end
    finally
        septwbackend!(old)
    end

    t = time() - t0; won1, served1, tgpu1 = hy_counts()
    won = round(Int, won1 - won0); served = round(Int, served1 - served0)

    if route === :race
        hy_push!(model, HyRow(n, dens, t, status != BT_TIMEOUT, won, served, tgpu1 - tgpu0))
    end

    winner = ispositive(won) ? :gpu : :cpu
    return order, (; width, lower), status, (; route = Symbol(route, :_, stage), winner, tcpu = t, tgpu = tgpu1 - tgpu0, won, served)
end

hy_density(graph::AbstractGraph) = (n = nv(graph); n < 2 ? 1.0 : de(graph) / (n * (n - 1)))

# ---------------------------------------------------------------------------
# The routing model
#
# A row per raced atom: its size and density, the seconds of its search,
# whether it answered, the rounds that the GPU answered first (won) and
# that it served, and the seconds on the GPU. Atoms are neighbours if their
# sizes are within a factor sqrt 2 and their densities within a factor 2.
# An atom with at least HY_MIN neighbours, every one of which was answered
# without a round won by the GPU, skips the race (:cpu): the backend learns
# which rounds to send to the GPU by size alone, and this saves the queue,
# the warm-up of a new size, and the GPU time of the rounds it loses.
# ---------------------------------------------------------------------------

struct HyRow
    n::Int
    density::Float64
    t::Float64
    done::Bool
    won::Int
    served::Int
    tgpu::Float64
end

struct HyModel
    rows::Vector{HyRow}
    lock::ReentrantLock
    new::Vector{HyRow}
end

const HY_MODEL = HyModel(HyRow[], ReentrantLock(), HyRow[])
const HY_LOADED = Set{String}()
const HY_MIN = 3
const HY_HEAD = "n\tdensity\tt\tdone\twon\tserved\ttgpu"

function hy_model!(path)
    if !isnothing(path) && !(path in HY_LOADED) && isfile(path)
        lock(HY_MODEL.lock) do
            for line in Iterators.drop(eachline(path), 1)
                f = split(line, '\t'); length(f) == 7 || continue
                push!(HY_MODEL.rows, HyRow(parse(Int, f[1]), parse(Float64, f[2]), parse(Float64, f[3]), f[4] == "1",
                    parse(Int, f[5]), parse(Int, f[6]), parse(Float64, f[7])))
            end
        end

        push!(HY_LOADED, path)
    end

    return HY_MODEL
end

function hy_push!(model::HyModel, row::HyRow)
    lock(model.lock) do
        push!(model.rows, row); push!(model.new, row)
    end

    return
end

function hy_save!(model::HyModel, path)
    isnothing(path) && return

    lock(model.lock) do
        isempty(model.new) && return
        new = !isfile(path)

        open(path, "a") do io
            new && println(io, HY_HEAD)

            for r in model.new
                println(io, join((r.n, r.density, r.t, Int(r.done), r.won, r.served, r.tgpu), '\t'))
            end
        end

        empty!(model.new); push!(HY_LOADED, path)
    end

    return
end

function hy_route(model::HyModel, n::Integer, density::Real)
    nall = ncpu = 0

    lock(model.lock) do
        for r in model.rows
            if abs(log2(r.n / n)) <= 0.5 && abs(log2(max(r.density, 1.0e-9) / max(density, 1.0e-9))) <= 1.0
                nall += 1; ncpu += r.done && iszero(r.won)
            end
        end
    end

    return nall >= HY_MIN && ncpu == nall ? :cpu : :race
end

"""
    hybridmodel()

The rows of the routing model of [`HybridTW`](@ref), learned from the races so far: a named
tuple for each raced atom, with its size and density, the seconds of its search, whether it
answered, the rounds that the GPU won and served, and the seconds on the GPU.
"""
function hybridmodel()
    lock(HY_MODEL.lock) do
        return [(; r.n, r.density, r.t, r.done, r.won, r.served, r.tgpu) for r in HY_MODEL.rows]
    end
end

# ---------------------------------------------------------------------------
# Huge graphs
#
# Safe separators and exact searches do not pay on a graph with millions
# of vertices. The ordering is nested dissection on the top levels (METISND,
# with the package extension of Metis.jl), with the leaves ordered by
# alg.leaf; then HBT, from that ordering, for the time that is left, if it
# is at least twice that of the dissection.
# ---------------------------------------------------------------------------

function hy_huge(weights::AbstractVector, graph::AbstractGraph{V}, alg::HybridTW, t0::Float64, dl::Float64, gpu::Bool, model) where {V}
    lb = lowerbound(weights, graph, alg.lb); W = promote_type(eltype(weights), typeof(lb))
    weights = convert(Vector{W}, collect(view(weights, oneto(nv(graph))))); lb = convert(W, lb)
    run = HyRun{W}(t0, Tuple{Float64, W}[], Any[], alg.verbose)
    leaf = alg.leaf
    nd = isnothing(Base.get_extension(CliqueTrees, :MetisExt)) ? leaf : ND(leaf, METISND(; seed = 1); width = 16, level = alg.ndlevel, scale = 100)
    order, _ = permutation(weights, graph, nd)
    order = convert(Vector{V}, order)
    width = treewidth(weights, graph, order)
    hy_trace!(run, width)
    alg.verbose && println("hybrid: huge graph, $(nv(graph)) vertices, nested dissection in $(round(time() - t0; digits = 3)) s")

    # HBT only if the time left is at least twice that of the dissection
    if dl - time() > 2 * (time() - t0) && width > lb
        hbt = hbt_new(weights, graph, alg.hbt; seed = alg.seed, lb, init = order, deadline = dl)

        while time() < dl
            mt = min(dl - time(), max(0.25, (time() - t0) / 4))
            horder, _, kh = hbtrun!(hbt; mt, nt = alg.threads, lb)

            if kh < width
                order = convert(Vector{V}, horder); width = kh; hy_trace!(run, width)
            end

            kh <= lb && break
        end
    end

    push!(run.atoms, (; n = nv(graph), m = ne(graph), route = :huge, status = :anytime, lo = lb, width, winner = :cpu, t = time() - t0, tcpu = time() - t0, tgpu = NaN, won = 0, served = 0))
    return hy_result(order, width, lb, run, t0)
end

# The result: the width is certified if it meets the lower bound (with
# float weights, up to rounding: the bound and the width are sums of the
# weights in other orders).
function hy_result(order::Vector{V}, width::W, lb::W, run::HyRun{W}, t0::Float64) where {V, W}
    certified = hy_le(width, lb)
    return HybridResult{V, W}(order, width, certified ? width : lb, certified, run.trace, run.atoms, time() - t0)
end

hy_le(a::Integer, b::Integer) = a <= b
hy_le(a::Real, b::Real) = a <= b + 1.0e-10 * abs(b)

# The nested dissection of an atom by alg.dissection (see HybridTW), or
# nothing.
function hy_dissect(weights::AbstractVector, graph::AbstractGraph, alg::HybridTW)
    dis = alg.dissection
    dis === false && return nothing

    if isnothing(dis)
        if !isnothing(Base.get_extension(CliqueTrees, :KaHyParExt))
            dis = KaHyParND()
        elseif !isnothing(Base.get_extension(CliqueTrees, :MetisExt))
            dis = METISND(; seed = 1)
        else
            return nothing
        end
    end

    order, _ = permutation(weights, graph, ND(MF(; strategy = 1), dis; scale = 100))
    return order
end

# ---------------------------------------------------------------------------
# Contraction cost
#
# For a tensor network, the width of the line graph is the size of the
# largest intermediate, but what a contraction costs is the sum of its
# steps (branchcost). hybridcost returns the cheapest of the contraction
# trees (branchtree) of the ordering of the pipeline and of the orderings
# of alg.portfolio, by total cost, so that it is never worse than a fast
# greedy ordering; alg.refine (a cost-based refinement of a contraction
# tree, for instance a GPU search scored by cost) is applied to each.
# ---------------------------------------------------------------------------

"""
    hybridcost(weights, hypergraph, point, alg::HybridTW=HybridTW())

Run the pipeline [`HybridTW`](@ref) on the line graph of a tensor network, and return the
cheapest contraction tree, by total cost, of its ordering and of the orderings of
`alg.portfolio`, each refined by `alg.refine` (if it is not `nothing`). The network is a
pointed hypergraph, as in [`branchtree`](@ref): `weights` are the log2 dimensions of the
indices, `hypergraph` has a hyperedge per tensor and the hyperedge `point` of the output.

Returns a named tuple: `parent` (the contraction tree, see [`branchtree`](@ref)), `order`
(its ordering of the indices), `cost`, `width`, `peak` (see [`branchstats`](@ref)), `source`
(`:pipeline` or the index of the ordering in the portfolio), `result` (the
[`HybridResult`](@ref) of the pipeline), and `candidates` (the cost of each).
"""
function hybridcost(weights::AbstractVector, hypergraph::BipartiteGraph, point::Integer, alg::HybridTW = HybridTW())
    n = nov(hypergraph); keep = findall(w -> !iszero(w), view(weights, oneto(n)))
    line = linegraph(reverse(hypergraph), hypergraph)

    # the pipeline orders the indices of positive weight; the others cost
    # nothing and are put last
    sub = length(keep) == n ? line : hy_induced(line, keep)
    wsub = Float64[weights[v] for v in keep]
    result = hybridtw(wsub, sub, alg)

    function full(order)
        out = Int[keep[v] for v in order]; mark = falses(n); mark[out] .= true
        append!(out, findall(!, mark))
        return out
    end

    orders = Any[(:pipeline, full(result.order))]

    for (j, a) in enumerate(alg.portfolio)
        push!(orders, (j, full(first(permutation(wsub, sub, a)))))
    end

    best = nothing; candidates = Pair{Any, Float64}[]

    for (source, order) in orders
        parent = branchtree(weights, hypergraph, point; alg = order)
        isnothing(alg.refine) || (parent = alg.refine(weights, hypergraph, point, parent))
        st = branchstats(weights, hypergraph, point, parent)
        push!(candidates, source => st.cost)

        if isnothing(best) || (st.cost, st.width) < (best.cost, best.width)
            best = (; parent, order, st.cost, st.width, st.peak, source)
        end
    end

    return (; best..., result, candidates)
end

# The subgraph induced by the vertices keep (sorted), numbered 1, 2, ... in
# their order.
function hy_induced(graph::AbstractGraph, keep::AbstractVector{Int})
    new = zeros(Int, nv(graph)); new[keep] = eachindex(keep)
    ptr = [1]; tgt = Int[]

    for v in keep
        for x in neighbors(graph, v)
            (x != v && ispositive(new[x])) && push!(tgt, new[x])
        end

        sort!(view(tgt, ptr[end]:length(tgt))); push!(ptr, length(tgt) + 1)
    end

    return BipartiteGraph{Int, Int}(length(keep), length(keep), length(tgt), ptr, tgt)
end
