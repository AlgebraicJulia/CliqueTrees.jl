# Shared definitions for the TN pipeline profiling: the paper's methods, stage taps, and a
# checkpointed copy of HBT's batch driver. Reads CliqueTrees (itay/mt) without modifying it.
using CliqueTrees, Graphs, Metis, Random, Printf, SparseArrays
const CT = CliqueTrees
include("/home/itaykadosh/Projects/GATAS/tn-work/tncorpus/load.jl")
using .TNCorpus

# ----------------------------------------------------------------------------- methods (draft Sec. 5.2)
bwalg() = BestWidth(MF(), MF(; strategy = 1), AMF(), MMD())
nd1(s; alg = bwalg()) = ND(alg, METISND(; seed = s); width = 16, level = 12, scale = 100, imbalance = 300)
nd8(; alg = bwalg()) = BestWidth(ntuple(s -> nd1(s; alg), 8)...)
# four minimal triangulations for SafeSeparators (assumption: the four greedy orderings of bw, made minimal)
minalgs() = (MinimalChordal(MF()), MinimalChordal(MF(; strategy = 1)), MinimalChordal(AMF()), MinimalChordal(MMD()))
pipeline(alg) = SafeRules(SafeSeparators(alg, minalgs(); maxmissing = 50))

# ----------------------------------------------------------------------------- taps
# Tap(tag, alg): records (tag, nv, ne, seconds) for every permutation / batch call through it.
const TAPLOG = Vector{NamedTuple{(:tag, :nv, :ne, :t), Tuple{Symbol, Int, Int, Float64}}}()
const BATCHLOG = Vector{Any}()   # captured batches (weights, graphs) when CAPTURE[] is true
const CAPTURE = Ref(false)

struct Tap{A <: CT.EliminationAlgorithm} <: CT.EliminationAlgorithm
    tag::Symbol
    alg::A
end

function CT.permutation(weights::AbstractVector, graph::AbstractGraph, alg::Tap; verbose::Bool = false)
    t0 = time_ns()
    r = CT.permutation(weights, graph, alg.alg; verbose)
    push!(TAPLOG, (; tag = alg.tag, nv = nv(graph), ne = ne(graph), t = (time_ns() - t0) / 1e9))
    return r
end

function CT.batch_permutation(weights::AbstractVector{<:AbstractVector}, graphs::AbstractVector{<:AbstractGraph}, alg::Tap; verbose::Bool = false)
    CAPTURE[] && push!(BATCHLOG, (map(copy, weights), map(g -> CT.BipartiteGraph{Int, Int}(g), graphs)))
    t0 = time_ns()
    r = CT.batch_permutation(weights, graphs, alg.alg; verbose)
    push!(TAPLOG, (; tag = Symbol(alg.tag, :_batch), nv = sum(nv, graphs; init = 0), ne = sum(ne, graphs; init = 0), t = (time_ns() - t0) / 1e9))
    return r
end

CT.isconcurrent(::Tap) = false

# ----------------------------------------------------------------------------- checkpointed HBT batch
# A copy of batch_permutation(::HBT) (src/bt/hbt.jl) that runs hbtrun! in short slices and
# records (seconds since batch entry, batch width) at every improvement. Differences from the
# library: (1) the exact solve of a widest graph with <= base vertices gets at most
# min(half the remaining time, max(2 s, elapsed)) instead of half the remaining time, so a long
# budget does not freeze the early curve; (2) widths are re-scored with treewidth().
struct HBTCurve{H <: HBT} <: CT.EliminationAlgorithm
    hbt::H
    events::Vector{Tuple{Float64, Float64, String}}
end
HBTCurve(h::HBT) = HBTCurve(h, Tuple{Float64, Float64, String}[])
CT.isconcurrent(::HBTCurve) = false

function CT.batch_permutation(weights::AbstractVector{<:AbstractVector{W}}, graphs::AbstractVector{<:AbstractGraph{V}}, cur::HBTCurve; verbose::Bool = false) where {W, V}
    alg = cur.hbt; ev = cur.events
    t0 = time(); dl = t0 + alg.time; ngraph = length(graphs)
    orders = Vector{Vector{V}}(undef, ngraph)
    widths = Vector{W}(undef, ngraph)
    lowers = Vector{W}(undef, ngraph)
    inits = Vector{Union{Nothing, Vector{V}}}(nothing, ngraph)
    tried = falses(ngraph)
    searches = Vector{Any}(nothing, ngraph)
    grd = (; rng = Xoshiro(alg.seed), par = (; alg = alg.alg))
    ninit = alg.init isa CT.EliminationAlgorithm ? alg.ninit : max(1, alg.ninit)
    best = Ref(Inf)
    note(tag) = (k = maximum(widths; init = zero(W)); k < best[] && (best[] = k; push!(ev, (time() - t0, Float64(k), tag))))

    for i in 1:ngraph
        weight = weights[i]; graph = graphs[i]
        lowers[i] = convert(W, lowerbound(weight, graph, alg.lb))
        widths[i] = typemax(W)
        if alg.init isa CT.EliminationAlgorithm
            order, _ = permutation(weight, graph, MinimalChordal(alg.init))
            inits[i] = orders[i] = order
            widths[i] = treewidth(weight, graph, order)
        end
        for k in 1:ninit
            widths[i] <= lowers[i] && break
            k > 1 && isassigned(orders, i) && time() > dl && break
            order = CT.hbt_grd(grd, weight, graph)
            width = treewidth(weight, graph, order)
            if width < widths[i]
                orders[i] = order; widths[i] = width
            end
        end
    end
    lb = maximum(lowers; init = zero(W))
    push!(ev, (time() - t0, Float64(lb), "lb"))
    note("init")

    while ngraph > 0 && time() < dl
        k, i = findmax(widths)
        k <= lb && break
        ub = nothing
        for j in 1:ngraph
            j == i || (ub = isnothing(ub) ? widths[j] : max(ub, widths[j]))
        end
        weight = weights[i]; graph = graphs[i]
        if !tried[i] && nv(graph) <= alg.base
            tried[i] = true
            bgraph, wgt = CT.hbt_graph(weight, graph)
            xdl = time() + min((dl - time()) / 2, max(2.0, time() - t0))
            order, status = CT.hbt_safe(wgt, bgraph, lowers[i], lowers[i], typemax(W), xdl)
            if status == CT.BT_FOUND
                orders[i] = convert(Vector{V}, order)
                widths[i] = lowers[i] = treewidth(weight, graph, orders[i])
                lb = max(lb, lowers[i])
                note("exact")
            end
        else
            if isnothing(searches[i])
                searches[i] = CT.hbt_new(weight, graph, alg; lb = lowers[i], init = inits[i], deadline = dl)
            end
            # slices: ~5% of the elapsed time, 0.02 s .. 5 s
            while time() < dl
                slice = clamp(0.05 * (time() - t0), 0.02, 5.0)
                order, _, width = CT.hbtrun!(searches[i]; mt = min(slice, dl - time()), nt = 1, lb, ub)
                if width < widths[i]
                    o = convert(Vector{V}, order); tw = treewidth(weight, graph, o)
                    if tw < widths[i]
                        orders[i] = o; widths[i] = tw; note("hbt")
                    end
                end
                (widths[i] <= lb || (!isnothing(ub) && widths[i] < ub)) && break
            end
        end
    end
    push!(ev, (time() - t0, Float64(maximum(widths; init = zero(W))), "end"))
    return orders
end

# ----------------------------------------------------------------------------- utilities
function peakrss_mb()
    # VmHWM of this process (high-water mark of resident memory)
    for l in eachline("/proc/self/status")
        startswith(l, "VmHWM:") && return parse(Int, split(l)[2]) / 1024
    end
    return NaN
end

function warmup()
    inst = TNCorpus.load("qc_rand_google53_27_20")
    w = inst.weights; g = TNCorpus.bipartite(inst)
    permutation(w, g, bwalg()); permutation(w, g, pipeline(nd8()))
    permutation(w, g, pipeline(HBTCurve(HBT(; time = 0.5, threads = 1))))
    permutation(w, g, pipeline(HBTCurve(HBT(; time = 0.5, threads = 1, init = nd8()))))
    return
end

function CT.permutation(weights::AbstractVector, graph::AbstractGraph, cur::HBTCurve; verbose::Bool = false)
    o = CT.batch_permutation([weights], [graph], cur)[1]
    return o, invperm(o)
end
