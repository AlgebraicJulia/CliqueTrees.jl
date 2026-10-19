# GPUND: CliqueTrees' filled nested dissection (ND{1}: at every branch the
# narrower of the dissection and an ordering of the whole branch) with every
# node graph of at most `pmax` vertices ordered on the GPU, as one batch.
#
# The dissection tree does not depend on the node orderings (only the choice
# at a branch does), so it runs twice without modifying the library:
#   1. record: ND with a proxy node algorithm that records each small node
#      graph (and returns a placeholder); METIS separators are cached;
#   2. GPU:    one batched search over all recorded node graphs;
#   3. replay: ND again; the proxy returns the GPU orderings, the separators
#      come from the cache, and nodes larger than `pmax` are ordered by bw on
#      the CPU, as in plain nd.

using CliqueTrees: DissectionAlgorithm, pointers, targets

const ME = Base.get_extension(CliqueTrees, :MetisExt)
const BW = BestWidth(MF(), MF(; strategy = 1), AMF(), MMD())

function graphkey(weights, graph, extra)
    n = nv(graph)
    ptr = pointers(graph); tgt = targets(graph)
    h = hash(extra, hash(n))
    @inbounds for i in 1:n+1
        h = hash(ptr[i] - ptr[1], h)
    end
    @inbounds for p in ptr[1]:ptr[n+1]-1
        h = hash(tgt[p], h)
    end
    @inbounds for i in 1:n
        h = hash(weights[i], h)
    end
    return h
end

# ---------------------------------------------------- cached separators --

struct CachedMETIS <: DissectionAlgorithm
    inner::METISND
end

const SEPCACHE = Dict{UInt64, Tuple{Vector{Metis.idx_t}, Metis.idx_t}}()
const SEPLOCK = ReentrantLock()
const SEPSTATS = zeros(Int, 2)   # hits, misses

function ME.separator!(options::AbstractVector{ME.INT}, sepsize, part::AbstractVector{ME.INT}, weights::AbstractVector{ME.INT}, graph::BipartiteGraph{ME.INT, ME.INT}, imbalance::ME.INT, alg::CachedMETIS)
    n = nv(graph)
    key = graphkey(weights, graph, (imbalance, alg.inner.seed, alg.inner.nseps))
    hit = lock(SEPLOCK) do
        if haskey(SEPCACHE, key)
            p, s = SEPCACHE[key]
            copyto!(part, 1, p, 1, n); sepsize[] = s
            SEPSTATS[1] += 1
            return true
        end
        return false
    end
    hit && return
    ME.separator!(options, sepsize, part, weights, graph, imbalance, alg.inner)
    lock(SEPLOCK) do
        SEPCACHE[key] = (part[1:n], sepsize[]); SEPSTATS[2] += 1
    end
    return
end

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph, alg::ND{<:Any, <:EliminationAlgorithm, CachedMETIS}; verbose::Bool = false)
    order = ME.dissect(weights, graph, alg)
    return order, invperm(order)
end

# ------------------------------------------------------ proxy node order --

struct NodeAlg{A <: EliminationAlgorithm} <: EliminationAlgorithm
    pmax::Int
    big::A          # CPU ordering of the nodes larger than pmax (replay pass)
end

CliqueTrees.isconcurrent(::NodeAlg) = false

const NODEMODE = Ref(:off)
const NODEREC = Dict{UInt64, Tuple{SimpleGraph{Int}, Vector{Int32}}}()
const NODERES = Dict{UInt64, Vector{Int}}()
const NODESTATS = zeros(Int, 3)   # small nodes, big nodes (CPU bw), replay hits

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::NodeAlg; verbose::Bool = false) where {V}
    n = nv(graph)
    if 2 < n <= alg.pmax
        key = graphkey(weights, graph, 0)
        if NODEMODE[] === :record
            if !haskey(NODEREC, key)
                NODEREC[key] = (tosimple(graph), Int32[weights[v] for v in 1:n])
                NODESTATS[1] += 1
            end
            o = collect(V, 1:n)
            return o, copy(o)
        elseif haskey(NODERES, key)
            NODESTATS[3] += 1
            o = convert(Vector{V}, NODERES[key])
            return o, invperm(o)
        end
    end
    NODEMODE[] === :record && return permutation(weights, graph, MMD())
    NODESTATS[2] += 1
    return permutation(weights, graph, alg.big)
end

# ------------------------------------------------------------ the method --

"""
    GPUND(; time, pmax, width, level, scale, imbalance, seed)

ND{1}(bw-like node orderings, METIS) with the node graphs of at most `pmax`
vertices ordered by a batched GPU search of `time` seconds. With the defaults
(width 16, level 12, scale 100) the dissection is the paper's `nd`.
"""
Base.@kwdef struct GPUND <: EliminationAlgorithm
    time::Float64 = 1.0
    pmax::Int = 256
    width::Int = 16
    level::Int = 12
    scale::Int = 100
    imbalance::Int = 130
    seed::Int = 0
    rounds::Int = 1                 # at least this many GPU rounds (time = 0: exactly)
    big::Symbol = :bw               # :bw, or :fast = BestWidth(AMF(), MMD()) for nodes > pmax
end

CliqueTrees.isconcurrent(::GPUND) = false
bigalg(a::GPUND) = a.big === :bw ? BW : BestWidth(AMF(), MMD())

ndalg(a::GPUND) = ND{1}(NodeAlg(a.pmax, bigalg(a)), CachedMETIS(METISND(; seed = a.seed)); width = a.width, level = a.level, scale = a.scale, imbalance = a.imbalance)

function gpund(weights::AbstractVector, graphs::AbstractVector, a::GPUND)
    st = STATS[]
    empty!(SEPCACHE); empty!(NODEREC); empty!(NODERES); SEPSTATS .= 0; NODESTATS .= 0
    alg = ndalg(a)
    t0 = time()
    NODEMODE[] = :record
    try
        for k in eachindex(graphs)
            permutation(weights[k], graphs[k], alg)
        end
    finally
        NODEMODE[] = :off
    end
    st.tsplit += time() - t0

    keys_ = collect(keys(NODEREC))
    pieces = Piece[]
    for key in keys_
        g, w = NODEREC[key]
        n = nv(g)
        push!(pieces, Piece(collect(1:n), n, g, 1, w, Int32[round(Int32, SQ * x / a.scale) for x in w]))
    end
    if !isempty(pieces)
        porders, _ = gpusearch(pieces, nothing; time = a.time, seed = a.seed + 1, focus = false, minrounds = a.rounds)
        for (i, key) in enumerate(keys_)
            NODERES[key] = porders[i]
        end
    end

    t1 = time()
    NODEMODE[] = :replay
    out = try
        [first(permutation(weights[k], graphs[k], alg)) for k in eachindex(graphs)]
    finally
        NODEMODE[] = :off
    end
    st.tasm += time() - t1
    return out
end

function CliqueTrees.batch_permutation(weights::AbstractVector{<:AbstractVector}, graphs::AbstractVector{<:AbstractGraph}, alg::GPUND; verbose::Bool = false)
    V = eltype(graphs[1])
    return [convert(Vector{V}, o) for o in gpund(weights, graphs, alg)]
end

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::GPUND; verbose::Bool = false) where {V}
    o = convert(Vector{V}, only(gpund([weights], [graph], alg)))
    return o, invperm(o)
end
