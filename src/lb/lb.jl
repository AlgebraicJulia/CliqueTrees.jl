"""
    LowerBoundAlgorithm

An algorithm for computing a lower bound to the treewidth of a graph. The options are

| type          | name            | time | space    |
|:--------------|:----------------|:-----|:---------|
| [`MMW`](@ref) | minor-min-width |      | O(m + n) |
| [`LBN`](@ref) | LBN+            |      | O(m + n) |
| [`LBP`](@ref) | LBP+            |      | O(m + n) |
"""
abstract type LowerBoundAlgorithm end

"""
    WidthOrAlgorithm = Union{Number, LowerBoundAlgorithm}
"""
const WidthOrAlgorithm = Union{Number, LowerBoundAlgorithm}

"""
    lowerbound([weights, ]graph;
        alg::WidthOrAlgorithm=DEFAULT_LOWER_BOUND_ALGORITHM)

Compute a lower bound to the treewidth of a graph.

```jldoctest
julia> using CliqueTrees

julia> graph = [
           0 1 0 0 0 0 0 0
           1 0 1 0 0 1 0 0
           0 1 0 1 0 1 1 1
           0 0 1 0 0 0 0 0
           0 0 0 0 0 1 1 0
           0 1 1 0 1 0 0 0
           0 0 1 0 1 0 0 1
           0 0 1 0 0 0 1 0
       ];

julia> lowerbound(graph)
2
```
"""
function lowerbound(graph; alg::WidthOrAlgorithm = DEFAULT_LOWER_BOUND_ALGORITHM)
    return lowerbound(graph, alg)
end

function lowerbound(graph, alg::WidthOrAlgorithm)
    return lowerbound(BipartiteGraph(graph), alg)
end

function lowerbound(graph::AbstractGraph{V}, width::Integer) where {V}
    return convert(V, width)
end

function lowerbound(graph::AbstractGraph{V}, alg::LowerBoundAlgorithm) where {V}
    n = nv(graph); weights = Ones{V}(n)
    return lowerbound(weights, graph, alg) - one(V)
end

function lowerbound(weights::AbstractVector, graph; alg::WidthOrAlgorithm = DEFAULT_LOWER_BOUND_ALGORITHM)
    return lowerbound(weights, graph, alg)
end

function lowerbound(weights::AbstractVector, graph, alg::WidthOrAlgorithm)
    return lowerbound(weights, BipartiteGraph(graph), alg)
end

function lowerbound(weights::AbstractVector{W}, graph::AbstractGraph, width::Number) where {W}
    return convert(W, width)
end

# Call `f(V, E, totdeg)`, where `totdeg` is the total weight of the
# vertices of `graph`, and `V` and `E` are the integer types of the
# working arrays of a lower bound algorithm: 32-bit integers when they
# suffice. If `grow` is true, then the algorithm adds edges to the
# graph, so its arc storage may grow to a multiple of n².
function lbcall(f::F, weights::AbstractVector, graph::AbstractGraph{V}, grow::Bool) where {F, V}
    @assert nv(graph) <= length(weights)

    n = Int(nv(graph)); m = Int(de(graph))

    # `totdeg` is the total weight of the
    # vertices in the graph
    totdeg = 0

    @inbounds for v in oneto(n)
        totdeg += trunc(Int, weights[v])
    end

    # `cap` bounds the size of the arc storage
    cap = mmwcapacity(n, m) + 4n
    grow && (cap = max(cap, 4n * n))

    if n < typemax(Int32) ÷ 4 && cap < typemax(Int32) && totdeg < typemax(Int32)
        return f(Int32, Int32, totdeg)
    else
        return f(promote_type(V, Int), promote_type(etype(graph), Int), totdeg)
    end
end

include("mmw.jl")
include("lbn.jl")
include("lbp.jl")

"""
    DEFAULT_LOWER_BOUND_ALGORITHM = MMW()
"""
const DEFAULT_LOWER_BOUND_ALGORITHM = MMW()
