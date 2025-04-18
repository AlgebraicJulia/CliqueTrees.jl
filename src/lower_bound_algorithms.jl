"""
    LowerBoundAlgorithm

An algorithm for computing a lower bound to the treewidth of a graph. The options are

| type          | name            | time | space |
|:--------------|:----------------|:-----|:------|
| [`MMW`](@ref) | minor-min-width |      |       |
"""
abstract type LowerBoundAlgorithm end

"""
    WidthOrAlgorithm = Union{Number, LowerBoundAlgorithm}
"""
const WidthOrAlgorithm = Union{Number, LowerBoundAlgorithm}

"""
    MMW{S} <: LowerBoundAlgorithm

    MMW{1}() # min-d heuristic

    MMW{2}() # max-d heuristic

    MMW{3}() # least-c heuristic

    MMW()

The minor-min-width heuristic.

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

julia> alg = MMW{1}()
MMW{1}

julia> lowerbound(graph; alg)
2
```

# References

  - Gogate, Vibhav, and Rina Dechter. "A complete anytime algorithm for treewidth." *Proceedings of the 20th conference on Uncertainty in artificial intelligence.* 2004.
  - Bodlaender, Hans, Thomas Wolle, and Arie Koster. "Contraction and treewidth lower bounds." *Journal of Graph Algorithms and Applications* 10.1 (2006): 5-49.
"""
struct MMW{S} <: LowerBoundAlgorithm end

function MMW()
    return MMW{3}()
end

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

function lowerbound(weights::AbstractVector, graph; alg::WidthOrAlgorithm = DEFAULT_LOWER_BOUND_ALGORITHM)
    return lowerbound(weights, graph, alg)
end

# method ambiguity
function lowerbound(weights::AbstractVector, ::Number)
    error()
end

# method ambiguity
function lowerbound(weights::AbstractVector, ::MMW)
    error()
end

function lowerbound(graph, alg::Number)
    return lowerbound(BipartiteGraph(graph), alg)
end

function lowerbound(graph::AbstractGraph{V}, alg::Number) where {V}
    width::V = alg
    return width
end

function lowerbound(weights::AbstractVector{W}, graph, alg::Number) where {W}
    width::W = alg
    return width
end

function lowerbound(graph, ::MMW{S}) where {S}
    return mmw(graph, Val(S))
end

function lowerbound(weights::AbstractVector, graph, ::MMW{S}) where {S}
    return mmw(weights, graph, Val(S))
end

# Contraction and Treewidth Lower Bounds
# Bodlaender, Koster, and Wolle
# MMD+ (min-d) heuristic
#
# A Complete Anytime Algorithm for Treewidth
# Gogate and Dechter
# minor-min-width
function mmw(graph, ::Val{S}) where {S}
    return mmw(BipartiteGraph(graph), Val(S))
end

function mmw(graph::AbstractGraph, ::Val{S}) where {S}
    return mmw!(Graph(graph), Val(S))
end

function mmw!(graph::Graph{V}, ::Val{S}) where {V, S}
    n = nv(graph)
    label = zeros(V, n); tag = zero(V)

    # bucket queue data structure
    head = zeros(V, n)
    prev = Vector{V}(undef, n)
    next = Vector{V}(undef, n)

    function set(i)
        return DoublyLinkedList(view(head, i + one(V)), prev, next)
    end

    mindegree = n
    maxdegree = zero(V)

    for v in vertices(graph)
        degree = eltypedegree(graph, v)
        mindegree = min(degree, mindegree)
        maxdegree = max(degree, maxdegree)
        pushfirst!(set(degree), v)
    end

    maxmindegree = mindegree

    while mindegree < maxdegree
        if !isempty(set(mindegree))
            v = first(set(mindegree))
            delete!(set(mindegree), v)

            if !iszero(eltypedegree(graph, v))
                tag, w = mmwnbr!(graph, label, tag, v, Val(S))
                delete!(set(eltypedegree(graph, w)), w)
                rem_edge!(graph, v, w)

                tag += one(V)

                for x in neighbors(graph, w)
                    label[x] = tag
                end

                for ww in neighbors(graph, v)
                    delete!(set(eltypedegree(graph, ww)), ww)

                    if label[ww] < tag
                        add_edge!(graph, w, ww)
                    end
                end

                degree = eltypedegree(graph, w)
                pushfirst!(set(degree), w)
                mindegree = min(mindegree, degree)
                maxdegree = max(maxdegree, degree)
            end

            while !isempty(neighbors(graph, v))
                w = last(neighbors(graph, v))
                rem_edge!(graph, v, w)
                degree = eltypedegree(graph, w)
                pushfirst!(set(degree), w)
                mindegree = min(mindegree, degree)
                maxdegree = max(maxdegree, degree)
            end
        end

        while isempty(set(mindegree))
            mindegree += one(V)
        end

        while isempty(set(maxdegree))
            maxdegree -= one(V)
        end

        maxmindegree = max(maxmindegree, mindegree)
    end

    return maxmindegree
end

# Safe Reduction Rules for Weighted Treewidth
# Eijkhof, Bodlaender, and Koster
# MMNW+ heuristic
function mmw(weights::AbstractVector, graph, ::Val{S}) where {S}
    return mmw(weights, BipartiteGraph(graph), Val(S))
end

function mmw(weights::AbstractVector, graph::AbstractGraph, ::Val{S}) where {S}
    return mmw!(weights, Graph(graph), Val(S))
end

function mmw!(weights::AbstractVector{W}, graph::Graph{V}, ::Val{S}) where {W, V, S}
    n = nv(graph)
    tol = tolerance(W)
    label = zeros(V, n); tag = zero(V)

    # remove self-loops
    for v in vertices(graph)
        rem_edge!(graph, v, v)
    end

    # heap data structure
    heap = Heap{V, W}(n)
    maxminweight = zero(W)

    for v in vertices(graph)
        nw = weights[v]

        for w in neighbors(graph, v)
            nw += weights[w]
        end

        push!(heap, v => nw)
    end

    hfall!(heap)

    while !isempty(heap)
        v = argmin(heap)
        delete!(heap, v)
        maxminweight = max(maxminweight, heap[v])
        tag, w = mmwnbr!(heap, weights, graph, label, tag, v, Val(S))

        if ispositive(w)
            rem_edge!(graph, v, w)
            heap[w] -= weights[v]

            tag += one(V)

            for x in neighbors(graph, w)
                label[x] = tag
            end

            for ww in neighbors(graph, v)
                if label[ww] < tag
                    add_edge!(graph, w, ww)
                    heap[w] += weights[ww]
                    heap[ww] += weights[w]
                end
            end

            hrise!(heap, w)
            hfall!(heap, w)
        end

        while !isempty(neighbors(graph, v))
            w = last(neighbors(graph, v))
            rem_edge!(graph, v, w)
            heap[w] -= weights[v]
            hrise!(heap, w)
        end
    end

    return maxminweight
end

# min-d
# u* := argmin { |N(u)| | u ∈ N(v) }
function mmwnbr!(graph::Graph{V}, label::Vector{V}, tag::V, v::V, ::Val{1}) where {V}
    w = zero(V); nw = typemax(V)

    for ww in neighbors(graph, v)
        nww = eltypedegree(graph, ww)

        if nww < nw
            w, nw = ww, nww
        end
    end

    return tag, w
end

# weighted min-d
# u* := argmin { Σ { w(t) | t ∈ N(u) } | u ∈ N(v) and w(u) ≤ w(v) }
function mmwnbr!(heap::Heap{V, W}, weights::AbstractVector{W}, graph::Graph{V}, label::Vector{V}, tag::V, v::V, ::Val{1}) where {W, V}
    tol = tolerance(W); w = zero(V); nw = typemax(W)

    for ww in neighbors(graph, v)
        nww = heap[ww]

        if weights[ww] < weights[v] + tol && nww <= nw - tol
            w, nw = ww, nww
        end
    end

    return tag, w
end

# max-d
# u* := argmax { |N(u)| | u ∈ N(v) }
function mmwnbr!(graph::Graph{V}, label::Vector{V}, tag::V, v::V, ::Val{2}) where {V}
    w = zero(V); nw = -one(V)

    for ww in neighbors(graph, v)
        nww = eltypedegree(graph, ww)

        if nww > nw
            w, nw = ww, nww
        end
    end

    return tag, w
end

# weighted max-d
# u* := argmin { Σ { w(t) | t ∈ N(u) } | u ∈ N(v) and w(u) ≤ w(v) }
function mmwnbr!(heap::Heap{V, W}, weights::AbstractVector{W}, graph::Graph{V}, label::Vector{V}, tag::V, v::V, ::Val{2}) where {W, V}
    tol = tolerance(W); w = zero(V); nw = -tol

    for ww in neighbors(graph, v)
        nww = heap[ww]

        if weights[ww] < weights[v] + tol && nww - tol >= nw
            w, nw = ww, nww
        end
    end

    return tag, w
end

# least-c
# u* := argmin { |N(u) ∩ N(v)| | u ∈ N(v) }
function mmwnbr!(graph::Graph{V}, label::Vector{V}, tag::V, v::V, ::Val{3}) where {V}
    w = zero(V); nw = typemax(V)
    label[neighbors(graph, v)] .= tag += one(V)

    for ww in neighbors(graph, v)
        nww = zero(V)

        for xx in neighbors(graph, ww)
            if label[xx] == tag
                nww += one(V)
            end
        end

        if nww < nw
            w, nw = ww, nww
        end
    end

    return tag, w
end

# weighted least-c
# u* := argmin { Σ { w(s)w(t) | s ∈ N(u) and t ∈ N(v) } | u ∈ N(v) }
function mmwnbr!(heap::Heap{V, W}, weights::AbstractVector{W}, graph::Graph{V}, label::Vector{V}, tag::V, v::V, ::Val{3}) where {W, V}
    tol = tolerance(W); w = zero(V); nw = typemax(W)
    label[neighbors(graph, v)] .= tag += one(V)

    for ww in neighbors(graph, v)
        if weights[ww] < weights[v]
            nww = zero(W)

            for xx in neighbors(graph, ww)
                if label[xx] == tag
                    nww += weights[xx]
                end
            end

            if nww <= nw - tol
                w, nw = ww, nww
            end
        end
    end

    return tag, w
end

function Base.show(io::IO, ::MIME"text/plain", alg::MMW{S}) where {S}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "MMW{$S}")
    return nothing
end

"""
    DEFAULT_LOWER_BOUND_ALGORITHM = MMW()
"""
const DEFAULT_LOWER_BOUND_ALGORITHM = MMW()
