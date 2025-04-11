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
    MMW <: LowerBoundAlgorithm

    MMW()

The minor-min-width heuristic.

# References

  - Gogate, Vibhav, and Rina Dechter. "A complete anytime algorithm for treewidth." *Proceedings of the 20th conference on Uncertainty in artificial intelligence.* 2004.
  - Bodlaender, Hans, Thomas Wolle, and Arie Koster. "Contraction and treewidth lower bounds." *Journal of Graph Algorithms and Applications* 10.1 (2006): 5-49.
"""
struct MMW <: LowerBoundAlgorithm end

"""
    lowerbound([weights, ]graph;
        alg::WidthOrAlgorithm=DEFAULT_LOWER_BOUND_ALGORITHM)

Compute a lower bound to the treewidth of a graph.
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
    return with
end

function lowerbound(graph, ::MMW)
    return mmw(graph)
end

function lowerbound(weights::AbstractVector, graph, ::MMW)
    return mmw(weights, graph)
end

# Contraction and Treewidth Lower Bounds
# Bodlaender, Koster, and Wolle
# MMD+ (min-d) heuristic
#
# A Complete Anytime Algorithm for Treewidth
# Gogate and Dechter
# minor-min-width
function mmw(graph)
    return mmw(BipartiteGraph(graph))
end

function mmw(graph::AbstractGraph)
    return mmw!(Graph(graph))
end

function mmw!(graph::Graph{V}) where {V}
    label = zeros(V, nv(graph))
    tag = zero(V)

    # bucket queue data structure
    head = zeros(V, nv(graph))
    prev = Vector{V}(undef, nv(graph))
    next = Vector{V}(undef, nv(graph))

    function set(i)
        return DoublyLinkedList(view(head, i + one(V)), prev, next)
    end

    mindegree = nv(graph)
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
                w = argmin(v -> eltypedegree(graph, v), neighbors(graph, v))
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
function mmw(weights::AbstractVector, graph)
    return mmw(weights, BipartiteGraph(graph))
end

function mmw(weights::AbstractVector, graph::AbstractGraph)
    return mmw!(weights, Graph(graph))
end

function mmw!(weights::AbstractVector{W}, graph::Graph{V}) where {W, V}
    n = nv(graph)
    tol = tolerance(weights)
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

        w = zero(V)
        nw = typemax(W)

        for ww in neighbors(graph, v)
            nww = heap[ww]

            if weights[ww] < weights[v] + tol && nww <= nw - tol
                w, nw = ww, nww
            end
        end

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

function Base.show(io::IO, ::MIME"text/plain", alg::MMW)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "MMW")
    return nothing
end

"""
    DEFAULT_LOWER_BOUND_ALGORITHM = MMW()
"""
const DEFAULT_LOWER_BOUND_ALGORITHM = MMW()
