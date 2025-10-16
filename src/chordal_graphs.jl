"""
    ischordal(graph)

Determine whether a simple graph is [chordal](https://en.wikipedia.org/wiki/Chordal_graph).
"""
function ischordal(graph)
    return isperfect(graph, permutation(graph; alg = MCS()))
end

function isperfect(graph, pair::Tuple)
    return isperfect(graph, pair...)
end

"""
    isperfect(graph, order::AbstractVector[, index::AbstractVector])

Determine whether an fill-reducing permutation is perfect.
"""
function isperfect(graph, order::AbstractVector, index::AbstractVector = invperm(order))
    return isperfect(BipartiteGraph(graph), order, index)
end

# Simple Linear-Time Algorithms to Test Chordality of BipartiteGraphs, Test Acyclicity of Hypergraphs, and Selectively Reduce Acyclic Hypergraphs
# Tarjan and Yannakakis
# Test for Zero Fill-In.
#
# Determine whether a fill-reducing permutation is perfect.
# The complexity is O(m + n), where m = |E| and n = |V|.
function isperfect(graph::AbstractGraph{V}, order::AbstractVector{V}, index::AbstractVector{V}) where {V}
    @assert nv(graph) <= length(order)
    @assert nv(graph) <= length(index)

    n = nv(graph)
    f = FVector{V}(undef, n)
    findex = FVector{V}(undef, n)

    @inbounds for i in oneto(n)
        w = order[i]; f[w] = w; findex[w] = i

        for v in neighbors(graph, w)
            if index[v] < i
                findex[v] = i

                if f[v] == v
                    f[v] = w
                end
            end
        end

        for v in neighbors(graph, w)
            if index[v] < i && findex[f[v]] < i
                return false
            end
        end
    end

    return true
end

"""
    color(graph[, order::AbstractVector, index::AbstractVector])

Compute a minimal vertex coloring of a chordal graph.
"""
function color(graph, pair::Tuple = permutation(graph; alg = MCS()))
    return color(graph, pair...)
end

function color(graph, order::AbstractVector, index::AbstractVector = invperm(order))
    return color(BipartiteGraph(graph), order, index)
end

function color(
        graph::AbstractGraph{V}, order::AbstractVector, index::AbstractVector
    ) where {V}
    k = zero(V)
    label = fill(nv(graph) + one(V), nv(graph))
    color = Vector{V}(undef, nv(graph))
    lower = sympermute(graph, index, Reverse)

    @inbounds for v in reverse(vertices(lower))
        if outdegree(lower, v) >= k
            color[order[v]] = k += one(V)
        else
            for w in outneighbors(lower, v)
                label[color[order[w]]] = v
            end

            for i in oneto(k)
                if label[i] > v
                    color[order[v]] = i
                    break
                end
            end
        end
    end

    return Coloring(k, color)
end
