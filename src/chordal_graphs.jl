"""
    eliminationgraph(tree::CliqueTree)

Construct the [subtree graph](https://en.wikipedia.org/wiki/Chordal_graph) of a clique tree. The result is stored in `graph`.
"""
function eliminationgraph(tree::CliqueTree{V,E}) where {V,E}
    graph = BipartiteGraph{V,E}(nv(tree), nv(tree), ne(tree))
    return eliminationgraph!(graph, tree)
end

"""
    eliminationgraph!(graph, tree::CliqueTree)

See [`eliminationgraph`](@ref). The result is stored in `graph`.
"""
function eliminationgraph!(graph::BipartiteGraph, tree::CliqueTree)
    empty!(targets(graph))
    push!(empty!(pointers(graph)), 1)

    for bag in tree
        res = residual(bag)
        sep = separator(bag)

        for i in eachindex(res)
            append!(targets(graph), res[(i + 1):end])
            append!(targets(graph), sep)
            push!(pointers(graph), length(targets(graph)) + 1)
        end
    end

    return graph
end

"""
    eliminationgraph(graph;
        alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM,
        snd::SupernodeType=DEFAULT_SUPERNODE_TYPE)

Construct the elimination graph of a simple graph.

```julia
julia> using CliqueTrees, SparseArrays

julia> graph = [
           0 1 1 0 0 0 0 0
           1 0 1 0 0 1 0 0
           1 1 0 1 1 0 0 0
           0 0 1 0 1 0 0 0
           0 0 1 1 0 0 1 1
           0 1 0 0 0 0 1 0
           0 0 0 0 1 1 0 1
           0 0 0 0 1 0 1 0
       ];

julia> label, filledgraph = eliminationgraph(graph);

julia> sparse(filledgraph)
8×8 SparseMatrixCSC{Bool, Int64} with 13 stored entries:
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 1  1  1  1  ⋅  ⋅  ⋅  ⋅
 1  ⋅  ⋅  ⋅  1  1  ⋅  ⋅
 ⋅  ⋅  ⋅  1  1  1  1  ⋅

julia> isfilled(filledgraph)
true
```
"""
function eliminationgraph(
    graph;
    alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM,
    snd::SupernodeType=DEFAULT_SUPERNODE_TYPE,
)
    return eliminationgraph(graph, alg, snd)
end

function eliminationgraph(graph, alg::PermutationOrAlgorithm, snd::SupernodeType)
    label, tree = cliquetree(graph, alg, snd)
    return label, eliminationgraph(tree)
end

function eliminationgraph(graph, alg::PermutationOrAlgorithm, snd::Nodal)
    label, tree = cliquetree(graph, alg, snd)
    return label, tree.sep
end

"""
    nv(tree::CliqueTree)

Compute the number of vertices in the intersection graph of a clique tree.
"""
function Graphs.nv(tree::CliqueTree{V,E}) where {V,E}
    n::V = pointers(residuals(tree))[end] - one(E)
    return n
end

"""
    ne(tree::CliqueTree)

Compute the number of edges in the intersection graph of a clique tree.
"""
function Graphs.ne(tree::CliqueTree{<:Any,E}) where {E}
    m::E = sum(tree; init=0) do bag
        m = length(residual(bag))
        n = length(separator(bag))
        (n)m + (m - 1)m ÷ 2
    end

    return m
end

"""
    ischordal(graph)

Determine whether a simple graph is [chordal](https://en.wikipedia.org/wiki/Chordal_graph).
"""
function ischordal(graph)
    index, size = mcs(graph)
    return isperfect(graph, invperm(index), index)
end

"""
    isfilled(graph)

Determine whether a directed graph is filled.
"""
function isfilled(graph)
    return isfilled(BipartiteGraph(graph))
end

function isfilled(graph::AbstractGraph)
    return isperfect(graph, vertices(graph), vertices(graph))
end

"""
    isperfect(graph, order::AbstractVector[, index::AbstractVector])

Determine whether an fill-reducing permutation is perfect.
"""
function isperfect(graph, order::AbstractVector, index::AbstractVector=invperm(order))
    return isperfect(BipartiteGraph(graph), order, index)
end

# Simple Linear-Time Algorithms to Test Chordality of BipartiteGraphs, Test Acyclicity of Hypergraphs, and Selectively Reduce Acyclic Hypergraphs
# Tarjan and Yannakakis
# Test for Zero Fill-In.
#
# Determine whether a fill-reducing permutation is perfect.
# The complexity is O(m + n), where m = |E| and n = |V|.
function isperfect(
    graph::AbstractGraph{V}, order::AbstractVector{V}, index::AbstractVector{V}
) where {V}
    # validate arguments
    vertices(graph) != eachindex(index) &&
        throw(ArgumentError("vertices(graph) != eachindex(index)"))
    eachindex(order) != eachindex(index) &&
        throw(ArgumentError("eachindex(order) != eachindex(index)"))

    # run algorithm
    f = Vector{V}(undef, nv(graph))
    findex = Vector{V}(undef, nv(graph))

    for (i, w) in enumerate(order)
        f[w] = w
        findex[w] = i

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
