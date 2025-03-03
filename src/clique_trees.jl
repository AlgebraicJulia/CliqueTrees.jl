"""
    CliqueTree{V, E} <: AbstractVector{Clique{V, E}}

A [clique tree](https://en.wikipedia.org/wiki/Tree_decomposition) with vertices of type `V` and edges of type `E`.
This type implements the [indexed tree interface](https://juliacollections.github.io/AbstractTrees.jl/stable/#The-Indexed-Tree-Interface).
"""
struct CliqueTree{V,E} <: AbstractVector{Clique{V,E}}
    tree::SupernodeTree{V}
    count::Vector{V}
    sep::BipartiteGraph{V,E,Vector{E},Vector{V}}
end

function Tree(tree::CliqueTree)
    return Tree(tree.tree)
end

function Tree{V}(tree::CliqueTree) where {V}
    return Tree{V}(tree.tree)
end

"""
    cliquetree(graph;
        alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM,
        snd::SupernodeType=DEFAULT_SUPERNODE_TYPE)

Construct a tree decomposition of a simple graph.
The vertices of the graph are first ordered by a fill-reducing permutation computed by the algorithm `alg`.
The size of the resulting decomposition is determined by the supernode partition `snd`.

```julia
julia> using CliqueTrees

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

julia> label, tree = cliquetree(graph);

julia> tree
6-element CliqueTree{Int64, Int64}:
 [6, 7, 8]
 ├─ [1, 6, 7]
 ├─ [4, 6, 8]
 │  └─ [3, 4, 6]
 │     └─ [2, 3, 6]
 └─ [5, 7, 8]
```
"""
function cliquetree(
    graph;
    alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM,
    snd::SupernodeType=DEFAULT_SUPERNODE_TYPE,
)
    return cliquetree(graph, alg, snd)
end

@views function cliquetree(graph, alg::PermutationOrAlgorithm, snd::SupernodeType)
    # construct supernodal elimination tree
    label, tree, count, index, ptr, lower, upper = supernodetree(graph, alg, snd)
    lower = sympermute!(upper, lower, index, Reverse)

    # compute separators
    function diff(col, res)
        i = 1

        while i in eachindex(col) && col[i] in res
            i += 1
        end

        return col[i:end]
    end

    V = eltype(lower)
    sep = BipartiteGraph(nv(lower), ptr, Vector{V}(undef, ptr[end] - 1))
    cache = Vector{V}(undef, Δout(sep))

    for (j, res) in enumerate(tree)
        # get representative vertex
        vertex = res[begin]

        # subtract residual from higher neighbors
        column = diff(neighbors(lower, vertex), res)

        # initialize separator
        state = neighbors(sep, j)[eachindex(column)] .= column

        # iterate over children
        i = firstchildindex(tree, j)

        while !isnothing(i) && length(state) < outdegree(sep, j)
            # subtract residual from child separator
            column = diff(neighbors(sep, i), res)

            # update separator
            union = mergesorted!(cache, state, column)
            state = neighbors(sep, j)[eachindex(union)] .= union

            # update child
            i = nextsiblingindex(tree, i)
        end
    end

    # construct clique tree
    return label, CliqueTree(tree, invpermute!(count, index), sep)
end

"""
    treewidth(tree::CliqueTree)

Compute the [width](https://en.wikipedia.org/wiki/Treewidth) of a clique tree.
"""
function treewidth(tree::CliqueTree{V}) where {V}
    n::V = maximum(length, tree; init=1) - 1
    return n
end

"""
    treewidth(graph;
        alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM)

Compute an upper bound to the [tree width](https://en.wikipedia.org/wiki/Treewidth) of a simple graph.
"""
function treewidth(graph; alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM)
    return treewidth(graph, alg)
end

function treewidth(graph, alg::PermutationOrAlgorithm)
    label, tree, upper = eliminationtree(graph, alg)
    rowcount, colcount = supcnt(reverse(upper), tree)
    V = eltype(colcount)
    return maximum(colcount; init=one(V)) - one(V)
end

"""
    residual(tree::CliqueTree, i::Integer)

Get the residual at node `i`.
"""
function residual(tree::CliqueTree, i::Integer)
    return tree.tree[i]
end

"""
    separator(tree::CliqueTree, i::Integer)

Get the separator at node `i`.
"""
function separator(tree::CliqueTree, i::Integer)
    return neighbors(separators(tree), i)
end

"""
    residuals(tree::CliqueTree)

Get the residuals of a clique tree.
"""
function residuals(tree::CliqueTree)
    return residuals(tree.tree)
end

"""
    separators(tree::CliqueTree)

Get the separators of a clique tree.
"""
function separators(tree::CliqueTree)
    return tree.sep
end

"""
    relatives(tree::CliqueTree)

Compute the relative indices of a clique tree.
"""
function relatives(tree::CliqueTree{V}) where {V}
    graph = BipartiteGraph{V}(
        pointers(residuals(tree))[end] - 1, pointers(separators(tree)), ne(separators(tree))
    )

    for (j, clique) in enumerate(tree)
        for i in childindices(tree, j)
            indexinsorted!(neighbors(graph, i), separator(tree, i), clique)
        end
    end

    return graph
end

##########################
# Indexed Tree Interface #
##########################

function AbstractTrees.rootindex(tree::CliqueTree)
    return rootindex(tree.tree)
end

function AbstractTrees.parentindex(tree::CliqueTree, i::Integer)
    return parentindex(tree.tree, i)
end

function firstchildindex(tree::CliqueTree, i::Integer)
    return firstchildindex(tree.tree, i)
end

function AbstractTrees.nextsiblingindex(tree::CliqueTree, i::Integer)
    return nextsiblingindex(tree.tree, i)
end

function rootindices(tree::CliqueTree)
    return rootindices(tree.tree)
end

function AbstractTrees.childindices(tree::CliqueTree, i::Integer)
    return childindices(tree.tree, i)
end

function ancestorindices(tree::CliqueTree, i::Integer)
    return ancestorindices(tree.tree, i)
end

#############################
# Abstract Vector Interface #
#############################

function Base.getindex(tree::CliqueTree, i::Integer)
    return Clique(residual(tree, i), separator(tree, i))
end

function Base.IndexStyle(::Type{<:CliqueTree})
    return IndexLinear()
end

function Base.size(tree::CliqueTree)
    return size(tree.tree)
end
