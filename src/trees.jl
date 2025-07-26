"""
    Tree{I, Prnt, Ptr, Tgt} <: AbstractUnitRange{I}

A rooted forest T = (V, E) with edges oriented from leaf to root.
This type implements the [indexed tree interface](https://juliacollections.github.io/AbstractTrees.jl/stable/#The-Indexed-Tree-Interface).
"""
struct Tree{I, Prnt, Ptr, Tgt} <: AbstractUnitRange{I}
    """
    The forest T.
    """
    tree::Parent{I, Prnt}

    """
    A bipartite graph G = (V, V ∪ {r}, E'). E' contains an arc

        (v, w) ∈ E'

    for each arc (w, v) ∈ E. Additionally, E' contains an arc

        (r, w) ∈ E'

    for each root vertex w of T.
    """
    graph::BipartiteGraph{I, I, Ptr, Tgt}

    function Tree{I, Prnt, Ptr, Tgt}(tree::Parent{I, Prnt}, graph::BipartiteGraph{I, I, Ptr, Tgt}) where {I, Prnt, Ptr, Tgt}
        @argcheck last(tree) == nov(graph)
        @argcheck last(tree) == nv(graph) - one(I)
        @argcheck last(tree) == ne(graph)
        return new{I, Prnt, Ptr, Tgt}(tree, graph)
    end
end

function Tree{I, Prnt, Ptr, Tgt}(tree::Parent{I, Prnt}) where {I, Prnt, Ptr, Tgt}
    n = last(tree); nn = n + one(I)
    count = FVector{I}(undef, nn)
    graph = BipartiteGraph{I, I, Ptr, Tgt}(n, nn, n)
    return tree_impl!(count, graph, tree)
end

function Tree(tree::Tree)
    return Tree(tree.tree, tree.graph)
end

function Tree(tree::Parent{I, Prnt}, graph::BipartiteGraph{I, I, Ptr, Tgt}) where {I, Prnt, Ptr, Tgt}
    return Tree{I, Prnt, Ptr, Tgt}(tree, graph)
end

function Tree(tree::Parent{I, Prnt}) where {I, Prnt}
    return Tree{I, Prnt, FVector{I}, FVector{I}}(tree)
end

"""
    eliminationtree([weights, ]graph;
        alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM)

Construct a [tree-depth decomposition](https://en.wikipedia.org/wiki/Tr%C3%A9maux_tree) of a simple graph.

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

julia> label, tree = eliminationtree(graph);

julia> tree
8-element Tree{Int64, Vector{Int64}, Array{Int64, 0}, Vector{Int64}, Vector{Int64}}:
 8
 └─ 7
    ├─ 5
    └─ 6
       ├─ 1
       ├─ 3
       │  └─ 2
       └─ 4
```
"""
function eliminationtree(graph; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM)
    return eliminationtree(graph, alg)
end

function eliminationtree(weights::AbstractVector, graph; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM)
    return eliminationtree(weights, graph, alg)
end

function eliminationtree(graph, alg::PermutationOrAlgorithm)
    return eliminationtree(BipartiteGraph(graph), alg)
end

function eliminationtree(graph::AbstractGraph, alg::PermutationOrAlgorithm)
    return eliminationtree(graph, permutation(graph, alg))
end

function eliminationtree(graph::AbstractGraph{V}, (order, index)::Tuple{Vector{V}, Vector{V}}) where {V}
    E = etype(graph); n = nv(graph); m = half(de(graph)); nn = n + one(V)
    count = FVector{E}(undef, n)
    ancestor = FVector{V}(undef, n)
    pointer = FVector{E}(undef, nn)
    target = FVector{V}(undef, m)
    tree = Parent{V}(n)
    eliminationtree_impl!(count, ancestor, pointer, target, tree, graph, index)
    return order, tree
end

function eliminationtree(weights::AbstractVector, graph, alg::PermutationOrAlgorithm)
    return eliminationtree(weights, BipartiteGraph(graph), alg)
end

function eliminationtree(weights::AbstractVector, graph::AbstractGraph, alg::PermutationOrAlgorithm)
    return eliminationtree(graph, permutation(weights, graph, alg))
end

function eliminationtree_impl!(
        count::AbstractVector{E},
        ancestor::AbstractVector{V},
        pointer::AbstractVector{E},
        target::AbstractVector{V},
        tree::Parent{V},
        graph,
        index::AbstractVector{V},
    ) where {V, E}
    upper = sympermute!_impl!(count, pointer, target, graph, index, Forward)
    etree_impl!(tree, ancestor, upper)
    return upper
end

function tree_impl!(
        count::AbstractVector{V},
        graph::BipartiteGraph{V, V},
        tree::Parent{V},
    ) where {V}
    reverse!_impl!(count, graph, tree)
    return Tree(tree, graph)
end

function tree_impl!(
        count::AbstractVector{V},
        pointer::AbstractVector{V},
        target::AbstractVector{V},
        tree::Parent{V},
    ) where {V}
    n = last(tree); nn = n + one(V)
    graph = BipartiteGraph(n, nn, n, pointer, target)
    return tree_impl!(count, graph, tree)
end

function Base.copy(tree::Tree)
    return Tree(copy(tree.tree), copy(tree.graph))
end

function Base.copy!(dst::Tree, src::Tree)
    copy!(dst.tree, src.tree)
    copy!(dst.graph, src.graph)
    return dst
end

function Base.:(==)(left::Tree, right::Tree)
    return left.tree == right.tree && left.graph == right.graph
end

##########################
# Indexed Tree Interface #
##########################

@propagate_inbounds function AbstractTrees.parentindex(tree::Tree, i::Integer)
    @boundscheck checkbounds(tree, i)
    @inbounds j = parentindex(tree.tree, i)
    return j
end

function rootindices(tree::Tree)
    n = nv(tree.graph)
    @inbounds roots = neighbors(tree.graph, n)
    return roots
end

function AbstractTrees.rootindex(tree::Tree)
    roots = rootindices(tree)

    if isempty(roots)
        root = nothing
    else
        root = first(roots)
    end

    return root
end

@propagate_inbounds function AbstractTrees.childindices(tree::Tree, i::Integer)
    @boundscheck checkbounds(tree, i)
    @inbounds children = neighbors(tree.graph, i)
    return children
end

@propagate_inbounds function ancestorindices(tree::Tree, i::Integer)
    @boundscheck checkbounds(tree, i)
    @inbounds ancestors = ancestorindices(tree.tree, i)
    return ancestors
end

#################################
# Abstract Unit Range Interface #
#################################

function Base.first(tree::Tree)
    return first(tree.tree)
end

function Base.last(tree::Tree)
    return last(tree.tree)
end
