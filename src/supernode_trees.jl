"""
    SupernodeTree{V} <: AbstractVector{UnitRange{V}}

A rooted forest T = (V, E) and a function snd: U → V.
This type implements the [indexed tree interface](https://juliacollections.github.io/AbstractTrees.jl/stable/#The-Indexed-Tree-Interface).
"""
struct SupernodeTree{V} <: AbstractVector{UnitRange{V}}
    """
    The rooted forest T.
    """
    tree::Tree{V, FVector{V}, FVector{V}, FVector{V}}

    """
    A directed bipartite graph G = (U, V, E') with an arc

        (snd(u), u) ∈ E'

    for all vertices u ∈ U.
    """
    graph::BipartiteGraph{V, V, FVector{V}, OneTo{V}}

    function SupernodeTree{V}(tree::Tree, graph::BipartiteGraph) where {V}
        @argcheck last(tree) == nv(graph)
        @argcheck ne(graph) == nov(graph)
        return new{V}(tree, graph)
    end
end

function SupernodeTree(tree::Tree{V}, graph::BipartiteGraph{V}) where {V}
    return SupernodeTree{V}(tree, graph)
end

function Tree(tree::SupernodeTree)
    return Tree(tree.tree)
end

"""
    supernodetree(graph;
        alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM,
        snd::SupernodeType=DEFAULT_SUPERNODE_TYPE)

Construct a supernodal elimination tree.
"""
function supernodetree(
        graph;
        alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM,
        snd::SupernodeType = DEFAULT_SUPERNODE_TYPE,
    )
    label, tree, index, ptr, lower, upper = supernodetree(graph, alg, snd)
    return label, tree
end

function supernodetree(
        weights::AbstractVector,
        graph;
        alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM,
        snd::SupernodeType = DEFAULT_SUPERNODE_TYPE,
    )
    label, tree, index, ptr, lower, upper = supernodetree(weights, graph, alg, snd)
    return label, tree
end

function supernodetree(graph, alg::PermutationOrAlgorithm, snd::SupernodeType)
    return supernodetree(BipartiteGraph(graph), alg, snd)
end

function supernodetree(graph::AbstractGraph{V}, alg::PermutationOrAlgorithm, snd::SupernodeType) where {V}
    n = nv(graph); weights = Ones{V}(n)
    return supernodetree(weights, graph, alg, snd)
end

function supernodetree(weights::AbstractVector, graph, alg::PermutationOrAlgorithm, snd::SupernodeType)
    return supernodetree(weights, BipartiteGraph(graph), alg, snd)
end

function supernodetree(weights::AbstractVector, graph::AbstractGraph{V}, alg::PermutationOrAlgorithm, snd::SupernodeType) where {V}
    E = etype(graph); n = nv(graph); m = half(de(graph))
    nn = n + one(V); nnn = nn + one(V)

    target1 = FVector{V}(undef, max(m, nn))
    target2 = FVector{V}(undef, m)
    target3 = FVector{V}(undef, n)

    ework1 = FVector{E}(undef, n)
    pointer1 = FVector{E}(undef, nn)
    pointer2 = FVector{E}(undef, nn)
    pointer3 = FVector{V}(undef, nnn)

    colcount = FVector{V}(undef, n)
    elmorder = FVector{V}(undef, n)
    elmindex = FVector{V}(undef, n)
    sndptr = FVector{V}(undef, nn)
    sepptr = FVector{E}(undef, nn)
    new = FVector{V}(undef, n)
    parent = FVector{V}(undef, n)
    elmtree = Parent{V}(n)

    order, index = permutation(weights, graph, alg)

    sndtree, upper, lower = supernodetree_impl!(target1, target2,
        target3, ework1, pointer1, pointer2, pointer3, colcount,
        elmorder, elmindex, sndptr, sepptr, new, parent, elmtree, graph,
        order, index, snd)
        
    return order, sndtree, elmindex, sepptr, lower, upper 
end

function supernodetree_impl!(
        target1::AbstractVector{V},
        target2::AbstractVector{V},
        target3::AbstractVector{V},
        ework1::AbstractVector{E},
        pointer1::AbstractVector{E},
        pointer2::AbstractVector{E},
        pointer3::AbstractVector{V},
        colcount::AbstractVector{V},
        elmorder::AbstractVector{V},
        elmindex::AbstractVector{V},
        sndptr::AbstractVector{V},
        sepptr::AbstractVector{E},
        new::AbstractVector{V},
        parent::AbstractVector{V},
        elmtree::Parent{V},
        graph::AbstractGraph{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
        snd::SupernodeType,
    ) where {V, E}

    n = nv(graph)

    upper = eliminationtree_impl!(ework1, elmorder,
        pointer1, target1, elmtree, graph, index)

    lower = reverse!_impl!(ework1, pointer2, target2, upper)

    supcnt_impl!(colcount, new, parent, index, elmorder,
        elmindex, sndptr, UnionFind(target1, pointer3, target3),
        Ones{V}(n), lower, elmtree)

    ancestor = index

    tree = stree_impl!(new, parent, ancestor, elmorder,
        tree_impl!(target1, pointer3, target3, elmtree),
        colcount, snd)

    postorder!_impl!(target1, sndptr, elmindex, elmorder, tree)

    for i in tree
        elmorder[elmindex[i]] = i
    end

    sndptr[begin] = one(V)
    sepptr[begin] = one(E)

    for i in tree
        ii = i + one(V); j = elmorder[i]
        u = new[j]
        p = elmindex[u] = sndptr[i]

        for v in ancestorindices(elmtree, u)
            v == ancestor[j] && break
            elmindex[v] = p += one(V)
        end

        sepptr[ii] = sepptr[i] + convert(E, sndptr[i] + colcount[u] - p) - one(E)
        sndptr[ii] = p + one(V)
    end

    for v in oneto(n)
        elmorder[v] = order[v]
    end

    for v in oneto(n)
        order[elmindex[v]] = elmorder[v]
    end

    h = last(tree)
    residual = BipartiteGraph(n, h, n, sndptr, oneto(n))
    sndtree = SupernodeTree(tree_impl!(target1, pointer3, target3, tree), residual)
    return sndtree, upper, lower
end

"""
    residuals(tree::SupernodeTree)

Get the residuals of a supernodal elimination tree.
"""
function residuals(tree::SupernodeTree)
    return tree.graph
end

function Base.copy(tree::SupernodeTree)
    return SupernodeTree(copy(tree.tree), copy(tree.graph))
end

function Base.copy!(dst::SupernodeTree, src::SupernodeTree)
    copy!(dst.tree, src.tree)
    copy!(dst.graph, src.graph)
    return dst
end

function Base.:(==)(left::SupernodeTree, right::SupernodeTree)
    return left.tree == right.tree && left.graph == right.graph
end

##########################
# Indexed Tree Interface #
##########################

function AbstractTrees.rootindex(tree::SupernodeTree)
    return rootindex(tree.tree)
end

function AbstractTrees.parentindex(tree::SupernodeTree, i::Integer)
    return parentindex(tree.tree, i)
end

function rootindices(tree::SupernodeTree)
    return rootindices(tree.tree)
end

function AbstractTrees.childindices(tree::SupernodeTree, i::Integer)
    return childindices(tree.tree, i)
end

function ancestorindices(tree::SupernodeTree, i::Integer)
    return ancestorindices(tree.tree, i)
end

#############################
# Abstract Vector Interface #
#############################

function Base.getindex(tree::SupernodeTree{V}, i::Integer) where {V}
    return neighbors(tree.graph, i)
end

function Base.IndexStyle(::Type{<:SupernodeTree})
    return IndexLinear()
end

function Base.size(tree::SupernodeTree)
    return size(tree.tree)
end
