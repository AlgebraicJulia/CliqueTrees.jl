"""
    SupernodeTree{V} <: AbstractVector{UnitRange{V}}

A supernodal elimination tree with vertices of type `V`.
This type implements the [indexed tree interface](https://juliacollections.github.io/AbstractTrees.jl/stable/#The-Indexed-Tree-Interface).
"""
struct SupernodeTree{V} <: AbstractVector{UnitRange{V}}
    tree::Tree{V, Vector{V}, Scalar{V}, Vector{V}, Vector{V}}
    res::BipartiteGraph{V, V, Vector{V}, OneTo{V}}
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
    label, tree, count, index, ptr, lower, upper = supernodetree(graph, alg, snd)
    return label, tree
end

function supernodetree(
        weights::AbstractVector,
        graph;
        alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM,
        snd::SupernodeType = DEFAULT_SUPERNODE_TYPE,
    )
    label, tree, count, index, ptr, lower, upper = supernodetree(weights, graph, alg, snd)
    return label, tree
end

function supernodetree(graph, alg::PermutationOrAlgorithm, snd::SupernodeType)
    return supernodetree(snd, eliminationtree(graph, alg)...)
end

function supernodetree(weights::AbstractVector, graph, alg::PermutationOrAlgorithm, snd::SupernodeType)
    return supernodetree(snd, eliminationtree(weights, graph, alg)...)
end

function supernodetree(snd::SupernodeType, label::Vector{V}, etree::Tree{V}, upper::BipartiteGraph{V, E}) where {V, E}
    lower = reverse(upper)
    rowcount, colcount = supcnt(lower, etree)
    new, ancestor, tree = stree(etree, colcount, snd)

    n = nv(lower); m = last(tree); mm = m + one(V)
    eindex = Vector{V}(undef, n)
    sndptr = Vector{V}(undef, mm); sndptr[begin] = one(V)
    sepptr = Vector{E}(undef, mm); sepptr[begin] = one(E)
    eorder = invperm(postorder!(tree))

    for i in tree
        ii = i + one(V); j = eorder[i]
        u = new[j]
        p = eindex[u] = sndptr[i]

        for v in ancestorindices(etree, u)
            v == ancestor[j] && break
            eindex[v] = p += one(V)
        end

        sepptr[ii] = sepptr[i] + convert(E, sndptr[i] + colcount[u] - p) - one(E)
        sndptr[ii] = p + one(V)
    end

    invpermute!(label, eindex)
    sndtree = SupernodeTree(tree, BipartiteGraph(n, m, n, sndptr, oneto(n)))
    return label, sndtree, rowcount, eindex, sepptr, lower, upper
end

function supernodetree(snd::Nodal, label::Vector{V}, etree::Tree{V}, upper::BipartiteGraph{V, E}) where {V, E}
    lower = reverse(upper)
    rowcount, colcount = supcnt(lower, etree)

    n = nv(lower); m = last(etree); mm = m + one(V)
    eindex = postorder!(etree); eorder = invperm(eindex)
    sndptr = Vector{V}(undef, mm); sndptr[begin] = one(V)
    sepptr = Vector{E}(undef, mm); sepptr[begin] = one(E)

    for i in etree
        ii = i + one(V); j = eorder[i]
        sepptr[ii] = sepptr[i] + convert(E, colcount[j]) - one(E)
        sndptr[ii] = sndptr[i] + one(V)
    end

    invpermute!(label, eindex)
    sndtree = SupernodeTree(etree, BipartiteGraph(n, m, n, sndptr, oneto(n)))
    return label, sndtree, rowcount, eindex, sepptr, lower, upper
end

"""
    residuals(tree::SupernodeTree)

Get the residuals of a supernodal elimination tree.
"""
function residuals(tree::SupernodeTree)
    return tree.res
end

function Base.copy(tree::SupernodeTree)
    return SupernodeTree(copy(tree.tree), copy(tree.res))
end

function Base.copy!(dst::SupernodeTree, src::SupernodeTree)
    copy!(dst.tree, src.tree)
    copy!(dst.res, src.res)
    return dst
end

function Base.:(==)(left::SupernodeTree, right::SupernodeTree)
    return left.tree == right.tree && left.res == right.res
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

function firstchildindex(tree::SupernodeTree, i::Integer)
    return firstchildindex(tree.tree, i)
end

function AbstractTrees.nextsiblingindex(tree::SupernodeTree, i::Integer)
    return nextsiblingindex(tree.tree, i)
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
    return neighbors(tree.res, i)
end

function Base.IndexStyle(::Type{<:SupernodeTree})
    return IndexLinear()
end

function Base.size(tree::SupernodeTree)
    return size(tree.tree)
end
