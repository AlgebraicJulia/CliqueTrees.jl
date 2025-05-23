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
    return supernodetree(graph, snd, eliminationtree(graph, alg)...)
end

function supernodetree(weights::AbstractVector, graph, alg::PermutationOrAlgorithm, snd::SupernodeType)
    return supernodetree(graph, snd, eliminationtree(weights, graph, alg)...)
end

function supernodetree(graph, snd, label, etree, upper)
    lower = reverse(upper)
    rowcount, colcount = supcnt(lower, etree)
    new, ancestor, tree = stree(etree, colcount, snd)

    V = eltype(lower)
    E = etype(lower)
    eindex = Vector{V}(undef, length(etree))
    sndptr = Vector{V}(undef, length(tree) + 1)
    sepptr = Vector{E}(undef, length(tree) + 1)
    sndptr[begin] = sepptr[begin] = 1

    for (i, j) in enumerate(invperm(postorder!(tree)))
        u = new[j]
        p = eindex[u] = sndptr[i]

        for v in takewhile(v -> v != ancestor[j], ancestorindices(etree, u))
            eindex[v] = p += 1
        end

        sepptr[i + 1] = sndptr[i] + sepptr[i] + colcount[u] - p - 1
        sndptr[i + 1] = p + 1
    end

    invpermute!(label, eindex)
    sndtree = SupernodeTree(tree, BipartiteGraph(nv(lower), sndptr, vertices(lower)))
    return label, sndtree, rowcount, eindex, sepptr, lower, upper
end

function supernodetree(graph, snd::Nodal, label, etree, upper)
    lower = reverse(upper)
    rowcount, colcount = supcnt(lower, etree)

    V = eltype(lower)
    E = etype(lower)
    eindex = postorder!(etree)
    sndptr = Vector{V}(undef, length(etree) + 1)
    sepptr = Vector{E}(undef, length(etree) + 1)
    sndptr[begin] = sepptr[begin] = 1

    for (i, j) in enumerate(invperm(eindex))
        sepptr[i + 1] = sepptr[i] + colcount[j] - 1
        sndptr[i + 1] = sndptr[i] + 1
    end

    invpermute!(label, eindex)
    sndtree = SupernodeTree(etree, BipartiteGraph(nv(lower), sndptr, vertices(lower)))
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
