"""
    AbstractTree{V} = Union{Tree{V}, SupernodeTree{V}, CliqueTree{V}}

A rooted forest. This type implements the [indexed tree interface](https://juliacollections.github.io/AbstractTrees.jl/stable/#The-Indexed-Tree-Interface).
"""
const AbstractTree{V} = Union{Tree{V},SupernodeTree{V},CliqueTree{V}}

function Base.show(io::IO, ::MIME"text/plain", tree::T) where {T<:AbstractTree}
    n = length(tree)
    println(io, "$n-element $T:")

    for (i, root) in enumerate(take(rootindices(tree), MAX_ITEMS_PRINTED + 1))
        if i <= MAX_ITEMS_PRINTED
            node = IndexNode(tree, root)

            for line in eachsplit(strip(repr_tree(node)), "\n")
                println(io, " $line")
            end
        else
            println(io, " ⋮")
        end
    end
end

##########################
# Indexed Tree Interface #
##########################

"""
    rootindices(tree::AbstractTree)

Get the roots of a rooted forest.
"""
rootindices(tree::AbstractTree)

"""
    firstchildindex(tree::AbstractTree, i::Integer)

Get the first child of node `i`. Returns `nothing` if `i` is a leaf.
"""
firstchildindex(tree::AbstractTree, i::Integer)

"""
    ancestorindices(tree::AbstractTree, i::Integer)

Get the proper ancestors of node `i`.
"""
ancestorindices(tree::AbstractTree, i::Integer)

function AbstractTrees.ParentLinks(::Type{IndexNode{T,V}}) where {V,T<:AbstractTree{V}}
    return StoredParents()
end

function AbstractTrees.SiblingLinks(::Type{IndexNode{T,V}}) where {V,T<:AbstractTree{V}}
    return StoredSiblings()
end

function AbstractTrees.NodeType(::Type{IndexNode{T,V}}) where {V,T<:AbstractTree{V}}
    return HasNodeType()
end

function AbstractTrees.nodetype(::Type{IndexNode{T,V}}) where {V,T<:AbstractTree{V}}
    return IndexNode{T,V}
end
