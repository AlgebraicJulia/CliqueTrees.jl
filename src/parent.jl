"""
    Parent{V, Prnt} <: AbstractUnitRange{V}

A rooted forest T = (V, E) with edges oriented from
leaf to root.
"""
struct Parent{V <: Integer, Prnt <: AbstractVector{V}} <: AbstractUnitRange{V}
    """
    V is equal to the set

        V := {1, 2, ..., nv}
    """
    nv::V

    """
    E is equal to the set

        E := {(v, prnt[v]) : v ∈ V and prnt[v] > 0}

    In particular, v is a root if an only if prnt[v] is nonpositive.
    """
    prnt::Prnt

    function Parent{V, Prnt}(nv::Integer, prnt::AbstractVector) where {V <: Integer, Prnt <: AbstractVector{V}}
        @argcheck !isnegative(nv)
        @argcheck nv <= length(prnt)
        return new{V, Prnt}(nv, prnt)
    end
end

const AbstractParent{V} = Parent{V}

function Parent(nv::V, prnt::Prnt) where {V <: Integer, Prnt <: AbstractVector{V}}
    tree = Parent{V, Prnt}(nv, prnt)
    return tree
end

function Parent{V}(nv::Integer) where {V}
    prnt = FVector{V}(undef, nv)
    tree = Parent(convert(V, nv), prnt)
    return tree
end

# A Compact Row Storage Scheme for Cholesky Factors Using Elimination Trees
# Liu
# Algorithm 4.2: Elimination Tree by Path Compression.
#
# Construct the elimination tree of an ordered graph.
# The complexity is O(m log n), where m = |E| and n = |V|.
function etree(upper::AbstractGraph{V}) where {V}
    n = nv(upper)
    tree = Parent(n, Vector{V}(undef, n))
    ancestor = Vector{V}(undef, n)
    etree_impl!(tree, ancestor, upper)
    return tree
end

function etree_impl!(
        tree::Parent{V},
        ancestor::AbstractVector{V},
        upper::AbstractGraph{V},
    ) where {V}
    @argcheck nv(upper) == length(tree)
    @argcheck nv(upper) <= length(ancestor)
    n = nv(upper); parent = tree.prnt

    @inbounds for i in oneto(n)
        parent[i] = zero(V)
        ancestor[i] = zero(V)

        for k in neighbors(upper, i)
            r = k

            while !iszero(ancestor[r]) && ancestor[r] != i
                t = ancestor[r]
                ancestor[r] = i
                r = t
            end

            if iszero(ancestor[r])
                ancestor[r] = i
                parent[r] = i
            end
        end
    end

    return
end

# Compute the `root`, `child`, and `brother` fields of a forest.
function lcrs_impl!(
        brother::AbstractVector{V},
        child::AbstractVector{V},
        tree::AbstractParent{V},
    ) where {V}
    @argcheck length(tree) <= length(brother)
    @argcheck length(tree) <= length(child)
    root = zero(V)

    @inbounds for i in tree
        child[i] = zero(V)
    end

    @inbounds for i in reverse(tree)
        j = parentindex(tree, i)

        if isnothing(j)
            brother[i] = root; root = i
        else
            brother[i] = child[j]; child[j] = i
        end
    end

    return root
end

##########################
# Indexed Tree Interface #
##########################

@propagate_inbounds function AbstractTrees.parentindex(tree::Parent, i::Integer)
    @boundscheck checkbounds(tree, i)
    @inbounds j = tree.prnt[i]

    if !ispositive(j)
        j = nothing
    end

    return j
end

@propagate_inbounds function ancestorindices(tree::Parent, i::Integer)
    @boundscheck checkbounds(tree, i)
    @inbounds head = view(tree.prnt, i)
    return SinglyLinkedList(head, tree.prnt)
end

#################################
# Abstract Unit Range Interface #
#################################

function Base.first(tree::Parent{V}) where {V}
    return one(V)
end

function Base.last(tree::Parent{V}) where {V}
    return tree.nv
end
