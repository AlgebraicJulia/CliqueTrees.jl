module Utilities

using Base: @propagate_inbounds, oneto
using Base.Iterators
using Base.Order
using FillArrays: Ones
using FixedSizeArrays
using Graphs

# arrays
export AbstractScalar, Scalar, FScalar, FVector, FMatrix, FArray

# arithmetic
export tolerance, twice, half, two, three, four, five, six, eight, ispositive, isnegative, istwo, isthree, isfour

# graphs
export eltypedegree, de, quotientlists!, quotientdegree!

# printing
export MAX_ITEMS_PRINTED, printiterator

const AbstractScalar{T} = AbstractArray{T, 0}
const Scalar{T} = Array{T, 0}

const FArray{T} = FixedSizeArrayDefault{T}
const FMatrix{T} = FArray{T, 2}
const FVector{T} = FArray{T, 1}
const FScalar{T} = FArray{T, 0}

const MAX_ITEMS_PRINTED = 5

# `tol = tolerance(W)` should satisfy
#     v < w iff v ≤ w - tol
#     v ≤ w iif v < w + tol
# for all weights v and w.
function tolerance(::Type{W}) where {W <: AbstractFloat}
    return convert(W, 1.0e-5)
end

function tolerance(::Type{W}) where {W <: Integer}
    return one(W)
end

function twice(i::I) where {I}
    return i + i
end

function half(i::I) where {I <: Integer}
    return i ÷ two(I)
end

function half(i::W) where {W <: AbstractFloat}
    return i / two(W)
end

function eight(::Type{I}) where {I}
    return twice(four(I))
end

function eight(::I) where {I}
    return eight(I)
end

function six(::Type{I}) where {I}
    return twice(three(I))
end

function six(::I) where {I}
    return six(I)
end

function five(::Type{I}) where {I}
    return one(I) + four(I)
end

function five(::I) where {I}
    return five(I)
end

function four(::Type{I}) where {I}
    return one(I) + three(I)
end

function four(::I) where {I}
    return four(I)
end

function three(::Type{I}) where {I}
    return one(I) + two(I)
end

function three(::I) where {I}
    return three(I)
end

function two(::Type{I}) where {I}
    return one(I) + one(I)
end

function two(::I) where {I}
    return two(I)
end

@static if isdefined(Base, :ispositive)
    import Base.ispositive
else
    function ispositive(i::I) where {I}
        return i > zero(I)
    end
end

@static if isdefined(Base, :isnegative)
    import Base.isnegative
else
    function isnegative(i::I) where {I}
        return i < zero(I)
    end
end

function isfour(i::I) where {I}
    return i == four(I)
end

function isthree(i::I) where {I}
    return i == three(I)
end

function istwo(i::I) where {I}
    return i == two(I)
end

function eltypedegree(graph::AbstractGraph{V}, i::Integer) where {V}
    n::V = outdegree(graph, i)
    return n
end

function de(graph::AbstractGraph)
    m = ne(graph)

    if !is_directed(graph)
        m = twice(m)
    end

    return m
end

function printiterator(io::IO, iterator::T) where {T}
    return printiterator(io, iterator, Base.IteratorSize(T))
end

function printiterator(io::IO, iterator::T, ::Any) where {T}
    print(io, "$T:")
    printelements(io, iterator)
    return
end

# The lists of a quotient graph, as `hamf_impl!` takes them: the variables 1:n,
# joined by the edges (`xadj`, `adjncy`), and the elements n + 1:n + m, where
# the members of the element e are etgt[eptr[e]:eptr[e + 1] - 1]. The list of
# the node i is iw[pe[i]:pe[i] + len[i] - 1]: for a variable, the `elen[i]`
# elements that contain it, and then its neighbors; for an element, its
# members. Repeated members are dropped, and an element with fewer than two
# members is left empty. Returns one past the last entry of `iw`.
function quotientlists!(
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        marker::AbstractVector{V},
        n::V,
        m::V,
        xadj::AbstractVector{E},
        adjncy::AbstractVector{V},
        eptr::AbstractVector{E},
        etgt::AbstractVector{V},
    ) where {V, E}
    nnode = n + m

    @inbounds for i in oneto(nnode)
        len[i] = zero(V)
    end

    @inbounds for v in oneto(n)
        marker[v] = zero(V); elen[v] = zero(V)
    end

    # `len[n + e]` is the number of distinct members of the element `e`,
    # or zero if it is less than two
    @inbounds for e in oneto(m)
        k = zero(V)

        for p in eptr[e]:eptr[e + one(V)] - one(E)
            v = etgt[p]

            if marker[v] != e
                marker[v] = e; k += one(V)
            end
        end

        if k < two(V)
            k = zero(V)
        end

        len[n + e] = k
    end

    # `elen[v]` is the number of nonempty elements containing the variable
    # `v`, and `len[v]` adds the number of its neighbors
    @inbounds for e in oneto(m)
        if ispositive(len[n + e])
            for p in eptr[e]:eptr[e + one(V)] - one(E)
                v = etgt[p]

                if marker[v] != m + e
                    marker[v] = m + e; elen[v] += one(V)
                end
            end
        end
    end

    @inbounds for v in oneto(n)
        len[v] = elen[v] + convert(V, xadj[v + one(V)] - xadj[v])
    end

    # `pe[i + 1]` is the tail of the list of `i`: on return, it is
    # the head of the list of `i + 1`
    @inbounds pe[begin] = pfree = one(E)

    @inbounds for i in oneto(nnode)
        pe[i + one(V)] = pfree; pfree += convert(E, len[i])
    end

    # the elements come first in the list of each variable
    @inbounds for e in oneto(m)
        x = n + e

        if ispositive(len[x])
            for p in eptr[e]:eptr[e + one(V)] - one(E)
                v = etgt[p]

                if marker[v] != twice(m) + e
                    marker[v] = twice(m) + e
                    iw[pe[x + one(V)]] = v; pe[x + one(V)] += one(E)
                    iw[pe[v + one(V)]] = x; pe[v + one(V)] += one(E)
                end
            end
        end
    end

    # and then its neighbors
    @inbounds for v in oneto(n)
        for p in xadj[v]:xadj[v + one(V)] - one(E)
            iw[pe[v + one(V)]] = adjncy[p]; pe[v + one(V)] += one(E)
        end
    end

    return pfree
end

# The external degree of each variable of the quotient graph (`pe`, `len`,
# `elen`, `iw`) (see `quotientlists!`): the total of the `weights` of its
# neighbors and of the members of its elements, less its own. The members
# of the elements are marked, in O(Σ len + Σ |C|²) time and O(n) space.
function quotientdegree!(
        degree::AbstractVector{W},
        marker::AbstractVector{V},
        weights::AbstractVector,
        pe::AbstractVector{E},
        len::AbstractVector{V},
        elen::AbstractVector{V},
        iw::AbstractVector{V},
        n::V,
    ) where {W, V, E}
    @inbounds for v in oneto(n)
        marker[v] = zero(V)
    end

    @inbounds for v in oneto(n)
        marker[v] = v; deg = zero(W)
        p1 = pe[v]; p2 = p1 + convert(E, elen[v]) - one(E); p3 = p1 + convert(E, len[v]) - one(E)

        # the members of its elements
        for p in p1:p2
            e = iw[p]

            for q in pe[e]:pe[e] + convert(E, len[e]) - one(E)
                u = iw[q]

                if marker[u] != v
                    marker[u] = v; deg += weights[u]
                end
            end
        end

        # its neighbors
        for p in (p2 + one(E)):p3
            u = iw[p]

            if marker[u] != v
                marker[u] = v; deg += weights[u]
            end
        end

        degree[v] = deg
    end

    return
end

function printiterator(io::IO, iterator::T, ::Base.HasLength) where {T}
    n = length(iterator)
    print(io, "$n-element $T:")
    printelements(io, iterator)
    return
end

function printelements(io::IO, iterator)
    count = 0

    for v in iterator
        count += 1

        if count <= MAX_ITEMS_PRINTED
            print(io, "\n $v")
        else
            print(io, "\n ⋮")
            break
        end
    end

    return
end

end
