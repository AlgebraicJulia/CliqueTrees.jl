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
export eltypedegree, de, coverincidence!, coverdegree!

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

# The incidence graph of a clique cover: the vertices 1:n and the cliques
# n + 1:n + m, where the members of the clique c are
# ctgt[cptr[c]:cptr[c + 1] - 1]. The neighbors of the node i are
# iw[pe[i]:pe[i] + len[i] - 1]: the cliques containing a vertex, or the
# members of a clique. Repeated members are dropped, and a clique with fewer
# than two members is left empty. Returns one past the last entry of `iw`.
function coverincidence!(
        pe::AbstractVector{E},
        len::AbstractVector{V},
        iw::AbstractVector{V},
        marker::AbstractVector{V},
        n::V,
        m::V,
        cptr::AbstractVector{E},
        ctgt::AbstractVector{V},
    ) where {V, E}
    nnode = n + m

    @inbounds for i in oneto(nnode)
        len[i] = zero(V)
    end

    @inbounds for v in oneto(n)
        marker[v] = zero(V)
    end

    # `len[n + c]` is the number of distinct members of the clique `c`,
    # or zero if it is less than two
    @inbounds for c in oneto(m)
        k = zero(V)

        for p in cptr[c]:cptr[c + one(V)] - one(E)
            v = ctgt[p]

            if marker[v] != c
                marker[v] = c; k += one(V)
            end
        end

        if k < two(V)
            k = zero(V)
        end

        len[n + c] = k
    end

    # `len[v]` is the number of nonempty cliques containing the vertex `v`
    @inbounds for c in oneto(m)
        if ispositive(len[n + c])
            for p in cptr[c]:cptr[c + one(V)] - one(E)
                v = ctgt[p]

                if marker[v] != m + c
                    marker[v] = m + c; len[v] += one(V)
                end
            end
        end
    end

    # `pe[i + 1]` is the tail of the list of `i`: on return, it is
    # the head of the list of `i + 1`
    @inbounds pe[begin] = pfree = one(E)

    @inbounds for i in oneto(nnode)
        pe[i + one(V)] = pfree; pfree += convert(E, len[i])
    end

    @inbounds for c in oneto(m)
        e = n + c

        if ispositive(len[e])
            for p in cptr[c]:cptr[c + one(V)] - one(E)
                v = ctgt[p]

                if marker[v] != twice(m) + c
                    marker[v] = twice(m) + c
                    iw[pe[e + one(V)]] = v; pe[e + one(V)] += one(E)
                    iw[pe[v + one(V)]] = e; pe[v + one(V)] += one(E)
                end
            end
        end
    end

    return pfree
end

# The external degree of each vertex of the graph covered by the cliques of
# the incidence graph (`pe`, `len`, `iw`) (see `coverincidence!`): the weight
# `nv` of the union of its cliques, less its own. The vertices of the
# cliques are marked, in O(Σ |C|²) time and O(n) space.
function coverdegree!(
        degree::AbstractVector{V},
        marker::AbstractVector{V},
        nv::AbstractVector{V},
        pe::AbstractVector{E},
        len::AbstractVector{V},
        iw::AbstractVector{V},
        n::V,
    ) where {V, E}
    @inbounds for v in oneto(n)
        marker[v] = zero(V)
    end

    @inbounds for v in oneto(n)
        marker[v] = v; deg = zero(V)

        for p in pe[v]:pe[v] + convert(E, len[v]) - one(E)
            e = iw[p]

            for q in pe[e]:pe[e] + convert(E, len[e]) - one(E)
                u = iw[q]

                if marker[u] != v
                    marker[u] = v; deg += nv[u]
                end
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
