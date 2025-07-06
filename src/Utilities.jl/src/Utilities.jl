module Utilities

using Base: @propagate_inbounds
using Base.Iterators
using Base.Order
using FixedSizeArrays
using Graphs

# arrays
export AbstractScalar, Scalar, FScalar, FVector

# arithmetic
export tolerance, twice, half, two, three, four, five, six, eight, ispositive, isnegative, istwo, isthree, isfour

# graphs
export eltypedegree, de

# sorted collections
export mergesorted!, indexinsorted!, swap!

# printing
export MAX_ITEMS_PRINTED, printiterator

const AbstractScalar{T} = AbstractArray{T, 0}
const Scalar{T} = Array{T, 0}
const FScalar{T} = FixedSizeArrayDefault{T, 0}
const FVector{T} = FixedSizeArrayDefault{T, 1}
const MAX_ITEMS_PRINTED = 5

# `tol = tolerance(W)` should satisfy
#     v < w iff v ≤ w - tol
#     v ≤ w iif v < w + tol
# for all weights v and w.
function tolerance(::Type{W}) where {W <: AbstractFloat}
    return W(1.0e-5)
end

function tolerance(::Type{W}) where {W <: Integer}
    return one(W)
end

function twice(i::I) where {I}
    return i + i
end

function half(i::I) where {I}
    return i ÷ two(I)
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

function ispositive(i::I) where {I}
    return i > zero(I)
end

function isnegative(i::I) where {I}
    return i < zero(I)
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

# Compute the union of sorted sets `source1` and `source2`.
# The result is written to `target`.
function mergesorted!(
        target::AbstractVector{I},
        source1::AbstractVector{I},
        source2::AbstractVector{I},
        order::Ordering = Forward,
    ) where {I}
    s1 = firstindex(source1)
    s2 = firstindex(source2)
    t = firstindex(target)

    @inbounds while s1 in eachindex(source1) && s2 in eachindex(source2)
        x1 = source1[s1]
        x2 = source2[s2]

        if isequal(x1, x2)
            target[t] = x1
            s1 += 1
            s2 += 1
        elseif lt(order, x1, x2)
            target[t] = x1
            s1 += 1
        else
            target[t] = x2
            s2 += 1
        end

        t += 1
    end

    @inbounds while s1 in eachindex(source1)
        target[t] = source1[s1]
        s1 += 1
        t += 1
    end

    @inbounds while s2 in eachindex(source2)
        target[t] = source2[s2]
        s2 += 1
        t += 1
    end

    return @view target[begin:(t - 1)]
end

function indexinsorted!(
        target::AbstractVector{I},
        source1::AbstractVector{I},
        source2::AbstractVector{I},
        order::Ordering = Forward,
    ) where {I}
    s1 = firstindex(source1)
    s2 = firstindex(source2)

    while s1 in eachindex(source1)
        x1 = source1[s1]
        x2 = source2[s2]

        if !lt(order, x2, x1)
            target[s1] = s2
            s1 += 1
        end

        s2 += 1
    end

    return target
end

@propagate_inbounds function swap!(vector::AbstractVector, i::Integer, j::Integer)
    @boundscheck checkbounds(vector, i)
    @boundscheck checkbounds(vector, j)
    @inbounds v = vector[i]
    @inbounds vector[i] = vector[j]
    @inbounds vector[j] = v
    return vector
end

function printiterator(io::IO, iterator::T) where {T}
    return printiterator(io, iterator, Base.IteratorSize(T))
end

function printiterator(io::IO, iterator::T, ::Any) where {T}
    print(io, "$T:")
    printelements(io, iterator)
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
