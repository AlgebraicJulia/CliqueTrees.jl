function double(i::I) where {I}
    return i * two(I)
end

function halve(i::I) where {I}
    return i ÷ two(I)
end

function three(::Type{I}) where {I}
    return one(I) + two(I)
end

function two(::Type{I}) where {I}
    return one(I) + one(I)
end

function ispositive(i::I) where {I}
    return i > zero(I)
end

function isnegative(i::I) where {I}
    return i < zero(I)
end

function printiterator(io::IO, iterator::T) where {T}
    print(io, "$T:")

    for (i, v) in enumerate(take(iterator, MAX_ITEMS_PRINTED + 1))
        if i <= MAX_ITEMS_PRINTED
            print(io, "\n $v")
        else
            print(io, "\n ⋮")
        end
    end

    return
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
