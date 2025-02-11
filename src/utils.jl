# Compute the union of sorted sets `source1` and `source2`.
# The result is written to `target`.
function mergesorted!(
    target::AbstractVector{I},
    source1::AbstractVector{I},
    source2::AbstractVector{I},
    order::Ordering=ForwardOrdering(),
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

    @view target[begin:(t - 1)]
end

function indexinsorted!(
    target::AbstractVector{I},
    source1::AbstractVector{I},
    source2::AbstractVector{I},
    order::Ordering=ForwardOrdering(),
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
