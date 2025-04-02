"""
    sortingnetwork!(solver::Solver, var::AbstractVector)

Encode a cardinality constraint using a sorting network.
    - at least k: `solver[length(var) - k + 1] = 1`
    - at most k: `solver[length(var) - k] = -1`
"""
function sortingnetwork!(solver::Solver, var::AbstractVector)
    n = length(var)

    mergesort(n) do i, j
        min = length(solver) + one(Int32)
        max = length(solver) + two(Int32)
        resize!(solver, max)

        # min ↔ var(i) ∧ var(j)
        clause!(solver, var[i], -min)
        clause!(solver, var[j], -min)
        clause!(solver, -var[i], -var[j], min)

        # max ↔ var(i) ∨ var(j)
        clause!(solver, -var[i], max)
        clause!(solver, -var[j], max)
        clause!(solver, var[i], var[j], -max)

        var[i] = min
        var[j] = max
    end

    return solver
end

# Batcher's Odd-Even Merge Sort
# https://gist.github.com/stbuehler/883635
function mergesort(f::Function, n::Integer)
    mergesort(f, oneto(n))
    return
end

function mergesort(f::Function, slice::AbstractRange)
    if length(slice) <= 2
        sorttwo(f, slice)
    else
        lhs, rhs = halves(slice)
        mergesort(f, lhs)
        mergesort(f, rhs)
        oddevenmerge(f, slice)
    end

    return
end

function is2pot(n::Integer)
    return ispositive(n) && iszero(n & (n - one(n)))
end

function is2pot(slice::AbstractRange)
    return is2pot(length(slice))
end

function odd(slice::AbstractRange)
    return (first(slice) + step(slice)):twice(step(slice)):last(slice)
end

function even(slice::AbstractRange)
    return first(slice):twice(step(slice)):last(slice)
end

function halves(slice::AbstractRange{I}) where {I}
    if length(slice) <= 1
        lhs = slice
        rhs = oneto(zero(I))
    else
        if is2pot(slice)
            mid = first(slice) + half(length(slice)) * step(slice)
        else
            len = two(I)

            while len < length(slice)
                len = twice(len)
            end

            mid = first(slice) + half(len) * step(slice)
        end

        lhs = first(slice):step(slice):(mid - one(I))
        rhs = mid:step(slice):last(slice)
    end

    return lhs, rhs
end

function sorttwo(f::Function, slice::AbstractRange)
    if istwo(length(slice))
        f(slice[1], slice[2])
    end

    return
end

function oddevenmerge(f::Function, slice::AbstractRange)
    if length(slice) <= 2
        sorttwo(f, slice)
    else
        oddevenmerge(f, odd(slice))
        oddevenmerge(f, even(slice))

        for i in 2:2:(length(slice) - 1)
            f(slice[i], slice[i + 1])
        end
    end

    return
end
