# Range minimum (or maximum) queries: blocks of 32 values, and a sparse table
# over the blocks. The values have type T, and positions type V. Queries
# return positions.
const RMQBLOCK = 32

struct RMQ{V, T}
    val::FVector{T}
    tab::FVector{V}
    nb::Int
    ismax::Bool
end

@inline function rmqbetter(val::AbstractVector, i::Integer, j::Integer, ismax::Bool)
    return ismax ? val[i] > val[j] : val[i] < val[j]
end

function RMQ(val::FVector{V}, ismax::Bool) where {V}
    return RMQ(V, val, ismax)
end

function RMQ(::Type{V}, val::FVector{T}, ismax::Bool) where {V, T}
    n = length(val); nb = cld(n, RMQBLOCK); nl = 1

    while (1 << nl) <= nb
        nl += 1
    end

    tab = FVector{V}(undef, max(nl * nb, 1))

    for j in 1:nb
        lo = (j - 1) * RMQBLOCK + 1; hi = min(j * RMQBLOCK, n); best = lo

        for i in lo + 1:hi
            rmqbetter(val, i, best, ismax) && (best = i)
        end

        tab[j] = best
    end

    for k in 1:nl - 1, j in 1:nb - (1 << k) + 1
        p = tab[(k - 1) * nb + j]; q = tab[(k - 1) * nb + j + (1 << (k - 1))]
        tab[k * nb + j] = rmqbetter(val, q, p, ismax) ? q : p
    end

    return RMQ{V, T}(val, tab, nb, ismax)
end

# the position of the minimum (maximum) of val[i:j], i <= j
function rmq(r::RMQ, i::Integer, j::Integer)
    val = r.val; ismax = r.ismax
    bi = (i - 1) ÷ RMQBLOCK + 1; bj = (j - 1) ÷ RMQBLOCK + 1; best = i

    if bj - bi <= 1
        for t in i + 1:j
            rmqbetter(val, t, best, ismax) && (best = t)
        end
    else
        for t in i + 1:bi * RMQBLOCK
            rmqbetter(val, t, best, ismax) && (best = t)
        end

        for t in (bj - 1) * RMQBLOCK + 1:j
            rmqbetter(val, t, best, ismax) && (best = t)
        end

        lo = bi + 1; hi = bj - 1; k = 0

        while (2 << k) <= hi - lo + 1
            k += 1
        end

        p = r.tab[k * r.nb + lo]; q = r.tab[k * r.nb + hi - (1 << k) + 1]
        rmqbetter(val, p, best, ismax) && (best = p)
        rmqbetter(val, q, best, ismax) && (best = q)
    end

    return best
end
