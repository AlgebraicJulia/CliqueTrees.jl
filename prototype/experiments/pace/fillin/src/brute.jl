# Exact minimum fill-in by DP over vertex subsets (elimination orderings), for testing (n <= ~16).
# Eliminating v after the set S adds the edges from v to Q(S, v) = vertices outside S ∪ {v}
# reachable from v through S; the filled graph has sum |Q| edges, so fill = min sum |Q| - |E|.
function brute_minfill(adjl::Vector{Vector{Int}})
    n = length(adjl); n <= 3 && return 0
    nb = [reduce(|, (1 << (u - 1) for u in adjl[v]); init = 0) for v in 1:n]
    function q(S, v)
        seen = 1 << (v - 1); front = seen; out = 0
        while front != 0
            x = trailing_zeros(front) + 1; front &= front - 1
            nx = nb[x] & ~seen
            seen |= nx
            out |= nx & ~S
            front |= nx & S
        end
        return count_ones(out)
    end
    f = fill(typemax(Int), 1 << n); f[1] = 0
    for S in 0:((1 << n) - 1)
        f[S + 1] == typemax(Int) && continue
        for v in 1:n
            (S >> (v - 1)) & 1 == 1 && continue
            T = S | (1 << (v - 1))
            c = f[S + 1] + q(S, v)
            c < f[T + 1] && (f[T + 1] = c)
        end
    end
    return f[end] - sum(length, adjl) ÷ 2
end
