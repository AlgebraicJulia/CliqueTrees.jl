using CliqueTrees, Graphs, Random
const CT = CliqueTrees
include(joinpath(@__DIR__, "..", "src", "pidref.jl"))

# brute force weighted treewidth: TW(S) = min_v max(TW(S - v), w(v) + w(Q(S - v, v)))
function brute_tw(adjl, w)
    n = length(adjl); N = 1 << n
    adjm = [sum(1 << (u - 1) for u in adjl[v]; init = 0) for v in 1:n]
    wset(s) = sum(w[v] for v in 1:n if (s >> (v - 1)) & 1 == 1; init = 0.0)
    function Q(S, v)   # vertices outside S ∪ {v} reachable from v through S
        seen = 1 << (v - 1); front = seen; out = 0
        while front != 0
            nb = 0; f = front
            while f != 0
                u = trailing_zeros(f) + 1; f &= f - 1; nb |= adjm[u]
            end
            nb &= ~seen; seen |= nb
            out |= nb & ~S; front = nb & S
        end
        return out
    end
    TW = fill(Inf, N); TW[1] = -Inf
    for S in 1:N-1
        best = Inf; s = S
        while s != 0
            v = trailing_zeros(s) + 1; s &= s - 1
            S2 = S & ~(1 << (v - 1))
            best = min(best, max(TW[S2 + 1], w[v] + wset(Q(S2, v))))
        end
        TW[S + 1] = best
    end
    return TW[N]
end

function randconn(rng, n, p)
    while true
        g = erdos_renyi(n, p; rng)
        is_connected(g) && return g
    end
end

rng = Xoshiro(1); nfail = 0; ntest = 0
for trial in 1:600
    n = rand(rng, 4:14); p = rand(rng, [0.2, 0.3, 0.45, 0.6])
    g = randconn(rng, n, p)
    adjl = [sort(neighbors(g, v)) for v in 1:n]
    w = trial % 2 == 0 ? ones(n) : Float64.(rand(rng, [1.0, 1.0, log2(3), 2.0], n))
    pg = PGraph(adjl, w)
    lo = maximum(w)
    r = pid_tw(pg, lo)
    tw = treewidth(w, g, r.order)
    bf = brute_tw(adjl, w)
    _, _, kct, st = CT.septw(w, g, lo, Inf)
    global ntest += 1
    if abs(tw - bf) > 1e-9 || abs(r.width - tw) > 1e-9 || abs(kct - bf) > 1e-9
        global nfail += 1
        println("FAIL n=$n p=$p pid=$(r.width) order=$tw brute=$bf septw=$kct w=$w")
    end
end
println("tests=$ntest failures=$nfail")
