include(joinpath(@__DIR__, "Rounds.jl")); using .Rounds
include(joinpath(@__DIR__, "GRounds.jl")); using .GRounds
include(joinpath(@__DIR__, "evalutil.jl"))
using CUDA
rng = Xoshiro(3)
function linegraph_like(rng, nt, k)
    nidx = nt * k ÷ 2
    g = SimpleGraph(nidx)
    for t in 1:nt
        ix = unique(rand(rng, 1:nidx, k))
        for a in ix, b in ix
            a < b && add_edge!(g, a, b)
        end
    end
    g
end
for trial in 1:18
    n = rand(rng, 20:400)
    g = trial % 3 == 0 ? random_regular_graph(n - n%2, 3; rng) : trial % 3 == 1 ? erdos_renyi(n, 3.0/n; rng) : linegraph_like(rng, n ÷ 3, 4)
    w = [rand(rng, (1.0, log2(3.0), 2.0)) for _ in 1:nv(g)]
    for sc in (:md, :mf, :mf1), relax in (0.0, 0.5, Inf), comp in (false, true)
        alg = RoundsAlg(; score = sc, select = :luby, relax, compress = comp)
        o, ix = GRounds.grounds(w, g, alg; check = true, poolfactor = 1.2)
        @assert isperm(o)
        o2, _ = rounds(w, g, alg)
        q1 = quality(w, g, o)[1]; q2 = quality(w, g, o2)[1]
        trial <= 2 && println((n, sc, relax, comp, q1, q2))
    end
end
println("ok")
