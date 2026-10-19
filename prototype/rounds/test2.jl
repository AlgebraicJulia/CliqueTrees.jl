include(joinpath(@__DIR__, "Rounds.jl")); using .Rounds
include(joinpath(@__DIR__, "evalutil.jl"))
rng = Xoshiro(3)
function linegraph_like(rng, nt, k)
    # random hypergraph: nt tensors with k random indices each -> line graph cliques
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
for trial in 1:24
    n = rand(rng, 20:300)
    g = trial % 3 == 0 ? random_regular_graph(n - n%2, 3; rng) : trial % 3 == 1 ? erdos_renyi(n, 3.0/n; rng) : linegraph_like(rng, n ÷ 3, 4)
    w = [rand(rng, (1.0, log2(3.0), 2.0)) for _ in 1:nv(g)]
    for sc in (:md, :mf, :mf1), sel in (:greedy, :luby), relax in (0.0, 0.5, Inf), comp in (false, true)
        alg = RoundsAlg(; score = sc, select = sel, relax, compress = comp, tailsel = 1, tailmax = 30)
        o, ix = rounds(w, g, alg; check = true)
        @assert isperm(o)
        o2, _ = rounds(w, g, RoundsAlg(; score = sc, select = sel, relax, compress = comp, ntasks = 1, tailsel = 1, tailmax = 30))
        @assert o == o2 "thread dependence"
    end
end
println("ok")
