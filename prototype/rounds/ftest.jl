include(joinpath(@__DIR__, "Rounds.jl")); using .Rounds
include(joinpath(@__DIR__, "FRounds.jl")); using .FRounds
include(joinpath(@__DIR__, "evalutil.jl")); include(joinpath(@__DIR__, "gens.jl"))
rng = Xoshiro(3)
function linegraph_like(rng, nt, k)
    nidx = nt * k ÷ 2; g = SimpleGraph(nidx)
    for t in 1:nt
        ix = unique(rand(rng, 1:nidx, k))
        for a in ix, b in ix; a < b && add_edge!(g, a, b); end
    end
    g
end
for trial in 1:18
    n = rand(rng, 20:300)
    g = trial % 3 == 0 ? random_regular_graph(n - n%2, 3; rng) : trial % 3 == 1 ? erdos_renyi(n, 3.0/n; rng) : linegraph_like(rng, n ÷ 3, 4)
    w = [rand(rng, (1.0, log2(3.0), 2.0)) for _ in 1:nv(g)]
    for sc in (:md, :mf, :mf1), relax in (0.0, 0.5, Inf), comp in (false, true)
        alg = RoundsAlg(; score = sc, select = :luby, relax, compress = comp, tailsel = 2, tailmax = 40)
        o, _ = FRounds.frounds(w, g, alg; check = true, poolfactor = 1.1)
        @assert isperm(o)
        o2, _ = FRounds.frounds(w, g, RoundsAlg(; score = sc, select = :luby, relax, compress = comp, ntasks = 1, tailsel = 2, tailmax = 40); poolfactor = 1.1)
        @assert o == o2 "thread dependence"
    end
end
println("ok")
g = lat2(256); w = randw(nv(g))
for sc in (:md, :mf)
    FRounds.frounds(w, g, RoundsAlg(; score = sc, relax = 0.5))
    tm = Dict{Symbol,Any}()
    t = @elapsed o = FRounds.frounds(w, g, RoundsAlg(; score = sc, relax = 0.5); timing = tm)[1]
    t2 = @elapsed o2 = rounds(w, g, RoundsAlg(; score = sc, relax = 0.5))[1]
    println("$sc flat t=$(round(t,digits=3)) w=$(quality(w,g,o)[1])  old t=$(round(t2,digits=3)) w=$(quality(w,g,o2)[1]) ", tm)
end
