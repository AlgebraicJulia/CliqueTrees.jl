include(joinpath(@__DIR__, "Rounds.jl")); using .Rounds
include(joinpath(@__DIR__, "GRounds.jl")); using .GRounds
include(joinpath(@__DIR__, "evalutil.jl")); include(joinpath(@__DIR__, "gens.jl"))
include(joinpath(@__DIR__, "..", "tncorpus", "load.jl")); using .TNCorpus
using CUDA
g0 = lat2(20); w0 = randw(nv(g0)); GRounds.grounds(w0, g0, RoundsAlg(; score=:mf, relax=0.5, tailsel=32, tailmax=10^9))
for (nm, ld) in (("qaoaamp_N4000_p5_s0", () -> (i = TNCorpus.load("qaoaamp_N4000_p5_s0"); (i.graph, i.weights))),
                 ("lat3d_32", () -> (i = TNCorpus.load("lat3d_32"); (i.graph, i.weights))),
                 ("lat2_512", () -> (g = lat2(512); (g, randw(nv(g))))))
    g, w = ld()
    for sc in (:md, :mf, :mf1), ts in (0, 32, 128)
        alg = RoundsAlg(; score = sc, relax = 0.5, tailsel = ts, tailmax = 10^9)
        GRounds.grounds(w, g, alg)
        tm = Dict{Symbol,Any}()
        t = @elapsed o = GRounds.grounds(w, g, alg; timing = tm)[1]
        println(rpad(nm, 22), " $sc tailsel=$ts w=$(round(quality(w, g, o)[1], digits=1)) t=$(round(t, digits=3)) gpu=$(round(tm[:gpu],digits=3)) cpu_post=$(round(tm[:cpu_post],digits=3)) rounds=$(tm[:rounds]) tail=$(tm[:tail])")
    end
end
