include(joinpath(@__DIR__, "Rounds.jl")); using .Rounds
include(joinpath(@__DIR__, "GRounds.jl")); using .GRounds
include(joinpath(@__DIR__, "evalutil.jl")); include(joinpath(@__DIR__, "gens.jl"))
using CUDA
for L in (12, 30)
    g = lat2(L); w = randw(nv(g))
    for (tail, tailsel, tailmax) in ((0,0,0), (100,0,0), (0, 4, 500))
        o, _ = GRounds.grounds(w, g, RoundsAlg(; score=:mf, relax=0.5, tail, tailsel, tailmax); check = true)
        @assert isperm(o)
        println((L, tail, tailsel, quality(w, g, o)[1]))
    end
end
for L in (32, 64, 128)
    g = lat2(L); w = randw(nv(g))
    for rep in 1:3
        tm = Dict{Symbol,Any}()
        t = @elapsed GRounds.grounds(w, g, RoundsAlg(; score=:mf, relax=0.5); timing = tm)
        rep == 3 && println("L=$L t=$(round(t,digits=4)) rounds=$(tm[:rounds]) per-round=$(round(1e6*tm[:gpu]/tm[:rounds]))us sel=$(round(tm[:sel],digits=4)) elim=$(round(tm[:elim],digits=4))")
    end
end
