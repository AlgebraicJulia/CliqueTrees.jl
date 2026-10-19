include(joinpath(@__DIR__, "Rounds.jl")); using .Rounds
include(joinpath(@__DIR__, "evalutil.jl"))
include(joinpath(@__DIR__, "..", "tncorpus", "load.jl")); using .TNCorpus
using Metis
src = get(ENV, "SRC", "einsum"); nmin = parse(Int, get(ENV, "NMIN", "2000"))
rows = src == "einsum" ? TNCorpus.select(; source = "einsum", paper = true) : TNCorpus.select(; source = "ladder", nmax = parse(Int, get(ENV, "NMAX", "40000")))
rows = sort(filter(r -> r.n >= nmin, rows); by = r -> r.n)
out = open(ENV["OUT"], "w")
println(out, "name\tfamily\tn\talg\twidth\tlcost\ttime")
R(sc) = RoundsAlg(; score = sc, select = :luby, relax = 0.5)
nd(a, seed) = ND(a, METISND(; seed); width = 16, level = 12, scale = 100)
algs = ["bw" => BestWidth(MF(), MF(; strategy = 1), AMF(), MMD()),
        "MF1" => MF(; strategy = 1),
        "R-mf1" => R(:mf1),
        "R-bw" => BestWidth(R(:mf), R(:mf1), R(:md)),
        "R-mf1+AMF" => BestWidth(R(:mf1), AMF())]
for r in rows
    inst = TNCorpus.load(r); g = inst.graph; w = inst.weights
    for (an, a) in algs, (kind, alg) in (("flat", a), ("nd", nd(a, 1)))
        t = @elapsed o = try permutation(w, g, alg)[1] catch e; @warn e; nothing end
        wd, lc = o === nothing ? (NaN, NaN) : quality(w, g, o)
        println(out, join((r.name, r.family, r.n, "$kind-$an", wd, lc, t), '\t')); flush(out)
    end
end
close(out)
