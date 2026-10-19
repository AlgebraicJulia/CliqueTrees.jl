# loss from batching: rounds vs the same greedy run one vertex at a time
include(joinpath(@__DIR__, "Rounds.jl")); using .Rounds
include(joinpath(@__DIR__, "evalutil.jl"))
include(joinpath(@__DIR__, "..", "tncorpus", "load.jl")); using .TNCorpus
rows = sort(filter(r -> r.n <= parse(Int, get(ENV, "NMAX", "20000")), TNCorpus.select(; source = "einsum", paper = true)); by = r -> r.n)
out = open(ENV["OUT"], "w")
println(out, "name\tfamily\tn\tscore\tvariant\twidth\tlcost\trounds")
for r in rows
    inst = TNCorpus.load(r); g = inst.graph; w = inst.weights
    nv(g) < 2 && continue
    for sc in (:md, :mf, :mf1)
        for (vn, alg) in (("seq", RoundsAlg(; score = sc, select = :greedy, relax = 0.0, lim = 1)),
                          ("luby0", RoundsAlg(; score = sc, select = :luby, relax = 0.0)),
                          ("luby0.5", RoundsAlg(; score = sc, select = :luby, relax = 0.5)),
                          ("luby2", RoundsAlg(; score = sc, select = :luby, relax = 2.0)))
            st = []
            o = rounds(w, g, alg; stats = st)[1]
            wd, lc = quality(w, g, o)
            println(out, join((r.name, r.family, r.n, sc, vn, wd, lc, length(st)), '\t')); flush(out)
        end
    end
end
close(out)
