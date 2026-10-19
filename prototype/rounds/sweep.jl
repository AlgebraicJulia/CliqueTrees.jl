# knob sweep: serial MF/MF1/AMF/MMD/bw vs CPU rounds (luby/greedy) across relax
include(joinpath(@__DIR__, "Rounds.jl")); using .Rounds
include(joinpath(@__DIR__, "evalutil.jl"))
include(joinpath(@__DIR__, "..", "tncorpus", "load.jl")); using .TNCorpus
src = get(ENV, "SRC", "einsum")
rows = src == "einsum" ? TNCorpus.select(; source = "einsum", paper = true) : TNCorpus.select(; source = src, nmax = parse(Int, get(ENV, "NMAX", "20000")))
sort!(rows; by = r -> r.n)
out = open(ENV["OUT"], "w")
println(out, "name\tfamily\tn\talg\tscore\trelax\twidth\tlcost\ttime\trounds")
relaxes = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, Inf)
serials = ["MF" => MF(), "MF1" => MF(; strategy = 1), "AMF" => AMF(), "MMD" => MMD()]
g0 = TNCorpus.load(rows[1])
for (sc) in (:md, :mf, :mf1), sel in (:luby, :greedy); rounds(g0.weights, g0.graph, RoundsAlg(; score = sc, select = sel)); end
for (_, a) in serials; permutation(g0.weights, g0.graph, a); end
for r in rows
    inst = TNCorpus.load(r); g = inst.graph; w = inst.weights
    nv(g) < 2 && continue
    function rec(alg, sc, relax, f)
        t = @elapsed o = try f() catch e; nothing end
        if o === nothing
            println(out, join((r.name, r.family, r.n, alg, sc, relax, NaN, NaN, t, 0), '\t'))
        else
            o, R = o
            wd, lc = quality(w, g, o)
            println(out, join((r.name, r.family, r.n, alg, sc, relax, wd, lc, t, R), '\t'))
        end
        flush(out)
    end
    for (an, a) in serials
        rec(an, "-", 0, () -> (permutation(w, g, a)[1], 0))
    end
    for sc in (:md, :mf, :mf1), relax in relaxes
        rec("luby", sc, relax, () -> begin st = []; o = rounds(w, g, RoundsAlg(; score = sc, select = :luby, relax); stats = st)[1]; (o, length(st)) end)
    end
    for sc in (:md, :mf, :mf1), relax in (0.0, 0.5, 2.0)
        rec("greedy", sc, relax, () -> begin st = []; o = rounds(w, g, RoundsAlg(; score = sc, select = :greedy, relax); stats = st)[1]; (o, length(st)) end)
    end
end
close(out)
