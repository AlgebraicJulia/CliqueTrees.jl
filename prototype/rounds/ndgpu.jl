# ND with the GPU rounds as the greedy algorithm on large graphs
const DIR = @__DIR__
include(joinpath(DIR, "Rounds.jl")); using .Rounds
using CUDA
include(joinpath(DIR, "GRounds.jl")); using .GRounds
include(joinpath(DIR, "evalutil.jl")); include(joinpath(DIR, "gens.jl"))
include(get(ENV, "CORPUS", joinpath(DIR, "..", "tncorpus")) * "/load.jl"); using .TNCorpus
using Metis
out = open(ENV["OUT"], "a")
nd(a) = ND(a, METISND(; seed = 1); width = 16, level = 12, scale = 100)
R(sc) = RoundsAlg(; score = sc, select = :luby, relax = 0.5)
H(sc, small) = GPUHybrid(RoundsAlg(; score = sc, select = :luby, relax = 0.5, tailsel = 128, tailmax = 10^9, tailalg = AMF()), small; nmin = parse(Int, get(ENV, "NMIN", "20000")))
algs = ["bw" => BestWidth(MF(), MF(; strategy = 1), AMF(), MMD()),
        "MF1" => MF(; strategy = 1),
        "AMF" => AMF(),
        "H-mf1" => H(:mf1, MF(; strategy = 1)),
        "H-bw" => BestWidth(H(:mf, MF()), H(:mf1, MF(; strategy = 1)), AMF(), H(:md, MMD()))]
insts = Any[]
for L in parse.(Int, filter(!isempty, split(get(ENV, "LAT2", "128,256,512"), ",")))
    push!(insts, ("lat2_$L", () -> (g = lat2(L); (g, randw(nv(g))))))
end
for nm in split(get(ENV, "NAMES", ""), ",")
    isempty(nm) && continue
    push!(insts, (nm, () -> (i = TNCorpus.load(String(nm)); (i.graph, i.weights))))
end
let g = lat2(40), w = randw(nv(g))
    GRounds.grounds(w, g, RoundsAlg(; score = :mf1, relax = 0.5, tailsel = 128, tailmax = 10^9, tailalg = AMF()))
    GRounds.grounds(w, g, RoundsAlg(; score = :mf, relax = 0.5, tailsel = 128, tailmax = 10^9, tailalg = AMF()))
    GRounds.grounds(w, g, RoundsAlg(; score = :md, relax = 0.5, tailsel = 128, tailmax = 10^9, tailalg = AMF()))
    for (_, a) in algs; permutation(w, g, nd(a)); permutation(w, g, a); end
end
for (name, ld) in insts
    g, w = ld()
    for (an, a) in algs, (kind, alg) in (("flat", a), ("nd", nd(a)))
        t = @elapsed o = permutation(w, g, alg)[1]
        wd, lc = quality(w, g, o)
        println(out, join((gethostname(), Threads.nthreads(), name, nv(g), "$kind-$an", wd, lc, t), '\t')); flush(out)
        println(rpad(name, 22), rpad("$kind-$an", 12), " w=", round(wd, digits = 1), " t=", round(t, digits = 3)); flush(stdout)
    end
end
