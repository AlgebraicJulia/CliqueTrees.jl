# usage: julia cpu_one.jl NAME METH   (METH in AMD AMF MMD MF MF1 METIS-ND)
using Metis, AMD
include(joinpath(@__DIR__, "fcommon.jl"))
name, meth = ARGS[1], ARGS[2]
alg(m) = m == "AMD" ? CliqueTrees.AMD() : m == "AMF" ? AMF() : m == "MMD" ? MMD() : m == "MF" ? MF() :
         m == "MF1" ? MF(strategy = 1) : m == "METIS-ND" ? CliqueTrees.METIS() : error("bad method")
let bg = BipartiteGraph(lat2(12)); permutation(bg, alg(meth)); end
g = loadg(name); n = nv(g); bg = BipartiteGraph(g)
t = @elapsed(o = permutation(bg, alg(meth))[1])
println(name, " ", meth, " ordered in ", round(t, digits = 2), " s"); flush(stdout)
saveperm(name, meth, o)
row = evalrow(name, g, meth, o, t); println(row)
appendline(joinpath(RES, "results.tsv"), row)
