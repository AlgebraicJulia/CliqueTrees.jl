using CliqueTrees, CUDA, Graphs, Random
using CliqueTrees: GPUSeptw, CPUSeptw, septwbackend!, septw, BT_FOUND
ext = Base.get_extension(CliqueTrees, :CUDAExt)
rng = MersenneTwister(1)
for (n, p) in ((20, 0.3), (40, 0.15), (80, 0.06))
    g = BipartiteGraph(erdos_renyi(n, p; rng)); conn = is_connected(Graph(g))
    w = rand(rng, 1:3, n)
    septwbackend!(CPUSeptw()); t = @elapsed c = septw(w, g, 0, typemax(Int), Inf, 1)
    for engine in (:septw, :pid)
        septwbackend!(GPUSeptw(; engine, budget = 0.0))
        tg = @elapsed r = septw(w, g, 0, typemax(Int), Inf, 1)
        println("n=$n conn=$conn $engine cpu=$(c[3:4]) $(round(t; digits=2))s gpu=$(r[3:4]) $(round(tg; digits=2))s sameperm=$(r[1] == c[1]) tw=$(treewidth(w, g; alg = r[1]))")
    end
end
println(ext.STATS)
