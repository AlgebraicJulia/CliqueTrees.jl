using CliqueTrees, CUDA, Graphs, Random
using CliqueTrees: GPUSeptw, CPUSeptw, septwbackend!, septw, BT_FOUND
ext = Base.get_extension(CliqueTrees, :CUDAExt)
rng = MersenneTwister(1)
for (n, p) in ((20, 0.3), (40, 0.15), (80, 0.06))
    g = BipartiteGraph(erdos_renyi(n, p; rng)); w = rand(rng, 1:3, n)
    n == 80 || continue
    for engine in (:pid, :septw)
        ext.resetstats!()
        septwbackend!(GPUSeptw(; engine, budget = 0.0))
        r = septw(w, g, 0, typemax(Int), Inf, 1)
        println(engine, " ", r[3:4], " ", ext.STATS)
    end
    esc = ext.Esc(GPUSeptw(; engine = :pid), 0.0, true, false, 0.0, 0.0, nothing)
    for k in 30:34
        r = ext.pidround!(esc, w, g, k, Inf)
        println(k, " ", r isa Tuple ? r[2:3] : r)
    end
end
