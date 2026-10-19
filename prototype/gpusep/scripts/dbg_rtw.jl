using CliqueTrees, CUDA, Graphs, Random
using CliqueTrees: CPUSeptw, GPUSeptw, septwbackend!, septwwarmup!
const CT = CliqueTrees
ext = Base.get_extension(CliqueTrees, :CUDAExt)
function graphs(rng, n, p, m)
    out = BipartiteGraph{Int, Int}[]
    while length(out) < m
        g = erdos_renyi(n, p; rng); is_connected(g) && push!(out, BipartiteGraph(g))
    end
    return out
end
for e in (:septw, :pid); septwwarmup!(GPUSeptw(; engine = e); nmax = 64); end
rng = MersenneTwister(11)
gs = [graphs(rng, 24, 0.25, 6); graphs(rng, 40, 0.12, 6)]
for engine in (:septw, :pid), (i, g) in enumerate(gs)
    w = rand(rng, 1:4, nv(g)); tw = treewidth(w, g; alg = PIDBT())
    septwbackend!(GPUSeptw(; engine, budget = 0.0)); ext.resetstats!()
    t = @elapsed (perm, _, k, st, cert) = CT.rtw(w, g, 0, typemax(Int); mt = 10.0)
    septwbackend!(CPUSeptw())
    tc = @elapsed (_, _, kc, stc, _) = CT.rtw(w, g, 0, typemax(Int); mt = 10.0)
    println("$engine g$i n=$(nv(g)) tw=$tw gpu: k=$k $st $(round(t;digits=2))s cpu: k=$kc $stc $(round(tc;digits=2))s ", ext.STATS); flush(stdout)
end
