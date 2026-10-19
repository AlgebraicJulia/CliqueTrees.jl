using CliqueTrees, Graphs, Serialization, Printf
const CT = CliqueTrees
include(joinpath(@__DIR__, "..", "src", "gpupid.jl"))
name = ARGS[1]; tl = parse(Float64, ARGS[2])
a = deserialize(joinpath(@__DIR__, "..", "atoms", name * ".jls"))
g = SimpleGraph(a.n); for v in 1:a.n, u in a.adj[v]; u > v && add_edge!(g, v, u); end
pg = PGraph(a.adj, a.weights)
let n = 64 * (cld(a.n, 64) - 1) + 10, h = cycle_graph(n)
    gpu_tw(PGraph([sort(neighbors(h, v)) for v in 1:n], ones(n)), 2.0)
end
gg = GPUGraph(pg); K = max(a.lb, CT.bt_bas(a.weights, g)); t00 = time()
while time() - t00 < tl
    t0 = time(); r = gpu_decide(gg, K; deadline = t00 + tl); t = time() - t0
    s = r.stats; tt = s[:t]
    @printf("K=%.4f yes=%d t=%.2f rounds=%d I=%d O=%d P=%d pairs=%.3g x2=%d x3=%d | %s\n", K, r.yes, t, s[:rounds], s[:I], s[:O], s[:P], s[:pairs], s[:xcand2], s[:xcand3],
        join(["$k=$(round(v, digits=2))" for (k, v) in sort(collect(tt), by = x -> -x[2])], " "))
    flush(stdout)
    (r.yes || !isfinite(r.nextK) || get(s, :timeout, false)) && break
    global K = r.nextK
end
