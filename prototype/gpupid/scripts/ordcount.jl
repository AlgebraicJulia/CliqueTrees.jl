# Object counts of the decisions at the optimum (YES) and just below (NO), for several vertex orders.
using CliqueTrees, Graphs, Serialization, Printf
include(joinpath(@__DIR__, "..", "src", "gpupid.jl"))
for arg in ARGS
    name, ks = split(arg, '='); K = parse(Float64, ks)
    a = deserialize(joinpath(@__DIR__, "..", "atoms", name * ".jls"))
    for o in (:id, :bfs, :rbfs)
        perm = vorder(a.adj, o); pg = relabel(a.adj, a.weights, perm); gg = GPUGraph(pg)
        for (tag, k) in (("NO", K - 1e-6), ("YES", K))
            r = try gpu_decide(gg, k; deadline = time() + 300) catch e; println("$name $o $tag ERROR"); continue end
            s = r.stats
            @printf("%-40s %-5s %-3s yes=%d I=%d O=%d P=%d pairs=%.3g rounds=%d%s\n", name, o, tag, r.yes, s[:I], s[:O], s[:P], s[:pairs], s[:rounds], get(s, :timeout, false) ? " TIMEOUT" : "")
            flush(stdout); r = nothing; GC.gc(true); CUDA.reclaim()
        end
    end
end
