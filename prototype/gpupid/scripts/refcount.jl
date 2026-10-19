using CliqueTrees, Graphs, Serialization
const CT = CliqueTrees
include(joinpath(@__DIR__, "..", "src", "pidref.jl"))
function loadatom(path)
    a = deserialize(path)
    g = SimpleGraph(a.n); for v in 1:a.n, u in a.adj[v]; u > v && add_edge!(g, v, u); end
    return a, g
end
for name in ARGS
    a, g = loadatom(joinpath(@__DIR__, "..", "atoms", name * ".jls"))
    pg = PGraph(a.adj, a.weights)
    t0 = time(); res = CT.septw(a.weights, g, a.lb, Inf; mt = 120, nt = 1); t = time() - t0; kct = res[3]; st = res[4]
    println("$name n=$(a.n) lb=$(a.lb) ub=$(a.ub) septw k=$kct status=$st t=$(round(t, digits=2))")
    st == CT.BT_FOUND || continue
    # decision at the optimum and just below
    for K in (kct, kct - 1e-6)
        t = @elapsed r = pid_decide(pg, K; verbose = true)
        println("  K=$K yes=$(r.yes) next=$(r.nextK) t=$(round(t, digits=2)) ", r.stats)
    end
    t = @elapsed r = pid_tw(pg, a.lb)
    println("  pid_tw width=$(r.width) check=$(treewidth(a.weights, g, r.order)) runs=$(r.runs) t=$(round(t, digits=2))")
end
