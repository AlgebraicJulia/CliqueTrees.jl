using CliqueTrees, Graphs, Serialization, Printf
const CT = CliqueTrees
include(joinpath(@__DIR__, "..", "src", "gpupid.jl"))
name = ARGS[1]; tl = parse(Float64, ARGS[2]); ords = split(ARGS[3], ',')
a = deserialize(joinpath(@__DIR__, "..", "atoms", name * ".jls"))
g = SimpleGraph(a.n); for v in 1:a.n, u in a.adj[v]; u > v && add_edge!(g, v, u); end
function bfsorder(g)
    # BFS from a pseudo-peripheral vertex (two sweeps), neighbours by increasing degree
    far(s) = (d = gdistances(g, s); argmax(d))
    s = far(far(1)); order = Int[s]; seen = falses(nv(g)); seen[s] = true; h = 1
    while h <= length(order)
        u = order[h]; h += 1
        for x in sort(neighbors(g, u); by = x -> degree(g, x))
            seen[x] || (seen[x] = true; push!(order, x))
        end
    end
    return order
end
function relabel(order)   # new vertex i is old vertex order[i]
    inv = invperm(order)
    adj = [sort([inv[u] for u in a.adj[order[i]]]) for i in 1:a.n]
    return PGraph(adj, a.weights[order])
end
let n = 64 * (cld(a.n, 64) - 1) + 10, h = cycle_graph(n)
    gpu_tw(PGraph([sort(neighbors(h, v)) for v in 1:n], ones(n)), 2.0)
end
for o in ords
    order = o == "id" ? collect(1:a.n) : o == "bfs" ? bfsorder(g) : o == "rbfs" ? reverse(bfsorder(g)) :
        o == "degdesc" ? sortperm(degree(g); rev = true) : o == "degasc" ? sortperm(degree(g)) :
        o == "mf" ? permutation(a.weights, g, MF())[1] : o == "rmf" ? reverse(permutation(a.weights, g, MF())[1]) : error(o)
    pg = relabel(order)
    t0 = time(); r = gpu_tw(pg, max(a.lb, CT.bt_bas(a.weights, g)); tlimit = tl); t = time() - t0
    s = r.stats[end][3]
    @printf("%-8s status=%s width=%s t=%.2f lastK: I=%d O=%d P=%d pairs=%.3g join=%.2f\n", o, r.status, string(r.width), t, s[:I], s[:O], s[:P], s[:pairs], get(s[:t], :join, 0.0))
    flush(stdout)
end
