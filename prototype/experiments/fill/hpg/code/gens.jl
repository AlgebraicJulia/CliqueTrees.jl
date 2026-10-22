using Graphs, Random
# line graph of a graph: one vertex per edge, adjacent when they share an endpoint
function linegraph(g::SimpleGraph)
    es = collect(edges(g)); m = length(es)
    id = Dict{Tuple{Int,Int},Int}()
    for (k, e) in enumerate(es)
        id[(src(e), dst(e))] = k
    end
    inc = [Int[] for _ in 1:nv(g)]
    for (k, e) in enumerate(es)
        push!(inc[src(e)], k); push!(inc[dst(e)], k)
    end
    L = SimpleGraph(m)
    for v in 1:nv(g)
        I = inc[v]
        for a in 1:length(I), b in (a + 1):length(I)
            add_edge!(L, I[a], I[b])
        end
    end
    return L
end
lat2(L) = linegraph(Graphs.grid([L, L]))
lat3(L) = linegraph(Graphs.grid([L, L, L]))
randw(n; seed = 1) = (rng = Xoshiro(seed); [rand(rng, (1.0, 1.0, log2(3.0), 2.0)) for _ in 1:n])
