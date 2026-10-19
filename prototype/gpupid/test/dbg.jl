using CliqueTrees, Graphs, Random
include(joinpath(@__DIR__, "..", "src", "gpupid.jl"))
g = cycle_graph(6); add_edge!(g, 1, 4)
adjl = [sort(neighbors(g, v)) for v in 1:nv(g)]; w = ones(nv(g))
pg = PGraph(adjl, w)
for K in (2.0, 3.0)
    a = pid_decide(pg, K; verbose = true); println("ref K=$K yes=$(a.yes) ", a.stats)
    println("calling gpu"); flush(stdout)
    b = gpu_decide(GPUGraph(pg), K; verbose = true, deadline = time() + 20); println("gpu K=$K yes=$(b.yes) next=$(b.nextK) ", b.stats)
end
