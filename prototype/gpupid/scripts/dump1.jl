using CliqueTrees, Graphs, Serialization
include(joinpath(@__DIR__, "..", "src", "gpupid.jl"))
name = ARGS[1]; K = parse(Float64, ARGS[2])
a = deserialize(joinpath(@__DIR__, "..", "atoms", name * ".jls"))
pg = PGraph(a.adj, a.weights)
r = gpu_decide(GPUGraph(pg), K; dump = true)
serialize(joinpath(@__DIR__, "..", "dump_$(name)_$(K).jls"), (; a, K, d = r.stats[:dump]))
println("dumped I=$(length(r.stats[:dump].I)) O=$(length(r.stats[:dump].O))")
