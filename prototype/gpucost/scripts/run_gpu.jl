# Method (b): our pipeline's orderings + GPU mass scoring by C + GPU batched subtree reconfiguration.
#   julia --project=env scripts/run_gpu.jl <list> <orddir> <outdir> <B> <hbt tag> [fmass] [K1] [K2]
# Seeds: <name>__<hbt tag>.ord plus the cheap greedy (MF, MF1, AMF, MMD) and nd1 (8 seeds) orderings.
# The GPU stage gets B minus the seeds' own (CPU) seconds, at least 0.25 s.
# Writes <outdir>/<name>__gpuB<B>.tree (meta log2C, W) and appends a row to <outdir>/gpu.tsv.
include(joinpath(@__DIR__, "..", "src", "Host.jl")); using .Host
include(joinpath(@__DIR__, "..", "src", "GPUTree.jl")); using .GPUTree
include(joinpath(@__DIR__, "..", "src", "GPUReconf.jl")); using .GPUReconf
include(joinpath(@__DIR__, "..", "src", "Reconf.jl")); using .Reconf
include(joinpath(@__DIR__, "..", "src", "Search.jl")); using .Search
using CUDA, Random

names = readlines(ARGS[1]); orddir = ARGS[2]; out = ARGS[3]; B = parse(Float64, ARGS[4]); htag = ARGS[5]
fmass = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 0.4
K1 = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 10
K2 = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 12
mkpath(out)
readord(f) = (ls = readlines(f); (parse(Float64, split(ls[1], '\t')[3]), parse.(Int, split(ls[2]))))

# warm-up (compilation) on a small network
let net = readnet("rnd_mixed_03"), f = joinpath(orddir, "rnd_mixed_03__MF.ord")
    for _ in 1:3
        gpu_pipeline(net, [readord(f)[2]]; budget = 2.0)
    end
end

for name in names
    fh = joinpath(orddir, "$(name)__$(htag).ord")
    isfile(fh) || (println("skip ", name); continue)
    tags = vcat([htag], ["MF", "MF1", "AMF", "MMD"], ["nd1s$k" for k in 1:8])
    seeds = Vector{Int}[]; tseeds = 0.0
    for t in tags
        f = joinpath(orddir, "$(name)__$(t).ord"); isfile(f) || continue
        s, o = readord(f); tseeds += s; push!(seeds, o)
    end
    net = readnet(name)
    bud = max(0.25, B - tseeds)
    GC.gc()
    best, st = gpu_pipeline(net, seeds; budget = bud, fmass, K1, K2, log = stdout)
    f = joinpath(out, "$(name)__gpuB$(Int(B)).tree")
    Host.write_tree(f, best; meta = Dict("log2C" => best.log2C, "W" => best.W))
    open(joinpath(out, "gpu.tsv"), "a") do io
        println(io, join((name, net.family, B, round(tseeds; digits = 3), round(st.total; digits = 3), round(tseeds + st.total; digits = 3),
            st.ncand, st.ngen, round(st.ncand / st.tmass; digits = 0), st.chunks, round(st.tgpu; digits = 3),
            round(st.seedraw * log10(2); digits = 4), round(st.rawbest * log10(2); digits = 4), round(st.mid * log10(2); digits = 4),
            round(best.log2C * log10(2); digits = 4), round(best.W; digits = 3)), '\t'))
    end
    println(rpad(name, 48), " B=$B  log10C ", round(best.log2C * log10(2); digits = 4), "  W ", round(best.W; digits = 2), "  wall ", round(tseeds + st.total; digits = 2), " s")
    flush(stdout)
end
