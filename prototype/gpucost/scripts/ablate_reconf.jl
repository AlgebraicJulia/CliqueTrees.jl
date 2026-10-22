# Ablation: GPU batched reconfiguration alone, on the same seed ordering as method (a) (hbt 1 s),
# vs cotengra's subtree_reconfigure (ftn.refine, in results/base/*__ctgref1.tree).
#   julia --project=env scripts/ablate_reconf.jl <list> <orddir> <outdir>
# Writes <outdir>/<name>__gpurK<K>.tree for K = 8 (cotengra's subtree size), 10, 12 (each from the raw
# tree, run to convergence: 3 rounds without improvement) and a TSV row (name K seconds chunks log10C).
include(joinpath(@__DIR__, "..", "src", "Host.jl")); using .Host
include(joinpath(@__DIR__, "..", "src", "GPUReconf.jl")); using .GPUReconf
include(joinpath(@__DIR__, "..", "src", "Reconf.jl")); using .Reconf
using CUDA, Random
names = readlines(ARGS[1]); orddir = ARGS[2]; out = ARGS[3]; mkpath(out)
let net = readnet("rnd_mixed_03")
    o = open_last(net, parse.(Int, split(readlines(joinpath(orddir, "rnd_mixed_03__MF.ord"))[2])))
    for K in (8, 10, 12); reconf!([cost!(bucket_tree(net, o), net)], net; K); end
end
for name in names
    net = readnet(name)
    o = open_last(net, parse.(Int, split(readlines(joinpath(orddir, "$(name)__hbtT1.0s1.ord"))[2])))
    for K in (8, 10, 12)
        t0 = time()
        tr = cost!(bucket_tree(net, o), net); raw = tr.log2C
        st = reconf!([tr], net; K, rounds = 200, patience = 3, rng = Xoshiro(1))
        dt = time() - t0
        Host.write_tree(joinpath(out, "$(name)__gpurK$(K).tree"), tr; meta = Dict("log2C" => tr.log2C, "W" => tr.W, "t" => dt))
        open(joinpath(out, "ablate.tsv"), "a") do io
            println(io, join((name, K, round(dt; digits = 3), st.chunks, round(st.tgpu; digits = 3), round(raw * log10(2); digits = 4), round(tr.log2C * log10(2); digits = 4)), '\t'))
        end
        println(rpad(name, 48), " K=$K  ", round(dt; digits = 2), " s  ", st.chunks, " chunks (", round(st.chunks / dt; digits = 0), "/s)  log10C ", round(raw * log10(2); digits = 3), " -> ", round(tr.log2C * log10(2); digits = 4))
    end
end
