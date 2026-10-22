# Seed orderings from CliqueTrees for one network.
#   julia -t 1 --project=env scripts/mkorders.jl <name | list file> <outdir> <spec,spec,...>
# spec: greedy (MF, MF1, AMF, MMD) | nd (8 nd1 seeds, each written separately) | bw | nd8
#       | hbt:<T>:<seed> | hbtnd:<T>:<seed>
# Writes <outdir>/<name>__<tag>.ord: line 1 "name tag seconds bagwidth", line 2 the ordering.
include(joinpath(@__DIR__, "common_pipeline.jl"))

names = isfile(ARGS[1]) ? readlines(ARGS[1]) : [ARGS[1]]; outdir = ARGS[2]; specs = split(ARGS[3], ',')
mkpath(outdir)

function wu()
    inst = TNCorpus.load("rnd_mixed_03"); w = inst.weights; g = TNCorpus.bipartite(inst)
    permutation(w, g, bwalg()); permutation(w, g, pipeline(HBT(; time = 0.2, threads = 1)))
    permutation(w, g, pipeline(HBT(; time = 0.2, threads = 1, init = nd8())))
end
wu()

function emit(name, w, g, tag, t, o)
    f = joinpath(outdir, "$(name)__$(tag).ord")
    open(f * ".tmp", "w") do io
        println(io, join((name, tag, round(t; digits = 4), treewidth(w, g, o)), '\t'))
        println(io, join(o, ' '))
    end
    mv(f * ".tmp", f; force = true)
end
function run1(name, w, g, tag, alg)
    GC.gc(); t = @elapsed((o, _) = permutation(w, g, alg)); emit(name, w, g, tag, t, o)
end

for name in names
    inst = TNCorpus.load(name); w = inst.weights; g = TNCorpus.bipartite(inst)
    for s in specs
        if s == "greedy"
            for (tag, a) in (("MF", MF()), ("MF1", MF(; strategy = 1)), ("AMF", AMF()), ("MMD", MMD()))
                run1(name, w, g, tag, a)
            end
        elseif s == "nd"
            for k in 1:8
                run1(name, w, g, "nd1s$k", nd1(k))
            end
        elseif s == "bw"
            run1(name, w, g, "bw", bwalg())
        elseif s == "nd8"
            run1(name, w, g, "nd8", nd8())
        else
            kind, T, seed = split(s, ':'); T = parse(Float64, T); seed = parse(Int, seed)
            h = kind == "hbt" ? HBT(; time = T, threads = 1, seed) : HBT(; time = T, threads = 1, seed, init = nd8())
            run1(name, w, g, "$(kind)T$(T)s$(seed)", pipeline(h))
        end
        println(name, " ", s, " done"); flush(stdout)
    end
end
