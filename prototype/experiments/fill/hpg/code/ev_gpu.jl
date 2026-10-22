# usage: julia ev_gpu.jl NAME : evaluate GPU-MF/GPU-MD permutations of NAME, times from gpu_times.tsv
include(joinpath(@__DIR__, "fcommon.jl"))
name = ARGS[1]; g = loadg(name); n = nv(g)
times = Dict{String,Float64}()
for l in eachline(joinpath(RES, "gpu_times.tsv")); f = split(l, '\t'); f[1] == name && (times[f[4]] = parse(Float64, f[5])); end
for m in ("GPU-MF", "GPU-MF1", "GPU-MD")
    haskey(times, m) && isfile(permfile(name, m)) || (println("missing $name $m"); continue)
    raw = Vector{Int32}(undef, n); read!(permfile(name, m), raw)
    row = evalrow(name, g, m, Int.(raw), times[m]); println(row)
    appendline(joinpath(RES, "results.tsv"), row)
end
