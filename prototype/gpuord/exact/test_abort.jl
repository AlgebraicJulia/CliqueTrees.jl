using CUDA, CliqueTrees, Graphs, Serialization, Printf
include(joinpath(@__DIR__, "SeptwGPU.jl"))
include(joinpath(@__DIR__, "exact_common.jl"))
rs = loadrounds(@__DIR__; nmax = 64)
cpu_round(rs[1], 1); gpu_round(rs[1].w, rs[1].g, rs[1].k)
cand = sort(collect(eachindex(rs)); by = j -> -nv(rs[j].g))[1:200]
tg = Dict(j => @elapsed(gpu_round(rs[j].w, rs[j].g, rs[j].k)) for j in cand)
i = argmax(tg); r = rs[i]
tc = @elapsed ref = cpu_round(r, 1)
full = gpu_round(r.w, r.g, r.k)
@printf("heaviest GPU round of 200: n=%d, GPU %.3f s, CPU %.3f s, equal: %s\n", nv(r.g), tg[i], tc, full[3] == ref[2] && full[1] == ref[1])
for frac in (0.1, 0.3, 0.6, 2.0)
    budget = frac * tg[i]
    t = @elapsed x = gpu_round(r.w, r.g, r.k, time() + budget)
    @printf("budget %.3f s: returned after %.3f s, %s\n", budget, t, x === :timeout ? "timeout" : "complete (equal: $(x[3] == ref[2] && x[1] == ref[1]))")
end
