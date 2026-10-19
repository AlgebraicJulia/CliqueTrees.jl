# Phase breakdown of one GPU exact round (the heaviest CPU rounds), to see where single-call latency goes.
# usage: julia latency.jl datadir cpuref.jls [ntop]
using CUDA
include(joinpath(@__DIR__, "exact_common.jl"))
include(joinpath(@__DIR__, "SeptwGPU.jl"))
rs = loadrounds(ARGS[1]); C = deserialize(ARGS[2]); ntop = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 10
println(CUDA.name(CUDA.device()), " | CPU ref ", C.cpu)
gpu_round(rs[1].w, rs[1].g, rs[1].k)
top = sortperm(C.t1; rev = true)[1:ntop]
for i in top
    r = rs[i]; n = nv(r.g); W = gbucket(n)
    TE = W == 1 ? UInt64 : EnumW.Bits{W}; TD = W == 1 ? UInt64 : DPW.Bits{W}
    gpu_round(r.w, r.g, r.k)   # warm buffers for this size
    t0 = time()
    bt = EnumW.prepare(TE, [r.g], [collect(Float64, r.w)], [r.k]); t1 = time()
    lists, ovf, tk, info = EnumW.gpu_sepenum(bt; split = 3); t2 = time()
    sep = EnumW.sepgraph(lists[1], n); t3 = time()
    res, d = DPW.gpu_sepdp(TD, [collect(Float64, r.w)], [r.g], [sep], [r.k]); t4 = time()
    ok = res[1][3] == C.refs[i][2] && res[1][1] == C.refs[i][1]
    @printf("n=%2d seps %7d | CPU 1t %.3f s | GPU round %.3f s = prep %.3f + sepenum %.3f (kernels %.3f: start %.3f tasks %d %.3f; sort/glue %.3f) + sepgraph %.3f + sepdp %.3f (CPU prep %.3f, upload %.3f, kernels %.3f, order %.3f) | ok %s\n",
        n, length(lists[1]), C.t1[i], t4 - t0, t1 - t0, t2 - t1, tk, info.t1, info.ntask, info.t2, (t2 - t1) - tk, t3 - t2, t4 - t3, d.tprep, d.tupload, d.tgpu, d.tfinish, ok)
end
