using CUDA
include(joinpath(@__DIR__, "exact_common.jl"))
include(joinpath(@__DIR__, "SepEnumGPU.jl")); using .SepEnumGPU
include(joinpath(@__DIR__, "SepDPGPU.jl")); using .SepDPGPU
rs = loadrounds(@__DIR__; nmax = 64)
t1, refs, tN = deserialize("cpu_ref_t16.jls")
println(length(rs), " ", length(t1))
heavy = sortperm(t1; rev = true)[1:20]
sub = rs[heavy]
println("sepenum..."); flush(stdout)
b = prepare([r.g for r in sub], [r.w for r in sub], [r.k for r in sub])
t = @elapsed lists, ovf, tk, info = gpu_sepenum(b; split = 3)
println("sepenum done $t s, overflows $(count(ovf))"); flush(stdout)
seps = [CliqueTrees.bt_out(lists[j], nv(sub[j].g), 1) for m in (2, 5, 20)
    println("batched sepdp of $m rounds..."); flush(stdout)
    t = @elapsed res, I, _ = gpu_sepdp([r.w for r in sub[1:m]], [r.g for r in sub[1:m]], seps[1:m], [r.k for r in sub[1:m]])
    println("   $t s gpu $(I.tgpu) ok=", all(res[j][3] == refs[heavy[j]][2] for j in 1:m)); flush(stdout)
end
