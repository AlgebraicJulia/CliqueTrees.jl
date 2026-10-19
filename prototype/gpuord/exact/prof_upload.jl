using CUDA, Profile
include(joinpath(@__DIR__, "exact_common.jl"))
include(joinpath(@__DIR__, "SeptwGPU.jl"))
rs = loadrounds("."); C = deserialize("cpu_ref_laptop_t16.jls")
i = sortperm(C.t1; rev = true)[3]; r = rs[i]; n = nv(r.g)
bt = EnumW.prepare(UInt64, [r.g], [collect(Float64, r.w)], [r.k])
lists, _, _, _ = EnumW.gpu_sepenum(bt; split = 3)
sep = EnumW.sepgraph(lists[1], n)
DPW.gpu_sepdp(UInt64, [r.w], [r.g], [sep], [r.k])
P = DPW.prep(r.w, r.g, sep)
println("blocks ", P.nblk)
t = @elapsed for _ in 1:3; DPW.gpu_sepdp(UInt64, [r.w], [r.g], [sep], [r.k]); end
println("3 calls ", t)
Profile.clear(); @profile for _ in 1:3; DPW.gpu_sepdp(UInt64, [r.w], [r.g], [sep], [r.k]); end
Profile.print(; format = :flat, sortedby = :count, mincount = 30, noisefloor = 2)
