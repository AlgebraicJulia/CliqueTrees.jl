include(joinpath(@__DIR__, "..", "src", "Host.jl")); using .Host
include(joinpath(@__DIR__, "..", "src", "GPUTree.jl")); using .GPUTree
using CUDA, Random
for name in ARGS
    net = readnet(name); g = gnet(net)
    o = open_last(net, parse.(Int, split(readlines("data/ord/$(name)__hbtT0.5s1.ord")[2])))
    for J in (256, 1024, 4096)
        sc = scratch(g, J)
        base = CuArray(reshape(Int32.(o) .- Int32(1), :, 1))
        args = (CuArray(ones(Int32, J)), CuArray(fill(Int32(8), J)), CuArray(fill(Int32(16), J)), CuArray(rand(UInt64, J)))
        perturb!(sc, g, base, args...); build!(sc, g); gcost!(sc, g); CUDA.synchronize()
        tp = CUDA.@elapsed perturb!(sc, g, base, args...)
        tb = CUDA.@elapsed build!(sc, g)
        tc = CUDA.@elapsed gcost!(sc, g)
        println(rpad(name, 40), " J=$J perturb ", round(tp*1e3; digits=2), " ms build ", round(tb*1e3; digits=2), " ms cost ", round(tc*1e3; digits=2), " ms  -> ", round(J/(tp+tb+tc); digits=0), " cand/s")
    end
end
