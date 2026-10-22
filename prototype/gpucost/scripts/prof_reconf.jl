# Time breakdown of one reconfiguration round: host chunk building, GPU DP (+transfers), apply + re-cost.
include(joinpath(@__DIR__, "..", "src", "Host.jl")); using .Host
include(joinpath(@__DIR__, "..", "src", "GPUReconf.jl")); using .GPUReconf
include(joinpath(@__DIR__, "..", "src", "Reconf.jl")); using .Reconf
using CUDA, Random
for name in ARGS
    net = readnet(name)
    o = open_last(net, parse.(Int, split(readlines("data/ord/$(name)__hbtT0.5s1.ord")[2])))
    for K in (8, 10, 12)
        trees = [cost!(bucket_tree(net, o), net) for _ in 1:8]
        legs = [[Int[] for _ in 1:2net.T-1] for _ in trees]
        for (i, t) in enumerate(trees); cost!(t, net; legs = legs[i]); end
        rng = Xoshiro(1)
        b = DPBatch(); descs = Reconf.Desc[]
        mask = zeros(UInt16, net.n); io = falses(net.n)
        for (i, t) in enumerate(trees); Reconf.addchunks!(b, descs, i, t, net, legs[i], K, rng, false, mask, io); end
        dp_solve!(b; K)
        b = DPBatch(); descs = Reconf.Desc[]
        th = @elapsed for (i, t) in enumerate(trees); Reconf.addchunks!(b, descs, i, t, net, legs[i], K, rng, false, mask, io); end
        tg = @elapsed (best, arg) = dp_solve!(b; K)
        tk = CUDA.@elapsed dp_solve!(b; K)
        tc = @elapsed for (i, t) in enumerate(trees); cost!(t, net; legs = legs[i]); end
        println(rpad(name, 40), " K=$K chunks ", length(descs), " host build ", round(th * 1e3; digits = 1), " ms, dp_solve ", round(tg * 1e3; digits = 1),
            " ms (", round(length(descs) / tg; digits = 0), " chunks/s), recost 8 trees ", round(tc * 1e3; digits = 1), " ms")
    end
end
