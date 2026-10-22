# GPU tree cost vs the CPU reference (Host.bucket_tree + cost!): same trees, same C and W.
#   julia --project=env scripts/test_gpu.jl <list> <orddir> [nperturb]
include(joinpath(@__DIR__, "..", "src", "Host.jl")); using .Host
include(joinpath(@__DIR__, "..", "src", "GPUTree.jl")); using .GPUTree
using CUDA, Random
names = readlines(ARGS[1]); orddir = ARGS[2]; np = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 64
maxdc = 0.0; maxdw = 0.0; nt = 0; nsame = 0
for name in names
    files = filter(f -> startswith(f, name * "__") && endswith(f, ".ord"), readdir(orddir))
    isempty(files) && continue
    net = readnet(name); g = gnet(net)
    base = reduce(hcat, [parse.(Int32, split(readlines(joinpath(orddir, f))[2])) .- Int32(1) for f in files])
    nb = size(base, 2); J = nb + np
    sc = scratch(g, J)
    db = CuArray(base)
    bidx = CuArray(Int32[(j - 1) % nb + 1 for j in 1:J])
    nm = CuArray(Int32[j <= nb ? 0 : rand(1:64) for j in 1:J]); md = CuArray(Int32[rand(1:32) for j in 1:J])
    perturb!(sc, g, db, bidx, nm, md, CuArray(rand(UInt64, J)))
    score_orders!(sc, g); CUDA.synchronize()
    orders = Array(sc.orders) .+ 1; lc = Array(sc.log2C); W = Array(sc.W)
    for j in 1:J
        @assert sort(orders[:, j]) == 1:net.n
        tr = bucket_tree(net, orders[:, j]); cost!(tr, net)
        l, r = fetch_tree(sc, g, j)
        global nsame += (l == tr.left && r == tr.right)
        global maxdc = max(maxdc, abs(tr.log2C - lc[j])); global maxdw = max(maxdw, abs(tr.W - W[j])); global nt += 1
    end
    # timing
    t = CUDA.@elapsed score_orders!(sc, g)
    println(rpad(name, 48), " J=$J  host-vs-gpu max|dlog2C|=$(round(maxdc; sigdigits=3)) max|dW|=$(round(maxdw; sigdigits=3))  $(round(J / t; digits=0)) trees/s")
end
println("trees ", nt, " identical structure ", nsame, " max|dlog2C| ", maxdc, " max|dW| ", maxdw)
