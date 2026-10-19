using CUDA, CliqueTrees, Graphs, Random
include(joinpath(@__DIR__, "src", "GPUOrd.jl")); using .GPUOrd
rng = Xoshiro(3); P = 2; K = 4; W = 1; N = 64
gs = [random_regular_graph(30, 3; rng) for _ in 1:P]; ws = [ones(Int32, 30) for _ in 1:P]
b = pack(gs, ws, W)
best = CUDA.zeros(Int16, N, P); bestw = CUDA.fill(typemax(Int32), P); bestc = CUDA.fill(Inf32, P); bt = CUDA.zeros(Int32, P)
pidx = CuArray(Int32.(repeat(1:P; inner = K))); bstart = CuArray(Int32.(1:K:P*K+1)); bpiece = CuArray(Int32.(1:P))
cand = CUDA.zeros(Int16, N, P * K); width = CUDA.zeros(Int32, P * K); lcost = CUDA.zeros(Float32, P * K); tm = CUDA.zeros(Int32, P * K)
greedy!(cand, width, lcost, tm, b, pidx, CuArray([GreedyParams(0, 0, 1, 0.1f0, rand(rng, UInt32)) for _ in 1:P*K]); invq = 1e-3)
@show Array(width) Array(lcost) Array(tm)
selectbest!(bstart, bpiece, cand, width, lcost, tm, best, bestw, bestc, bt, b)
@show Array(bestw) Array(bestc) Array(bt)
@show Array(best)[1:30, 1]'
for r in 1:3
mutate!(cand, width, lcost, tm, b, pidx, best, bt, CuArray([MutateParams(2, 0.5f0, 0, rand(rng, UInt32)) for _ in 1:P*K]); invq = 1e-3)
@show Array(width) Array(lcost)
selectbest!(bstart, bpiece, cand, width, lcost, tm, best, bestw, bestc, bt, b)
@show Array(bestw) Array(bestc) Array(bt)
end
