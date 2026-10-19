using CUDA, CliqueTrees, Graphs, Random
include(joinpath(@__DIR__, "src", "GPUOrd.jl")); using .GPUOrd
rng = Xoshiro(3); P = 2; K = 4; W = 1; N = 64
gs = [random_regular_graph(60, 3; rng) for _ in 1:P]; ws = [ones(Int32, 60) for _ in 1:P]
b = pack(gs, ws, W)
best = CuArray(Int16.(hcat([vcat(randperm(rng, 60) .- 1, zeros(Int, 4)) for _ in 1:P]...)))
bt = CuArray(Int32[5, 7])
pidx = CuArray(Int32.(repeat(1:P; inner = K)))
cand = CUDA.zeros(Int16, N, P * K); width = CUDA.zeros(Int32, P * K); lcost = CUDA.zeros(Float32, P * K); tm = CUDA.zeros(Int32, P * K)
mutate!(cand, width, lcost, tm, b, pidx, best, bt, CuArray([MutateParams(2, 0.5f0, 0, UInt32(j)) for j in 1:P*K]); invq = 1e-3)
hc = Array(cand); hb = Array(best)
for j in 1:P*K
  p = (j-1)÷K+1
  println(j, " perm=", isperm(Int.(hc[1:60, j]) .+ 1), " diffpos=", findall(hc[1:60, j] .!= hb[1:60, p]), " w=", Array(width)[j])
end
