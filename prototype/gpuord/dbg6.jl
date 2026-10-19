using CUDA, CliqueTrees, Graphs, Random
include(joinpath(@__DIR__, "src", "GPUOrd.jl")); using .GPUOrd
rng = Xoshiro(1); n=100; W=2; N=128
g = erdos_renyi(n, 0.08; rng); ord = randperm(rng, n); w = ones(Int32, n)
b = pack([g], [w], W)
ho = zeros(Int16, N, 1); ho[1:n,1] .= ord .- 1
width = CUDA.zeros(Int32, 1); lc = CUDA.zeros(Float32, 1)
kernel = @cuda launch=false GPUOrd.score_kernel!(Val(W), b.adj, b.wts, b.nvs, CuArray(ho), Int32(1), width, lc, Int32(1), 1f0)
open("score.ptx","w") do io; CUDA.code_ptx(io, GPUOrd.score_kernel!, typeof.((Val(W), CUDA.cudaconvert(b.adj), CUDA.cudaconvert(b.wts), CUDA.cudaconvert(b.nvs), CUDA.cudaconvert(CuArray(ho)), Int32(1), CUDA.cudaconvert(width), CUDA.cudaconvert(lc), Int32(1), 1f0))); end
