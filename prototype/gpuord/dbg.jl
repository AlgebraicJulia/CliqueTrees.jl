using CUDA, CliqueTrees, Graphs, Random
include(joinpath(@__DIR__, "src", "GPUOrd.jl")); using .GPUOrd
rng = Xoshiro(1)
for (n, W) in ((60, 1), (60, 2), (60, 4), (100, 2), (33, 1), (33,2), (31,2))
g = erdos_renyi(n, 0.08; rng); w = ones(Int32, n)
b = pack([g], [w], W)
ho = zeros(Int16, 64W, 1); ord = randperm(rng, n); ho[1:n, 1] .= ord .- 1
width = CUDA.zeros(Int32, 1); lc = CUDA.zeros(Float32, 1)
score!(width, lc, b, CuArray(ho), 1)
println((n, W, Array(width), treewidth(w, g; alg = ord), GPUOrd.cpu_score_bitset(GPUOrd.cpurows(g, W), w, ord)))
end
