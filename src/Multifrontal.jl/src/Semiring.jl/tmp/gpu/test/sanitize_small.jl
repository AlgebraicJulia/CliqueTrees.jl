# Small workload touching every GPU kernel, for compute-sanitizer (memcheck / racecheck / synccheck / initcheck).
include(joinpath(@__DIR__, "..", "src", "SemiringGPU.jl"))
include(joinpath(@__DIR__, "graphs.jl"))
using .SemiringGPU
using .SemiringGPU: Semiring
using .Semiring: MinPlus, PlusProd, ChordalSLU, mlu, szero, sone, sgemx!
using CUDA, SparseArrays, Random
Random.seed!(11)

s = MinPlus()
for v in (1, 2), (m, n, k) in ((70, 33, 17), (130, 20, 9), (64, 64, 64))
    SemiringGPU.GEMM_VERSION[] = v
    A = CuArray(Float32.(rand(1:9, m, k))); B = CuArray(Float32.(rand(1:9, k, n))); C = CUDA.fill(100f0, m, n)
    sgemx_gpu!(s, C, A, B)
end
SemiringGPU.GEMM_VERSION[] = 2
X = CuArray(Float32.(rand(1:9, 130, 130))); sgetrf_gpu!(s, X)

A = grid3(9, Float32)                                       # 729 vertices: dense and batched paths, several levels
n = size(A, 1)
for (flarge, ns) in ((16, 8), (64, 1))
    F = ChordalSLU(s, A); copyto!(F, A)
    P = FactorPlan(F; large = flarge, graph = false, nstreams = ns); factorize!(P)
    for slarge in (16, typemax(Int)), ops in (false, true), pers in (false, true), warp in (false, true)
        SemiringGPU.PERSISTENT[] = pers; SemiringGPU.PATH_WARP[] = warp
        G = GPUSLU(P; large = slarge); ops && precompute_ops!(G)
        k = 37; src = rand(1:n, k)
        sssp_gpu!(CuMatrix{Float32}(undef, k, n), G, CuVector(src))
        B = fill(Inf32, k, n); for t in 1:k; B[t, src[t]] = 0f0; end
        rmul_gpu!(CuArray(B), G)
        closure_gpu(G)
    end
end
CUDA.synchronize()
println("sanitize workload done")
