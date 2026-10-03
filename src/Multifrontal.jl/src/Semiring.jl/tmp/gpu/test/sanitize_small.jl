# Small workload touching every GPU kernel and every setting that selects a different kernel path, for
# compute-sanitizer:
#   compute-sanitizer --tool memcheck|racecheck|synccheck|initcheck julia --project=. test/sanitize_small.jl
include(joinpath(@__DIR__, "..", "src", "SemiringGPU.jl"))
include(joinpath(@__DIR__, "graphs.jl"))
using .SemiringGPU
using .SemiringGPU: Semiring
using .Semiring: MinPlus, MaxPlus, PlusProd, ChordalSLU, mlu, szero, sone
using CUDA, SparseArrays, Random
Random.seed!(11)

s = MinPlus()
# dense kernels: every GEMM version and tiling, in place and not, odd sizes
for v in (2, 4, 6, 7, 8), tl in (SemiringGPU.TILING_LARGE, SemiringGPU.TILING_MID, SemiringGPU.TILING_SMALL, SemiringGPU.TILING_N32, SemiringGPU.TILING_N16)
    with_config(gemm_kernel = v) do
        for (m, n, k) in ((70, 33, 17), (130, 20, 9), (64, 64, 64), (200, 129, 31))
            A = CuArray(Float32.(rand(1:9, m, k))); B = CuArray(Float32.(rand(1:9, k, n))); C = CUDA.fill(100f0, m, n)
            sgemx_gpu!(s, C, A, B; tiling = tl)
            sgemx_gpu!(s, C, A, B; tiling = tl, overwrite = true)
        end
        X = CuArray(Float32.(rand(1:9, 90, 40))); T = CuArray(Float32.(rand(1:9, 40, 40)))
        sgemx_gpu!(s, X, X, T; overwrite = true)              # in place (one tile wide)
    end
end
X = CuArray(Float32.(rand(1:9, 130, 130))); sgetrf_gpu!(s, X)

# solver: factorization and solve settings, both stream schedules, with and without precomputed operators
A = grid3(9, Float32)                                       # 729 vertices: dense and batched paths, several levels
n = size(A, 1)
for fm in (1, 128), fused in (false, true), direct in (false, true), (flarge, ns) in ((16, 8), (64, 1))
    with_config(factor_merge = fm, fused_front = fused, direct_assembly = direct) do
        F = ChordalSLU(s, A); copyto!(F, A)
        P = FactorPlan(F; large = flarge, graph = false, nstreams = ns); factorize!(P)
        for slarge in (16, typemax(Int)), ops in (false, true), (rows, skip) in ((1, true), (1, false), (10^9, false)), merge in (1, 128)
            with_config(layered_min_rows = rows, skip_fill = skip, merge = merge) do
                G = GPUSLU(P; large = slarge); ops && precompute_ops!(G)
                k = 37; src = rand(1:n, k)
                sssp_gpu!(CuMatrix{Float32}(undef, k, n), G, CuVector(src))
                B = fill(Inf32, k, n); for t in 1:k; B[t, src[t]] = 0f0; end
                rmul_gpu!(CuArray(B), G)
                closure_gpu(G)
            end
        end
    end
end

# multi-GPU closure (two blocks on the same device when only one is present)
let F = ChordalSLU(s, A)
    copyto!(F, A); P = FactorPlan(F; large = 64, graph = false); factorize!(P)
    MG = MultiGPUSLU(P; devices = [CUDA.device(), CUDA.device()])
    closure_multigpu!(MG)
end

# plus-times (atomic add, scaled stars) on a small system
let A = 0.01 .* grid3(5, Float64)
    F = ChordalSLU(PlusProd(), A); copyto!(F, A)
    P = FactorPlan(F; large = 16, graph = false); factorize!(P)
    G = GPUSLU(P; large = 16); precompute_ops!(G)
    k = 9; src = rand(1:size(A, 1), k)
    sssp_gpu!(CuMatrix{Float64}(undef, k, size(A, 1)), G, CuVector(src))
end

CUDA.synchronize()
println("sanitize workload done")
