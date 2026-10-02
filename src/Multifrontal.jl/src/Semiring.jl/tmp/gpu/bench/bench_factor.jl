# Numeric factorization: our CPU solver vs our hybrid CPU+GPU factorization,
# and, for real arithmetic (the M-matrix I − W), against UMFPACK, CHOLMOD and
# NVIDIA cuDSS. Dense LU: ours (CPU and GPU) vs LAPACK and cuSOLVER.
#
#   julia --project=. -t auto bench/bench_factor.jl [dense] [minplus] [real]

include(joinpath(@__DIR__, "bench_solve.jl"))
include(joinpath(@__DIR__, "..", "test", "graphs.jl"))

using .Semiring: ChordalSLU, PlusProd
using CUDSS
import cuSPARSE

const NT = Threads.nthreads()
BLAS.set_num_threads(NT)   # all threads

best(f; reps = 3) = (f(); minimum(f() for _ in 1:reps))

# ----- graphs (same generators and seeds as bench_solve.jl) -----

grid3d(nx) = (Random.seed!(1); grid3(nx, Float32))

const SPARSE = [
    "USA-road-t.NY" => () -> GRAPHS["USA-road-t.NY"](),
    "grid2d-500"    => () -> grid(500, 500),
    "grid2d-1000"   => () -> grid(1000, 1000),
    "grid3d-30"     => () -> grid3d(30),
    "grid3d-40"     => () -> grid3d(40),
]

# ----- dense LU -----

function bench_dense()
    println("\n== dense LU, G multiply-adds/s (n³/3 per factorization) ==")
    @printf("  %5s | %10s %10s | %10s %10s %10s %10s\n", "n", "ours CPU", "ours GPU", "ours GPU", "cuSOLVER", "LAPACK", "ours CPU")
    @printf("  %5s | %10s %10s | %10s %10s %10s %10s\n", "", "MinPlus32", "MinPlus32", "real F32", "real F32", "real F32", "real F64")
    for n in (512, 1024, 2048, 4096)
        r(t) = n^3 / 3 / t / 1e9
        A = Float32.(rand(1:100, n, n))
        cpu = best(() -> @elapsed(Semiring.sgetrf!(MinPlus(), copy(A); nt = NT)); reps = n >= 4096 ? 1 : 3)
        Ag = CuArray(A); Bg = similar(Ag)
        gpu = best(() -> (copyto!(Bg, Ag); CUDA.@elapsed sgetrf_gpu!(MinPlus(), Bg)))
        # real: W* = (I − W)⁻¹ with W ≥ 0 small; ours factors W, LAPACK/cuSOLVER factor I − W (with pivoting)
        W = rand(Float32, n, n) ./ (2n); M = I - W
        Wg = CuArray(W); Mg = CuArray(M); Xg = similar(Wg)
        ours_real = best(() -> (copyto!(Xg, Wg); CUDA.@elapsed sgetrf_gpu!(PlusProd(), Xg)))
        cusolver = best(() -> (copyto!(Xg, Mg); CUDA.@elapsed lu!(Xg)))
        lapack = best(() -> (X = copy(M); @elapsed lu!(X)))
        W64 = Float64.(W)
        ours_cpu64 = best(() -> @elapsed(Semiring.sgetrf!(PlusProd(), copy(W64); nt = NT)); reps = n >= 4096 ? 1 : 3)
        @printf("  %5d | %10.0f %10.0f | %10.0f %10.0f %10.0f %10.0f\n", n, r(cpu), r(gpu), r(ours_real), r(cusolver), r(lapack), r(ours_cpu64))
    end
end

# ----- sparse, min-plus -----

function cpu_numeric(s, A; reps = 3)
    F = ChordalSLU(s, A)
    t = best(() -> (copyto!(F, A); @elapsed lu!(F; nt = NT)); reps)
    return F, t
end

# ours: subtree-parallel CPU and hybrid CPU+GPU, best over the top threshold
const LARGES = (64, 128, 256)

function cpu_parallel_numeric(s, A; reps = 3)
    best_t = Inf; best_l = 0
    F = ChordalSLU(s, A)
    for large in LARGES
        P = FactorPlan(F; large, graph = false, nstreams = 1)
        t = best(() -> (copyto!(F, A); sum(SemiringGPU.factorize_cpu!(P))); reps)
        t < best_t && ((best_t, best_l) = (t, large))
    end
    return best_t, best_l
end

# first: a fresh plan's first factorization (direct launches); replay: refactorization with the captured graph
function hybrid_numeric(s, A; reps = 3)
    F = ChordalSLU(s, A)
    res = map(LARGES) do large
        P = FactorPlan(F; large, graph = true, nstreams = 8)
        copyto!(F, A); h1 = factorize!(P)
        hs = [(copyto!(F, A); factorize!(P)) for _ in 1:reps]
        h = hs[argmin([x.cpu + x.upload + x.gpu for x in hs])]
        (large, h1.cpu + h1.upload + h1.gpu, h)
    end
    i = argmin([r[3].cpu + r[3].upload + r[3].gpu for r in res])
    return res[i]
end

function bench_minplus()
    println("\n== sparse numeric factorization, MinPlus Float32 (symbolic phase excluded; same AMF ordering for all) ==")
    println("   upstream CPU: CliqueTrees sgetrf! (fronts in sequence, multithreaded dense kernels)")
    println("   ours CPU ∥: same fronts, bottom subtrees in parallel (factorize_cpu!)")
    println("   hybrid: bottom subtrees in parallel on the CPU, top on the GPU (8 streams); first call / refactorization (CUDA graph)")
    A0 = grid(30, 30); cpu_numeric(MinPlus(), A0); cpu_parallel_numeric(MinPlus(), A0; reps = 1); hybrid_numeric(MinPlus(), A0; reps = 1)
    @printf("  %-14s %8s | %9s %10s | %9s %9s | %22s | %8s %8s\n", "graph", "n", "upstream", "ours CPU ∥", "hyb first", "hyb refac", "refac: cpu+upload+gpu", "vs upstr", "vs CPU ∥")
    for (name, mk) in SPARSE
        A = mk()
        _, tup = cpu_numeric(MinPlus(), A)
        tpar, lpar = cpu_parallel_numeric(MinPlus(), A)
        large, tfirst, h = hybrid_numeric(MinPlus(), A)
        th = h.cpu + h.upload + h.gpu
        @printf("  %-14s %8d | %8.1fms %9.1fms | %8.1fms %8.1fms | %6.1f + %5.1f + %6.1f | %7.2f× %7.2f×   (top=%d, %.0f%% of work, large=%d)\n",
            name, size(A, 1), 1e3tup, 1e3tpar, 1e3tfirst, 1e3th, 1e3h.cpu, 1e3h.upload, 1e3h.gpu, tup / th, tpar / th, h.ntop, 100h.topwork, large)
    end
end

# ----- sparse, real arithmetic -----

# W ≥ 0 symmetric with row sums < 1, so M = I − W is a nonsingular (symmetric) M-matrix and W* = M⁻¹
function mmatrix(A)
    W = Float64.(A)
    W = W ./ (1.01 * maximum(sum(W; dims = 2)))
    return W, sparse(1.0I, size(W)...) - W
end

function cudss_numeric(M::SparseMatrixCSC{T}, structure) where {T}
    Mg = cuSPARSE.CuSparseMatrixCSR(M)
    n = size(M, 1)
    x = CUDA.zeros(T, n); b = CUDA.ones(T, n)
    solver = CudssSolver(Mg, structure, 'F')
    cudss("analysis", solver, x, b); CUDA.synchronize()
    cudss("factorization", solver, x, b); CUDA.synchronize()
    t = best(() -> CUDA.@elapsed(cudss("refactorization", solver, x, b)))
    cudss("solve", solver, x, b); CUDA.synchronize()
    return t, Array(x)
end

function bench_real()
    println("\n== sparse numeric factorization, real arithmetic: M = I − W (ours factor W in PlusProd) ==")
    println("   each solver uses its own ordering; times are numeric factorization only (refactorization with the analysis reused)")
    A0, _ = mmatrix(grid(30, 30)); cpu_numeric(PlusProd(), A0); hybrid_numeric(PlusProd(), A0; reps = 1)
    hybrid_numeric(PlusProd(), Float32.(A0); reps = 1); cpu_parallel_numeric(PlusProd(), A0; reps = 1)

    println("   ours GPU = hybrid refactorization (CUDA graph replay); ours CPU ∥ = subtree-parallel CPU")
    @printf("  %-14s | %9s %9s %9s %9s | %9s %9s | %9s %9s %9s\n", "", "upstream", "ours CPU∥", "ours GPU", "ours GPU", "UMFPACK", "CHOLMOD", "cuDSS LU", "cuDSS LU", "cuDSS LLᵀ")
    @printf("  %-14s | %9s %9s %9s %9s | %9s %9s | %9s %9s %9s\n", "graph (ms)", "F64", "F64", "F64", "F32", "F64", "F64", "F64", "F32", "F64")
    for (name, mk) in SPARSE
        A = mk()
        W, M = mmatrix(A)
        F, ours = cpu_numeric(PlusProd(), W)
        par, _ = cpu_parallel_numeric(PlusProd(), W)
        h64 = hybrid_numeric(PlusProd(), W)[3]; g64 = h64.cpu + h64.upload + h64.gpu
        h32 = hybrid_numeric(PlusProd(), Float32.(W))[3]; g32 = h32.cpu + h32.upload + h32.gpu
        U = lu(M); umf = best(() -> @elapsed(lu!(U, M)))
        C = cholesky(M); chol = best(() -> @elapsed(cholesky!(C, M)))
        d64, x = cudss_numeric(M, "G")
        d32, _ = cudss_numeric(SparseMatrixCSC{Float32, Int}(M), "G")
        dll, _ = cudss_numeric(M, "SPD")
        # check: (I − W)⁻¹ 1 from cuDSS against our factor (rmul! of a row of ones; M is symmetric)
        y = vec(rmul!(ones(1, size(W, 1)), F))
        err = maximum(abs.(y .- x)) / maximum(abs.(x))
        @printf("  %-14s | %9.1f %9.1f %9.1f %9.1f | %9.1f %9.1f | %9.1f %9.1f %9.1f   (ours vs cuDSS rel. err %.1e)\n",
            name, 1e3ours, 1e3par, 1e3g64, 1e3g32, 1e3umf, 1e3chol, 1e3d64, 1e3d32, 1e3dll, err)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println(CUDA.name(CUDA.device()), ", ", NT, " CPU threads, BLAS threads ", BLAS.get_num_threads())
    which = isempty(ARGS) ? ["dense", "minplus", "real"] : ARGS
    "dense" in which && bench_dense()
    "minplus" in which && bench_minplus()
    "real" in which && bench_real()
end
