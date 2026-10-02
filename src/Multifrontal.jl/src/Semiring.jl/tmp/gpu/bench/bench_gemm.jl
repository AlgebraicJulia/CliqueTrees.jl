# Correctness and throughput of sgemx_gpu! against the CPU kernel
# Semiring.sgemx! and cuBLAS SGEMM.
#
#   julia --project=. -t auto bench/bench_gemm.jl

include(joinpath(@__DIR__, "..", "src", "SemiringGPU.jl"))

using .SemiringGPU
using .SemiringGPU: Semiring
using .Semiring: MinPlus, MaxPlus, MaxMin, PlusProd, sgemx!
using CUDA, LinearAlgebra, Printf, Random

Random.seed!(1)

rand_entries(::Type{T}, dims...) where {T <: AbstractFloat} = rand(T, dims...) .* T(100)
rand_entries(::Type{T}, dims...) where {T <: Integer} = rand(T(1):T(100), dims...)

function check(s, ::Type{T}, m, n, k) where {T}
    A = rand_entries(T, m, k)
    B = rand_entries(T, k, n)
    C = rand_entries(T, m, n) .+ T(100)
    ref = sgemx!(s, Val(:N), Val(:N), copy(C), A, B)
    out = Array(sgemx_gpu!(s, CuArray(C), CuArray(A), CuArray(B)))
    return s isa PlusProd ? isapprox(out, ref; rtol = 1e-4) : out == ref
end

# median-free timing: CUDA events around `reps` back-to-back launches
function gpu_rate(f, m, n, k; reps = 20)
    f(); CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:reps
        f()
    end
    return reps * m * n * k / t / 1e9
end

function cpu_rate(s, ::Type{T}, m, n, k; reps = 3) where {T}
    A = rand_entries(T, m, k); B = rand_entries(T, k, n); C = rand_entries(T, m, n)
    sgemx!(s, Val(:N), Val(:N), C, A, B)
    t = @elapsed for _ in 1:reps
        sgemx!(s, Val(:N), Val(:N), C, A, B)
    end
    return reps * m * n * k / t / 1e9
end

function gpu_semiring_rate(s, ::Type{T}, m, n, k; tiling = SemiringGPU.choose_tiling(m, n)) where {T}
    A = CuArray(rand_entries(T, m, k)); B = CuArray(rand_entries(T, k, n)); C = CuArray(rand_entries(T, m, n))
    return gpu_rate(() -> sgemx_gpu!(s, C, A, B; tiling), m, n, k)
end

function cublas_rate(::Type{T}, m, n, k) where {T}
    A = CUDA.rand(T, m, k); B = CUDA.rand(T, k, n); C = CUDA.rand(T, m, n)
    return gpu_rate(() -> mul!(C, A, B, true, true), m, n, k)
end

println(CUDA.name(CUDA.device()), ", ", Threads.nthreads(), " CPU threads, math mode ", CUDA.math_mode())
println()

# ----- correctness, including ragged sizes -----

println("correctness vs CPU sgemx!")
for (s, T) in [(MinPlus(), Float32), (MaxPlus(), Float32), (MaxMin(), Float32), (PlusProd(), Float32),
               (MinPlus(), Int32), (MinPlus(), Float64)]
    ok = all(check(s, T, m, n, k) for (m, n, k) in [(1, 1, 1), (7, 5, 3), (128, 128, 8), (130, 129, 9), (1000, 777, 333), (64, 2000, 17), (1000, 7, 333), (5000, 16, 100), (3000, 30, 50), (513, 33, 9)])
    @printf("  %-10s %-8s %s\n", s isa MaxMin ? "MaxMin" : s isa MaxPlus ? "MaxPlus" : nameof(typeof(s)), T, ok ? "ok" : "FAIL")
end
println()

# ----- square -----

println("square, G multiply-adds/s")
@printf("  %6s  %10s %10s %10s %10s %10s %10s | %10s %10s\n", "n", "MinPlus32", "MaxMin32", "MinPlusI32", "MinPlus64", "PlusProd32", "cuBLAS32", "CPU MinPl32", "GPU/CPU")
for n in (256, 512, 1024, 2048, 4096)
    g  = gpu_semiring_rate(MinPlus(), Float32, n, n, n)
    mm = gpu_semiring_rate(MaxMin(), Float32, n, n, n)
    gi = gpu_semiring_rate(MinPlus(), Int32, n, n, n)
    gd = gpu_semiring_rate(MinPlus(), Float64, n, n, n)
    pp = gpu_semiring_rate(PlusProd(), Float32, n, n, n)
    cb = cublas_rate(Float32, n, n, n)
    c  = n <= 2048 ? cpu_rate(MinPlus(), Float32, n, n, n) : NaN
    @printf("  %6d  %10.0f %10.0f %10.0f %10.0f %10.0f %10.0f | %10.1f %9.1f×\n", n, g, mm, gi, gd, pp, cb, c, g / c)
end
println()

# ----- frontal update shapes: M = F22 + L21 U12, (a × nj)(nj × a) -----

println("frontal update F22 ← F22 ⊕ L21 ⊗ U12 (a × nj times nj × a), MinPlus Float32, G multiply-adds/s")
@printf("  %6s %5s  %10s %10s %10s\n", "a", "nj", "GPU", "CPU", "GPU/CPU")
for (a, nj) in [(256, 16), (512, 32), (1024, 32), (1024, 128), (2048, 64), (4096, 128)]
    g = gpu_semiring_rate(MinPlus(), Float32, a, a, nj)
    c = cpu_rate(MinPlus(), Float32, a, a, nj)
    @printf("  %6d %5d  %10.0f %10.1f %9.1f×\n", a, nj, g, c, g / c)
end
