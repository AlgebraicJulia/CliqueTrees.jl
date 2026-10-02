# Experiment: Julia GEMM kernel (sgemx_kernel2!) with maxregs = 128 vs C++, interleaved bursts.
#   julia --project=. cuda/experiments/maxregs.jl
include(joinpath(@__DIR__, "..", "..", "src", "SemiringGPU.jl")); include(joinpath(@__DIR__, "..", "..", "src", "cuda_backend.jl"))
using .SemiringGPU, .SemiringCUDA, CUDA, Printf
using .SemiringGPU.Semiring: MinPlus
const SG = SemiringGPU
function rates(fs, mnk; rounds = 7, burst = 0.01)
    reps = map(fs) do f; f(); CUDA.synchronize(); t = CUDA.@elapsed f(); clamp(round(Int, burst / t), 1, 1000); end
    best = fill(Inf, length(fs))
    for _ in 1:rounds, (i, f) in enumerate(fs)
        sleep(0.05); t = CUDA.@elapsed for _ in 1:reps[i]; f(); end; best[i] = min(best[i], t / reps[i])
    end
    mnk ./ best ./ 1e9
end
s = MinPlus()
for (m, n, k) in [(2048, 2048, 2048), (4096, 4096, 4096), (27000, 64, 64)]
    A = CUDA.rand(Float32, m, k); B = CUDA.rand(Float32, k, n); C = CUDA.rand(Float32, m, n)
    jl(t, mr) = begin
        BM, BN, BK, TM, TN = typeof(t).parameters
        () -> @cuda threads = (BM ÷ TM) * (BN ÷ TN) blocks = (cld(m, BM), cld(n, BN)) maxregs = mr SG.sgemx_kernel2!(s, C, A, B, Val(BM), Val(BN), Val(BK), Val(TM), Val(TN))
    end
    fs = [jl(SG.TILING_LARGE, 255), jl(SG.TILING_LARGE, 128), jl(SG.TILING_SMALL, 255), () -> sgemx_cuda!(s, C, A, B; tiling = 1), () -> sgemx_cuda!(s, C, A, B; tiling = 7)]
    r = rates(fs, m * n * k)
    @printf("%-16s Julia 128×128: %5.0f, maxregs=128: %5.0f, Julia 64×64: %5.0f | C++ 128×128: %5.0f, C++ 128×64k16: %5.0f\n", "$m×$n×$k", r...)
    k1 = @cuda launch = false maxregs = 128 SG.sgemx_kernel2!(s, C, A, B, Val(128), Val(128), Val(8), Val(8), Val(8))
    m == 2048 && println("   maxregs=128 kernel: ", CUDA.registers(k1), " regs, local ", CUDA.memory(k1).local, " B")
end
