# register counts of the Julia (CUDA.jl) kernels, MinPlus Float32, for cuda/NOTES.md
include(joinpath(@__DIR__, "..", "..", "src", "SemiringGPU.jl"))
using .SemiringGPU, CUDA
using .SemiringGPU.Semiring: MinPlus
const SG = SemiringGPU
s = MinPlus(); T = Float32
C = CUDA.rand(T, 256, 256); A = CUDA.rand(T, 256, 256); B = CUDA.rand(T, 256, 256)
for t in (SG.TILING_LARGE, SG.TILING_SMALL, SG.TILING_N32, SG.TILING_N16)
    BM, BN, BK, TM, TN = typeof(t).parameters
    k = @cuda launch = false SG.sgemx_kernel2!(s, C, A, B, Val(BM), Val(BN), Val(BK), Val(TM), Val(TN))
    println(rpad("sgemx_kernel2! $(BM)x$(BN) k$BK", 40), "REG ", CUDA.registers(k), " LOCAL ", CUDA.memory(k).local, " SHARED ", CUDA.memory(k).shared)
end
I = CUDA.ones(Int, 10); Bv = CUDA.zeros(Bool, 10); V = CUDA.rand(T, 10)
k = @cuda launch = false SG.downward_kernel_simple!(s, Val(:N), C, I, 0, I, I, I, I, I, V, V); println(rpad("downward_kernel_simple!", 40), "REG ", CUDA.registers(k))
k = @cuda launch = false SG.upward_kernel!(s, Val(:N), Val(true), C, I, 0, I, I, I, I, I, V, V); println(rpad("upward_kernel!", 40), "REG ", CUDA.registers(k))
k = @cuda launch = false SG.upward_path_kernel!(s, Val(:N), Val(true), C, I, I, I, I, Bv, I, I, I, I, I, V, V); println(rpad("upward_path_kernel!", 40), "REG ", CUDA.registers(k))
if isdefined(SG, :upward_path_warp_kernel!)
    k = @cuda launch = false SG.upward_path_warp_kernel!(s, Val(:N), Val(true), C, I, I, I, I, Bv, I, I, I, I, I, V, V); println(rpad("upward_path_warp_kernel!", 40), "REG ", CUDA.registers(k))
    U = CUDA.zeros(UInt32, 10)
    k = @cuda launch = false SG.persistent_down_kernel!(s, Val(:N), C, I, 1, 1, I, Bv, U, U, I, I, I, I, I, V, V); println(rpad("persistent_down_kernel!", 40), "REG ", CUDA.registers(k), " SHARED ", CUDA.memory(k).shared)
end
open(joinpath(@__DIR__, "jl_downward_kernel_simple.sass"), "w") do io
    CUDA.code_sass(io, SG.downward_kernel_simple!, Tuple{typeof(s), Val{:N}, typeof(cudaconvert(C)), typeof(cudaconvert(I)), Int, ntuple(_ -> typeof(cudaconvert(I)), 5)..., typeof(cudaconvert(V)), typeof(cudaconvert(V))})
end
