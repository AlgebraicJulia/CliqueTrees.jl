#   julia --project=. -t 16 cuda/experiments/libcmp.jl cuda/build/libsemiring_cuda.so cuda/build-ptx/libsemiring_cuda.so
# compare sr_gemm from several builds of the library, interleaved, in one process
include(joinpath(@__DIR__, "..", "..", "bench", "bench_cuda_backend.jl"))
using Libdl
libs = ARGS
fps = [dlsym(dlopen(l), :sr_gemm) for l in libs]
gemm(fp, C, A, B, tiling) = (r = ccall(fp, Cint, (Cint, Cint, Cint, Cint, Cint, Cint, CuPtr{Cvoid}, Int64, CuPtr{Cvoid}, Int64, CuPtr{Cvoid}, Int64, Ptr{Cvoid}),
    0, 0, tiling, size(C, 1), size(C, 2), size(A, 2), pointer(A), stride(A, 2), pointer(B), stride(B, 2), pointer(C), stride(C, 2), Ptr{Cvoid}(CUDA.stream().handle)); @assert r == 0)
for (m, n, k) in [(15625, 64, 64), (27000, 64, 64), (27000, 20, 500), (27000, 500, 20), (15625, 300, 64), (2048, 2048, 2048)]
    A = CUDA.rand(Float32, m, k); B = CUDA.rand(Float32, k, n); C = CUDA.rand(Float32, m, n)
    wait_gpu()
    tilings = (3, 4, 5, 7, 8, 9, 10)
    fs = Any[]
    for t in tilings, fp in fps; push!(fs, () -> gemm(fp, C, A, B, t)); end
    push!(fs, () -> sgemx_gpu!(MinPlus(), C, A, B; tiling = SemiringGPU.TILING_SMALL))
    r = rates(fs, m, n, k)
    println("$m×$n×$k  Julia 64×64 $(round(Int, r[end]))")
    for (i, t) in enumerate(tilings)
        println("   tiling $(rpad(CU_TILINGS[t], 11)): ", join([string(round(Int, r[(i - 1) * length(fps) + j])) for j in 1:length(fps)], " / "))
    end
end
