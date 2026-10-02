# Experiment: U path walk kernels in isolation (thread and warp versions), Julia vs C++.
#   julia --project=. -t 16 cuda/experiments/path.jl grid2d-180 grid3d-25
include(joinpath(@__DIR__, "..", "..", "bench", "bench_cuda_backend.jl"))
const SG = SemiringGPU; const SC = SemiringCUDA
for name in ARGS
    A = read_mtx(joinpath(MTX, name * ".mtx")); n = size(A, 1)
    F = ChordalSLU(MinPlus(), A); copyto!(F, A)
    P = FactorPlan(F; large = 256, graph = false, nstreams = 8); factorize!(P)
    G = GPUSLU(P; large = 8192)
    D = CuMatrix{Float32}(undef, n, n)
    src = SG.upload(Array(G.rperm)); s = G.s; tb = 64; nb = cld(n, tb)
    jl() = @cuda threads = tb blocks = nb SG.upward_path_kernel!(s, Val(:N), Val(true), D, src, G.cinvp, G.idx, G.pnt, G.istop, G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.UDval, G.ULval)
    jlw() = @cuda threads = 128 blocks = cld(32 * n, 128) SG.upward_path_warp_kernel!(s, Val(:N), Val(true), D, src, G.cinvp, G.idx, G.pnt, G.istop, G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.UDval, G.ULval)
    fs = [jl, () -> SC.upward_path_cuda!(G, D, src, tb, 0), () -> SC.upward_path_cuda!(G, D, src, tb, 1), jlw,
          () -> SC.upward_path_warp_cuda!(G, D, src, 0), () -> SC.upward_path_warp_cuda!(G, D, src, 1)]
    fill!(D, Inf32)
    wait_gpu()
    t = map(fs) do f
        f(); CUDA.synchronize()
        minimum(CUDA.@elapsed(f()) for _ in 1:5)
    end
    @printf("%-10s U_path ms: thread: Julia %.1f  C++ port %.1f  C++ reg %.1f | warp: Julia %.1f  C++ port %.1f  C++ reg %.1f\n", name, (1e3 .* t)...)
    k = @cuda launch = false SG.upward_path_kernel!(s, Val(:N), Val(true), D, src, G.cinvp, G.idx, G.pnt, G.istop, G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.UDval, G.ULval)
    println("   Julia thread path kernel regs: ", CUDA.registers(k))
    global D = G = P = F = nothing; GC.gc(); CUDA.reclaim()
end
