# Persistent L sweep only (C++ port / reg vs Julia), for comparing library builds:
#   SEMIRING_CUDA_LIB=$PWD/cuda/build-p64/libsemiring_cuda.so julia --project=. -t 16 cuda/experiments/persist.jl grid3d-25
include(joinpath(@__DIR__, "..", "..", "bench", "bench_cuda_backend.jl"))
for name in ARGS
    A = read_mtx(joinpath(MTX, name * ".mtx")); n = size(A, 1)
    F = ChordalSLU(MinPlus(), A); copyto!(F, A)
    P = FactorPlan(F; large = 256, graph = false, nstreams = 8); factorize!(P)
    G = GPUSLU(P; large = 8192)
    D = CuMatrix{Float32}(undef, n, n); M = CuMatrix{Float32}(undef, n, G.maxna)
    setschedule!(true); closure_gpu!(D, G; M); CUDA.synchronize()
    plan = SemiringGPU.persistent_plan(G)
    jl() = begin   # the Julia persistent launch, as in downward_sweep!
        nchunk = cld(n, SemiringGPU.PERSIST_TB); nitems = plan.nrest * nchunk
        fill!(plan.counters, zero(UInt32)); fill!(plan.ticket, zero(UInt32))
        nblk = min(nitems, SemiringCUDA.nsm() * SemiringGPU.PERSIST_BLOCKS_PER_SM)
        @cuda threads = SemiringGPU.PERSIST_TB blocks = nblk SemiringGPU.persistent_down_kernel!(G.s, Val(:N), D, plan.rest, plan.nrest, nchunk,
            G.pnt, G.istop, plan.counters, plan.ticket, G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.LDval, G.LLval)
    end
    fs = [jl, () -> SemiringCUDA.persistent_down_cuda!(G, D, plan, 0), () -> SemiringCUDA.persistent_down_cuda!(G, D, plan, 1)]
    wait_gpu()
    t = map(fs) do f
        f(); CUDA.synchronize()
        minimum(CUDA.@elapsed(f()) for _ in 1:5)
    end
    @printf("%-10s L_persistent ms: Julia %.1f | C++ port %.1f | C++ reg %.1f   (%s)\n", name, (1e3 .* t)..., basename(dirname(SemiringCUDA.LIB)))
    global D = M = G = P = F = nothing; GC.gc(); CUDA.reclaim()
end
