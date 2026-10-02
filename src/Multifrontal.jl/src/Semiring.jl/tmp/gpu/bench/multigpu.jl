# Multi-GPU closure (rows split over GPUs) vs the single-GPU closure, with an exact check.
#
#   julia --project=. -t auto bench/multigpu.jl graph [graph ...]
#   DEVICES=0,1,2,3 ...      which GPUs (default: all); repeating a device runs several blocks on it (testing)
#   CHECK=0                  skip the comparison with the single-GPU closure (needs n² memory on GPU 0)
include(joinpath(@__DIR__, "bench_apsp.jl"))
using CUDA, Printf

const REPS = parse(Int, get(ENV, "REPS", "3"))
devs = haskey(ENV, "DEVICES") ? [CuDevice(parse(Int, x)) for x in split(ENV["DEVICES"], ",")] : collect(CUDA.devices())
check = get(ENV, "CHECK", "1") == "1"
println("devices: ", join(["$(CUDA.deviceid(d)):$(CUDA.name(d))" for d in devs], ", "), " | threads ", Threads.nthreads())

for name in ARGS
    CUDA.device!(devs[1])
    A = read_mtx(joinpath(MTX, name * ".mtx")); n = size(A, 1)
    F = ChordalSLU(MinPlus(), A); copyto!(F, A)
    P = FactorPlan(F; large = 256, graph = false, nstreams = 8); factorize!(P)
    tsetup = @elapsed (MG = MultiGPUSLU(P; devices = devs))
    blocks = closure_multigpu!(MG)                          # compile, tune, warm
    tm = zeros(length(devs))
    t = minimum(@elapsed(closure_multigpu!(MG; out = [b[2] for b in blocks], timer = tm)) for _ in 1:REPS)
    closure_multigpu!(MG; out = [b[2] for b in blocks], timer = tm)
    chk = sum(b -> (CUDA.device!(CUDA.device(b[2])); mapreduce(x -> isfinite(x) ? Float64(x) : 0.0, +, b[2])), blocks)
    CUDA.device!(devs[1])
    @printf("%-12s n=%6d | %d GPU blocks: closure %8.2f ms (per device %s ms) | setup %.0f ms | checksum %.6e\n",
        name, n, length(devs), 1e3t, join([@sprintf("%.1f", 1e3x) for x in tm], "/"), 1e3tsetup, chk)

    if check && 1.2 * n^2 * sizeof(T) < CUDA.free_memory()
        G = GPUSLU(P; large = 8192); precompute_ops!(G)
        D = CuMatrix{T}(undef, n, n); M = CuMatrix{T}(undef, n, G.maxna)
        closure_gpu!(D, G; M); CUDA.synchronize()
        t1 = minimum(@elapsed((closure_gpu!(D, G; M); CUDA.synchronize())) for _ in 1:REPS)
        ok = all(b -> Array(b[2]) == Array(view(D, b[1], :)), blocks)
        @printf("             single GPU closure %8.2f ms | speedup %.2f× | identical: %s\n", 1e3t1, t1 / t, ok)
        CUDA.unsafe_free!(D); CUDA.unsafe_free!(M)
    end

    for b in blocks; CUDA.unsafe_free!(b[2]); end
    CUDA.reclaim()
end
