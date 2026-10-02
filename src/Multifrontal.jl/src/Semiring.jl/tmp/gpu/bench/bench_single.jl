# Single-source queries (k = 1): the CPU with Richard's fast path and subtree
# parallelism (tmp/subtree_tmp.jl + tmp/sssp_tmp.jl) against the GPU.
#
#   julia --project=. -t auto bench/bench_single.jl [graph ...]

include(joinpath(@__DIR__, "bench_solve.jl"))

const SR = Semiring
const TMP = joinpath(pkgdir(SemiringGPU.CliqueTrees), "src", "Multifrontal.jl", "src", "Semiring.jl", "tmp")
Base.include(SR, joinpath(TMP, "subtree_tmp.jl"))
Base.include(SR, joinpath(TMP, "sssp_tmp.jl"))

const NQ = 100

function run_single(name)
    s = MinPlus()
    A = GRAPHS[name]()
    n = size(A, 1)
    nt = Threads.nthreads()
    F = mlu(s, A; nt)
    G = GPUSLU(F)
    srcs = rand(Xoshiro(3), 1:n, NQ)

    # CPU, fast path + subtree parallelism (row k of A*: side :R)
    cpu = Dict{Int, Float64}()
    ref = Dict{Int, Vector{T}}()

    for t in unique((1, nt))
        W, x, pool, sched = SR.sgetrs_elem_workspace_tmp(F; nt = t)
        b = Vector{T}(undef, n)
        SR.sgetrs_tmp!(F, Val(:R), Val(:N), b, srcs[1], W, x, pool, sched; nt = t)
        ts = map(srcs) do k
            @elapsed SR.sgetrs_tmp!(F, Val(:R), Val(:N), b, k, W, x, pool, sched; nt = t)
        end
        cpu[t] = median(ts)

        if t == nt
            for k in srcs[1:5]
                SR.sgetrs_tmp!(F, Val(:R), Val(:N), b, k, W, x, pool, sched; nt = t)
                ref[k] = copy(b)
            end
        end
    end

    # CPU, dense solve (no fast path), now with the subtree-parallel overlay
    B = fill(szero(s, T, Val(:N)), 1, n)
    dense = median(map(srcs[1:20]) do k
        fill!(B, szero(s, T, Val(:N))); B[1, k] = sone(s, T, Val(:N))
        @elapsed rmul!(B, F; nt)
    end)

    # GPU, path-walk U sweep, as a CUDA graph
    P = SSSPPlan(G, 1)
    P([srcs[1]]); CUDA.synchronize()
    gpu = median(map(srcs) do k
        CUDA.@elapsed P([k])
    end)
    ok = all(Array(P([k]))[1, :] == ref[k] for k in srcs[1:5])

    @printf("  %-15s %8d | %9.3f %9.3f %9.3f | %9.3f | %6.2f× | %s\n", name, n, 1e3cpu[1], 1e3cpu[nt], 1e3dense, 1e3gpu, cpu[nt] / gpu, ok ? "ok" : "MISMATCH")
end

if abspath(PROGRAM_FILE) == @__FILE__
    nt = Threads.nthreads()
    println(CUDA.name(CUDA.device()), ", ", nt, " CPU threads, ", T, "; median of $NQ random sources, ms per query")
    println("  CPU fast = tmp/sssp_tmp.jl (path forward solve + subtree-parallel backward solve); CPU dense = rmul! with the subtree overlay")
    @printf("  %-15s %8s | %9s %9s %9s | %9s | %7s | %s\n", "graph", "n", "fast 1thr", "fast $(nt)thr", "dense", "GPU", "CPU/GPU", "check")
    for name in (isempty(ARGS) ? ["USA-road-t.NY", "USA-road-t.FLA", "roadNet-PA", "grid2d-500"] : ARGS)
        run_single(name)
    end
end
