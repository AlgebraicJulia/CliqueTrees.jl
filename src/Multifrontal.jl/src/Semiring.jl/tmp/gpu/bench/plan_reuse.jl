# Reusing the structure: apsp_plan(A) once, then apsp_gpu(plan, A′) for new weights on the same pattern,
# against one-shot apsp_gpu(A′) calls. The ordering, symbolic analysis and plans are made once; a call
# copies the entries, refactors (CUDA graph replay) and solves. Results checked against the one-shot call.
#
#   OUT=plan.jsonl TAG=k3 julia --project=. -t 16 bench/plan_reuse.jl [--reps=5] graph ...
include(joinpath(@__DIR__, "bench_apsp.jl"))
using CUDA, Printf, Statistics

const S = SemiringGPU
const OUT = get(ENV, "OUT", "plan_reuse.jsonl")
const TAG = get(ENV, "TAG", "")
const REPS = something(tryparse(Int, replace(something(findfirst(a -> startswith(a, "--reps="), ARGS) |> i -> isnothing(i) ? nothing : ARGS[i], "--reps=5"), "--reps=" => "")), 5)
const GRAPHS = filter(a -> !startswith(a, "--"), ARGS)

# new weights, same pattern: w ↦ (37 w mod 100) + 1, the same map for (i, j) and (j, i)
reweight(A, k) = (B = copy(A); B.nzval .= mod.(round.(B.nzval) .* (37 + 2k), 100) .+ 1; B)
checksum(D) = (mapreduce(x -> isfinite(x) ? Float64(x) : 0.0, +, D), count(!isfinite, D))

for name in GRAPHS
    try
        A = read_mtx(joinpath(MTX, name * ".mtx")); n = size(A, 1)
        Bs = [reweight(A, k) for k in 1:REPS]
        oneshot(B) = (D = S.apsp_gpu(B); CUDA.synchronize(); D)
        free(D) = CUDA.unsafe_free!(D)
        free(oneshot(Bs[1])); free(oneshot(Bs[1]))
        t1 = [begin GC.gc(false); t = @elapsed(D = oneshot(B)); free(D); t end for B in Bs]
        tplan = @elapsed(plan = S.apsp_plan(A)); CUDA.synchronize()
        free(S.apsp_gpu(plan, Bs[1])); free(S.apsp_gpu(plan, Bs[1]))      # (the first call records the CUDA graph)
        tp = Float64[]; ok = true

        for B in Bs
            GC.gc(false)
            t = @elapsed(D = (X = S.apsp_gpu(plan, B); CUDA.synchronize(); X)); push!(tp, t)
            ok &= checksum(D) == (R = oneshot(B); c = checksum(R); free(R); c)
            free(D)
        end

        S.free_plan!(plan)
        @printf("%-6s %-16s n=%7d  one-shot best %7.1f median %7.1f ms | plan once %7.1f ms | reuse best %7.1f median %7.1f ms | %.2f× | %s\n",
            TAG, name, n, 1e3minimum(t1), 1e3median(t1), 1e3tplan, 1e3minimum(tp), 1e3median(tp), minimum(t1) / minimum(tp), ok ? "results match" : "MISMATCH")
        open(OUT, "a") do io
            println(io, "{\"tag\":\"$TAG\",\"graph\":\"$name\",\"n\":$n,\"oneshot\":[", join(t1, ","), "],\"plan\":$tplan,\"reuse\":[", join(tp, ","), "],\"ok\":$ok}")
        end
    catch e
        println(TAG, " ", name, ": failed: ", first(sprint(showerror, e), 300))
    end

    GC.gc(); CUDA.reclaim()
end
