# Full closure on the GPU (closure_gpu!, elimination coordinates) vs ROME's numbers.
#   julia --project=. -t auto bench/bench_closure.jl graph [large]
include(joinpath(@__DIR__, "bench_apsp.jl"))

name = ARGS[1]; large = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 256
solve_large = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 8192
length(ARGS) > 3 && (SemiringGPU.DOWN_VARIANT[] = Symbol(ARGS[4]))
use_ops = length(ARGS) > 4 && ARGS[5] == "ops"
# LATENCY_FIX=0 turns off the persistent L sweep and the warp-cooperative path walk
SemiringGPU.PERSISTENT[] = SemiringGPU.PATH_WARP[] = get(ENV, "LATENCY_FIX", "1") == "1"
A = read_mtx(joinpath(MTX, name * ".mtx")); n = size(A, 1)
# warm up all kernels on a small graph
let A0 = read_mtx(joinpath(MTX, "grid3d-25.mtx"))[1:2000, 1:2000]
    F0 = ChordalSLU(MinPlus(), A0); copyto!(F0, A0); P0 = FactorPlan(F0; large = 64, graph = false, nstreams = 8); factorize!(P0)
    closure_gpu(GPUSLU(P0; large = 64)); CUDA.reclaim()
end
ChordalSLU(MinPlus(), A)
const REPS = parse(Int, get(ENV, "REPS", "3"))
t_sym = minimum(@elapsed(ChordalSLU(MinPlus(), A)) for _ in 1:REPS)
F = ChordalSLU(MinPlus(), A)
numeric() = (copyto!(F, A); P = FactorPlan(F; large, graph = false, nstreams = 8); factorize!(P); G = GPUSLU(P; large = solve_large); use_ops && precompute_ops!(G); CUDA.synchronize(); G)
numeric()                                  # compile for this graph's shapes (Julia JIT; ROME is compiled ahead of time)
t_num = minimum(@elapsed(numeric()) for _ in 1:REPS)
G = numeric()
D = CuMatrix{T}(undef, n, n); M = CuMatrix{T}(undef, n, G.maxna)
closure_gpu!(D, G; M); CUDA.synchronize()                  # warm (this graph's shapes)
t_clo = minimum(@elapsed((closure_gpu!(D, G; M); CUDA.synchronize())) for _ in 1:REPS)
tm = Dict{Symbol, Float64}(); get(ENV, "PHASES", "1") == "1" && closure_gpu!(D, G; M, timer = tm)
@printf("%-11s %-4s %-7s n=%6d | symbolic %.3f  numeric %.3f  closure %.3f  → total %.3f s | phases (synced): ", name, use_ops ? "ops" : "-", SemiringGPU.DOWN_VARIANT[], n, t_sym, t_num, t_clo, t_sym + t_num + t_clo)
for (k, v) in sort(collect(tm); by = last, rev = true); @printf("%s %.0f  ", k, 1e3v); end
println("ms")
if name == "grid3d-25"
    H = Array(D); p = Array(G.rperm); C = similar(H); C[p, p] = H
    open(joinpath(@__DIR__, "..", "external", "ours_rows_grid3d-25.txt"), "w") do io
        for i in 1:3; println(io, join(Int.(C[i, :]), " ")); end
    end
end
