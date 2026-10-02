# Per-kernel GPU time of one closure (CUDA.@profile: CUPTI activity records, no hardware counters).
# Solver settings: environment variables SEMIRINGGPU_<SETTING> (see GPUConfig).
#   julia --project=. bench/prof_closure.jl graph [large] [solve_large]
include(joinpath(@__DIR__, "bench_apsp.jl"))
using CUDA, Printf
name = ARGS[1]; large = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 256
solve_large = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 8192
if get(ENV, "CAS", "0") == "1"           # A/B: the compare-and-swap ⊕ instead of native reductions
    @eval SemiringGPU atomic_kind(::Semiring.MinPlus, ::Val{:N}, ::Type{<:Union{Float32, Float64, Int32, Int64}}) = Val(:cas)
end
A = read_mtx(joinpath(MTX, name * ".mtx")); n = size(A, 1)
F = ChordalSLU(MinPlus(), A); copyto!(F, A)
P = FactorPlan(F; large, graph = false, nstreams = 8); factorize!(P)
G = GPUSLU(P; large = solve_large); precompute_ops!(G)
D = CuMatrix{T}(undef, n, n); M = CuMatrix{T}(undef, n, G.maxna)
for _ in 1:3; closure_gpu!(D, G; M); end; CUDA.synchronize()
t = minimum(@elapsed((closure_gpu!(D, G; M); CUDA.synchronize())) for _ in 1:5)
@printf("%s n=%d closure %.2f ms (wall, min of 5)\n", name, n, 1e3t)
show(stdout, CUDA.@profile(trace = false, closure_gpu!(D, G; M)))
println()
