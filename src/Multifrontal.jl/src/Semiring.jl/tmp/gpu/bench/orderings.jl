# Orderings for the symbolic phase: time to order, the symbolic phase's total time, the fill of the
# factor (entries of L + U) and the closure's work n·nnz(L + U), per graph, for several orderings.
# Needs Metis.jl and AMD.jl loadable (e.g. a stacked environment: JULIA_LOAD_PATH="@:<env with them>:@stdlib").
#   julia --project=. bench/orderings.jl graph ...
using Metis, AMD
include(joinpath(@__DIR__, "bench_apsp.jl"))
include(joinpath(@__DIR__, "suite.jl"))
using Printf
const SR = SemiringGPU.Semiring
const CT = SemiringGPU.CliqueTrees
const ALGS = ["AMF (default)" => CT.AMF(), "AMD" => CT.AMD(), "METIS ND" => CT.ND(), "MF" => CT.MF()]
best(f, r = 3) = (f(); minimum(@elapsed(f()) for _ in 1:r))

for name in (isempty(ARGS) ? suite(:quick) : ARGS)
    A = read_mtx(joinpath(MTX, name * ".mtx")); n = size(A, 1)
    @printf("\n%-16s n = %6d  nnz = %8d  (%s)\n", name, n, nnz(A), graph_class(name))
    for (label, alg) in ALGS
        try
            t_ord = best(() -> CT.permutation(A; alg))
            t_sym = best(() -> SR.ssymbolic(A; alg))
            Q, S = SR.ssymbolic(A; alg)
            F = SR.ChordalSLU(SR.MinPlus(), Float32, S, Q.perm, Q.invp, Q.perm, Q.invp)
            fill = length(F.LDval) + length(F.LLval) + length(F.ULval)
            @printf("  %-14s ordering %8.1f ms   symbolic %8.1f ms   nnz(L+U) %8.2fM   closure work n·nnz %8.2e\n",
                label, 1e3t_ord, 1e3t_sym, fill / 1e6, n * fill)
        catch e
            @printf("  %-14s failed: %s\n", label, first(sprint(showerror, e), 120))
        end
    end
end
