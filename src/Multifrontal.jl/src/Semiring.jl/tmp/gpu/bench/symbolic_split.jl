# The CPU steps of the symbolic phase and of filling the factor (CliqueTrees: ssymbolic, the ChordalSLU
# storage, copyto!), each timed alone by replaying ssymbolic's body (best of REPS; GC time and allocation).
#   julia --project=. -t auto bench/symbolic_split.jl graph ...
include(joinpath(@__DIR__, "bench_apsp.jl"))
using Printf
const SR = SemiringGPU.Semiring
const CT = SemiringGPU.CliqueTrees
const REPS = parse(Int, get(ENV, "REPS", "3"))

function steps(A::SparseMatrixCSC{T, I}) where {T, I}
    alg = SR.DEFAULT_ELIMINATION_ALGORITHM
    t = Pair{String, Tuple{Float64, Float64, Float64}}[]
    tm(f, k) = (s = @timed f(); push!(t, k => (s.time, s.gctime, s.bytes / 2^20)); s.value)
    n = convert(I, size(A, 2))
    scc = tm(() -> SR.sccs(A), "strongly connected components")
    Bptr = SR.pointers(scc); invp = SR.targets(scc); nBptr = SR.nv(scc)
    B = tm(() -> SR.permute(A, invp, invp), "permute (components)")
    graph = tm(() -> CT.BipartiteGraph(B), "graph of A")
    # scliquetree = per component: the ordering, then the elimination tree and supernodes
    sub = tm(() -> SR.symmetric(SR.subgraph(graph, one(I), Bptr[2] - one(I)), 'N'), "component subgraph (first)")
    tm(() -> CT.permutation(sub; alg), "ordering ($(nameof(typeof(alg))), first component)")
    perm, tree = tm(() -> SR.scliquetree(B, Bptr, nBptr; alg), "clique tree (ordering + elimination tree + supernodes)")
    S = tm(() -> SR.ChordalSymbolic(tree), "symbolic structure (ChordalSymbolic)")
    tm(() -> SR.sccfronts(S, Bptr, nBptr), "component fronts")
    C = tm(() -> SR.permute(B, perm, perm), "permute (ordering)")
    tm(() -> SR.soffd(C, Bptr, nBptr), "off-diagonal coupling")
    tm(() -> SR.ssymbolic(A; alg), "ssymbolic, whole")
    Q, SS = SR.ssymbolic(A; alg)
    F = tm(() -> SR.ChordalSLU(SR.MinPlus(), T, SS, Q.perm, Q.invp, Q.perm, Q.invp), "factor storage")
    tm(() -> copyto!(F, A), "copyto! (entries of A into the factor)")
    nnzf = length(F.LDval) + length(F.LLval) + length(F.ULval) + length(F.UDval)
    return t, nBptr, nnzf
end

for name in ARGS
    A = read_mtx(joinpath(MTX, name * ".mtx"))
    steps(A)
    runs = [steps(A) for _ in 1:REPS]
    t = runs[1][1]
    best = [k => (minimum(r[1][i][2][1] for r in runs), minimum(r[1][i][2][2] for r in runs), runs[1][1][i][2][3]) for (i, (k, _)) in enumerate(t)]
    @printf("\n%s  n = %d, nnz(A) = %d, components = %d, factor entries = %.1fM\n", name, size(A, 1), nnz(A), runs[1][2], runs[1][3] / 1e6)
    for (k, (s, g, mb)) in best
        @printf("  %-58s %8.1f ms%s  %7.1f MiB\n", k, 1e3s, g > 5e-4 ? @sprintf("  [GC %.1f]", 1e3g) : "", mb)
    end
end
