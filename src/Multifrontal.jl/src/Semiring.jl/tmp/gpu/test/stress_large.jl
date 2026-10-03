# Large randomized stress tests (n up to ~30k) against an independent single-source oracle, plus
# a determinism check (the same run repeated must be bit-identical: the persistent sweep, the
# atomic scatter and the multi-stream factorization are where races would hide).
#
#   julia --project=. -t auto test/stress_large.jl [ncases] [seed]
#
# Oracle: a binary-heap Dijkstra written here (min-plus, nonnegative weights) and its widest-path
# variant (max-min); it shares no code with the solver.

# SKIP_INT=1 leaves out integer element types (upstream min-plus Int32 saturation was removed in CliqueTrees fb01ee7)
const SKIP_INT = get(ENV, "SKIP_INT", "0") == "1"
const REJECTED = Ref(0)
semiring_ok(s, T) = try SemiringGPU.check_semiring(s, T); true catch e; e isa ArgumentError || rethrow(); false end
haskey(ENV, "SEMIRINGGPU_TUNE_FILE") || (ENV["SEMIRINGGPU_TUNE_FILE"] = tempname())
include(joinpath(@__DIR__, "..", "src", "SemiringGPU.jl"))
using .SemiringGPU
using .SemiringGPU: Semiring
using .Semiring: MinPlus, MaxMin, ChordalSLU, mlu, szero, sone
using CUDA, SparseArrays, Random, Printf

# ===== oracle =====

# label-setting from source r: minimize Σ (min-plus) or maximize min (max-min, widest path)
function dijkstra(A::SparseMatrixCSC{T}, r::Int, widest::Bool) where {T}
    n = size(A, 1)
    better(a, b) = widest ? a > b : a < b
    d = fill(widest ? typemin(T) : typemax(T), n)
    d[r] = widest ? typemax(T) : zero(T)
    done = falses(n)
    heap = [(d[r], r)]                                 # binary heap of (label, vertex), lazy deletion
    up!(h, i) = while i > 1 && better(h[i][1], h[i ÷ 2][1]); h[i], h[i ÷ 2] = h[i ÷ 2], h[i]; i ÷= 2; end

    function pop!(h)
        top = h[1]; h[1] = h[end]; Base.pop!(h); i = 1
        while true
            l = 2i; c = i
            l <= length(h) && better(h[l][1], h[c][1]) && (c = l)
            l + 1 <= length(h) && better(h[l + 1][1], h[c][1]) && (c = l + 1)
            c == i && break
            h[i], h[c] = h[c], h[i]; i = c
        end
        return top
    end

    rows = rowvals(A); vals = nonzeros(A)              # A[i, j] = arc i → j, stored by column: use the transpose
    At = sparse(transpose(A))
    rows = rowvals(At); vals = nonzeros(At)

    while !isempty(heap)
        (du, u) = pop!(heap)
        done[u] && continue
        done[u] = true

        for p in nzrange(At, u)                        # arcs u → v
            v = rows[p]; w = vals[p]
            cand = widest ? min(du, w) : du + w
            if better(cand, d[v])
                d[v] = cand
                push!(heap, (cand, v)); up!(heap, length(heap))
            end
        end
    end

    return d
end

# ===== graphs (undirected, nonnegative integer weights) =====

function graph(rng, ::Type{T}, kind, n) where {T}
    I = Int[]; J = Int[]
    if kind == :grid2
        a = isqrt(n); n = a * a
        for j in 1:a, i in 1:a; v = i + (j - 1) * a; i < a && (push!(I, v); push!(J, v + 1)); j < a && (push!(I, v); push!(J, v + a)); end
    elseif kind == :grid3
        a = round(Int, cbrt(n)); n = a^3
        for l in 1:a, j in 1:a, i in 1:a
            v = i + (j - 1) * a + (l - 1) * a^2
            i < a && (push!(I, v); push!(J, v + 1)); j < a && (push!(I, v); push!(J, v + a)); l < a && (push!(I, v); push!(J, v + a^2))
        end
    elseif kind == :er
        for _ in 1:(2n); u = rand(rng, 1:n); v = rand(rng, 1:n); u != v && (push!(I, u); push!(J, v)); end
    elseif kind == :tree
        for v in 2:n; push!(I, v); push!(J, rand(rng, max(1, v - 50):(v - 1))); end
    elseif kind == :pieces
        for _ in 1:(n ÷ 2); u = rand(rng, 1:n); v = clamp(u + rand(rng, -20:20), 1, n); u != v && (push!(I, u); push!(J, v)); end
    end
    W = T.(rand(rng, 0:100, length(I)))
    return sparse(vcat(I, J), vcat(J, I), vcat(W, W), n, n, min)
end

# ===== cases =====

function run_case(seed)
    rng = Xoshiro(seed)
    s, T = rand(rng, filter(x -> !(SKIP_INT && x[2] <: Integer), [(MinPlus(), Float32), (MinPlus(), Float64), (MinPlus(), Int32), (MaxMin(), Float32)]))
    kind = rand(rng, (:grid2, :grid3, :er, :tree, :pieces))
    A = graph(rng, T, kind, rand(rng, (2_000, 5_000, 12_000, 30_000)))
    n = size(A, 1)
    widest = s isa MaxMin

    settings = (gemm_kernel = rand(rng, (0, 2, 4, 6, 7, 8)), gemm_tune = rand(rng, Bool), skip_fill = rand(rng, Bool),
                merge = rand(rng, (1, 2, 8, 32, 128)), layered_min_rows = rand(rng, (1, 4096)), layer_size = rand(rng, (0, 1, 4, 32)),
                factor_merge = rand(rng, (1, 8, 128)), fused_front = rand(rng, Bool), direct_assembly = rand(rng, Bool))
    flarge = rand(rng, (16, 64, 256, typemax(Int)))
    slarge = rand(rng, (64, 2048, typemax(Int)))
    F = ChordalSLU(s, A); copyto!(F, A)

    if !semiring_ok(s, T)                       # must be rejected with an ArgumentError, not answered
        rejected = try FactorPlan(F; large = flarge, graph = false); false catch e; e isa ArgumentError || rethrow(); true end
        rejected ? (REJECTED[] += 1) : @printf("  FAIL seed=%d: accepted an unsupported %s %s\n", seed, nameof(typeof(s)), T)
        return rejected
    end

    return with_config(; settings...) do
    P = FactorPlan(F; large = flarge, graph = false, nstreams = rand(rng, (1, 8))); factorize!(P)
    G = GPUSLU(P; large = slarge)
    rand(rng, Bool) && precompute_ops!(G)
    tag = @sprintf("%s %s %s n=%d flarge=%s slarge=%s %s", nameof(typeof(s)), T, kind, n, flarge, slarge,
        join(("$k=$(Int(v))" for (k, v) in pairs(settings)), " "))

    k = rand(rng, (1, 63, 129, 300))
    src = rand(rng, 1:n, k)
    X = Array(sssp_gpu!(CuMatrix{T}(undef, k, n), G, CuVector(src)))
    nbad = 0

    for t in rand(rng, 1:k, min(k, 8))                 # check a sample of rows against the oracle
        d = dijkstra(A, src[t], widest)
        nbad += count(.!isequal.(X[t, :], d))
    end

    # determinism: the same solve again, bit for bit
    X2 = Array(sssp_gpu!(CuMatrix{T}(undef, k, n), G, CuVector(src)))
    ndiff = count(.!isequal.(X, X2))

    # closure rows (when it fits)
    nclo = 0
    if n <= 12_000
        C = Array(closure_gpu(G)); p = Array(G.rperm); pos = invperm(p)
        for r in rand(rng, 1:n, 4)
            d = dijkstra(A, r, widest)
            nclo += count(.!isequal.(C[pos[r], pos], d))
        end
    end

    ok = nbad == 0 && ndiff == 0 && nclo == 0
    ok || @printf("  FAIL seed=%d %s: oracle mismatches %d, nondeterministic %d, closure mismatches %d\n", seed, tag, nbad, ndiff, nclo)
    return ok
    end
end

ncases = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 40
seed0 = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 1
nfail = 0; t0 = time()

for c in 0:(ncases - 1)
    try
        global nfail += !run_case(seed0 + c)
    catch e
        global nfail += 1
        println("  ERROR seed=$(seed0 + c): ", sprint(showerror, e)[1:min(end, 300)])
    end
    GC.gc(); CUDA.reclaim()
end

@printf("stress_large: %d cases, %d failures, %d rejected as unsupported (%.0f s)\n", ncases, nfail, REJECTED[], time() - t0)
