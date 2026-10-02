# Randomized stress tests: every GPU path against an independent oracle.
#
#   julia --project=. -t auto test/stress.jl [ncases] [seed]
#
# The oracle is a dense Floyd–Warshall–Kleene closure written here from
# splus / sprod / sstar only; it shares no code with the solver (no
# symbolic phase, no factorization, no kernels). Each case draws a graph
# family, weights (including zeros, negative arcs, negative cycles,
# duplicates and self-loops), a semiring and element type, and a random
# configuration of every GPU switch, then checks the queries (rmul_gpu!,
# sssp_gpu!, SSSPPlan), the closure (closure_gpu!) and the factorization
# (CPU mlu, hybrid FactorPlan with graph replay) against the oracle.
# Failures print the case seed; rerun one case with `stress.jl 1 <seed>`.

include(joinpath(@__DIR__, "..", "src", "SemiringGPU.jl"))
using .SemiringGPU
using .SemiringGPU: Semiring
using .Semiring: MinPlus, MaxPlus, MaxMin, PlusProd, ChordalSLU, mlu, splus, sprod, sstar, szero, sone
using CUDA, SparseArrays, LinearAlgebra, Random, Printf

# ===== oracle =====

# A* = I ⊕ A⁺ by Kleene's algorithm (out of place at each pivot, so the
# update order does not matter)
function oracle(s, A::SparseMatrixCSC{T}) where {T}
    n = size(A, 1)
    z = szero(s, T, Val(:N))
    D = fill(z, n, n)

    for (i, j, v) in zip(findnz(A)...)
        D[i, j] = splus(s, D[i, j], v, Val(:N))
    end

    E = similar(D)

    for k in 1:n
        t = sstar(s, D[k, k])

        for j in 1:n
            tkj = sprod(s, t, D[k, j], Val(:N), Val(:N))

            for i in 1:n
                E[i, j] = splus(s, D[i, j], sprod(s, D[i, k], tkj, Val(:N), Val(:N)), Val(:N))
            end
        end

        D, E = E, D
    end

    for i in 1:n
        D[i, i] = splus(s, D[i, i], sone(s, T, Val(:N)), Val(:N))
    end

    return D
end

# X = B ⊗ D in the semiring (dense)
function smul(s, B::Matrix{T}, D::Matrix{T}) where {T}
    X = fill(szero(s, T, Val(:N)), size(B, 1), size(D, 2))

    for j in axes(D, 2), k in axes(B, 2), i in axes(B, 1)
        X[i, j] = splus(s, X[i, j], sprod(s, B[i, k], D[k, j], Val(:N), Val(:N)), Val(:N))
    end

    return X
end

# ===== random graphs =====

function edges(rng, kind, n)
    I = Int[]; J = Int[]
    add(i, j) = (push!(I, i); push!(J, j))

    if kind == :er
        d = rand(rng, (1.5, 3.0, 6.0))
        for _ in 1:round(Int, d * n / 2); add(rand(rng, 1:n), rand(rng, 1:n)); end
    elseif kind == :grid
        a = max(1, isqrt(n)); n = a * a
        for j in 1:a, i in 1:a
            v = i + (j - 1) * a
            i < a && add(v, v + 1); j < a && add(v, v + a)
        end
    elseif kind == :tree
        for v in 2:n; add(v, rand(rng, 1:(v - 1))); end
    elseif kind == :path
        for v in 2:n; add(v - 1, v); end
    elseif kind == :star
        for v in 2:n; add(1, v); end
    elseif kind == :dense
        for i in 1:n, j in (i + 1):n; rand(rng) < 0.5 && add(i, j); end
    elseif kind == :pieces                                  # disconnected, with isolated vertices
        for _ in 1:n; u = rand(rng, 1:n); v = rand(rng, max(1, u - 5):min(n, u + 5)); rand(rng) < 0.6 && add(u, v); end
    end

    return I, J, n
end

const KINDS = (:er, :grid, :tree, :path, :star, :dense, :pieces)

# symmetric pattern (the GPU path needs no coupling between strongly connected components)
function graph(rng, s, ::Type{T}, kind, n; negative = false, negcycle = false, selfloops = false, dups = false) where {T}
    I, J, n = edges(rng, kind, n)
    keep = I .!= J
    I = I[keep]; J = J[keep]
    m = length(I)
    w() = T <: Integer ? T(rand(rng, 0:100)) : T(rand(rng, 0:100))

    if s isa PlusProd
        W = T.(rand(rng, m)) ./ (4 * max(1, n))            # small, so that I − A is a nonsingular M-matrix
        A = sparse(vcat(I, J), vcat(J, I), vcat(W, W), n, n, +)
        return A
    end

    V = [w() for _ in 1:m]
    II = vcat(I, J); JJ = vcat(J, I); VV = vcat(V, V)

    if negative && s isa MinPlus                            # potential reweighting: no new negative cycles
        φ = T.(rand(rng, 0:200, n))
        VV = T[VV[e] + φ[II[e]] - φ[JJ[e]] for e in eachindex(VV)]
    end

    if negcycle && s isa MinPlus && m > 0                   # one arc pair with negative total weight
        e = rand(rng, 1:m)
        push!(II, I[e]); push!(JJ, J[e]); push!(VV, T(-150))
    end

    if dups && m > 0
        for _ in 1:max(1, m ÷ 5)
            e = rand(rng, 1:length(II)); push!(II, II[e]); push!(JJ, JJ[e]); push!(VV, w())
        end
    end

    if selfloops
        for v in rand(rng, 1:n, max(1, n ÷ 10))
            push!(II, v); push!(JJ, v); push!(VV, s isa MinPlus && negcycle ? T(-3) : w())
        end
    end

    combine(a, b) = splus(s, a, b, Val(:N))
    return sparse(II, JJ, VV, n, n, combine)
end

# ===== comparison =====

same(s, x, y) = s isa PlusProd ? isapprox(x, y; rtol = 1e-6, atol = 1e-9) : isequal(x, y)

function check!(fails, label, s, got, want, seed)
    if !same(s, got, want)
        d = findall(.!(s isa PlusProd ? isapprox.(got, want; rtol = 1e-6, atol = 1e-9) : isequal.(got, want)))
        i = first(d)
        push!(fails, (seed, label))
        @printf("  FAIL seed=%d %s: %d entries differ, first %s got %s want %s\n", seed, label, length(d), Tuple(i), got[i], want[i])
        return false
    end
    return true
end

# ===== one case =====

const SEMIRINGS = [(MinPlus(), Float32), (MinPlus(), Float64), (MinPlus(), Int32), (MaxPlus(), Float32),
                   (MaxMin(), Float32), (PlusProd(), Float64)]

function run_case(seed, fails)
    rng = Xoshiro(seed)
    s, T = rand(rng, SEMIRINGS)
    kind = rand(rng, KINDS)
    n = rand(rng, (1, 2, 3, 5, 8, 17, 33, 60, 100, 160, 250))
    neg = rand(rng) < 0.25; negcyc = rand(rng) < 0.1; loops = rand(rng) < 0.2; dups = rand(rng) < 0.2
    T <: Integer && (negcyc = false)                         # -∞ is not representable in Int32
    A = graph(rng, s, T, kind, n; negative = neg, negcycle = negcyc, selfloops = loops, dups)
    n = size(A, 1)
    D = oracle(s, A)

    # random switches
    SemiringGPU.GEMM_VERSION[] = rand(rng, (1, 2))
    SemiringGPU.PERSISTENT[] = rand(rng, Bool)
    SemiringGPU.PATH_WARP[] = rand(rng, Bool)
    flarge = rand(rng, (1, 8, 64, typemax(Int)))
    slarge = rand(rng, (1, 16, 2048, typemax(Int)))
    ops = rand(rng, Bool)
    hybrid = rand(rng, Bool)
    tag = @sprintf("%s %s %s n=%d neg=%d negcyc=%d loops=%d dups=%d gemm=%d pers=%d warp=%d flarge=%s slarge=%s ops=%d hybrid=%d",
        nameof(typeof(s)), T, kind, n, neg, negcyc, loops, dups, SemiringGPU.GEMM_VERSION[], SemiringGPU.PERSISTENT[],
        SemiringGPU.PATH_WARP[], flarge == typemax(Int) ? "∞" : flarge, slarge == typemax(Int) ? "∞" : slarge, ops, hybrid)
    ok = true

    # factorization: CPU, or hybrid with a replayed graph after new weights
    if hybrid
        F = ChordalSLU(s, A)
        P = FactorPlan(F; large = flarge, graph = true, nstreams = rand(rng, (1, 8)))
        copyto!(F, A); factorize!(P)
        copyto!(F, A); factorize!(P; download = true)        # graph replay
        G = GPUSLU(P; large = slarge)
        R = mlu(s, A)
        ok &= check!(fails, "hybrid factor == CPU factor | $tag", s, vcat(F.LDval, F.LLval, F.UDval, F.ULval), vcat(R.LDval, R.LLval, R.UDval, R.ULval), seed)
    else
        F = mlu(s, A)
        G = GPUSLU(F; large = slarge)
    end

    ops && precompute_ops!(G)
    k = rand(rng, (1, 2, 7, 31, 33, 64, 65, 100))
    src = rand(rng, 1:n, k)
    z = szero(s, T, Val(:N)); u = sone(s, T, Val(:N))

    # queries from unit sources
    B = fill(z, k, n); for t in 1:k; B[t, src[t]] = u; end
    want = D[src, :]
    ok &= check!(fails, "sssp_gpu! | $tag", s, Array(sssp_gpu!(CuMatrix{T}(undef, k, n), G, CuVector(src))), want, seed)
    Bg = CuArray(B); rmul_gpu!(Bg, G)
    ok &= check!(fails, "rmul_gpu! (unit) | $tag", s, Array(Bg), want, seed)

    # a dense right-hand side
    Bd = T <: Integer ? T.(rand(rng, 0:50, k, n)) : (s isa PlusProd ? T.(rand(rng, k, n)) : T.(rand(rng, 0:50, k, n)))
    Bg = CuArray(Bd); rmul_gpu!(Bg, G)
    ok &= check!(fails, "rmul_gpu! (dense) | $tag", s, Array(Bg), smul(s, Bd, D), seed)

    # CUDA-graph plan, two replays with new sources
    Pl = SSSPPlan(G, k)
    for _ in 1:2
        src2 = rand(rng, 1:n, k)
        ok &= check!(fails, "SSSPPlan | $tag", s, Array(Pl(src2)), D[src2, :], seed)
    end

    # closure
    C = Array(closure_gpu(G)); p = Array(G.rperm); H = similar(C); H[p, p] = C
    ok &= check!(fails, "closure_gpu! | $tag", s, H, D, seed)
    return ok
end

# ===== main =====

ncases = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 300
seed0 = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 1
fails = Tuple{Int, String}[]
t0 = time()

for c in 0:(ncases - 1)
    seed = seed0 + c
    try
        run_case(seed, fails)
    catch e
        push!(fails, (seed, "ERROR"))
        println("  ERROR seed=$seed: ", sprint(showerror, e)[1:min(end, 300)])
        get(ENV, "STRESS_BT", "0") == "1" && Base.display_error(stderr, e, catch_backtrace())
    end
    (c + 1) % 50 == 0 && @printf("%d cases, %d failures, %.0f s\n", c + 1, length(fails), time() - t0)
end

@printf("\nstress: %d cases, %d failures (%.0f s)\n", ncases, length(fails), time() - t0)
isempty(fails) || println("failing seeds: ", unique(first.(fails)))
