# GEMM variants on the closure's real shapes (min-plus, Float32), in G multiply-adds/s.
#
#   julia --project=. bench/gemm_shapes.jl
#
# Every variant of one shape is timed in short bursts, round-robin over several rounds, and keeps its
# best burst: the laptop's clock drifts under its power cap, so back-to-back timing of one variant
# after another is not comparable. Each variant's result is checked against kernel v2 (" WRONG" marks a
# difference). VARIANTS (a Julia expression) is a vector of (name, settings) pairs, settings a NamedTuple
# of GPUConfig fields (e.g. (gemm_kernel = 4,)); a name "t:BM,BN,BK,TM,TN" (or "u<v>:…") forces that
# tiling, and "old" / "new" are the tuner's picks among the plain tilings only (fixed widths, one slice,
# one launch: the candidates before fitted widths, split-K and remainder launches) and among all
# candidates. The default is old against new.
#
# Shapes are (m, n, k, kind): kind :acc (C ← C ⊕ A B), :ow (C ← A B), :sep (C ⊕= through an index view
# of C's columns, the solve's U separator update), :inplace (C ← C B in place, the solve's U₁₁* step)
# and :inplaceidx (C₁ ← C[:, [res; sep]] B in place through an index view, the solve's L step).
include(joinpath(@__DIR__, "..", "src", "SemiringGPU.jl"))
using .SemiringGPU, CUDA, Printf, Random
const S = SemiringGPU; const MP = S.Semiring.MinPlus()

const SHAPES = [
    # in place, one tile wide (L and U₁₁* steps of fronts up to 128 columns)
    (27000, 71, 1200, :inplaceidx), (27000, 80, 1200, :inplaceidx), (27000, 105, 249, :inplaceidx), (32400, 75, 315, :inplaceidx),
    (32400, 106, 324, :inplaceidx), (23133, 128, 900, :inplaceidx), (27000, 40, 400, :inplaceidx), (40421, 80, 3135, :inplaceidx),
    (40421, 114, 3199, :inplaceidx), (23133, 70, 1684, :inplaceidx), (27000, 15, 740, :inplaceidx),
    (27000, 73, 73, :inplace), (27000, 105, 105, :inplace), (27000, 128, 128, :inplace), (32400, 96, 96, :inplace), (27000, 13, 13, :inplace),
    # separator updates through an index view
    (27000, 144, 105, :sep), (27000, 109, 73, :sep), (27000, 1173, 340, :sep), (27000, 840, 440, :sep), (32400, 170, 143, :sep), (27000, 17, 40, :sep),
    (23133, 1792, 121, :sep), (23133, 1666, 106, :sep), (27000, 306, 96, :sep), (27000, 510, 87, :sep), (40421, 2462, 318, :sep),
    # plain (fronts wider than 128 columns), and the shapes of earlier versions of this benchmark
    (27000, 137, 966, :acc), (27000, 129, 875, :acc), (27000, 222, 778, :acc), (27000, 340, 1513, :acc), (27000, 253, 1399, :acc),
    (27000, 598, 598, :acc), (23133, 840, 840, :acc), (27000, 440, 1280, :acc), (15625, 163, 948, :acc), (27000, 1173, 340, :acc),
    (23133, 1806, 45, :acc), (23133, 1555, 74, :acc), (15625, 251, 251, :acc), (22500, 64, 64, :acc), (32400, 32, 120, :acc),
    (27000, 598, 598, :ow), (27000, 440, 440, :ow)]
const ROUNDS = parse(Int, get(ENV, "ROUNDS", "5"))
const ONLY = get(ENV, "KINDS", "")             # e.g. KINDS=inplace,sep: only those kinds

parse_tiling(s) = (v = parse.(Int, split(s, ",")); S.Tiling{v...}())
variants = eval(Meta.parse(get(ENV, "VARIANTS", "[(\"old\", (;)), (\"new\", (;))]")))
CPP = any(startswith(v[1], "cpp") for v in variants)
CPP && (include(joinpath(@__DIR__, "..", "cuda", "cuda_backend.jl")); @eval using .SemiringCUDA)

colv(P, idx) = SubArray(P, (Base.Slice(axes(P, 1)), idx))

# the operands of a shape: (C, A, B, overwrite, inplace, P0, P): C (and A, in place) are views into P,
# whose initial value is P0
function operands(m, n, k, kind)
    rnd(r, c) = round.(CUDA.rand(Float32, r, c) .* 100)
    if kind === :inplace
        P0 = rnd(m, n + 3); B = rnd(n, n)
        return (; P0, n, k = n, B, ow = true, inplace = true, view = (P -> (view(P, :, 1:n), view(P, :, 1:n))))
    elseif kind === :inplaceidx
        P0 = rnd(m, k + 9); cols = CuVector([collect(1:n); collect((n + 10):(k + 9))]); B = rnd(k, n)
        return (; P0, n, k, B, ow = true, inplace = true, view = (P -> (view(P, :, 1:n), colv(P, cols))))
    elseif kind === :sep
        P0 = rnd(m, n + 64); idx = CuVector(sort(randperm(n + 64)[1:n])); A = rnd(m, k); B = rnd(k, n)
        return (; P0, n, k, B, ow = false, inplace = false, view = (P -> (colv(P, idx), A)))
    else
        P0 = rnd(m, n); A = rnd(m, k); B = rnd(k, n)
        return (; P0, n, k, B, ow = kind === :ow, inplace = false, view = (P -> (P, A)))
    end
end

# the tuner's pick on these operands among all candidates (new) or the plain tilings only (old)
function tuned(s, P, op, all::Bool)
    C, A = op.view(P)
    m, n = size(C); k = size(A, 2)
    v7 = !isnothing(S.gemm_layout(C)) && !isnothing(S.gemm_layout(A))
    cands = S.gemm_candidates(n, op.inplace, S.min3_ok(s, Float32), v7, v7 && S.pair_ok(s, Float32))
    all || filter!(S.classic, cands)
    more = all && v7 ? (cs -> S.gemm_refinements(cs, m, n, k, S.splitk_ok(s, Float32), device_profile().nsm)) : nothing
    return S.tune_gemm(s, C, A, op.B, op.ow, cands, more; op.inplace)
end

function runner(name, op, P)
    if startswith(name, "cpp")
        tl = parse(Int, name[4:end])
        return P -> ((C, A) = op.view(P); Base.invokelatest(sgemx_cuda!, MP, C, A, op.B; tiling = tl, overwrite = op.ow))
    elseif occursin(r"^(t|u\d*):", name)      # t: / u4: / u5: a forced tiling (the kernel version is set in setup)
        tl = parse_tiling(split(name, ":")[2])
        return P -> ((C, A) = op.view(P); S.sgemx_gpu!(MP, C, A, op.B; tiling = tl, overwrite = op.ow, inplace = op.inplace))
    elseif name in ("old", "new")
        c = tuned(MP, copy(P), op, name == "new")
        return (P -> ((C, A) = op.view(P); S.launch!(MP, C, A, op.B, c, Val(op.ow); op.inplace)), c)
    else
        return P -> ((C, A) = op.view(P); S.sgemx_gpu!(MP, C, A, op.B; overwrite = op.ow, inplace = op.inplace))
    end
end

function burst(f, P, P0)
    copy!(P, P0); f(P); CUDA.synchronize()
    reps = 0; t0 = time(); tg = 0.0
    while time() - t0 < 0.03 || reps < 2
        copy!(P, P0); CUDA.synchronize()
        tg += CUDA.@elapsed f(P); reps += 1
    end
    return tg / reps
end

cfgname(c) = c isa S.GemmConfig ? "v$(c.version) $(c.bm)×$(c.bn > 0 ? c.bn : "fit$(-c.bn)")×$(c.bk) 8×$(c.tn)$(c.lm == 8 ? " lm8" : "")$(c.split > 1 ? " split$(c.split)" : "")$(c.rem > 0 ? " +rem" : "")" : ""

p = measure!(device_profile()); println(p)
println("peak min-plus ", round(p.minplus / 1e9), " G/s; columns: G ops/s (best burst of $ROUNDS rounds)")
Random.seed!(1)
for (m, n, k, kind) in SHAPES
    isempty(ONLY) || string(kind) in split(ONLY, ",") || continue
    op = operands(m, n, k, kind)
    P = copy(op.P0)
    # reference: kernel v2 (as a fresh, not in place, output for the in-place shapes)
    R = copy(op.P0); C, A = op.view(R)
    if op.inplace
        W = similar(C); S.launch!(MP, W, A, op.B, S.GemmConfig(2, S.TILING_LARGE), Val(true)); copyto!(C, W)
    else
        S.launch!(MP, C, A, op.B, S.GemmConfig(2, S.TILING_LARGE), Val(op.ow))
    end
    R = Array(R)
    fs = map(variants) do (name, setup)
        r = with_config(() -> runner(name, op, P); setup...)
        r isa Tuple ? (name, setup, r[1], r[2]) : (name, setup, r, nothing)
    end
    best = fill(Inf, length(fs)); ok = trues(length(fs))
    for (i, (name, setup, f, _)) in enumerate(fs)
        copy!(P, op.P0); with_config(() -> f(P); setup...); ok[i] = Array(P) == R
    end
    sleep(0.05)
    for _ in 1:ROUNDS, (i, (name, setup, f, _)) in enumerate(fs)
        best[i] = min(best[i], with_config(() -> burst(f, P, op.P0); setup...))
    end
    @printf("%6d×%4d×%4d %-10s", m, n, k, kind)
    for (i, (name, _, _, c)) in enumerate(fs); @printf("| %s %5.0f%s %s ", name, m * n * k / best[i] / 1e9, ok[i] ? "" : " WRONG", cfgname(c)); end
    length(fs) == 2 && @printf("| %.2f×", best[1] / best[2])
    println()
end
