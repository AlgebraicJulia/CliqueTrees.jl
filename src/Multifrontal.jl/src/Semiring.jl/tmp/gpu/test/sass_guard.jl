# Guard against silent 2-10× regressions in the hot kernels: no device function calls (CALL), no local-memory
# spills (STL/LDL), and min-plus Float32 must use the FMNMX instruction (not FSETP + FSEL; FMNMX3, the 3-input
# min of kernel v8 on sm_100, counts too: a split-K v8 kernel has no other min).
include(joinpath(@__DIR__, "..", "src", "SemiringGPU.jl"))
include(joinpath(@__DIR__, "graphs.jl"))
using .SemiringGPU
using .SemiringGPU: Semiring
using .Semiring: MinPlus, ChordalSLU
using CUDA

function sass(f)
    io = IOBuffer()
    CUDA.@device_code_sass io = io f()
    return String(take!(io))
end

count_op(s, op) = count(l -> occursin(Regex("\\b" * op * "\\b"), l), split(s, '\n'))

checks = Pair{String, Function}[]
s = MinPlus()
A = CUDA.ones(Float32, 512, 512)

for (name, tiling) in (("gemm v2 large", SemiringGPU.TILING_LARGE), ("gemm v2 small", SemiringGPU.TILING_SMALL),
                       ("gemm v2 n16", SemiringGPU.TILING_N16), ("gemm v2 n32", SemiringGPU.TILING_N32))
    push!(checks, name => () -> sgemx_gpu!(s, A, A, A; tiling))
end

G = let A = grid3(10, Float32), F = ChordalSLU(s, A)
    copyto!(F, A); P = FactorPlan(F; large = 16, graph = false); factorize!(P); GPUSLU(P; large = 16)
end
n = G.n
push!(checks, "closure (sweeps, path walk, dense path)" => () -> closure_gpu(G))
# the L sweep below the top as the slot-cached layered walk (layered.jl), and the plain layered walk
push!(checks, "closure, layered slot sweep" => () -> with_config(() -> closure_gpu(G); layered_min_rows = 1))
push!(checks, "closure, layered walk" => () -> with_config(() -> closure_gpu(G); layered_min_rows = 1, layer_cache = false))
# the batched factorization and operator kernels (sgetrf.jl), compiled fresh for max-plus (min-plus is compiled by G
# above); without front merges, whose index-map kernels are not part of this check
push!(checks, "factorization (batched top), operators" => () -> with_config(; merge = 1, factor_merge = 1) do
    A = grid3(10, Float32); F = ChordalSLU(Semiring.MaxPlus(), A)
    copyto!(F, A); P = FactorPlan(F; large = 16, graph = false); factorize!(P); precompute_ops!(GPUSLU(P; large = 16)); CUDA.synchronize()
end)

# kernel v7 (CUTLASS structure), the tilings the tuner may pick, strided and through an index view
if SemiringGPU.v7_ok(Float32, 128, 64, 8, 8)
    for (BM, BN, BK, TN) in ((128, 128, 8, 8), (128, 64, 8, 8), (64, 64, 8, 8), (64, 128, 8, 8), (128, 64, 16, 4), (64, 64, 8, 4))
        push!(checks, "gemm v7 $(BM)×$(BN)×$(BK) 8×$TN" => () -> SemiringGPU.launch7!(s, A, A, A, Val(BM), Val(BN), Val(BK), Val(false); tn = TN))
    end
    idx = CuVector(collect(1:2:512))
    push!(checks, "gemm v7 128×64 indexed" => () -> SemiringGPU.launch7!(s, view(A, :, 1:256), SubArray(A, (Base.Slice(axes(A, 1)), idx)), view(A, 1:256, 1:256), Val(128), Val(64), Val(8), Val(false)))
    # warps of 64 rows (lm = 8): tile widths in steps of 16 (8 × 4 thread tiles) or 32 (8 × 8), as the
    # fitted widths, the remainder launches and the in-place L step (index view of A, overwrite) use them
    for (BM, BN, BK, TN) in ((64, 80, 8, 4), (128, 48, 8, 4), (64, 16, 8, 4), (64, 96, 8, 8), (128, 160, 8, 8))
        push!(checks, "gemm v7 $(BM)×$(BN)×$(BK) 8×$TN lm8" => () -> SemiringGPU.launch7!(s, A, A, A, Val(BM), Val(BN), Val(BK), Val(false); tn = TN, lm = 8))
    end
    push!(checks, "gemm v7 64×112 lm8 in place, indexed" => () -> SemiringGPU.launch7!(s, view(A, :, 1:112), SubArray(A, (Base.Slice(axes(A, 1)), idx)), view(A, 1:256, 1:112), Val(64), Val(112), Val(8), Val(true); tn = 4, lm = 8))
    # split-K: the slices add their partial tiles into C with the atomic ⊕ (overwrite: C filled first)
    for (BM, BN, BK, TN, LM) in ((128, 128, 8, 8, 4), (128, 64, 8, 8, 4), (128, 64, 16, 4, 4), (64, 80, 8, 4, 8))
        push!(checks, "gemm v7 $(BM)×$(BN)×$(BK) 8×$TN lm$LM split 3" => () -> SemiringGPU.launch7!(s, A, A, A, Val(BM), Val(BN), Val(BK), Val(false); tn = TN, lm = LM, split = 3))
    end
    push!(checks, "gemm v7 128×64 split 3, indexed C" => () -> SemiringGPU.launch7!(s, SubArray(A, (Base.Slice(axes(A, 1)), idx)), view(A, :, 1:256), view(A, 1:256, 1:256), Val(128), Val(64), Val(8), Val(false); split = 3))
    if SemiringGPU.pair_ok(s, Float32)        # kernel v8 (sm_100)
        for (BM, BN, BK, TN, LM, SP) in ((128, 128, 8, 8, 4, 1), (128, 64, 8, 8, 4, 1), (64, 80, 8, 4, 8, 1), (128, 64, 8, 8, 4, 3), (64, 96, 8, 8, 8, 2))
            push!(checks, "gemm v8 $(BM)×$(BN)×$(BK) 8×$TN lm$LM split $SP" => () -> SemiringGPU.launch7!(s, A, A, A, Val(BM), Val(BN), Val(BK), Val(false); tn = TN, lm = LM, split = SP, pair = true))
        end
    end
end

bad = 0
for (name, f) in checks
    code = sass(f)
    calls = count_op(code, "CALL.REL.NOINC") + count_op(code, "CALL.ABS.NOINC")
    spills = count_op(code, "STL") + count_op(code, "LDL")
    fmnmx = count_op(code, "FMNMX") + count_op(code, "FMNMX3")
    fsel = count_op(code, "FSEL")
    ok = calls == 0 && spills == 0 && fmnmx > 0
    global bad += !ok
    println(rpad(name, 42), ok ? "ok" : "CHECK", "  (CALL $calls, STL/LDL $spills, FMNMX(3) $fmnmx, FSEL $fsel)")
end
println(bad == 0 ? "sass guard: all ok" : "sass guard: $bad suspicious")
