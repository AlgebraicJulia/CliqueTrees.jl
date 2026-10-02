# GEMM variants on the closure's real shapes (min-plus, Float32), in G multiply-adds/s.
#
#   julia --project=. bench/gemm_shapes.jl
#
# Every variant of one shape is timed in short bursts, round-robin over several rounds, and keeps its
# best burst: the laptop's clock drifts under its power cap, so back-to-back timing of one variant
# after another is not comparable. Each variant's result is checked against the default kernel.
# VARIANTS (a Julia expression) is a vector of (name, settings) pairs, settings a NamedTuple of GPUConfig
# fields (e.g. (gemm_kernel = 4,)); a name "t:BM,BN,BK,TM,TN" (or "u<v>:…") forces that tiling.
include(joinpath(@__DIR__, "..", "src", "SemiringGPU.jl"))
using .SemiringGPU, CUDA, Printf
const S = SemiringGPU; const MP = S.Semiring.MinPlus()

const SHAPES = [(27000, 598, 598), (23133, 840, 840), (27000, 440, 1280), (15625, 163, 948), (27000, 137, 966),
                (27000, 1173, 340), (23133, 1806, 45), (23133, 1555, 74), (15625, 251, 251), (22500, 64, 64), (32400, 32, 120)]
const ROUNDS = parse(Int, get(ENV, "ROUNDS", "5"))

parse_tiling(s) = (v = parse.(Int, split(s, ",")); S.Tiling{v...}())
variants = eval(Meta.parse(get(ENV, "VARIANTS", "[(\"default\", (;))]")))
CPP = any(startswith(v[1], "cpp") for v in variants)
CPP && (include(joinpath(@__DIR__, "..", "cuda", "cuda_backend.jl")); @eval using .SemiringCUDA)

function runner(name)
    if startswith(name, "cpp")
        tl = parse(Int, name[4:end])
        return (C, A, B) -> Base.invokelatest(sgemx_cuda!, MP, C, A, B; tiling = tl)
    elseif occursin(r"^(t|u\d*):", name)      # t: / u4: / u5: a forced tiling (the kernel version is set in setup)
        tl = parse_tiling(split(name, ":")[2])
        return (C, A, B) -> S.sgemx_gpu!(MP, C, A, B; tiling = tl)
    else
        return (C, A, B) -> S.sgemx_gpu!(MP, C, A, B)
    end
end

function burst(f, C, C0, A, B)
    copy!(C, C0); f(C, A, B); CUDA.synchronize()
    reps = 0; t0 = time(); tg = 0.0
    while time() - t0 < 0.03 || reps < 2
        copy!(C, C0); CUDA.synchronize()
        tg += CUDA.@elapsed f(C, A, B); reps += 1
    end
    return tg / reps
end

p = measure!(device_profile()); println(p)
println("peak min-plus ", round(p.minplus / 1e9), " G/s; columns: G ops/s (best burst of $ROUNDS rounds)")
for (m, n, k) in SHAPES
    A = round.(CUDA.rand(Float32, m, k) .* 100); B = round.(CUDA.rand(Float32, k, n) .* 100); C0 = round.(CUDA.rand(Float32, m, n) .* 300)
    R = copy(C0); with_config(() -> S.sgemx_gpu!(MP, R, A, B; tiling = S.TILING_LARGE); gemm_kernel = 2)   # reference: the fixed v2 kernel
    C = similar(C0)
    fs = [(name, setup, runner(name)) for (name, setup) in variants]
    best = fill(Inf, length(fs)); ok = trues(length(fs))
    for (i, (name, setup, f)) in enumerate(fs)
        copy!(C, C0); with_config(() -> f(C, A, B); setup...); ok[i] = Array(C) == Array(R)
    end
    sleep(0.05)
    for _ in 1:ROUNDS, (i, (name, setup, f)) in enumerate(fs)
        best[i] = min(best[i], with_config(() -> burst(f, C, C0, A, B); setup...))
    end
    @printf("%6d×%4d×%4d ", m, n, k)
    for (i, (name, _, _)) in enumerate(fs); @printf("| %s %5.0f%s ", name, m * n * k / best[i] / 1e9, ok[i] ? "" : " WRONG"); end
    println()
end
