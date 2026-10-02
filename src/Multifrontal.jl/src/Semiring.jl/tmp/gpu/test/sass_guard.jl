# Guard against silent 2-10× regressions in the hot kernels: no device function calls (CALL), no local-memory
# spills (STL/LDL), and min-plus Float32 must use the FMNMX instruction (not FSETP + FSEL).
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

bad = 0
for (name, f) in checks
    code = sass(f)
    calls = count_op(code, "CALL.REL.NOINC") + count_op(code, "CALL.ABS.NOINC")
    spills = count_op(code, "STL") + count_op(code, "LDL")
    fmnmx = count_op(code, "FMNMX")
    fsel = count_op(code, "FSEL")
    ok = calls == 0 && spills == 0 && fmnmx > 0
    global bad += !ok
    println(rpad(name, 42), ok ? "ok" : "CHECK", "  (CALL $calls, STL/LDL $spills, FMNMX $fmnmx, FSEL $fsel)")
end
println(bad == 0 ? "sass guard: all ok" : "sass guard: $bad suspicious")
