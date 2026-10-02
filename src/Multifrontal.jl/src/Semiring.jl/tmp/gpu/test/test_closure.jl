# closure_gpu! (elimination coordinates) against the CPU closure Matrix(F).
include(joinpath(@__DIR__, "..", "src", "SemiringGPU.jl"))
include(joinpath(@__DIR__, "graphs.jl"))
using .SemiringGPU
using .SemiringGPU: Semiring
using .Semiring: MinPlus, MaxMin, PlusProd, mlu, ChordalSLU
using CUDA, LinearAlgebra, SparseArrays, Random
Random.seed!(5)
for (name, s, A) in [("grid 60×60", MinPlus(), grid(60, 60, Float32)), ("grid3 14³", MinPlus(), grid3(14, Float32)),
                     ("grid3 12³ MaxMin", MaxMin(), grid3(12, Float32)), ("grid 30×30 PlusProd", PlusProd(), grid(30, 30, Float64) ./ 500)]
    R = mlu(s, A); C = Matrix(R)                     # original labels
    for large in (typemax(Int), 64)
        F = ChordalSLU(s, A); copyto!(F, A)
        P = FactorPlan(F; large, graph = false, nstreams = 8); factorize!(P); G = GPUSLU(P; large = 64)
        D = Array(closure_gpu(G))
        p = Array(G.rperm)
        H = similar(D); H[p, p] = D                  # D[i, j] = A*[p[i], p[j]]
        ok = s isa PlusProd ? isapprox(H, C; rtol = 1e-10) : H == C
        println("  ", rpad(name, 22), " large=", rpad(large == typemax(Int) ? "∞" : large, 4), ": ", ok ? "ok" : "FAIL")
    end
end
