# CUDA C++ backend (src/cuda_backend.jl, cuda/*.cu) against the Julia kernels and the CPU.
#   cuda/build.sh && julia --project=. -t 16 test/test_cuda_backend.jl
include(joinpath(@__DIR__, "..", "src", "SemiringGPU.jl"))
include(joinpath(@__DIR__, "..", "cuda", "cuda_backend.jl"))
include(joinpath(@__DIR__, "graphs.jl"))
using .SemiringGPU, .SemiringCUDA
using .SemiringGPU: Semiring
using .Semiring: MinPlus, MaxMin, PlusProd, mlu, ChordalSLU, sgemx!
using CUDA, LinearAlgebra, SparseArrays, Random, Printf
Random.seed!(7)

nfail = 0
report(ok, msg) = (ok || (global nfail += 1); println("  ", rpad(msg, 84), ok ? "ok" : "FAIL"))

name(s) = s isa MinPlus ? "MinPlus" : s isa MaxMin ? "MaxMin" : "PlusProd"
# integer-valued entries (exact for MinPlus / MaxMin); some +∞ / -∞ to exercise the padding and NaN absorption
function entries(s, ::Type{T}, dims...) where {T}
    X = T.(rand(1:100, dims...))
    if !(s isa PlusProd)
        X[rand(length(X)) .< 0.05] .= T(Inf)
        X[rand(length(X)) .< 0.02] .= T(-Inf)
    else
        X ./= 1000
    end
    return X
end

println("GEMM: C++ vs Julia sgemx_gpu! vs CPU sgemx!")
shapes = [(1, 1, 1), (7, 5, 3), (128, 128, 8), (130, 129, 9), (1000, 777, 333), (64, 2000, 17), (1000, 7, 333), (5000, 16, 100),
          (3000, 30, 50), (513, 33, 9), (257, 64, 64), (1001, 64, 65), (300, 100, 1), (2048, 2048, 64), (5000, 17, 3), (33, 1000, 40)]
for (s, T) in [(MinPlus(), Float32), (MinPlus(), Float64), (MaxMin(), Float32), (PlusProd(), Float64), (PlusProd(), Float32)]
    okc = true; okj = true
    for (m, n, k) in shapes
        A = entries(s, T, m, k); B = entries(s, T, k, n); C = entries(s, T, m, n)
        ref = sgemx!(s, Val(:N), Val(:N), copy(C), A, B)
        jl = Array(sgemx_gpu!(s, CuArray(C), CuArray(A), CuArray(B)))
        for tiling in 0:10
            cu = Array(sgemx_cuda!(s, CuArray(C), CuArray(A), CuArray(B); tiling))
            if s isa PlusProd
                okc &= isapprox(cu, ref; rtol = T == Float32 ? 1e-4 : 1e-12)
                okj &= isapprox(cu, jl; rtol = T == Float32 ? 1e-4 : 1e-12)
            else
                okc &= isequal(cu, ref)
                okj &= isequal(cu, jl)
                if !isequal(cu, ref)
                    println("    mismatch $(name(s)) $T $((m, n, k)) tiling $tiling: ", count(.!isequal.(cu, ref)), " entries")
                end
            end
        end
    end
    report(okc, "$(name(s)) $T, $(length(shapes)) shapes × 11 tilings vs CPU")
    report(okj, "$(name(s)) $T vs Julia sgemx_gpu!")
end

# strided views: odd leading dimensions and offsets (the solver's column blocks and reshaped factor blocks)
let s = MinPlus(), T = Float32
    big = CuArray(entries(s, T, 999, 700))
    C = view(big, 3:500, 11:80); A = view(big, 501:998, 100:163); B = view(big, 200:263, 300:369)
    ref = sgemx!(s, Val(:N), Val(:N), Array(C), Array(A), Array(B))
    out = Array(sgemx_cuda!(s, C, A, B))
    report(isequal(out, ref), "MinPlus Float32 on strided views (ld = 999, unaligned)")
end

println("closure_cuda! vs closure_gpu! vs CPU closure")
# schedules of src/sgetrs.jl: level by level (the original), and the persistent L sweep +
# warp-per-source path walk (SemiringGPU.PERSISTENT[] / PATH_WARP[], if this version has them)
const SCHEDULES = isdefined(SemiringGPU, :PERSISTENT) ? [false, true] : [false]
setschedule!(on) = isdefined(SemiringGPU, :PERSISTENT) && (SemiringGPU.PERSISTENT[] = on; SemiringGPU.PATH_WARP[] = on)
for (gname, s, A) in [("grid 60×60", MinPlus(), grid(60, 60, Float32)), ("grid3 14³", MinPlus(), grid3(14, Float32)),
                      ("grid3 12³ MaxMin", MaxMin(), grid3(12, Float32)), ("grid 30×30 PlusProd", PlusProd(), grid(30, 30, Float64) ./ 500),
                      ("grid3 10³ F64", MinPlus(), grid3(10, Float64))]
    R = mlu(s, A); Cref = Matrix(R)
    for large in (typemax(Int), 64), ops in (false, true), sched in SCHEDULES
        setschedule!(sched)
        F = ChordalSLU(s, A); copyto!(F, A)
        P = FactorPlan(F; large, graph = false, nstreams = 8); factorize!(P); G = GPUSLU(P; large = 64)
        ops && precompute_ops!(G)
        Dj = Array(closure_gpu(G))
        p = Array(G.rperm)
        for variant in (0, 1)
            Dc = Array(closure_cuda(G; variant))
            H = similar(Dc); H[p, p] = Dc
            if s isa PlusProd
                okc = isapprox(H, Cref; rtol = 1e-10); okj = isapprox(Dc, Dj; rtol = 1e-12)
            else
                okc = H == Cref; okj = isequal(Dc, Dj)
            end
            report(okc && okj, "$(rpad(gname, 20)) large=$(rpad(large == typemax(Int) ? "∞" : large, 3)) ops=$(rpad(ops, 5)) $(sched ? "persistent" : "levels    ") var=$variant (CPU $(okc ? "=" : "≠"), Julia $(okj ? "=" : "≠"))")
        end
    end
end
setschedule!(true)

println(nfail == 0 ? "all tests passed" : "$nfail FAILED")
