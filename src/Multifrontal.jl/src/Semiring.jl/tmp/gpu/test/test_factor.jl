# GPU dense LU and hybrid sparse factorization against the CPU.
include(joinpath(@__DIR__, "..", "src", "SemiringGPU.jl"))
include(joinpath(@__DIR__, "graphs.jl"))

using .SemiringGPU
using .SemiringGPU: Semiring
using .Semiring: MinPlus, MaxMin, PlusProd, mlu, szero, sone
using CUDA, LinearAlgebra, SparseArrays, Random

Random.seed!(3)

println("dense sgetrf_gpu! vs Semiring.sgetrf!")
for (s, T) in [(MinPlus(), Float32), (MinPlus(), Float64), (MaxMin(), Float32), (PlusProd(), Float64)]
    oks = map((1, 5, 64, 65, 200, 700)) do n
        A = s isa PlusProd ? rand(T, n, n) ./ (2n) : T.(rand(1:100, n, n))
        ref = Semiring.sgetrf!(s, copy(A))
        out = Array(sgetrf_gpu!(s, CuArray(A)))
        s isa PlusProd ? isapprox(out, ref; rtol = 1e-10) : out == ref
    end
    println("  ", rpad(string(nameof(typeof(s)), " ", T), 22), " n=1,5,64,65,200,700: ", all(oks) ? "ok" : "FAIL $oks")
end

println("\nhybrid mlu_gpu vs CPU mlu (factor arrays and solves)")
for (name, s, A) in [("grid 60×60", MinPlus(), grid(60, 60, Float32)), ("grid 150×150", MinPlus(), grid(150, 150, Float32)),
                     ("grid3 16³", MinPlus(), grid3(16, Float32)), ("grid3 16³ F64", MinPlus(), grid3(16, Float64)),
                     ("grid3 14³ MaxMin", MaxMin(), grid3(14, Float32)), ("grid 40×40 PlusProd", PlusProd(), grid(40, 40, Float64) ./ 500)]
    T = eltype(A); n = size(A, 1)
    R = mlu(s, A)
    for large in (typemax(Int), 256, 64, 63, 16)
        F, G, times = mlu_gpu(s, A; large, download = true)
        cmp(x, y) = s isa PlusProd ? isapprox(x, y; rtol = 1e-10) : x == y
        okF = cmp(F.LDval, R.LDval) && cmp(F.LLval, R.LLval) && cmp(F.UDval, R.UDval) && cmp(F.ULval, R.ULval)
        k = 37; src = rand(1:n, k)
        B = fill(szero(s, T, Val(:N)), k, n); for t in 1:k; B[t, src[t]] = sone(s, T, Val(:N)); end
        okS = cmp(Array(sssp_gpu!(CuMatrix{T}(undef, k, n), G, CuVector(src))), rmul!(B, R))
        println("  ", rpad(name, 22), " large=", rpad(large == typemax(Int) ? "∞" : large, 5), " top fronts=", lpad(times.ntop, 5),
            " (", lpad(round(100times.topwork, digits = 1), 5), "% of work)  factor: ", okF ? "ok" : "FAIL", "  solve: ", okS ? "ok" : "FAIL")
    end
end

println("\nFactorPlan: refactorization with new weights, graph replay")
for (name, s, mk) in [("grid3 16³", MinPlus(), () -> grid3(16, Float32)), ("grid 150×150", MinPlus(), () -> grid(150, 150, Float32)),
                      ("grid3 14³ MaxMin", MaxMin(), () -> grid3(14, Float32))]
    A = mk()
    F = Semiring.ChordalSLU(s, A)
    for ns in (1, 8)
    P = FactorPlan(F; large = 64, graph = true, nstreams = ns)
    oks = map(1:4) do r
        Ar = r == 1 ? A : (B = copy(A); nonzeros(B) .= rand(1:100, nnz(B)); B = (B + B') ; B)   # new weights, same pattern
        copyto!(F, Ar)
        factorize!(P; download = true)
        R = mlu(s, Ar)
        F.LDval == R.LDval && F.LLval == R.LLval && F.UDval == R.UDval && F.ULval == R.ULval
    end
    println("  ", rpad(name, 22), " streams=", ns, ": 4 factorizations (1 direct + 3 graph replays): ", all(oks) ? "ok" : "FAIL $oks")
    end
end

println("\nfactorize_cpu! (subtree-parallel CPU reference) vs CPU mlu")
for (name, s, A) in [("grid3 16³", MinPlus(), grid3(16, Float32)), ("grid 150×150", MinPlus(), grid(150, 150, Float32)), ("grid 40×40 PlusProd", PlusProd(), grid(40, 40, Float64) ./ 500)]
    R = mlu(s, A)
    oks = map((64, 16)) do large
        F = Semiring.ChordalSLU(s, A); copyto!(F, A)
        P = FactorPlan(F; large); SemiringGPU.factorize_cpu!(P)
        cmp(x, y) = s isa PlusProd ? isapprox(x, y; rtol = 1e-10) : x == y
        cmp(F.LDval, R.LDval) && cmp(F.LLval, R.LLval) && cmp(F.UDval, R.UDval) && cmp(F.ULval, R.ULval)
    end
    println("  ", rpad(name, 22), " large=64,16: ", all(oks) ? "ok" : "FAIL $oks")
end
