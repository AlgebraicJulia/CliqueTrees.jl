# GPU rmul! against the CPU rmul! on small graphs.
include(joinpath(@__DIR__, "..", "src", "SemiringGPU.jl"))

using .SemiringGPU
using .SemiringGPU: Semiring
using .Semiring: MinPlus, MaxMin, PlusProd, mlu, szero, sone
using CUDA, LinearAlgebra, SparseArrays, Random

Random.seed!(1)

function grid(nx, ny, ::Type{T}) where {T}
    id(i, j) = i + (j - 1) * nx
    I = Int[]; J = Int[]; V = T[]
    for j in 1:ny, i in 1:nx
        for (di, dj) in ((1, 0), (0, 1))
            i2, j2 = i + di, j + dj
            if i2 <= nx && j2 <= ny
                w = T(rand(1:100))
                push!(I, id(i, j)); push!(J, id(i2, j2)); push!(V, w)
                push!(I, id(i2, j2)); push!(J, id(i, j)); push!(V, w)
            end
        end
    end
    return sparse(I, J, V, nx * ny, nx * ny)
end

function sources(s, ::Type{T}, k, n) where {T}
    B = fill(szero(s, T, Val(:N)), k, n)
    for t in 1:k
        B[t, rand(1:n)] = sone(s, T, Val(:N))
    end
    return B
end

for (s, A) in [(MinPlus(), grid(30, 20, Float32)), (MinPlus(), grid(60, 60, Float32)), (MinPlus(), grid(150, 150, Float32)),
               (MinPlus(), grid(40, 40, Float64)), (MaxMin(), grid(40, 40, Float32)),
               (PlusProd(), grid(30, 30, Float64) ./ 500)]
    T = eltype(A)
    F = mlu(s, A)
    for large in (typemax(Int), 4096, 64, 1)
        G = GPUSLU(F; large)
        oks = map((1, 7, 64, 100)) do k
            B = sources(s, T, k, size(A, 1))
            ref = rmul!(copy(B), F)
            out = Array(rmul_gpu!(CuArray(B), G))
            s isa PlusProd ? isapprox(out, ref; rtol = 1e-10) : out == ref
        end
        println(rpad(string(nameof(typeof(s)), " ", T, " n=", size(A, 1)), 28), " large=", rpad(large == typemax(Int) ? "∞" : large, 5),
            " (", lpad(SemiringGPU.nlarge(G), 4), "/", G.nf, " fronts dense)  k=1,7,64,100: ", all(oks) ? "ok" : "FAIL $(oks)")
    end
end

println("\nsssp_gpu! (path-walk U sweep) vs CPU rmul! on unit sources")
for (s, A) in [(MinPlus(), grid(60, 60, Float32)), (MinPlus(), grid(150, 150, Float32)),
               (MinPlus(), grid(40, 40, Float64)), (MaxMin(), grid(40, 40, Float32)),
               (PlusProd(), grid(30, 30, Float64) ./ 500)]
    T = eltype(A); n = size(A, 1)
    F = mlu(s, A)
    for large in (typemax(Int), 64), ops in (false, true)
        G = GPUSLU(F; large); ops && precompute_ops!(G)
        oks = map((1, 7, 64, 100)) do k
            src = rand(1:n, k)
            B = fill(szero(s, T, Val(:N)), k, n)
            for t in 1:k; B[t, src[t]] = sone(s, T, Val(:N)); end
            ref = rmul!(B, F)
            out = Array(sssp_gpu!(CuMatrix{T}(undef, k, n), G, CuVector(src)))
            s isa PlusProd ? isapprox(out, ref; rtol = 1e-10) : out == ref
        end
        println(rpad(string(nameof(typeof(s)), " ", T, " n=", n), 28), " large=", rpad(large == typemax(Int) ? "∞" : large, 5), " ops=", rpad(ops, 5), " k=1,7,64,100: ", all(oks) ? "ok" : "FAIL $(oks)")
    end
end

println("\nSSSPPlan (CUDA graph replay) vs CPU rmul!")
for (s, A) in [(MinPlus(), grid(150, 150, Float32)), (MaxMin(), grid(40, 40, Float32)), (PlusProd(), grid(30, 30, Float64) ./ 500)]
    T = eltype(A); n = size(A, 1)
    F = mlu(s, A)
    G = GPUSLU(F; large = 64)
    oks = map((1, 7, 100)) do k
        P = SSSPPlan(G, k)
        all(1:3) do _   # replay with fresh sources each time
            src = rand(1:n, k)
            B = fill(szero(s, T, Val(:N)), k, n)
            for t in 1:k; B[t, src[t]] = sone(s, T, Val(:N)); end
            ref = rmul!(B, F)
            out = Array(P(src))
            s isa PlusProd ? isapprox(out, ref; rtol = 1e-10) : out == ref
        end
    end
    println(rpad(string(nameof(typeof(s)), " ", T, " n=", n), 28), " k=1,7,100 × 3 replays: ", all(oks) ? "ok" : "FAIL $(oks)")
end
