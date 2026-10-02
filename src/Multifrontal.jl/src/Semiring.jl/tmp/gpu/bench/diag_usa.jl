# Diagnose the k = 256 mismatch on USA-road-t.USA: precision (Float32 rounding past 2^24) or a bug (e.g. k n > 2^31)?
include(joinpath(@__DIR__, "bench_solve.jl"))
const NT = Threads.nthreads()

function rows(s, ::Type{X}, src, n) where {X}
    B = fill(szero(s, X, Val(:N)), length(src), n)
    for (t, v) in enumerate(src); B[t, v] = sone(s, X, Val(:N)); end
    return B
end

function report(label, gpu, cpu)
    d = gpu .!= cpu
    nd = count(d)
    fin = isfinite.(cpu)
    @printf("  %-34s mismatches %12d of %d", label, nd, length(cpu))
    if nd > 0
        δ = abs.(Float64.(gpu[d]) .- Float64.(cpu[d]))
        rel = δ ./ max.(abs.(Float64.(cpu[d])), 1)
        big = count(abs.(cpu[d]) .> 2.0^24)
        @printf(" | max |Δ| %.3g, max rel %.3g, %d of them have distance > 2^24, finite-gpu/finite-cpu disagree %d", maximum(δ), maximum(rel), big, count(isfinite.(gpu) .!= fin))
        idx = findfirst(d); @printf(" | first at %s: gpu %s cpu %s", Tuple(CartesianIndices(d)[idx]), gpu[idx], cpu[idx])
    end
    println()
end

s = MinPlus()
A = GRAPHS["USA-road-t.USA"](); n = size(A, 1)
println(CUDA.name(CUDA.device()), "; n = ", n, ", ", NT, " threads; max arc weight ", maximum(nonzeros(A)))
for k in (128, 256)
    src = rand(Xoshiro(7), 1:n, k)
    for X in (Float32, Float64)
        AX = X.(A)
        F = mlu(s, AX; nt = NT)
        cpu = rmul!(rows(s, X, src, n), F; nt = NT)
        @printf("k=%d %s: max finite distance %.6g (2^24 = %.6g)\n", k, X, maximum(x for x in cpu if isfinite(x)), 2.0^24)
        G = GPUSLU(F)
        g1 = Array(sssp_gpu!(CuMatrix{X}(undef, k, n), G, CuVector(src))); report("sssp_gpu! vs CPU rmul!", g1, cpu); g1 = nothing
        Bg = CuArray(rows(s, X, src, n)); rmul_gpu!(Bg, G); g2 = Array(Bg); Bg = nothing
        report("rmul_gpu! vs CPU rmul!", g2, cpu); g2 = nothing
        if X == Float32
            global cpu32 = cpu
        else
            report("CPU Float32 vs CPU Float64", X.(cpu32), cpu)
        end
        G = nothing; F = nothing; GC.gc(); CUDA.reclaim()
    end
end
