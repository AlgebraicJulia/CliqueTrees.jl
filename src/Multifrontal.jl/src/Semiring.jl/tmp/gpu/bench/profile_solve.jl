# Where does the GPU solve spend its time? Per-phase CUDA event timing.
include(joinpath(@__DIR__, "bench_solve.jl"))

function phases(f!; reps = 3)
    ts = Float64[]
    for r in 0:reps
        t = CUDA.@elapsed f!(nothing)
        r > 0 && push!(ts, t)
    end
    timer = Dict{Symbol, Float64}()
    f!(timer)
    return median(ts), timer
end

function show_phases(label, k, t, timer)
    @printf("  %-10s k=%-4d %7.2f ms | ", label, k, 1e3t)
    for (key, v) in sort(collect(timer); by = first)
        @printf("%s %.2f  ", key, 1e3v)
    end
    println()
end

for name in (isempty(ARGS) ? ["USA-road-t.FLA", "grid2d-500"] : ARGS)
    s = MinPlus(); A = GRAPHS[name](); n = size(A, 1)
    F = mlu(s, A)
    G = GPUSLU(F)
    println("\n$name: n=$n, fronts=$(G.nf), levels=$(SemiringGPU.nlevels(G)), dense fronts=$(SemiringGPU.nlarge(G))   (phase times synced per launch)")

    for k in (1, 16, 256)
        src = rand(Xoshiro(2), 1:n, k)
        B0 = fill(szero(s, T, Val(:N)), k, n); for t in 1:k; B0[t, src[t]] = sone(s, T, Val(:N)); end
        B0g = CuArray(B0); Bg = similar(B0g); Wg = similar(B0g); srcg = CuVector(src)

        t, timer = phases(tm -> (copyto!(Bg, B0g); rmul_gpu!(Bg, G; W = Wg, timer = tm)))
        show_phases("rmul_gpu!", k, t, timer)
        ref = Array(Bg)

        t, timer = phases(tm -> sssp_gpu!(Bg, G, srcg; W = Wg, timer = tm))
        show_phases("sssp_gpu!", k, t, timer)
        Array(Bg) == ref || println("  MISMATCH")
    end
end
