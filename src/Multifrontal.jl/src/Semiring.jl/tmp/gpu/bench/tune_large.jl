# GPU solve time vs the large-front threshold.
include(joinpath(@__DIR__, "bench_solve.jl"))

for name in (isempty(ARGS) ? ["USA-road-t.NY", "grid2d-500"] : ARGS)
    s = MinPlus(); A = GRAPHS[name](); n = size(A, 1)
    F = mlu(s, A)
    println("\n", name, ": GPU ms/query")
    @printf("  %10s %8s | %9s %9s %9s %9s\n", "large", "#dense", "k=1", "k=16", "k=64", "k=256")
    rng = Xoshiro(2)
    Bs = Dict(k => CuArray(sources!(Matrix{T}(undef, k, n), s, rng)) for k in KS)
    for large in (typemax(Int), 1 << 16, 1 << 14, 1 << 12, 1 << 10)
        G = GPUSLU(F; large)
        ts = map(KS) do k
            B0g = Bs[k]; Bg = similar(B0g); Wg = similar(B0g)
            1e3gpu_time(G, B0g, Bg, Wg; reps = 3) / k
        end
        @printf("  %10s %8d | %9.3f %9.3f %9.4f %9.4f\n", large == typemax(Int) ? "∞" : string(large), SemiringGPU.nlarge(G), ts...)
    end
end
