# How is numeric factorization work distributed over fronts, and how much sits in GPU-sized fronts?
include(joinpath(@__DIR__, "bench_solve.jl"))
using .Semiring: ChordalSLU

function grid3(nx)
    rng = Xoshiro(1); id(i, j, l) = i + (j - 1) * nx + (l - 1) * nx^2
    I = Int[]; J = Int[]; V = T[]
    for l in 1:nx, j in 1:nx, i in 1:nx, d in ((1,0,0),(0,1,0),(0,0,1))
        a, b, c = i + d[1], j + d[2], l + d[3]
        (a <= nx && b <= nx && c <= nx) || continue
        w = T(rand(rng, 1:100)); u = id(i, j, l); v = id(a, b, c)
        append!(I, (u, v)); append!(J, (v, u)); append!(V, (w, w))
    end
    return sparse(I, J, V, nx^3, nx^3)
end

graphs = ["USA-road-t.NY" => () -> GRAPHS["USA-road-t.NY"](), "grid2d-500" => () -> grid(500, 500),
          "grid2d-1000" => () -> grid(1000, 1000), "grid3d-30" => () -> grid3(30), "grid3d-40" => () -> grid3(40)]

for (name, mk) in graphs
    s = MinPlus(); A = mk(); n = size(A, 1)
    F = ChordalSLU(s, A); copyto!(F, A); lu!(F)          # compile + warm
    copyto!(F, A)
    tnum = @elapsed lu!(F; nt = Threads.nthreads())
    S = F.S.S; nf = Int(SemiringGPU.MF.nv(S.res))
    nn = Float64.(diff(SemiringGPU.MF.pointers(S.res)[1:nf+1])); na = Float64.(diff(SemiringGPU.MF.pointers(S.sep)[1:nf+1]))
    w = @. nn^3 / 3 + nn^2 * na + nn * na^2          # multiply-adds per front (paper, Sec. III)
    W = sum(w)
    @printf("\n%s: n=%d, fronts=%d, numeric factorization %.3f s (%d threads) = %.1f G mul-adds/s, total work %.2e\n",
        name, n, nf, tnum, Threads.nthreads(), W / tnum / 1e9, W)
    @printf("  largest front: nn=%d na=%d\n", maximum(nn), na[argmax(nn .+ na)])
    for thr in (64, 128, 256, 512)
        big = (nn .+ na) .>= thr
        @printf("  fronts with nn+na ≥ %4d: %6d fronts (%.3f%%) hold %5.1f%% of the work\n", thr, count(big), 100count(big) / nf, 100sum(w[big]) / W)
    end
end
