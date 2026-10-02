# CPU vs GPU blocked single-source queries, X = B A* (rmul!, row layout),
# in the min-plus semiring.
#
#   julia --project=. -t auto bench/bench_solve.jl [graph ...]
#
# graphs: roadNet-PA, USA-road-t.NY, USA-road-t.FLA, grid2d-500 (default: all)

include(joinpath(@__DIR__, "..", "src", "SemiringGPU.jl"))

using .SemiringGPU
using .SemiringGPU: Semiring
using .Semiring: MinPlus, mlu, szero, sone
using CUDA, LinearAlgebra, SparseArrays, Random, Printf, Statistics, CodecZlib

const DATA = joinpath(@__DIR__, "..", "data")
const T = Float32
const KS = (1, 16, 64, 256)

# ----- graphs -----

# SNAP edge list, undirected, weights uniform in [1, 100] (fixed seed), as in the paper
function load_snap(path)
    I = Int[]; J = Int[]
    for line in eachline(GzipDecompressorStream(open(path)))
        startswith(line, '#') && continue
        a, b = split(line)
        push!(I, parse(Int, a) + 1); push!(J, parse(Int, b) + 1)
    end
    n = max(maximum(I), maximum(J))
    rng = Xoshiro(1)
    E = sparse(min.(I, J), max.(I, J), true, n, n)
    E = triu(E, 1)
    i, j, _ = findnz(E)
    w = T.(rand(rng, 1:100, length(i)))
    return sparse(vcat(i, j), vcat(j, i), vcat(w, w), n, n, min)
end

# DIMACS .gr, travel-time arc weights
function load_dimacs(path)
    I = Int[]; J = Int[]; V = T[]; n = 0
    for line in eachline(GzipDecompressorStream(open(path)))
        if startswith(line, "p")
            n = parse(Int, split(line)[3])
        elseif startswith(line, "a")
            _, a, b, w = split(line)
            push!(I, parse(Int, a)); push!(J, parse(Int, b)); push!(V, parse(T, w))
        end
    end
    return sparse(I, J, V, n, n, min)
end

function grid(nx, ny)
    rng = Xoshiro(1)
    id(i, j) = i + (j - 1) * nx
    I = Int[]; J = Int[]; V = T[]
    for j in 1:ny, i in 1:nx, (di, dj) in ((1, 0), (0, 1))
        i2, j2 = i + di, j + dj
        (i2 <= nx && j2 <= ny) || continue
        w = T(rand(rng, 1:100))
        append!(I, (id(i, j), id(i2, j2))); append!(J, (id(i2, j2), id(i, j))); append!(V, (w, w))
    end
    return sparse(I, J, V, nx * ny, nx * ny)
end

const GRAPHS = Dict(
    "roadNet-PA"     => () -> load_snap(joinpath(DATA, "roadNet-PA.txt.gz")),
    "USA-road-t.NY"  => () -> load_dimacs(joinpath(DATA, "USA-road-t.NY.gr.gz")),
    "USA-road-t.FLA" => () -> load_dimacs(joinpath(DATA, "USA-road-t.FLA.gr.gz")),
    "grid2d-500"     => () -> grid(500, 500),
)

# ----- timing -----

function sources!(B, s, rng)
    fill!(B, szero(s, T, Val(:N)))
    for t in axes(B, 1)
        B[t, rand(rng, axes(B, 2))] = sone(s, T, Val(:N))
    end
    return B
end

function cpu_time(F, B0, Bc, nt; reps)
    ts = Float64[]
    for r in 0:reps
        copyto!(Bc, B0)
        t = @elapsed rmul!(Bc, F; nt)
        r > 0 && push!(ts, t)
    end
    return median(ts)
end

# column layout: lmul!(F, Bᵀ), Bᵀ is n × k (the transposition is not timed)
function cpu_time_col(F, B0, Bl, nt; reps)
    ts = Float64[]
    for r in 0:reps
        permutedims!(Bl, B0, (2, 1))
        t = @elapsed lmul!(F, Bl; nt)
        r > 0 && push!(ts, t)
    end
    return median(ts)
end

function gpu_time(G, B0g, Bg, Wg; reps)
    ts = Float64[]
    for r in 0:reps
        copyto!(Bg, B0g)
        t = CUDA.@elapsed rmul_gpu!(Bg, G; W = Wg)
        r > 0 && push!(ts, t)
    end
    return median(ts)
end

# ----- main -----

function gpu_median(f; reps)
    f(); CUDA.synchronize()
    return median([CUDA.@elapsed(f()) for _ in 1:reps])
end

function run(name)
    s = MinPlus()
    A = GRAPHS[name]()
    n = size(A, 1)
    nt = Threads.nthreads()

    mlu(s, grid(10, 10))   # compile
    tf = @elapsed F = mlu(s, A; nt)
    tg = @elapsed G = GPUSLU(F)
    nfac = length(F.LLval) + length(F.ULval) + length(F.LDval)

    @printf("\n%s: n = %d, m = %d, fill nnz(L+U)/m = %.1f, tree levels = %d, dense fronts = %d\n", name, n, nnz(A), nfac / nnz(A), SemiringGPU.nlevels(G), SemiringGPU.nlarge(G))
    @printf("  CPU factorization (%d threads) %.2f s; factor upload %.2f s\n", nt, tf, tg)
    println("  ms per query. CPU = best of rmul!/lmul! at 1 or $nt threads (dense B). GPU: rmul_gpu! (dense B), sssp_gpu! (path-walk U sweep), plan (sssp + CUDA graph)")
    @printf("  %5s | %9s | %9s %9s %9s | %8s | %s\n", "k", "CPU best", "rmul_gpu", "sssp_gpu", "plan", "speedup", "check")

    rng = Xoshiro(2)

    for k in KS
        src = rand(rng, 1:n, k)
        B0 = fill(szero(s, T, Val(:N)), k, n)
        for t in 1:k
            B0[t, src[t]] = sone(s, T, Val(:N))
        end
        Bc = similar(B0)
        reps = k <= 16 ? 5 : k >= 256 ? 1 : 3

        cpu = min(cpu_time(F, B0, Bc, 1; reps), cpu_time(F, B0, Bc, nt; reps))
        Bl = Matrix{T}(undef, n, k)
        cpu = min(cpu, cpu_time_col(F, B0, Bl, 1; reps), cpu_time_col(F, B0, Bl, nt; reps))
        Bl = nothing; GC.gc()

        B0g = CuArray(B0); Bg = similar(B0g); Wg = similar(B0g); srcg = CuVector(src)
        t_rmul = gpu_time(G, B0g, Bg, Wg; reps = max(reps, 3))
        ok = Array(Bg) == Bc
        t_sssp = gpu_median(() -> sssp_gpu!(Bg, G, srcg; W = Wg); reps = max(reps, 3))
        ok &= Array(Bg) == Bc
        B0g = Bg = Wg = nothing; GC.gc(); CUDA.reclaim()

        P = SSSPPlan(G, k)
        t_plan = gpu_median(() -> P(src); reps = max(reps, 5))
        ok &= Array(P.X) == Bc
        P = nothing

        @printf("  %5d | %9.4f | %9.4f %9.4f %9.4f | %7.1f× | %s\n", k, 1e3cpu / k, 1e3t_rmul / k, 1e3t_sssp / k, 1e3t_plan / k, cpu / t_plan, ok ? "ok" : "MISMATCH")

        B0 = Bc = nothing
        GC.gc(); CUDA.reclaim()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println(CUDA.name(CUDA.device()), ", ", Threads.nthreads(), " CPU threads, ", T)

    for name in (isempty(ARGS) ? ["USA-road-t.NY", "grid2d-500", "roadNet-PA", "USA-road-t.FLA"] : ARGS)
        run(name)
    end
end
