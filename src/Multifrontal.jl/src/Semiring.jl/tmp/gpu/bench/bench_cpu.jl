# CPU-only benchmarks (no GPU needed), with Threads.nthreads() threads:
#   julia -t 128 bench/bench_cpu.jl closure grid3d-25 ...       numeric factorization + full closure Matrix(F) (graphs in data/mtx)
#   julia -t 128 bench/bench_cpu.jl queries USA-road-t.NY ...   numeric factorization + single (fast path) and blocked queries
include(joinpath(@__DIR__, "bench_solve.jl"))
const SR = Semiring
const TMP = joinpath(pkgdir(SemiringGPU.CliqueTrees), "src", "Multifrontal.jl", "src", "Semiring.jl", "tmp")
Base.include(SR, joinpath(TMP, "subtree_tmp.jl"))
Base.include(SR, joinpath(TMP, "sssp_tmp.jl"))

function read_mtx(path)
    I = Int[]; J = Int[]; V = T[]; n = 0; header = true
    for line in eachline(path)
        startswith(line, '%') && continue
        a = split(line)
        if header; n = parse(Int, a[1]); header = false; continue; end
        push!(I, parse(Int, a[1])); push!(J, parse(Int, a[2])); push!(V, parse(T, a[3]))
    end
    return sparse(I, J, V, n, n, min)
end

load(name) = haskey(GRAPHS, name) ? GRAPHS[name]() : read_mtx(joinpath(DATA, "mtx", name * ".mtx"))
best(f, r) = (f(); minimum(f() for _ in 1:r))

function unit_sources(s, k, n)
    B = fill(szero(s, T, Val(:N)), k, n)
    for (t, v) in enumerate(rand(Xoshiro(2), 1:n, k)); B[t, v] = sone(s, T, Val(:N)); end
    return B
end

const NT = Threads.nthreads()
const R = parse(Int, get(ENV, "REPS", "2"))

if abspath(PROGRAM_FILE) == @__FILE__
    mode = ARGS[1]
    mlu(MinPlus(), grid(20, 20))                       # compile
    println("CPU: ", Sys.cpu_info()[1].model, ", ", NT, " threads")

    for name in ARGS[2:end]
        s = MinPlus(); A = load(name); n = size(A, 1)
        tsym = best(() -> @elapsed(SR.ChordalSLU(s, A)), R)
        F = SR.ChordalSLU(s, A)
        tnum = best(() -> (copyto!(F, A); @elapsed(lu!(F; nt = NT))), R)
        fr = (length(F.LLval) + length(F.ULval) + length(F.LDval)) / nnz(A)
        @printf("%-15s n=%9d m=%9d fill=%6.1f thr=%3d | symbolic %8.3f s  numeric %8.3f s", name, n, nnz(A), fr, NT, tsym, tnum)

        if mode == "closure"
            @printf("  closure %8.3f s\n", best(() -> @elapsed(Matrix(F; nt = NT)), R))
        else
            W, x, pool, sched = SR.sgetrs_elem_workspace_tmp(F; nt = NT)
            b = Vector{T}(undef, n); srcs = rand(Xoshiro(3), 1:n, 50)
            SR.sgetrs_tmp!(F, Val(:R), Val(:N), b, srcs[1], W, x, pool, sched; nt = NT)
            t1 = median([@elapsed(SR.sgetrs_tmp!(F, Val(:R), Val(:N), b, k, W, x, pool, sched; nt = NT)) for k in srcs])
            @printf("  single %8.3f ms", 1e3t1)

            for k in KS
                B0 = unit_sources(s, k, n); Bc = similar(B0); Bl = Matrix{T}(undef, n, k)
                t = min(cpu_time(F, B0, Bc, NT; reps = 1), cpu_time_col(F, B0, Bl, NT; reps = 1))
                @printf("  k=%d %8.4f ms/q", k, 1e3t / k)
                B0 = Bc = Bl = nothing; GC.gc()
            end

            println()
        end
    end
end
