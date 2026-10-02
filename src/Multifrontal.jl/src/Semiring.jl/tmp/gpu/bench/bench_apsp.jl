# All-pairs shortest paths (the closure A*), ours vs ROME, on the graphs in data/mtx.
#
# Ours, GPU: symbolic phase (CPU) + hybrid numeric factorization + solves for
# all n sources in blocks of k, written into a full n × n distance matrix on
# the GPU (as ROME keeps it), optionally copied to the host.
# Ours, CPU: Matrix(F), the closure from the factors (sgetri!), for reference.
#
#   julia --project=. -t auto bench/bench_apsp.jl [graph ...]

include(joinpath(@__DIR__, "bench_solve.jl"))

using .Semiring: ChordalSLU, ssymbolic

const MTX = joinpath(DATA, "mtx")

function read_mtx(path)
    I = Int[]; J = Int[]; V = T[]; n = 0; header = true
    for line in eachline(path)
        startswith(line, '%') && continue
        a = split(line)
        if header
            n = parse(Int, a[1]); header = false
        else
            push!(I, parse(Int, a[1])); push!(J, parse(Int, a[2])); push!(V, parse(T, a[3]))
        end
    end
    return sparse(I, J, V, n, n, min)
end

function apsp_gpu(A; k = 1024, large = 256, download = false, keep = false)
    s = MinPlus(); n = size(A, 1)
    t_sym = @elapsed F = ChordalSLU(s, A)
    copyto!(F, A)
    t_num = @elapsed begin
        P = FactorPlan(F; large, graph = false, nstreams = 8)   # one-shot: plan + first factorization, all timed
        factorize!(P)
        G = GPUSLU(P)
        CUDA.synchronize()
    end
    D = CuMatrix{T}(undef, n, n)
    t_solve = @elapsed begin
        P = SSSPPlan(G, k)                       # includes capture; timed, it is part of the run
        for r0 in 1:k:n
            r1 = min(r0 + k - 1, n)
            src = collect(r0:(r0 + k - 1)); src[src .> n] .= n
            X = P(src)
            @views D[r0:r1, :] .= X[1:(r1 - r0 + 1), :]
        end
        CUDA.synchronize()
    end
    t_d2h = 0.0
    H = nothing
    if download
        # D → host in row blocks through a pinned staging buffer (pinning all n² values fails for large n);
        # the host copy is kept only when `keep` (for checking against the CPU)
        H = keep ? Matrix{T}(undef, n, n) : nothing
        stage = Vector{T}(undef, k * n); CUDA.pin(stage)
        dev = CuVector{T}(undef, k * n)
        t_d2h = @elapsed for r0 in 1:k:n
            r1 = min(r0 + k - 1, n); m = r1 - r0 + 1
            reshape(view(dev, 1:(m * n)), m, n) .= view(D, r0:r1, :)        # gather the rows on the device
            copyto!(stage, 1, dev, 1, m * n)                                # one contiguous pinned copy
            isnothing(H) || (H[r0:r1, :] .= reshape(view(stage, 1:(m * n)), m, n))
        end
    end
    return (; t_sym, t_num, t_solve, t_d2h, D, H)
end

function apsp_cpu(A)
    s = MinPlus()
    t = @elapsed begin
        F = mlu(s, A; nt = Threads.nthreads())
        C = Matrix(F; nt = Threads.nthreads())
    end
    return t, C
end

if abspath(PROGRAM_FILE) == @__FILE__
    names = isempty(ARGS) ? ["grid3d-25", "grid2d-150", "grid3d-30", "grid2d-180"] : ARGS
    println(CUDA.name(CUDA.device()), ", ", Threads.nthreads(), " CPU threads, ", T)
    apsp_gpu(read_mtx(joinpath(MTX, "grid3d-25.mtx")); k = 256)   # compile
    @printf("  %-11s %6s | %8s %8s %8s %8s | %9s %9s | %9s | %s\n", "graph", "n", "symbolic", "numeric", "solves", "GPU tot", "D→H", "GPU+D→H", "CPU closure", "check")
    for name in names
        A = read_mtx(joinpath(MTX, name * ".mtx")); n = size(A, 1)
        r = apsp_gpu(A; download = true, keep = n <= 23000)
        tot = r.t_sym + r.t_num + r.t_solve
        ok = "-"; tc = NaN
        if n <= 23000
            GC.gc(); CUDA.reclaim()
            tc, C = apsp_cpu(A)
            ok = (!isnothing(r.H) && C == r.H) ? "ok" : "MISMATCH"
            C = nothing
        end
        @printf("  %-11s %6d | %7.2fs %7.2fs %7.2fs %7.2fs | %8.2fs %8.2fs | %8.2fs | %s\n", name, n, r.t_sym, r.t_num, r.t_solve, tot, r.t_d2h, tot + r.t_d2h, tc, ok)
        if name == "grid3d-25"
            open(joinpath(@__DIR__, "..", "external", "ours_rows_grid3d-25.txt"), "w") do io
                for i in 1:3
                    println(io, join(Int.(r.H[i, :]), " "))
                end
            end
        end
        r = nothing; GC.gc(); CUDA.reclaim()
    end
end
