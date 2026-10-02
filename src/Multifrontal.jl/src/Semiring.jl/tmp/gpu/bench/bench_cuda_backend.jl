# CUDA C++ backend vs the Julia (CUDA.jl) kernels on the same algorithms.
#
#   cuda/build.sh
#   julia --project=. -t 16 bench/bench_cuda_backend.jl gemm
#   julia --project=. -t 16 bench/bench_cuda_backend.jl closure [graph ...]
#
# GEMM: G multiply-adds/s (m n k / time), CUDA events around `reps` back-to-back
# calls, best of 3. Closure: wall time of closure_gpu! / closure_cuda! including a
# final synchronize, min of 3 runs, then one synchronized run per phase.
include(joinpath(@__DIR__, "..", "src", "SemiringGPU.jl"))
include(joinpath(@__DIR__, "..", "cuda", "cuda_backend.jl"))
using .SemiringGPU, .SemiringCUDA
using .SemiringGPU: Semiring
using .Semiring: MinPlus, MaxMin, PlusProd, ChordalSLU
using CUDA, LinearAlgebra, SparseArrays, Printf, Random

# the GPU is shared: wait (up to 10 min) while another process uses it
function wait_gpu()
    for i in 1:30
        others = filter(!=(string(getpid())), split(readchomp(`nvidia-smi --query-compute-apps=pid --format=csv,noheader`)))
        isempty(others) && return true
        i == 1 && println("  [GPU busy: pids $(join(others, ", ")); waiting]")
        sleep(20)
    end
    println("  [GPU still busy after 10 min: timings below may be contended]")
    return false
end

# Laptop GPU, 50 W cap: the SM clock swings between ~1.3 and ~2.1 GHz with power
# and temperature. So all candidates for one shape are timed in short bursts
# (~10 ms), interleaved round-robin after a short idle period, and each keeps its
# best burst: every candidate sees the same thermal/power conditions.
function rates(fs::Vector, m, n, k; rounds = 7, burst = 0.010)
    reps = map(fs) do f
        f(); CUDA.synchronize()
        t = CUDA.@elapsed f()
        clamp(round(Int, burst / max(t, 1e-6)), 1, 1000)
    end
    best = fill(Inf, length(fs))
    for _ in 1:rounds, (i, f) in enumerate(fs)
        sleep(0.05)
        t = CUDA.@elapsed for _ in 1:reps[i]
            f()
        end
        best[i] = min(best[i], t / reps[i])
    end
    return m * n * k ./ best ./ 1e9
end

const JL_TILINGS = (SemiringGPU.TILING_LARGE, SemiringGPU.TILING_SMALL, SemiringGPU.TILING_N32, SemiringGPU.TILING_N16)
tname(::SemiringGPU.Tiling{BM, BN}) where {BM, BN} = "$(BM)×$(BN)"
const CU_TILINGS = Dict(1 => "128×128", 2 => "128×64", 3 => "64×64", 4 => "128×32", 5 => "256×16", 6 => "128×128k16", 7 => "128×64k16", 8 => "128×16", 9 => "128×32k32", 10 => "128×16k32")

function bench_gemm()
    println(CUDA.name(CUDA.device()), "; Julia GEMM_VERSION = ", SemiringGPU.GEMM_VERSION[])
    shapes = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (27000, 64, 64), (27000, 20, 500), (27000, 500, 20),
              (8192, 8192, 64), (2048, 512, 512), (15625, 64, 64), (15625, 300, 64)]
    for (s, label) in [(MinPlus(), "MinPlus Float32"), (PlusProd(), "PlusProd Float32")]
        println("\n$label, G multiply-adds/s")
        @printf("  %-18s | %8s %-9s %8s %-9s | %8s %-9s %8s %-11s | %8s | %s\n", "m × n × k", "Julia", "(auto)", "Julia", "(best)", "C++", "(auto)", "C++", "(best)", "cuBLAS", "C++/Julia auto, best")
        for (m, n, k) in shapes
            T = Float32
            A = CuArray(rand(T, m, k) .* 100); B = CuArray(rand(T, k, n) .* 100); C = CuArray(rand(T, m, n) .* 100)
            wait_gpu()
            jt = SemiringGPU.choose_tiling(m, n)
            ct = SemiringCUDA.choose_tiling(m, n, k)
            fs = Any[() -> sgemx_gpu!(s, C, A, B), () -> sgemx_cuda!(s, C, A, B)]
            append!(fs, [() -> sgemx_gpu!(s, C, A, B; tiling) for tiling in JL_TILINGS])
            append!(fs, [() -> sgemx_cuda!(s, C, A, B; tiling) for tiling in 1:10])
            s isa PlusProd && push!(fs, () -> mul!(C, A, B, true, true))
            r = rates(fs, m, n, k)
            ja, ca = r[1], r[2]
            jb, ib = findmax(r[3:6]); jbt = tname(JL_TILINGS[ib])
            cb, ic = findmax(r[7:16]); cbt = CU_TILINGS[ic]
            cb32 = s isa PlusProd ? r[17] : NaN
            @printf("  %-18s | %8.0f %-9s %8.0f %-9s | %8.0f %-9s %8.0f %-11s | %8.0f | %.2f×, %.2f×\n", "$m×$n×$k", ja, tname(jt), jb, jbt,
                ca, CU_TILINGS[ct], cb, cbt, cb32, ca / ja, cb / jb)
            get(ENV, "VERBOSE", "0") == "1" && println("      Julia: ", join([@sprintf("%s %.0f", tname(t), r[2 + i]) for (i, t) in enumerate(JL_TILINGS)], ", "),
                "\n      C++:   ", join([@sprintf("%s %.0f", CU_TILINGS[i], r[6 + i]) for i in 1:10], ", "))
            A = B = C = nothing; GC.gc(); CUDA.reclaim()
        end
    end
end

# ----- closure -----

const MTX = joinpath(@__DIR__, "..", "data", "mtx")

function read_mtx(path, ::Type{T} = Float32) where {T}
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

# per-column 64-bit fingerprint of a Float32 matrix, on the GPU (two n × n copies on the
# host would not fit next to the developer's processes: earlyoom kills the run)
function colhash_kernel(h, D)
    j = blockIdx().x
    acc = UInt64(0)
    i = threadIdx().x
    @inbounds while i <= size(D, 1)
        b = UInt64(reinterpret(UInt32, D[i, j]))
        acc += ((b << 20) ⊻ b ⊻ UInt64(i)) * 0x9e3779b97f4a7c15
        i += blockDim().x
    end
    CUDA.@atomic h[j] += acc
    return
end

function fingerprint(D::CuMatrix{Float32})
    h = CUDA.zeros(UInt64, size(D, 2))
    @cuda threads = 256 blocks = size(D, 2) colhash_kernel(h, D)
    return Array(h)
end

function timed(f; trials = 3)
    f(); CUDA.synchronize()
    return minimum(@elapsed((f(); CUDA.synchronize())) for _ in 1:trials)
end

function phases(f)
    tm = Dict{Symbol, Float64}()
    f(tm); CUDA.synchronize()
    return join([@sprintf("%s %.1f", k, 1e3v) for (k, v) in sort(collect(tm); by = last, rev = true)], "  ")
end

# the two schedules of src/sgetrs.jl: level by level (as when this backend was written) and
# the developer's newer persistent L sweep + warp-per-source path walk (on by default)
const SCHEDULES = isdefined(SemiringGPU, :PERSISTENT) ? [false, true] : [false]
setschedule!(on) = isdefined(SemiringGPU, :PERSISTENT) && (SemiringGPU.PERSISTENT[] = on; SemiringGPU.PATH_WARP[] = on)

function bench_closure(names)
    println(CUDA.name(CUDA.device()))
    # warm up (compile) on a small graph
    let A0 = read_mtx(joinpath(MTX, "grid3d-25.mtx"))[1:2000, 1:2000]
        F0 = ChordalSLU(MinPlus(), A0); copyto!(F0, A0); P0 = FactorPlan(F0; large = 64, graph = false, nstreams = 8); factorize!(P0)
        G0 = GPUSLU(P0; large = 64)
        for sched in SCHEDULES
            setschedule!(sched); closure_gpu(G0); closure_cuda(G0); closure_cuda(G0; variant = 0)
        end
        precompute_ops!(G0); closure_gpu(G0); closure_cuda(G0); CUDA.reclaim()
    end
    for name in names
        A = read_mtx(joinpath(MTX, name * ".mtx")); n = size(A, 1)
        F = ChordalSLU(MinPlus(), A); copyto!(F, A)
        P = FactorPlan(F; large = 256, graph = false, nstreams = 8); factorize!(P)
        G = GPUSLU(P; large = 8192)
        D = CuMatrix{Float32}(undef, n, n); M = CuMatrix{Float32}(undef, n, G.maxna)
        println("\n$name  n = $n, $(SemiringGPU.nlarge(G)) large fronts, maxna = $(G.maxna)")
        for ops in (false, true), sched in SCHEDULES
            ops && isnothing(G.ops[]) && precompute_ops!(G)
            setschedule!(sched)
            wait_gpu()
            tj = timed(() -> closure_gpu!(D, G; M))
            hj = fingerprint(D)
            t0 = timed(() -> closure_cuda!(D, G; M, variant = 0))
            ok0 = fingerprint(D) == hj
            t1 = timed(() -> closure_cuda!(D, G; M, variant = 1))
            ok1 = fingerprint(D) == hj
            @printf("  ops=%-5s %-10s  Julia %.4f s | C++ port %.4f s (%.2f×, %s) | C++ reg %.4f s (%.2f×, %s)\n", ops, sched ? "persistent" : "levels",
                tj, t0, tj / t0, ok0 ? "=" : "MISMATCH", t1, tj / t1, ok1 ? "=" : "MISMATCH")
            println("      Julia    phases (ms, synced): ", phases(tm -> closure_gpu!(D, G; M, timer = tm)))
            println("      C++ port phases (ms, synced): ", phases(tm -> closure_cuda!(D, G; M, timer = tm, variant = 0)))
            println("      C++ reg  phases (ms, synced): ", phases(tm -> closure_cuda!(D, G; M, timer = tm, variant = 1)))
        end
        setschedule!(true)
        D = M = G = P = F = nothing; GC.gc(); CUDA.reclaim()
    end
end

mode = isempty(ARGS) ? "gemm" : ARGS[1]
if abspath(PROGRAM_FILE) == @__FILE__
if mode == "gemm"
    bench_gemm()
elseif mode == "closure"
    bench_closure(length(ARGS) > 1 ? ARGS[2:end] : ["grid3d-25", "grid2d-150", "grid3d-30", "grid2d-180"])
end
end
