# apsp_gpu under warm and cold conditions, for fair comparisons with codes that run once per process.
#
#   julia --project=. -t 16 bench/cold.jl [--reps=5] graph ...         warm and cold-memory calls
#   julia --project=. -t 16 bench/cold.jl --first graph                one first call in this process
#
#   warm         best/median of calls after two warm-up calls (the method of the other benchmarks):
#                compiled kernels, tuned GEMMs, CUDA.jl's memory pool and the pinned host arena all warm
#   cold memory  before each call the memory pool is returned to the driver (CUDA.reclaim after a full
#                collection) and the pinned arena is freed, so every call allocates its device and pinned
#                memory again, as a process that runs once does; compiled code and tuning stay warm
#   first        the first call of a fresh process (also compiling the kernels; the tuning file on disk is
#                read, as after any earlier run on this machine)
include(joinpath(@__DIR__, "bench_apsp.jl"))
using CUDA, Printf, Statistics

const S = SemiringGPU
const REPS = something(tryparse(Int, replace(something(findfirst(a -> startswith(a, "--reps="), ARGS) |> i -> isnothing(i) ? nothing : ARGS[i], "--reps=5"), "--reps=" => "")), 5)
const FIRST = "--first" in ARGS
const GRAPHS = filter(a -> !startswith(a, "--"), ARGS)

function cold!()
    GC.gc(true); CUDA.reclaim()
    isdefined(S, :settle_arena) && S.settle_arena()       # (an arena still being pinned in the background)
    isdefined(S, :ARENA) && lock(S.ARENA_LOCK) do         # (versions with the pinned host arena)
        isnothing(S.ARENA.mem) || S.CUDA.CUDACore.free(S.ARENA.mem)
        S.ARENA.mem = nothing; S.ARENA.size = 0
    end
    return
end

call(A) = (D = S.apsp_gpu(A); CUDA.synchronize(); CUDA.unsafe_free!(D))

for name in GRAPHS
    A = read_mtx(joinpath(MTX, name * ".mtx")); n = size(A, 1)

    if FIRST
        t = @elapsed call(A)
        @printf("first    %-16s n=%7d  first call in a fresh process %8.1f ms\n", name, n, 1e3t)
        continue
    end

    call(A); call(A)
    warm = [begin GC.gc(false); @elapsed(call(A)) end for _ in 1:REPS]
    cold = [begin cold!(); @elapsed(call(A)) end for _ in 1:REPS]
    @printf("coldmem  %-16s n=%7d  warm best %7.1f median %7.1f ms | cold memory best %7.1f median %7.1f ms | +%.1f ms\n",
        name, n, 1e3minimum(warm), 1e3median(warm), 1e3minimum(cold), 1e3median(cold), 1e3(minimum(cold) - minimum(warm)))
    GC.gc(); CUDA.reclaim()
end
