# Where one apsp_gpu(A) call spends its time, step by step (the @step / @phase marks in src/):
# symbolic, factor storage, copy, plan (and its parts), factorization (CPU bottom, upload, GPU top and
# its kernel phases), solve setup (and its parts), operators, closure (and its phases), relabelling,
# and with --host the copy to the host. Every step synchronizes the device, so GPU work is charged to
# the step that issued it; the plain call (no timer) is timed too, for the cost of that.
#
#   julia --project=. -t auto bench/breakdown.jl [--host] [--reps=3] graph ...      (graphs in data/mtx)
include(joinpath(@__DIR__, "bench_apsp.jl"))
using CUDA, Printf

const S = SemiringGPU
const SR = S.Semiring
const HOST = "--host" in ARGS
const REPS = something(tryparse(Int, replace(something(findfirst(a -> startswith(a, "--reps="), ARGS) |> i -> isnothing(i) ? nothing : ARGS[i], "--reps=3"), "--reps=" => "")), 3)
const GRAPHS = filter(a -> !startswith(a, "--"), ARGS)

output = HOST ? :host : :device
const OUT = get(ENV, "OUT", "")
const TAG = get(ENV, "TAG", "")
# one call; the checksum of the closure (exact for integer weights: sums of Float32 integers below 2^53)
function call(A; check = false)
    D = S.apsp_gpu(A; output); CUDA.synchronize()
    chk = check ? (D isa CuArray ? (mapreduce(x -> isfinite(x) ? Float64(x) : 0.0, +, D), count(!isfinite, D)) :
                   (sum(x -> isfinite(x) ? Float64(x) : 0.0, D), count(!isfinite, D))) : nothing
    D isa CuArray && CUDA.unsafe_free!(D)
    return chk
end
total(tm) = sum(v for (k, v) in tm.times if !occursin('/', k); init = 0.0)
json(x::AbstractString) = "\"" * replace(x, "\"" => "\\\"") * "\""
json(x::Real) = isfinite(x) ? string(x) : "null"
json(x::Tuple) = "[" * join(json.(x), ",") * "]"
json(d::AbstractDict) = "{" * join([json(string(k)) * ":" * json(v) for (k, v) in d], ",") * "}"

println("GPU: ", CUDA.name(CUDA.device()), " | CPU threads: ", Threads.nthreads(), " | output = :$output")

for name in (isempty(GRAPHS) ? ["grid3d-30", "delaunay_n15", "ca-CondMat"] : GRAPHS)
    try
        A = read_mtx(joinpath(MTX, name * ".mtx")); n = size(A, 1)
        chk = call(A; check = true); call(A)                   # compile, tune, warm (and the checksum)
        plain = minimum(begin GC.gc(false); @elapsed(call(A)) end for _ in 1:REPS)
        runs = [begin GC.gc(false); S.with_steps(() -> call(A))[2] end for _ in 1:REPS]
        tm = runs[argmin(total.(runs))]
        # the ordering, separately: ssymbolic computes it inside (SCCs first; one per undirected component)
        alg = SR.DEFAULT_ELIMINATION_ALGORITHM
        S.CliqueTrees.permutation(A; alg)
        t_ord = minimum(@elapsed(S.CliqueTrees.permutation(A; alg)) for _ in 1:REPS)
        @printf("\n%s  n = %d, nnz = %d | apsp_gpu %.1f ms (plain call), %.1f ms with the step timer | checksum %.6e/%d\n",
            name, n, nnz(A), 1e3plain, 1e3total(tm), chk...)
        S.print_steps(tm)
        @printf("  (ordering alone, %s: %.1f ms of the symbolic step)\n", nameof(typeof(alg)), 1e3t_ord)
        isempty(OUT) || open(OUT, "a") do io
            println(io, "{\"tag\":", json(TAG), ",\"gpu\":", json(CUDA.name(CUDA.device())), ",\"graph\":", json(name),
                ",\"n\":", n, ",\"nnz\":", nnz(A), ",\"plain\":", json(plain), ",\"ordering\":", json(t_ord),
                ",\"checksum\":", json(chk), ",\"order\":[", join(json.(tm.order), ","), "],\"times\":", json(tm.times),
                ",\"gc\":", json(tm.gc), ",\"bytes\":", json(Dict(k => Float64(v) for (k, v) in tm.bytes)),
                ",\"counts\":", json(Dict(k => Float64(v) for (k, v) in tm.counts)), "}")
        end
    catch e
        println("\n", name, ": failed: ", first(sprint(showerror, e), 300))
    end
    GC.gc(); CUDA.reclaim()
end
