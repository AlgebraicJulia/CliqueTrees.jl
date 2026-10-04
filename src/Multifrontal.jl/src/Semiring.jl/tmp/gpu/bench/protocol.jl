# Our solver under the comparison protocol (contenders/FAIRNESS.md): per graph, cold-memory and warm calls
# (median, min, max of REPS), the step tree of cold calls, and the checks against the reference.
#
#   OUT=protocol.jsonl TAG=k4 ROWHASH=dir julia --project=. -t 16 bench/protocol.jl [--reps=5] graph ...
#
#   cold   before each call the memory pool is returned to the driver and the pinned arena freed, so the
#          call allocates its memory as a process that runs once does (compiled code and the on-disk GEMM
#          tuning stay warm); the headline number
#   warm   calls in a warm process, reusing pools (the library-reuse case)
#   steps  the step timer's tree for cold calls (each step synchronizes; the relabel step is the time to
#          bring the result into the original vertex order)
#   check  checksum (sum of finite distances, unreachable count) and the position-sensitive full hash of
#          bench/rowhash.jl, compared with ROWHASH/<graph>.rowhash when given
#   elim   cold calls with columns = :elimination (no relabel: the result in the solver's column order, as
#          codes that return their own order), when the version has it; its row hashes, taken with each
#          column's original label, must equal the others
include(joinpath(@__DIR__, "bench_apsp.jl"))
using CUDA, Printf, Statistics

const S = SemiringGPU
const OUT = get(ENV, "OUT", "protocol.jsonl")
const TAG = get(ENV, "TAG", "")
const ROWHASH = get(ENV, "ROWHASH", "")
const REPS = something(tryparse(Int, replace(something(findfirst(a -> startswith(a, "--reps="), ARGS) |> i -> isnothing(i) ? nothing : ARGS[i], "--reps=5"), "--reps=" => "")), 5)
const GRAPHS = filter(a -> !startswith(a, "--"), ARGS)
const K1 = 0x9E3779B97F4A7C15; const K2 = 0x632BE59BD9B4E019; const K3 = 0xD6E8FEB86659FD93

function cold!()
    GC.gc(true); CUDA.reclaim()
    isdefined(S, :settle_arena) && S.settle_arena()       # (an arena still being pinned in the background)
    isdefined(S, :ARENA) && lock(S.ARENA_LOCK) do
        isnothing(S.ARENA.mem) || S.CUDA.CUDACore.free(S.ARENA.mem)
        S.ARENA.mem = nothing; S.ARENA.size = 0
    end
    return
end

val(x) = isfinite(x) ? UInt64(round(x)) : UInt64(0xFFFFFFFF)

function rowhashes(D, cols = 1:size(D, 2))
    n = size(D, 1); h = CUDA.zeros(UInt64, n); b = max(1, 2^29 ÷ n)
    for j0 in 1:b:n
        J = j0:min(n, j0 + b - 1)
        w = CuVector{UInt64}(UInt64.(cols[J]) .* K1 .+ K2)
        h .+= vec(sum(val.(view(D, :, J)) .* w'; dims = 2))
    end
    return Array(h)
end

stats(v) = (median = median(v), min = minimum(v), max = maximum(v), runs = v)
json(x::AbstractString) = "\"" * x * "\""
json(x::Real) = isfinite(x) ? string(x) : "null"
json(x::Bool) = string(x)
json(::Nothing) = "null"
json(x::Union{Tuple, AbstractVector}) = "[" * join(json.(x), ",") * "]"
json(d::AbstractDict) = "{" * join([json(string(k)) * ":" * json(v) for (k, v) in d], ",") * "}"
json(x::NamedTuple) = json(Dict(pairs(x)))
total(tm) = sum(v for (k, v) in tm.times if !occursin('/', k); init = 0.0)

println("GPU: ", CUDA.name(CUDA.device()), " | host threads: ", Threads.nthreads(), " | tag: ", TAG)

for name in GRAPHS
    try
        A = read_mtx(joinpath(MTX, name * ".mtx")); n = size(A, 1)
        call() = (D = S.apsp_gpu(A); CUDA.synchronize(); CUDA.unsafe_free!(D))
        # check (also the warm-up)
        D = S.apsp_gpu(A); CUDA.synchronize()
        chk = (mapreduce(x -> isfinite(x) ? Float64(x) : 0.0, +, D), count(!isfinite, D))
        rh = rowhashes(D); CUDA.unsafe_free!(D)
        full = sum(rh[i] * (UInt64(i) * K3 + one(UInt64)) for i in 1:n)
        refpath = joinpath(ROWHASH, name * ".rowhash")
        hashok = isempty(ROWHASH) || !isfile(refpath) ? nothing : rh == [parse(UInt64, l; base = 16) for l in eachline(refpath)]
        elim = try
            E, cols = S.apsp_gpu(A; columns = :elimination); CUDA.synchronize()
            ok = rowhashes(E, cols) == rh; CUDA.unsafe_free!(E); ok
        catch
            nothing                                           # (a version without columns = :elimination)
        end
        calle() = ((E, _) = S.apsp_gpu(A; columns = :elimination); CUDA.synchronize(); CUDA.unsafe_free!(E))
        call()
        warm = [begin GC.gc(false); @elapsed(call()) end for _ in 1:REPS]
        cold = [begin cold!(); @elapsed(call()) end for _ in 1:REPS]
        colde = elim === true ? [begin cold!(); @elapsed(calle()) end for _ in 1:REPS] : Float64[]
        steps = [begin cold!(); S.with_steps(call)[2] end for _ in 1:3]
        tm = steps[sortperm(total.(steps))[2]]               # (the run with the median total)
        @printf("%-6s %-16s n=%7d  cold median %7.1f [%7.1f–%7.1f] | warm median %7.1f [%7.1f–%7.1f] ms | relabel %5.1f ms | hash %s\n",
            TAG, name, n, 1e3median(cold), 1e3minimum(cold), 1e3maximum(cold), 1e3median(warm), 1e3minimum(warm), 1e3maximum(warm),
            1e3 * get(tm.times, "relabel columns", 0.0), hashok === nothing ? "(no reference)" : hashok ? "ok" : "MISMATCH")
        isempty(colde) || @printf("%-6s %-16s elimination-order columns: cold median %7.1f [%7.1f–%7.1f] ms | hash %s\n",
            TAG, name, 1e3median(colde), 1e3minimum(colde), 1e3maximum(colde), elim ? "ok" : "MISMATCH")
        elim === false && println(TAG, " ", name, ": elimination-order output MISMATCH")
        open(OUT, "a") do io
            println(io, "{\"tag\":", json(TAG), ",\"gpu\":", json(CUDA.name(CUDA.device())), ",\"graph\":", json(name), ",\"n\":", n,
                ",\"nnz\":", nnz(A), ",\"checksum\":", json(chk), ",\"fullhash\":", json(string(full; base = 16, pad = 16)),
                ",\"rowhash_ok\":", json(hashok), ",\"cold\":", json(stats(cold)), ",\"warm\":", json(stats(warm)),
                ",\"elim_ok\":", json(elim), ",\"cold_elim\":", isempty(colde) ? "null" : json(stats(colde)),
                ",\"steps\":", json(tm.times), ",\"order\":", json(tm.order), "}")
        end
    catch e
        println(TAG, " ", name, ": failed: ", first(sprint(showerror, e), 300))
    end

    GC.gc(); CUDA.reclaim()
end

