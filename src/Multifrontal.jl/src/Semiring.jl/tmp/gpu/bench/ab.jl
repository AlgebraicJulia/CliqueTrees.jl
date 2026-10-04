# A/B harness: what one apsp_gpu call costs, in enough detail to see where a change moved time.
#
#   OUT=results.jsonl TAG=base-r1 julia --project=. -t 16 bench/ab.jl [--plain=7] graph ...
#
# Per graph, one JSON line with:
#   plain        wall times of plain calls (after warm-up), and the GC time of each
#   steps        the step timer's tree (best of 3 by total), its GC time and host bytes per step
#   kernels      GPU time and launch count per kernel family (one profiled call, CUPTI)
#   api          host time and count per CUDA API call in that call
#   copies       bytes, time and bandwidth of each copy kind
#   gpu          the call's span, GPU busy time, and the longest stretches with the GPU idle, each with
#                the CUDA API calls the host made meanwhile (where host work holds the GPU up)
#   checksum     Float64 sum of the finite distances, and the number of infinite ones
# bench/ab_compare.py turns two tags into a table of differences.
include(joinpath(@__DIR__, "bench_apsp.jl"))
using CUDA, Printf, Statistics

const S = SemiringGPU
const OUT = get(ENV, "OUT", "ab.jsonl")
const TAG = get(ENV, "TAG", "")
const NPLAIN = something(tryparse(Int, replace(something(findfirst(a -> startswith(a, "--plain="), ARGS) |> i -> isnothing(i) ? nothing : ARGS[i], "--plain=7"), "--plain=" => "")), 7)
const GRAPHS = filter(a -> !startswith(a, "--"), ARGS)

json(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
json(x::Bool) = string(x)
json(x::Real) = isfinite(x) ? string(x) : "null"
json(::Nothing) = "null"
json(x::Union{Tuple, AbstractVector}) = "[" * join(json.(x), ",") * "]"
json(d::AbstractDict) = "{" * join([json(string(k)) * ":" * json(v) for (k, v) in d], ",") * "}"
json(x::NamedTuple) = json(Dict(pairs(x)))

family(name) = startswith(name, "[") ? name : rstrip(first(split(name, '(')), '_')

# the union of [start, stop) intervals, and the gaps between them longer than mingap
function busy_and_gaps(starts, stops, t0, t1; mingap = 2e-4)
    o = sortperm(starts); busy = 0.0; gaps = Tuple{Float64, Float64}[]
    cur0, cur1 = t0, t0

    for i in o
        a, b = starts[i], stops[i]

        if a > cur1
            busy += cur1 - cur0
            a - cur1 >= mingap && push!(gaps, (cur1, a))
            cur0, cur1 = a, b
        else
            cur1 = max(cur1, b)
        end
    end

    busy += cur1 - cur0
    t1 - cur1 >= mingap && push!(gaps, (cur1, t1))
    return busy, gaps
end

function profile_call(call)
    r = CUDA.@profile trace = false call()
    h, d = r.host, r.device
    t0 = minimum(h.start; init = Inf); t1 = maximum(h.stop; init = -Inf)
    kern = Dict{String, Vector{Float64}}(); copies = Dict{String, Vector{Float64}}(); api = Dict{String, Vector{Float64}}()

    for i in eachindex(d.name)
        nm = d.name[i]; dt = d.stop[i] - d.start[i]

        if startswith(nm, "[")
            c = get!(() -> [0.0, 0.0, 0.0], copies, nm)
            c[1] += dt; c[2] += 1; c[3] += coalesce(d.size[i], 0)
        else
            k = get!(() -> [0.0, 0.0], kern, family(nm))
            k[1] += dt; k[2] += 1
        end
    end

    for i in eachindex(h.name)
        a = get!(() -> [0.0, 0.0], api, h.name[i])
        a[1] += h.stop[i] - h.start[i]; a[2] += 1
    end

    busy, gaps = busy_and_gaps(d.start, d.stop, t0, t1)
    sort!(gaps; by = g -> g[1] - g[2])
    # what the host was calling during the longest idle stretches
    idle = map(gaps[1:min(end, 8)]) do (a, b)
        inside = Dict{String, Float64}()

        for i in eachindex(h.name)
            lo = max(a, h.start[i]); hi = min(b, h.stop[i])
            hi > lo && (inside[h.name[i]] = get(inside, h.name[i], 0.0) + (hi - lo))
        end

        top = sort(collect(inside); by = x -> -x[2])[1:min(end, 3)]
        Dict("at" => a - t0, "length" => b - a, "host" => Dict(top))
    end

    return Dict("span" => t1 - t0, "busy" => busy, "idle" => idle,
        "kernels" => Dict(k => (time = v[1], count = v[2]) for (k, v) in kern),
        "copies" => Dict(k => (time = v[1], count = v[2], bytes = v[3], gbs = v[1] > 0 ? v[3] / v[1] / 1e9 : 0.0) for (k, v) in copies),
        "api" => Dict(k => (time = v[1], count = v[2]) for (k, v) in api))
end

total(tm) = sum(v for (k, v) in tm.times if !occursin('/', k); init = 0.0)

println("GPU: ", CUDA.name(CUDA.device()), " | CPU threads: ", Threads.nthreads(), " | tag: ", TAG)

for name in GRAPHS
    try
        A = read_mtx(joinpath(MTX, name * ".mtx")); n = size(A, 1)
        chk = Ref{Any}(nothing)
        call(check = false) = begin
            D = S.apsp_gpu(A); CUDA.synchronize()
            check && (chk[] = (mapreduce(x -> isfinite(x) ? Float64(x) : 0.0, +, D), count(!isfinite, D)))
            CUDA.unsafe_free!(D)
        end
        call(true); call()
        plain = Float64[]; plaingc = Float64[]

        for _ in 1:NPLAIN
            GC.gc(false); g0 = Base.gc_time_ns()
            push!(plain, @elapsed(call())); push!(plaingc, (Base.gc_time_ns() - g0) / 1e9)
        end

        runs = [begin GC.gc(false); S.with_steps(() -> call())[2] end for _ in 1:3]
        tm = runs[argmin(total.(runs))]
        GC.gc(false)
        prof = profile_call(() -> call())
        @printf("%-8s %-16s n=%7d  plain min %7.1f  median %7.1f ms | stepped %7.1f ms | GPU busy %5.1f%% of %6.1f ms | checksum %.6e/%d\n",
            TAG, name, n, 1e3 * minimum(plain), 1e3 * median(plain), 1e3 * total(tm), 100 * prof["busy"] / prof["span"], 1e3 * prof["span"], chk[]...)

        open(OUT, "a") do io
            println(io, "{\"tag\":", json(TAG), ",\"gpu\":", json(CUDA.name(CUDA.device())), ",\"graph\":", json(name), ",\"n\":", n,
                ",\"nnz\":", nnz(A), ",\"checksum\":", json(chk[]), ",\"plain\":", json(plain), ",\"plaingc\":", json(plaingc),
                ",\"steps\":", json(tm.times), ",\"order\":", json(tm.order), ",\"stepgc\":", json(tm.gc),
                ",\"stepbytes\":", json(Dict(k => Float64(v) for (k, v) in tm.bytes)), ",\"profile\":", json(prof), "}")
        end
    catch e
        println(TAG, " ", name, ": failed: ", first(sprint(showerror, e), 300))
    end

    GC.gc(); CUDA.reclaim()
end
