# Portability harness: the same graphs on any NVIDIA GPU, reported as time and as a fraction of what
# this GPU can do (its measured min-plus throughput and copy bandwidth, src/device.jl).
#
#   julia --project=. -t auto bench/portable.jl [graph ...]        (default: every graph below that fits)
#
# Per graph: numeric phases (wall, synchronized), closure (wall, min of REPS), per-family GPU time of
# one closure and one numeric factorization (CUPTI), efficiency, and a checksum of the closure so that
# runs on different GPUs and kernel versions can be compared for identical results.
# Appends one JSON line per graph to bench/results_portable.jsonl.
include(joinpath(@__DIR__, "bench_apsp.jl"))
using CUDA, Printf

const GRAPHS = ["grid3d-25", "grid2d-150", "grid3d-30", "grid2d-180", "delaunay_n14", "delaunay_n15", "ca-HepTh", "ca-CondMat"]
const REPS = parse(Int, get(ENV, "REPS", "3"))
const OUT = get(ENV, "OUT", joinpath(@__DIR__, "results_portable.jsonl"))
const TAG = get(ENV, "TAG", "")
# solver settings: environment variables SEMIRINGGPU_<SETTING> (see GPUConfig), e.g. SEMIRINGGPU_MERGE=8

include(joinpath(@__DIR__, "portable_fam.jl"))

nnzL(G) = sum(f -> (nn = G.hRptr[f + 1] - G.hRptr[f]; na = G.hSptr[f + 1] - G.hSptr[f]; nn * (nn + 1) / 2 + nn * na), 1:G.nf)
json(x::AbstractString) = "\"" * replace(x, "\"" => "\\\"") * "\""
json(x::Real) = isfinite(x) ? string(x) : "null"
json(x::Dict) = "{" * join([json(string(k)) * ":" * json(v) for (k, v) in sort(collect(x); by = first)], ",") * "}"
json(x::Tuple) = "[" * join(json.(x), ",") * "]"

function bench(name, p)
    A = read_mtx(joinpath(MTX, name * ".mtx")); n = size(A, 1)
    need = 1.3 * n^2 * sizeof(T)
    need > CUDA.free_memory() && (println("$name: skipped, needs $(round(need / 2^30; digits = 1)) GB"); return)
    F = ChordalSLU(MinPlus(), A)
    ph = Dict{String, Float64}()
    function numeric()
        ph["copy_in"] = @elapsed copyto!(F, A)
        ph["plan"] = @elapsed (P = FactorPlan(F; large = 256, graph = false, nstreams = 8))
        h = factorize!(P); ph["cpu_bottom"] = h.cpu; ph["upload"] = h.upload; ph["gpu_top"] = h.gpu
        ph["gpuslu"] = @elapsed (G = GPUSLU(P; large = 8192); CUDA.synchronize())
        ph["precompute"] = @elapsed (precompute_ops!(G); CUDA.synchronize())
        return G
    end
    numeric(); G = numeric()                              # compile, then warm
    best = copy(ph)
    for _ in 2:REPS; numeric(); for (k, v) in ph; best[k] = min(best[k], v); end; end
    G = numeric()
    D = CuMatrix{T}(undef, n, n); M = CuMatrix{T}(undef, n, G.maxna)
    closure_gpu!(D, G; M); CUDA.synchronize()
    t_clo = minimum(@elapsed((closure_gpu!(D, G; M); CUDA.synchronize())) for _ in 1:REPS)
    fam_clo = families(CUDA.@profile(trace = true, (closure_gpu!(D, G; M); CUDA.synchronize())))
    fam_num = families(CUDA.@profile(trace = true, numeric()))
    # checksum: exact for integer weights (sums of Float32 integers well below 2^53)
    chk = (mapreduce(x -> isfinite(x) ? Float64(x) : 0.0, +, D), count(x -> !isfinite(x), D))
    work = n * nnzL(G)
    t_num = sum(values(best))
    eff = work / t_clo / p.minplus
    @printf("%-12s n=%6d nf=%5d | numeric %7.1f ms  closure %7.1f ms | %.2f T ops/s = %4.1f%% of min-plus peak | checksum %.6e/%d\n",
        name, n, G.nf, 1e3t_num, 1e3t_clo, work / t_clo / 1e12, 100eff, chk...)
    print("      numeric phases: "); for (k, v) in sort(collect(best); by = last, rev = true); @printf("%s %.1f  ", k, 1e3v); end; println("ms")
    for (lab, fm) in (("closure", fam_clo), ("numeric", fam_num))
        print("      $lab GPU time: ")
        for (k, (t, c)) in sort(collect(fm); by = x -> -x[2][1]); @printf("%s %.2f ms (%d)  ", k, 1e3t, c); end
        println()
    end
    open(OUT, "a") do io
        println(io, "{\"tag\":", json(TAG), ",\"gpu\":", json(p.name), ",\"graph\":", json(name), ",\"n\":", n, ",\"nf\":", G.nf,
            ",\"work\":", work, ",\"closure\":", t_clo, ",\"numeric\":", t_num, ",\"phases\":", json(best),
            ",\"fam_closure\":", json(Dict(k => v[1] for (k, v) in fam_clo)), ",\"fam_numeric\":", json(Dict(k => v[1] for (k, v) in fam_num)),
            ",\"minplus\":", p.minplus, ",\"bandwidth\":", p.bandwidth, ",\"checksum\":", json(chk), "}")
    end
    CUDA.unsafe_free!(D); CUDA.unsafe_free!(M); CUDA.reclaim()
end

p = measure!(device_profile())
println(p)
for g in (isempty(ARGS) ? GRAPHS : ARGS)
    isfile(joinpath(MTX, g * ".mtx")) ? bench(g, p) : println("$g: no data/mtx/$g.mtx")
end
