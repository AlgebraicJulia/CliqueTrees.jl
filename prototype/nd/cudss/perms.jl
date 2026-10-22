# CPU reference orders for cudss_bench.jl (amd / metis / mf1 configs): serial, one CliqueTrees version per process.
# usage: julia --project=ENV -t1 perms.jl TAG ALG OUTDIR name...
#   ALG: AMD (SuiteSparse through CliqueTrees, needs AMD.jl), METIS (CliqueTrees.METIS(), METIS_NodeND, needs Metis.jl),
#        MF1 (MF(strategy = 1): with TAG rmain this is Richard's origin/main serial MF1, env rr/renv)
#   writes OUTDIR/name.ALG.perm = serialize((order::Vector{Int32}, time::Float64, host, tag)); time = median of 3
#   (1 above n = 2M, METIS / MF1 above n = 1M), BipartiteGraph built outside the timed region (as pbase/pbench.jl)
using CliqueTrees, Graphs, SparseArrays, Serialization, Statistics
const HERE = @__DIR__
include(joinpath(ENV["SRC"], "gens.jl"))
tag, alg, outd = ARGS[1], ARGS[2], ARGS[3]
alg == "AMD" && @eval using AMD
alg == "METIS" && @eval using Metis
const DIRS = split(get(ENV, "DATA", ""), ":")
function loadg(name)
    for d in DIRS
        f = joinpath(d, name * ".bin")
        if isfile(f)
            n, I, J = deserialize(f)
            return SimpleGraph(sparse(vcat(I, J), vcat(J, I), true, n, n, |))
        end
    end
    startswith(name, "lat2_") && return lat2(parse(Int, name[6:end]))
    startswith(name, "lat3_") && return lat3(parse(Int, name[6:end]))
    error("no data for $name")
end
mk() = alg == "AMD" ? CliqueTrees.AMD() : alg == "METIS" ? CliqueTrees.METIS() : alg == "MF1" ? MF(; strategy = 1) : error(alg)
let bg = BipartiteGraph(lat2(10)); permutation(bg, mk()); end
mkpath(outd)
for name in ARGS[4:end]
    f = joinpath(outd, "$name.$alg.perm")
    isfile(f) && continue
    g = loadg(name); n = nv(g); bg = BipartiteGraph(g)
    reps = n > 2_000_000 || (alg != "AMD" && n > 1_000_000) ? 1 : 3
    ts = Float64[]; o = nothing
    for _ in 1:reps
        GC.gc()
        t = @elapsed (o = permutation(bg, mk())[1]); push!(ts, t)
    end
    @assert isperm(o)
    serialize(f * ".tmp", (Int32.(o), median(ts), gethostname(), tag)); mv(f * ".tmp", f; force = true)
    println(name, " ", alg, " n=", n, " t=", round(median(ts), digits = 3)); flush(stdout)
end
