# B200 sweep of the candidate rule of GPU-MF1 (LATENCY_NOTES.md, step 3): rounds, time and
# device fill counters (nnz(L), flops, tw; unit weights) per graph and config.
# usage: julia rsel.jl name1,name2,...   env: OUT, DATA, CORPUS, CFGS, EXTRA, REPS
#   cfg syntax: r<relax>u<urelax>  (e.g. r0.1u0 = the u0.1 quality config, r0.1u0.02),
#               r<relax>a<F>c<cap>  (adaptive untouched window: target F untouched candidates, cap),
#               v2 = relax 0.1 for all vertices
const DIR = @__DIR__
using CUDA, Serialization, SparseArrays, Random
include(joinpath(DIR, "Rounds.jl")); using .Rounds
include(joinpath(DIR, "evalutil.jl")); include(joinpath(DIR, "gens.jl"))
include(ENV["CORPUS"] * "/load.jl"); using .TNCorpus
include(joinpath(DIR, "GRounds.jl")); using .GRounds

const dirs = split(get(ENV, "DATA", ""), ":")
function loadg(name)
    for d in dirs
        f = joinpath(d, name * ".bin")
        if isfile(f)
            n, I, J = deserialize(f)
            return SimpleGraph(sparse(vcat(I, J), vcat(J, I), true, n, n, |))
        end
    end
    startswith(name, "lat2_") && return lat2(parse(Int, name[6:end]))
    startswith(name, "lat3_") && return lat3(parse(Int, name[6:end]))
    return TNCorpus.load(name).graph
end
function cfg(c)
    c == "v2" && return RoundsAlg(; score = :mf1, relax = 0.1), (;)
    m = match(r"^r([0-9.]+)a([0-9]+)c([0-9.]+)$", c)       # adaptive untouched window
    m === nothing || return RoundsAlg(; score = :mf1, relax = parse(Float64, m[1])),
        (; urelax = 0.0, uadapt = parse(Int, m[2]), umax = parse(Float64, m[3]))
    m = match(r"^r([0-9.]+)u([0-9.]+)$", c)
    m === nothing && error("cfg $c")
    return RoundsAlg(; score = :mf1, relax = parse(Float64, m[1])), (; urelax = parse(Float64, m[2]))
end
const CFGS = split(get(ENV, "CFGS", "v2,r0.1u0,r0.1u0.01,r0.1u0.02,r0.1u0.05"), ',')
const EXTRA = eval(Meta.parse(get(ENV, "EXTRA", "(;)")))     # trusted: set by our own sbatch files
const REPS = parse(Int, get(ENV, "REPS", "2"))
let g = lat2(12), u = ones(nv(g))
    for c in CFGS
        a, kw = cfg(c); GRounds.grounds(u, g, a; kw..., EXTRA...)
    end
end
println(CUDA.name(CUDA.device())); flush(stdout)
out = open(ENV["OUT"], "a")
for name in split(ARGS[1], ',')
    g = try loadg(String(name)) catch err; println("LOAD FAILED $name: $err"); continue end
    n = nv(g); u = ones(n)
    n <= 2_000_000 && let (a0, kw0) = cfg(CFGS[1]); GRounds.grounds(u, g, a0; kw0..., EXTRA...); end   # warm (allocator, first-touch)
    for c in CFGS
        try
            a, kw = cfg(c)
            t = Inf; tm = Dict{Symbol, Any}()
            for r in 1:(n < 200_000 ? max(REPS, 3) : n < 1_500_000 ? REPS : 1)
                tmi = Dict{Symbol, Any}()
                t1 = @elapsed GRounds.grounds(u, g, a; timing = tmi, kw..., EXTRA...)
                t1 < t && (t = t1; tm = tmi)
                t1 > 30 && break
            end
            println(out, join((name, n, ne(g), c, tm[:rounds], tm[:nnz], tm[:flops], tm[:tw], round(t, digits = 4), round(tm[:gpu], digits = 4)), '\t')); flush(out)
            println(rpad(name, 16), rpad(c, 12), " rounds=", tm[:rounds], " nnz=", tm[:nnz], " t=", round(t, digits = 4)); flush(stdout)
        catch err
            println("FAILED $name $c: ", first(sprint(showerror, err), 400)); flush(stdout)
        end
        GC.gc(); CUDA.reclaim()
    end
end
close(out)
