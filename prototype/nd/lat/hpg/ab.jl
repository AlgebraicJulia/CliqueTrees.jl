# B200 before/after of the latency changes (LATENCY_NOTES.md): GRoundsBase (lat/base) vs the
# lat/ GRounds with variants given by VARS, same graph and config, alternating runs.
# usage: julia ab.jl name1,name2,...   env: OUT, PROFOUT (optional phase profile of the last
#   variant), DATA, CORPUS, CFGS (v2,u0.1), VARS (label=NamedTuple;label=NamedTuple...),
#   REPS (default 3 below 200k vertices, 2 below 1.5M, else 1)
const DIR = @__DIR__
using CUDA, Serialization, SparseArrays, Random
include(joinpath(DIR, "Rounds.jl")); using .Rounds
include(joinpath(DIR, "evalutil.jl")); include(joinpath(DIR, "gens.jl"))
include(ENV["CORPUS"] * "/load.jl"); using .TNCorpus
include(joinpath(DIR, "GRounds.jl")); using .GRounds
include(joinpath(DIR, "base", "GRounds.jl")); using .GRoundsBase
include(joinpath(DIR, "pfsum.jl"))

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
const CFGS = split(get(ENV, "CFGS", "v2,u0.1"), ',')
cfgkw(c) = c == "v2" ? (;) : c == "u0.1" ? (; urelax = 0.0) : error("cfg $c")
const alg = RoundsAlg(; score = :mf1, relax = 0.1)
# variants: label => kwargs (trusted strings from our own sbatch files)
const VARS = [(String(a), eval(Meta.parse(b))) for (a, b) in (split(x, '=', limit = 2) for x in split(get(ENV, "VARS", "new=(;)"), ';'))]
let g = lat2(12), u = ones(nv(g))
    for c in CFGS
        GRoundsBase.grounds(u, g, alg; cfgkw(c)...)
        for (_, vk) in VARS
            GRounds.grounds(u, g, alg; cfgkw(c)..., vk...)
        end
        haskey(ENV, "PROFOUT") && GRounds.grounds(u, g, alg; timing = Dict{Symbol, Any}(), rprof = 100, cfgkw(c)..., VARS[end][2]...)
    end
end
println(CUDA.name(CUDA.device())); flush(stdout)
out = open(ENV["OUT"], "a")
pout = haskey(ENV, "PROFOUT") ? open(ENV["PROFOUT"], "a") : nothing
for name in split(ARGS[1], ',')
    g = try loadg(String(name)) catch err; println("LOAD FAILED $name: $err"); continue end
    n = nv(g); u = ones(n)
    reps = parse(Int, get(ENV, "REPS", n < 200_000 ? "3" : n < 1_500_000 ? "2" : "1"))
    for c in CFGS
        try
            kw = cfgkw(c)
            runs = Any[("base", tmi -> GRoundsBase.grounds(u, g, alg; timing = tmi, kw...))]
            for (lab, vk) in VARS
                push!(runs, (lab, tmi -> GRounds.grounds(u, g, alg; timing = tmi, kw..., vk...)))
            end
            if n <= 2_000_000      # warm
                for (_, f) in runs; f(Dict{Symbol, Any}()); end
            end
            best = fill(Inf, length(runs)); tms = Vector{Any}(undef, length(runs))
            for r in 1:reps, (k, (lab, f)) in enumerate(runs)
                tmi = Dict{Symbol, Any}()
                t1 = @elapsed f(tmi)
                t1 < best[k] && (best[k] = t1; tms[k] = tmi)
            end
            nb = tms[1][:nnz]
            for (k, (lab, _)) in enumerate(runs)
                tm = tms[k]
                println(out, join((name, n, c, lab, tm[:rounds], tm[:nnz], tm[:nnz] == nb, round(best[k], digits = 4),
                    round(tm[:gpu], digits = 4), round(best[1] / best[k], digits = 3)), '\t'))
            end
            flush(out)
            println(rpad(name, 16), rpad(c, 6), join(["$(runs[k][1])=$(round(best[k], digits=4))$(tms[k][:nnz] == nb ? "" : "(NNZ DIFF)")" for k in eachindex(runs)], " "))
            flush(stdout)
            if pout !== nothing
                tp = Dict{Symbol, Any}()
                GRounds.grounds(u, g, alg; timing = tp, rprof = 1 << 18, kw..., VARS[end][2]...)
                rows, tot = pfsummary(tp[:pf], tp[:rounds], tp[:grids], tp[:clockkhz])
                pfprint(pout, name, c * "-" * VARS[end][1], rows, tot); flush(pout)
            end
        catch err
            println("FAILED $name $c: ", first(sprint(showerror, err), 400)); flush(stdout)
        end
        GC.gc(); CUDA.reclaim()
    end
end
close(out)
pout === nothing || close(pout)
