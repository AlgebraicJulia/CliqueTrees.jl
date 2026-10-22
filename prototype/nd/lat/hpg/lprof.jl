# B200 per-round phase profile of GPU-MF1 (LATENCY_NOTES.md, step 1).
# usage: julia lprof.jl name1,name2,...   env: OUT (summary text), RDIR (per-round binaries), DATA, CORPUS,
#   CFGS (default v2,u0.1), BASE=1 also times lat/base (GRoundsBase) for the no-overhead check,
#   EXTRA (kwargs NamedTuple string for the lat code, e.g. "(; foo = 1)"), TAG (cfg label prefix)
# Per graph and cfg: one warm run (n < 200k), the timed run(s) without the profile (min of 3 below
# 200k vertices, else 1), then one run with rprof and stats. The per-round matrix (Float32,
# rounds × (1 + 3·12 + 3)) is written to RDIR/NAME_CFG.f32: round wall µs, per phase (wall µs,
# max item µs, items), then nal, nc, ns from the round log.
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
const EXTRA = eval(Meta.parse(get(ENV, "EXTRA", "(;)")))     # trusted: set by our own sbatch files
const TAG = get(ENV, "TAG", "")
const BASE = get(ENV, "BASE", "0") == "1"
cfgkw(c) = c == "v2" ? (;) : c == "u0.1" ? (; urelax = 0.0) : error("cfg $c")
const alg = RoundsAlg(; score = :mf1, relax = 0.1)

let g = lat2(12), u = ones(nv(g))
    for c in CFGS
        GRounds.grounds(u, g, alg; cfgkw(c)..., EXTRA...)
        GRounds.grounds(u, g, alg; timing = Dict{Symbol, Any}(), rprof = 1000, stats = Tuple{Int, Int, Int}[], cfgkw(c)..., EXTRA...)
        BASE && GRoundsBase.grounds(u, g, alg; cfgkw(c)...)
    end
end
println(CUDA.name(CUDA.device()), "  clock attr kHz ", CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_CLOCK_RATE)); flush(stdout)
out = open(ENV["OUT"], "a")
rdir = ENV["RDIR"]; mkpath(rdir)
for name in split(ARGS[1], ',')
    g = try loadg(String(name)) catch err; println("LOAD FAILED $name: $err"); continue end
    n = nv(g); u = ones(n)
    for c in CFGS
        try
            kw = cfgkw(c)
            n < 200_000 && GRounds.grounds(u, g, alg; kw..., EXTRA...)
            reps = n < 200_000 ? 3 : 1
            t = Inf; tm = Dict{Symbol, Any}()
            for _ in 1:reps
                tmi = Dict{Symbol, Any}()
                t1 = @elapsed GRounds.grounds(u, g, alg; timing = tmi, kw..., EXTRA...)
                t1 < t && (t = t1; tm = tmi)
            end
            tb = NaN; nb = -1
            if BASE
                tb = Inf
                for _ in 1:reps
                    tmi = Dict{Symbol, Any}()
                    t1 = @elapsed GRoundsBase.grounds(u, g, alg; timing = tmi, kw...)
                    t1 < tb && (tb = t1; nb = tmi[:nnz])
                end
            end
            tp = Dict{Symbol, Any}(); st = Tuple{Int, Int, Int}[]
            tpr = @elapsed GRounds.grounds(u, g, alg; timing = tp, rprof = 1 << 18, stats = st, kw..., EXTRA...)
            rows, tot = pfsummary(tp[:pf], tp[:rounds], tp[:grids], tp[:clockkhz])
            ns = [s[3] for s in st]; nc = [s[2] for s in st]; nal = [s[1] for s in st]
            lab = TAG * c
            println(out, "# $name n=$n m=$(ne(g)) cfg=$lab t=$(round(t, digits=4)) tbase=$(round(tb, digits=4)) tprof=$(round(tpr, digits=4)) rounds=$(tm[:rounds]) nnz=$(tm[:nnz]) nnzbase=$nb profnnz=$(tp[:nnz]) gpu=$(round(get(tm, :gpu, NaN), digits=4)) dense=$(tm[:dense]) gtail=$(tm[:gtail]) reb=$(tm[:rebuild]) syncs=$(tm[:syncs])")
            pfprint(out, name, lab, rows, tot)
            if !isempty(ns)
                q = sort(ns)
                pct(p) = q[clamp(round(Int, p * length(q)), 1, length(q))]
                println(out, "  ns/round mean $(round(sum(ns)/length(ns), digits=1)) p10 $(pct(0.1)) p50 $(pct(0.5)) p90 $(pct(0.9)) max $(q[end])  nc/round mean $(round(sum(nc)/length(nc), digits=1))  sel/cand $(round(sum(ns)/max(1,sum(nc)), digits=3))  rounds with ns<=4: $(count(<=(4), ns))")
            end
            flush(out)
            M = pfrounds(tp[:pf], tp[:rounds], tp[:clockkhz])
            R = size(M, 1)
            S = zeros(Float32, R, 3)
            for r in 1:min(R, length(st))
                S[r, 1] = nal[r]; S[r, 2] = nc[r]; S[r, 3] = ns[r]
            end
            open(joinpath(rdir, "$(name)_$(lab).f32"), "w") do io
                write(io, Int64(R), Int64(size(M, 2) + 3)); write(io, hcat(M, S))
            end
            println(rpad(name, 16), rpad(lab, 8), " n=", n, " t=", round(t, digits = 4), " tb=", round(tb, digits = 4), " rounds=", tm[:rounds],
                " nnz=", tm[:nnz], " base=", nb, " round=", round(tot.round_us, digits = 1), "µs"); flush(stdout)
        catch err
            println("FAILED $name $c: ", first(sprint(showerror, err), 400)); flush(stdout)
        end
        GC.gc(); CUDA.reclaim()
    end
end
close(out)
