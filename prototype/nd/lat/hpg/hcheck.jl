# exactness of lat/GRounds.jl against lat/base (GRoundsBase, the code before this work):
# nnz(L), flops, tw (unit weights) and width / log2 cost (instance weights) for the replay
# driver (several configs, esub 1 and default, bpsm), the persistent modes and the batch.
# usage: julia hcheck.jl name...   env QUICK=1, EXTRA, DATA
# HPG version of lat/check.jl (graphs from DATA .bin files); run from a code bundle dir
const DIR = @__DIR__
using CUDA, Serialization, SparseArrays, Random, Graphs
include(joinpath(DIR, "Rounds.jl")); using .Rounds
include(joinpath(DIR, "evalutil.jl")); include(joinpath(DIR, "gens.jl"))
include(joinpath(DIR, "GRounds.jl")); using .GRounds
include(joinpath(DIR, "base", "GRounds.jl")); using .GRoundsBase
const dirs = split(get(ENV, "DATA", ""), ":")
function loadg(name)
    startswith(name, "lat2_") && return lat2(parse(Int, name[6:end])), "tn"
    startswith(name, "lat3_") && return lat3(parse(Int, name[6:end])), "tn"
    for d in dirs
        f = joinpath(d, name * ".bin")
        if isfile(f)
            n, I, J = deserialize(f)
            return SimpleGraph(sparse(vcat(I, J), vcat(J, I), true, n, n, |)), "ssmc"
        end
    end
    error("no graph $name")
end
function fillq(g, o)
    cw = colweights(ones(Int, nv(g)), g, o)
    return sum(cw), sum(Float64(c)^2 for c in cw), maximum(cw) - 1
end
quick = get(ENV, "QUICK", "0") == "1"
extra = eval(Meta.parse(get(ENV, "EXTRA", "(;)")))      # trusted, set by hand
function inst(name)
    g, kind = loadg(name)
    kind == "ssmc" && return g, ones(nv(g))
    startswith(name, "lat") && return g, randw(nv(g))
    return g, randw(nv(g))
end
q(g, o, w) = (fillq(g, o), quality(w, g, o))
nsame = 0; nbad = 0
algs = Any[(RoundsAlg(; score = :mf1, relax = 0.1), (;)), (RoundsAlg(; score = :mf1, relax = 0.1), (; urelax = 0.0))]
if !quick
    for sc in (:md, :mf, :mf1), rx in (0.0, 0.5), cp in (true, false)
        push!(algs, (RoundsAlg(; score = sc, relax = rx, compress = cp), (;)))
    end
    push!(algs, (RoundsAlg(; score = :mf1, relax = 0.5, tailsel = 128, tailmax = 10^9), (; cpualpha = Inf)))
end
variants = [("replay", (;)), ("esub1", (; esub = 1)), ("bpsm1", (; bpsm = 1)), ("dg2", (; dgroup = 2))]
quick || append!(variants, [("block8", (; mode = :block, nw = 8)), ("grid3", (; mode = :grid, G = 3))])
for name in ARGS
    g, wt = inst(name)
    for (alg, kw) in algs, wts in (ones(nv(g)), wt)
        ref = q(g, GRoundsBase.grounds(wts, g, alg; kw...)[1], wts)
        for (lab, vk) in variants
            vk2 = haskey(vk, :dgroup) ? vk : merge(vk, extra)
            haskey(vk, :dgroup) && (ref2 = q(g, GRoundsBase.grounds(wts, g, alg; kw..., vk...)[1], wts))
            r = q(g, GRounds.grounds(wts, g, alg; kw..., vk2...)[1], wts)
            ok = haskey(vk, :dgroup) ? r == ref2 : r == ref
            global nsame += ok; global nbad += !ok
            ok || println("DIFF $name $(alg.score) relax=$(alg.relax) comp=$(alg.compress) $kw $lab: ", r, " vs ", ref)
        end
    end
    if !quick
        # batch: several copies of the graph in one launch
        gs = [g, g, g]
        os, _ = GRounds.grounds_batch(gs, RoundsAlg(; score = :mf1, relax = 0.1); weights = [ones(nv(g)) for _ in gs])
        rb = q(g, GRoundsBase.grounds(ones(nv(g)), g, RoundsAlg(; score = :mf1, relax = 0.1))[1], ones(nv(g)))
        for o in os
            ok = q(g, o, ones(nv(g))) == rb
            global nsame += ok; global nbad += !ok
            ok || println("DIFF batch $name")
        end
    end
    println(name, ": same ", nsame, " diff ", nbad); flush(stdout)
end
println("TOTAL same $nsame diff $nbad")
