# v2 (block-per-chain) sweep: julia --project=env scripts/sweep2.jl net=<tnc> seeds=<trees> time=<s> out=<tsv> [marks=..] [save=<prefix>] cfg ...
# cfg "k=v;k=v": beta0 beta1 rounds mix lam Wt nslice evalevery nch nth shared rseed
const HERE = @__DIR__
include(joinpath(HERE, "..", "src", "BlockAnneal.jl"))
using .BlockAnneal, CUDA, Printf
const BA = BlockAnneal

fixed = ("net", "seeds", "time", "out", "marks", "save", "slicefirst")
pos = filter(a -> occursin('=', a) && !occursin(';', a) && split(a, '=')[1] in fixed, ARGS)
kv = Dict(split(a, '='; limit = 2) |> x -> (x[1] => x[2]) for a in pos)
cfgs = filter(a -> !(a in pos), ARGS)
net = BA.readnet(kv["net"])
seeds = [(BA.tree_from_ssa(net, BA.read_tree(f)[2]), BA.read_tree(f)[3]) for f in split(kv["seeds"], ',')]
T = parse(Float64, kv["time"])
if haskey(kv, "slicefirst")   # greedy-slice every seed to width <= slicefirst before annealing
    wt0 = parse(Float64, kv["slicefirst"])
    seeds = [(CHs, BA.greedy_slice(net, CHs, wt0; S0 = sl)) for (CHs, sl) in seeds]
    for (CHs, sl) in seeds
        h = BA.hostcost(net, CHs, BA.slicemask(net, sl); Wt = wt0)
        @printf("greedy-sliced seed: log2C %.4f W %.1f nsliced %d\n", h.log2C, h.W, length(sl))
    end
end
marks = haskey(kv, "marks") ? parse.(Float64, split(kv["marks"], ',')) : [T / 10, T / 3, T]
out = open(kv["out"], "a")
let A = BA.BAnnealer(net, seeds; nch = 2); BA.run!(A; time = 0.01, rounds = 4); end
for (ci, cfg) in enumerate(cfgs)
    c = Dict(split(x, '='; limit = 2) |> y -> (y[1] => y[2]) for x in split(cfg, ';') if !isempty(x))
    gf(k, d) = haskey(c, k) ? parse(Float64, c[k]) : d
    gi(k, d) = round(Int, gf(k, d))
    Wt = gf("Wt", Inf)
    shared = haskey(c, "shared") ? c["shared"] == "1" : nothing
    A = BA.BAnnealer(net, seeds; nch = gi("nch", 0), nth = gi("nth", BA.default_threads(net)), seed = gi("rseed", 1), shared)
    p = (beta0 = gf("beta0", 5.0), beta1 = gf("beta1", 300.0), rounds = gi("rounds", 2000), lam = gf("lam", isfinite(Wt) ? 4.0 : 0.0),
         Wt, mix = gf("mix", 1.0), evalevery = gi("evalevery", 64), nslice = gi("nslice", isfinite(Wt) ? 4 : 0), kmax = gi("kmax", 8), nreconf = gi("nreconf", 0), lam1 = gf("lam1", gf("lam", isfinite(Wt) ? 4.0 : 0.0)))
    res = Float64[]; mk = Ref(1)
    tr(t, A) = while mk[] <= length(marks) && t >= marks[mk[]]
        push!(res, minimum(Array(A.BESTE))); mk[] += 1
    end
    t0 = time()
    ep = if get(c, "mode", "sa") == "pt"
        BA.temper!(A; time = T, betamin = p.beta0, betamax = p.beta1, rounds = p.rounds, lam = p.lam, Wt = p.Wt, mix = p.mix,
                   evalevery = p.evalevery, nslice = p.nslice, kmax = p.kmax, nreconf = p.nreconf, lam1 = p.lam1, trace = tr)
    else
        BA.run!(A; time = T, p..., trace = tr)
    end
    tt = time() - t0
    while length(res) < length(marks); push!(res, minimum(Array(A.BESTE))); end
    CH, sl, E, g = BA.best(A)
    h = BA.hostcost(net, CH, BA.slicemask(net, sl); Wt)
    nprop = sum(Array(A.NPROP)); nacc = sum(Array(A.NACC))
    rate = ep * p.rounds * A.nch * A.nth / tt
    line = join((net.name, cfg, A.nch, A.nth, A.shared, @sprintf("%.3g", rate), @sprintf("%.3g", nprop / tt), @sprintf("%.3f", nacc / max(nprop, 1)),
                 (@sprintf("%.4f", r) for r in res)..., @sprintf("%.4f", h.log2C), h.W, length(sl)), '\t')
    println(line); println(out, line); flush(out); flush(stdout)
    if haskey(kv, "save")
        BA.write_tree("$(kv["save"]).c$ci.tree", net.nt, BA.ssa_from_children(net, CH), sl;
            meta = Dict("gpu_E" => @sprintf("%.6f", E), "host_log2C" => @sprintf("%.6f", h.log2C), "host_W" => @sprintf("%.3f", h.W), "cfg" => cfg))
    end
    A = nothing; GC.gc(); CUDA.reclaim()
end
