# GPU annealing pipeline (v2 block kernel), anytime, with checkpoints re-costed on the host.
#   julia --project=env scripts/pipeline.jl net=<tnc> seeds=<trees> out=<prefix> time=<total s> [Wt=<cap>] [key=value ...]
# Unconstrained (no Wt): parallel tempering over trees for `time` seconds.
# With Wt (slicing to width <= Wt):
#   phase 1 (frac1 of the time): tempering of the unsliced trees with a soft width penalty
#            lam1p * sum_v max(0, w(L(v)) - Wt) (makes the trees slicing-friendly, as TreeSA's sc_target);
#   phase 2: every chain's current tree is greedily sliced on the host to width <= Wt (greedy_slice), the
#            best `keep` distinct results seed all chains, then joint tempering over trees and slice sets
#            (rotations under the hard-ish penalty lam2, add/remove/swap slice moves) for the rest of the time.
# Writes <prefix>.trace.tsv (wall time incl. host work, best feasible log2 C) and <prefix>.T<t>.tree at the
# checkpoint times plus <prefix>.final.tree; tree metadata carries the GPU energy and the host re-costing.
const HERE = @__DIR__
include(joinpath(HERE, "..", "src", "BlockAnneal.jl"))
using .BlockAnneal, CUDA, Printf, Random
const BA = BlockAnneal

kv = Dict(split(a, '='; limit = 2) |> x -> (x[1] => x[2]) for a in ARGS)
gf(k, d) = haskey(kv, k) ? parse(Float64, kv[k]) : d
gi(k, d) = round(Int, gf(k, d))
net = BA.readnet(kv["net"])
seedfiles = split(kv["seeds"], ',')
seeds = [(BA.tree_from_ssa(net, BA.read_tree(f)[2]), BA.read_tree(f)[3]) for f in seedfiles]
out = kv["out"]; T = gf("time", 30.0); Wt = gf("Wt", Inf)
ckpt = sort(filter(<(T), haskey(kv, "ckpt") ? parse.(Float64, split(kv["ckpt"], ',')) : [0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]))
P = (betamin = gf("beta0", 1.0), betamax = gf("beta1", 300.0), rounds = gi("rounds", 200), mix = gf("mix", 1.0),
     evalevery = gi("evalevery", 64), nreconf = gi("nreconf", 0), kmax = gi("kmax", 6))
frac1 = gf("frac1", 0.3); lam1p = gf("lam1p", 0.5); lam2 = gf("lam2", 2.0); lam2b = gf("lam2b", lam2); nslice = gi("nslice", 16)
keep = gi("keep", 8); nch = gi("nch", 0); rseed = gi("rseed", 1)

dev = CUDA.device()
@printf("%s n=%d nt=%d W=%d K=%d | %s | T=%.0f Wt=%s\n", net.name, net.n, net.nt, net.W, net.K, CUDA.name(dev), T, string(Wt))
for (i, (CHs, sl)) in enumerate(seeds)
    h = BA.hostcost(net, CHs, BA.slicemask(net, sl))
    @printf("seed %s: log2C %.4f W %.1f\n", basename(seedfiles[i]), h.log2C, h.W)
end
let A = BA.BAnnealer(net, seeds; nch = 2); BA.run!(A; time = 0.01, rounds = 4, nreconf = 1, kmax = 4); end   # compile

io = open(out * ".trace.tsv", "w"); println(io, "t\tphase\tbest_log2C")
best = Ref((Inf, zeros(Int32, 2, 2net.nt - 1), Int[]))
nck = Ref(1); t0 = Ref(0.0)
function record!(A, phase)
    BE = Array(A.BESTE); g = argmin(BE)
    if BE[g] < best[][1]
        CH, sl, E, _ = BA.best(A)
        best[] = (E, CH, sl)
    end
    t = time() - t0[]
    @printf(io, "%.3f\t%d\t%.6f\n", t, phase, best[][1]); flush(io)
    while nck[] <= length(ckpt) && t >= ckpt[nck[]]
        dump("T" * replace(string(ckpt[nck[]]), ".0" => ""), t); nck[] += 1
    end
end
function dump(tag, t)
    E, CH, sl = best[]
    isfinite(E) || return
    h = BA.hostcost(net, CH, BA.slicemask(net, sl); Wt)
    BA.write_tree("$out.$tag.tree", net.nt, BA.ssa_from_children(net, CH), sl; meta = Dict("t" => @sprintf("%.3f", t),
        "gpu_E" => @sprintf("%.6f", E), "host_log2C" => @sprintf("%.6f", h.log2C), "host_W" => @sprintf("%.3f", h.W),
        "host_maxop" => @sprintf("%.3f", h.maxop), "nsliced" => string(length(sl)), "Wt" => string(Wt), "device" => CUDA.name(dev)))
    @printf("[%8.2fs] %-7s log2C %.4f (gpu %.4f) W %.1f sliced %d\n", t, tag, h.log2C, E, h.W, length(sl)); flush(stdout)
end

t0[] = time()
shared = haskey(kv, "shared") ? kv["shared"] == "1" : nothing
A = BA.BAnnealer(net, seeds; nch, seed = rseed, shared, nth = gi("nth", BA.default_threads(net)))
@printf("chains %d x %d threads, shared=%s, %.1f KB shmem\n", A.nch, A.nth, A.shared, A.shmem / 1024)
if !isfinite(Wt)
    BA.temper!(A; time = T, P..., trace = (t, A) -> record!(A, 1))
elseif get(kv, "mode", "wsched") == "wsched"
    # width-target annealing: effective cap from the seeds' width down to Wt over wfrac of the time,
    # joint rotations + slice moves throughout (TreeSASlicer-like gradual slicing, in parallel)
    W0 = maximum(BA.hostcost(net, CHs, BA.slicemask(net, sl)).W for (CHs, sl) in seeds)
    BA.temper!(A; time = T, P..., lam = lam2, lam1 = lam2b, Wt, Wt0 = W0, wfrac = gf("wfrac", 0.6), nslice, pheur = gf("pheur", 0.75), resample = gf("resample", 0.0), resevery = gi("resevery", 10),
               trace = (t, A) -> record!(A, 2))
else
    T1 = frac1 * T
    BA.temper!(A; time = T1, P..., lam = lam1p, Wt, nslice = 0, trace = (t, A) -> record!(A, 1))
    # host: greedy-slice every chain's current tree, keep the best distinct ones
    CHall = Array(A.CH); E1 = Array(A.ST)[1, :]
    order = sortperm(E1)
    cand = []
    for g in order[1:min(end, 4keep)]
        CHg = CHall[:, :, g]
        sl = BA.greedy_slice(net, CHg, Wt)
        h = BA.hostcost(net, CHg, BA.slicemask(net, sl); Wt)
        push!(cand, (h.log2C, CHg, sl))
    end
    sort!(cand; by = first)
    @printf("[%8.2fs] phase 1 done; greedy-sliced candidates: %s\n", time() - t0[], join((@sprintf("%.2f", c[1]) for c in cand[1:min(end, keep)]), " "))
    if cand[1][1] < best[][1]
        best[] = (cand[1][1], cand[1][2], cand[1][3])
    end
    seeds2 = [(c[2], c[3]) for c in cand[1:min(end, keep)]]
    A = nothing; GC.gc(); CUDA.reclaim()
    A = BA.BAnnealer(net, seeds2; nch, seed = rseed + 1, shared, nth = gi("nth", BA.default_threads(net)))
    T2 = T - (time() - t0[])
    BA.temper!(A; time = T2, P..., lam = lam2, lam1 = lam2b, Wt, nslice, trace = (t, A) -> record!(A, 2))
end
record!(A, 9)
close(io)
dump("final", time() - t0[])
let BE = sort(Array(A.BESTE)), ST = Array(A.ST)
    println("per-chain best: ", join((@sprintf("%.1f", x) for x in BE), " "))
    println("per-chain current E: ", join((@sprintf("%.1f", x) for x in sort(ST[1, :])), " "))
    println("per-chain nsliced: ", join(sort([count_ones(reduce(+, count_ones.(Array(A.SL[:, g])))) >= 0 ? sum(count_ones, Array(A.SL[:, g])) : 0 for g in 1:A.nch]), " "))
end
