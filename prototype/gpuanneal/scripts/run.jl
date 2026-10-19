# GPU annealing driver.
#   julia --project=env scripts/run.jl net=<path.tnc> seeds=<a.tree,b.tree,...> out=<prefix> [key=value ...]
# keys: time (s), nch (chains; default fills the GPU), beta0, beta1, epoch (steps per launch), lam, Wt (width
#       target for slicing; Inf = unconstrained), pslice, mix, resample, rseed, ckpt (comma list of seconds)
# Writes <prefix>.trace.tsv (t, best feasible GPU energy, mean energy, acceptance) and, at every checkpoint
# time and at the end, <prefix>.T<t>.tree (best tree so far, with the GPU number and the exact host
# re-costing in its metadata).
const HERE = @__DIR__
include(joinpath(HERE, "..", "src", "GPUAnneal.jl"))
using .GPUAnneal, CUDA, Printf, Statistics

kv = Dict(split(a, '='; limit = 2) |> x -> (x[1] => x[2]) for a in ARGS)
getf(k, d) = haskey(kv, k) ? parse(Float64, kv[k]) : d
geti(k, d) = haskey(kv, k) ? parse(Int, kv[k]) : d

net = readnet(kv["net"])
seedfiles = split(kv["seeds"], ',')
seeds = map(seedfiles) do f
    nt, ssa, sl = read_tree(f)
    @assert nt == net.nt "$f: $nt tensors, network has $(net.nt)"
    (tree_from_ssa(net, ssa), sl)
end
out = kv["out"]
T = getf("time", 10.0)
Wt = getf("Wt", Inf)
lam = getf("lam", isfinite(Wt) ? 4.0 : 0.0)
pslice = getf("pslice", isfinite(Wt) ? 0.02 : 0.0)
pars = (beta0 = getf("beta0", 1.0), beta1 = getf("beta1", 100.0), epoch = geti("epoch", 4 * net.nt), lam, Wt,
        pslice, mix = getf("mix", 0.0), resample = getf("resample", 0.0))
nch = geti("nch", GPUAnneal.default_chains(net))
ckpt = haskey(kv, "ckpt") ? parse.(Float64, split(kv["ckpt"], ',')) : [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
ckpt = sort(filter(<(T), ckpt))

for (i, (CHs, sl)) in enumerate(seeds)
    h = hostcost(net, CHs, GPUAnneal.slicemask(net, sl); Wt)
    @printf("seed %d (%s): log2C %.4f W %.1f excess %.1f\n", i, basename(seedfiles[i]), h.log2C, h.W, h.excess)
end
dev = CUDA.device()
@printf("%s: n=%d nt=%d W=%d K=%d | %s, %d chains, G=%d, %.1f MB/chain\n", net.name, net.n, net.nt, net.W, net.K,
        CUDA.name(dev), nch, GPUAnneal.groupsize(net.W), GPUAnneal.chain_bytes(net) / 2^20)
println("params ", pars, " time ", T)
flush(stdout)

# warm-up compile on a tiny annealer (not timed)
let A = Annealer(net, seeds; nch = 4, seed = 99)
    run_anneal!(A; time = 0.01, pars..., epoch = 8)
    GPUAnneal.clone!(A, [2], [1]); CUDA.synchronize()
end

A = Annealer(net, seeds; nch, seed = geti("rseed", 1))
io = open(out * ".trace.tsv", "w")
println(io, "t\tbestE\tmeanE\tminE\tfeasfrac\tacc")
nextck = Ref(1)
function dump(tag, t)
    CH, sl, E, g = GPUAnneal.best(A)
    isfinite(E) || return
    h = hostcost(net, CH, GPUAnneal.slicemask(net, sl); Wt)
    ssa = ssa_from_children(net, CH)
    write_tree("$out.$tag.tree", net.nt, ssa, sl; meta = Dict("t" => @sprintf("%.3f", t), "gpu_E" => @sprintf("%.6f", E),
        "host_log2C" => @sprintf("%.6f", h.log2C), "host_W" => @sprintf("%.3f", h.W), "host_maxop" => @sprintf("%.3f", h.maxop),
        "nsliced" => string(length(sl)), "chain" => string(g), "nch" => string(nch), "params" => string(pars)))
    @printf("[%8.2fs] %s best log2C %.4f (gpu %.4f) W %.1f maxop %.1f sliced %d\n", t, tag, h.log2C, E, h.W, h.maxop, length(sl))
    flush(stdout)
end
nacc0 = Ref(0)
function tr(t, A)
    BE = Array(A.BESTE)
    E, ex = GPUAnneal.energies(A, lam)
    na = sum(Array(A.NACC)); acc = (na - nacc0[]) / (nch * pars.epoch); nacc0[] = na
    @printf(io, "%.3f\t%.6f\t%.6f\t%.6f\t%.4f\t%.5f\n", t, minimum(BE), mean(E), minimum(E), mean(ex .< 1e-3), acc)
    flush(io)
    while nextck[] <= length(ckpt) && t >= ckpt[nextck[]]
        dump("T" * string(round(Int, ckpt[nextck[]])), t)
        nextck[] += 1
    end
end
t0 = time()
ep = run_anneal!(A; time = T, pars..., trace = tr)
tt = time() - t0
close(io)
@printf("%d epochs, %.3g steps total, %.3g steps/s\n", ep, ep * pars.epoch * nch, ep * pars.epoch * nch / tt)
dump("final", tt)
