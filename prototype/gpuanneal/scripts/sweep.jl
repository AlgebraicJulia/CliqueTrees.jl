# Parameter sweep in one process: julia --project=env scripts/sweep.jl net=<tnc> seeds=<trees> time=<s> out=<tsv> cfg1 cfg2 ...
# each cfg is "k=v;k=v" (keys as in run.jl: beta0 beta1 epoch mix resample nch Wt lam pslice rseed)
const HERE = @__DIR__
include(joinpath(HERE, "..", "src", "GPUAnneal.jl"))
using .GPUAnneal, CUDA, Printf

pos = filter(a -> occursin('=', a) && !occursin(';', a) && split(a, '=')[1] in ("net", "seeds", "time", "out", "marks"), ARGS)
kv = Dict(split(a, '='; limit = 2) |> x -> (x[1] => x[2]) for a in pos)
cfgs = filter(a -> !(a in pos), ARGS)
net = readnet(kv["net"])
seeds = [(tree_from_ssa(net, read_tree(f)[2]), read_tree(f)[3]) for f in split(kv["seeds"], ',')]
T = parse(Float64, kv["time"])
marks = haskey(kv, "marks") ? parse.(Float64, split(kv["marks"], ',')) : [T / 10, T / 3, T]
out = open(kv["out"], "a")
let A = Annealer(net, seeds; nch = 4); run_anneal!(A; time = 0.01, epoch = 8); GPUAnneal.clone!(A, [2], [1]); end
for cfg in cfgs
    c = Dict(split(x, '='; limit = 2) |> y -> (y[1] => y[2]) for x in split(cfg, ';') if !isempty(x))
    gf(k, d) = haskey(c, k) ? parse(Float64, c[k]) : d
    nch = haskey(c, "nch") ? parse(Int, c["nch"]) : GPUAnneal.default_chains(net)
    Wt = gf("Wt", Inf)
    p = (beta0 = gf("beta0", 1.0), beta1 = gf("beta1", 100.0), epoch = round(Int, gf("epoch", 4net.nt)),
         lam = gf("lam", 0.0), Wt, pslice = gf("pslice", 0.0), mix = gf("mix", 0.0), resample = gf("resample", 0.0))
    A = Annealer(net, seeds; nch, seed = round(Int, gf("rseed", 1)))
    res = Float64[]; mk = Ref(1)
    tr(t, A) = while mk[] <= length(marks) && t >= marks[mk[]]
        push!(res, minimum(Array(A.BESTE))); mk[] += 1
    end
    run_anneal!(A; time = T, p..., trace = tr)
    while length(res) < length(marks); push!(res, minimum(Array(A.BESTE))); end
    CH, sl, E, g = GPUAnneal.best(A)
    h = hostcost(net, CH, GPUAnneal.slicemask(net, sl); Wt)
    line = join((net.name, cfg, nch, (@sprintf("%.4f", r) for r in res)..., @sprintf("%.4f", h.log2C), h.W), '\t')
    println(line); println(out, line); flush(out); flush(stdout)
    A = nothing; GC.gc(); CUDA.reclaim()
end
