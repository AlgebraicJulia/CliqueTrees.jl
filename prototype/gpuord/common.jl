using CUDA, CliqueTrees, Graphs, Random, Metis, Printf, Serialization
const GPUORD = @__DIR__
isdefined(Main, :GPUOrd) || include(joinpath(GPUORD, "src", "GPUOrd.jl"))
isdefined(Main, :Hybrid) || include(joinpath(GPUORD, "src", "Hybrid.jl"))
using .GPUOrd, .Hybrid
isdefined(Main, :TNCorpus) || include("/tmp/claude-1000/-home-itaykadosh-Projects-GATAS/2e762ebb-ae79-4e3b-a677-b7caf4118124/scratchpad/tncorpus/load.jl")
using .TNCorpus

const bw = BestWidth(MF(), MF(; strategy = 1), AMF(), MMD())
nd(s = 0) = ND(bw, METISND(; seed = s); width = 16, level = 12, scale = 100)
nd8() = BestWidth(ntuple(s -> nd(s), 8)...)
const MINS = (MinimalChordal(MF()), MinimalChordal(MF(; strategy = 1)), MinimalChordal(AMF()), MinimalChordal(MMD()))
safe(alg) = SafeRules(SafeSeparators(alg, MINS; maxmissing = 50))
hbt(T) = safe(HBT(; time = T, threads = 1))
gpu(T; pmax = 256, seed = 1) = Hybrid.GPUSearch(; time = T, pmax, seed)

function synth(name)
    rng = Xoshiro(1)
    g = name == "grid60" ? Graphs.grid([60, 60]) :
        name == "grid100" ? Graphs.grid([100, 100]) :
        name == "grid3d14" ? Graphs.grid([14, 14, 14]) :
        name == "rr3_3000" ? random_regular_graph(3000, 3; rng) :
        name == "geo4000" ? first(euclidean_graph(4000, 2; cutoff = 0.03, rng)) : error(name)
    w = [rand(rng, (1.0, 1.0, log2(3.0), 2.0)) for _ in 1:nv(g)]
    return g, w
end

# run one method; width is recomputed exactly on the CPU from the ordering
function runmethod(alg, w, g)
    Hybrid.STATS[] = Hybrid.Stats()
    t = @elapsed order, _ = permutation(w, g; alg)
    @assert isperm(order)
    width = treewidth(w, g; alg = order)
    return (; t, width, stats = deepcopy(Hybrid.STATS[]))
end

function warmup()
    g = Graphs.grid([12, 12]); w = ones(nv(g))
    for a in (bw, nd(), gnd(0.05; pmax = 64), safe(gnd(0.05; pmax = 64)), hbt(0.1))
        permutation(w, g; alg = a)
    end
    g = Graphs.grid([30, 30]); w = ones(nv(g))
    for pm in (64, 128, 256, 512)
        permutation(w, g; alg = gnd(0.05; pmax = pm, width = pm))
    end
end
gnd(T; pmax = 256, width = 16, level = 12, seed = 0, rounds = 1, big = :bw) = Hybrid.GPUND(; time = T, pmax, width, level, seed, rounds, big)
ndfast(s = 0) = ND(BestWidth(AMF(), MMD()), METISND(; seed = s); width = 16, level = 12, scale = 100)
