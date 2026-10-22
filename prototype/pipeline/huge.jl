# Stage 4 of HybridTW on huge graphs: nested dissection (METIS) on the top
# levels, the GPU rounds of MF with strategy 1 (prototype/rounds, GRounds) on
# the leaves, then HBT for the time left.
#   julia --project=<env with CUDA, Metis> -t 8,2 huge.jl <tncorpus name | lat2:L | lat3:L> <seconds> [cpu]
# With "cpu", the leaves are ordered by MF with strategy 1 on the CPU (the
# baseline of the same pipeline).
using CliqueTrees, CUDA, Graphs, Metis, Printf
const ROUNDS = joinpath(@__DIR__, "..", "rounds")
include(joinpath(ROUNDS, "Rounds.jl")); using .Rounds
include(joinpath(ROUNDS, "GRounds.jl")); using .GRounds
include(joinpath(get(ENV, "TNCORPUS", joinpath(homedir(), "Projects/GATAS/tn-work/tncorpus")), "load.jl")); using .TNCorpus

# the GPU rounds on leaves of at least nmin vertices, MF1 on the CPU below
gpuleaf(; nmin = 20_000) = GPUHybrid(RoundsAlg(; score = :mf1, select = :luby, relax = 0.5, tailsel = 128, tailmax = 10^9, tailalg = AMF()),
    MF(; strategy = 1); nmin)

function load(name)
    if startswith(name, "lat2:")
        L = parse(Int, name[6:end]); g = Graphs.grid([L, L])
    elseif startswith(name, "lat3:")
        L = parse(Int, name[6:end]); g = Graphs.grid([L, L, L])
    else
        inst = TNCorpus.load(name); return inst.weights, TNCorpus.bipartite(inst)
    end
    return ones(Float64, nv(g)), BipartiteGraph(g)
end

function main()
    name = ARGS[1]; T = parse(Float64, ARGS[2]); cpu = length(ARGS) >= 3 && ARGS[3] == "cpu"
    leaf = cpu ? MF(; strategy = 1) : gpuleaf()
    # compile (not charged)
    let (w, g) = load("lat2:300")
        hybridtw(w, g, HybridTW(; time = 1.0, leaf, nhuge = 1000))
    end
    w, g = load(name)
    r = hybridtw(w, g, HybridTW(; time = T, leaf, nhuge = 200_000, verbose = true))
    @assert isperm(r.order) && r.width == treewidth(w, g, r.order)
    @printf("%s\t%d\t%s\t%g\t%.3f\t%.6g\t%.6g\t%s\n", name, nv(g), cpu ? "nd-mf1" : "nd-gpumf1", T, r.time, r.width, r.lower,
        join((@sprintf("%.2f:%.6g", t, k) for (t, k) in r.trace), ","))
end
main()
