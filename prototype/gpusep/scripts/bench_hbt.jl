# HBT end to end on a whole network: CPU-only rounds vs GPU-escalated rounds, same time budget.
#   julia --project=<env> -t <nt> bench_hbt.jl <name> <seconds> <out.tsv> [modes] [seed]
# modes: comma list of cpu,esc-septw,esc-pid (default all). Pipeline as in tncorpus/prof/common.jl:
# SafeRules(SafeSeparators(HBT(time, threads = nt), four minimal greedy triangulations; maxmissing = 50)).
# Row: name n mode seconds width tseptw share rounds(cpu/gpu) escalated gpufailed busy tgpu budget
# tseptw = seconds inside the rounds of the exact searches (sepenum + sepdp, CPU and GPU), summed over tasks;
# share = tseptw / (seconds * nt).
using CliqueTrees, CUDA, Graphs, Metis, Printf
const CT = CliqueTrees
const EXT = Base.get_extension(CliqueTrees, :CUDAExt)
include(joinpath(@__DIR__, "..", "..", "tncorpus", "load.jl"))
using .TNCorpus

name, T, out = ARGS[1], parse(Float64, ARGS[2]), ARGS[3]
modes = length(ARGS) >= 4 ? split(ARGS[4], ',') : ["cpu", "esc-septw", "esc-pid"]
seed = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1
NT = Threads.nthreads()

minalgs() = (MinimalChordal(MF()), MinimalChordal(MF(; strategy = 1)), MinimalChordal(AMF()), MinimalChordal(MMD()))
pipeline(alg) = SafeRules(SafeSeparators(alg, minalgs(); maxmissing = 50))

# The CPU rounds, timed: the CPU backend, through the extension point of the rounds.
struct TimedCPU <: CT.SeptwBackend end
struct TimedRounds end
const TCPU = Threads.Atomic{Float64}(0.0)
const NCPU = Threads.Atomic{Int}(0)
CT.bt_esc(::TimedCPU, w::AbstractVector, g::AbstractGraph, dl::Real) = TimedRounds()
function CT.bt_round!(::TimedRounds, w::AbstractVector{W}, g::AbstractGraph, k::W, dl::Real, nt::Integer) where {W}
    t = time(); r = CT.bt_round!(nothing, w, g, k, dl, nt)
    Threads.atomic_add!(TCPU, time() - t); Threads.atomic_add!(NCPU, 1)
    return r
end

backend(mode) = mode == "cpu" ? TimedCPU() : CT.GPUSeptw(; engine = Symbol(mode[5:end]))

inst = TNCorpus.load(name); w = inst.weights; g = TNCorpus.bipartite(inst)

# warm-up (compile time not charged)
let h = BipartiteGraph(cycle_graph(12)), hw = ones(12)
    permutation(hw, h, pipeline(HBT(; time = 1.0, threads = NT)))
    for mode in modes
        b = backend(mode); b isa CT.GPUSeptw && CT.septwwarmup!(b)
    end
end

io = open(out, "a")
for mode in modes
    b = backend(mode); old = CT.septwbackend!(b); EXT.resetstats!(); TCPU[] = 0.0; NCPU[] = 0; GC.gc()
    t = @elapsed (order, _) = permutation(w, g, pipeline(HBT(; time = T, threads = NT, seed)))
    CT.septwbackend!(old)
    S = EXT.STATS; ngpu = get(S, :gpufound, 0) + get(S, :gpuabove, 0) + get(S, :gputimeout, 0)
    tsep = mode == "cpu" ? TCPU[] : get(S, :tcpu, 0.0) + get(S, :tgpu, 0.0)
    ncpu = mode == "cpu" ? NCPU[] : get(S, :cpu, 0)
    bud = b isa CT.GPUSeptw ? b.delta * get(EXT.COST, b.engine, NaN) : NaN
    @printf(io, "%s\t%d\t%s\t%d\t%.3f\t%.10g\t%.3f\t%.3f\t%d/%d\t%d\t%d\t%d\t%.3f\t%.4g\n", name, nv(g), mode, seed, t, treewidth(w, g; alg = order),
        tsep, tsep / (t * NT), ncpu, ngpu, get(S, :escalated, 0), get(S, :gpufailed, 0), get(S, :busy, 0), get(S, :tgpu, 0.0), bud)
    flush(io)
    println(name, " ", mode, " width=", treewidth(w, g; alg = order), " t=", round(t; digits = 1), " septw share=", round(tsep / (t * NT); digits = 3)); flush(stdout)
end
close(io)
