# CPU baseline: TreeSA (OMEinsumContractionOrders 1.3, Kalachev et al. App. A) on one network.
#   julia -t <threads> --project=treesa_env scripts/treesa_run.jl <tnc> <out prefix> <niters list> [ntrials] [sc_target] [seedtree]
# For each niters value: one optimize_code call (ntrials independent annealings, multithreaded over trials,
# betas 0.01:0.05:15 = package default), wall time measured; writes <prefix>.ni<niters>.tree (ssa path,
# same tensor order as the .tnc minus scalars) for re-costing with cotengra, and a TSV line.
# sc_target = Inf -> unconstrained (score = total time complexity only).
# seedtree: optional .tree to start from (initializer = :specified), otherwise :greedy.
using OMEinsumContractionOrders, Printf
module HostIO
using Printf
include(joinpath(@__DIR__, "..", "src", "hostio.jl"))
end
using .HostIO: readnet, read_tree, write_tree

net = readnet(ARGS[1]); out = ARGS[2]
nis = parse.(Int, split(ARGS[3], ','))
ntrials = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : Threads.nthreads()
sct = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : Inf
seedf = length(ARGS) >= 6 ? ARGS[6] : ""

ixs = [collect(t) for t in net.tens]
code = OMEinsumContractionOrders.EinCode(ixs, collect(net.open))
sizes = Dict(i => net.dims[i] for i in 1:net.n)
score = isfinite(sct) ? ScoreFunction(sc_target = sct, sc_weight = 1.0) : ScoreFunction(sc_target = 1e4, sc_weight = 0.0)

function tossa(ne, nt)
    ssa = Tuple{Int,Int}[]; nxt = Ref(nt)
    function go(x)
        if OMEinsumContractionOrders.isleaf(x)
            return x.tensorindex - 1
        end
        @assert length(x.args) == 2
        a = go(x.args[1]); b = go(x.args[2])
        push!(ssa, (a, b)); id = nxt[]; nxt[] += 1
        id
    end
    go(ne); ssa
end

function fromssa(ssa, nt)
    nodes = Any[OMEinsumContractionOrders.NestedEinsum{Int}(i) for i in 1:nt]
    legs = Vector{Vector{Int}}([collect(t) for t in ixs])
    cnt = Dict{Int,Int}(); for t in ixs, i in t; cnt[i] = get(cnt, i, 0) + 1; end
    for i in net.open; cnt[i] = get(cnt, i, 0) + 1; end
    # output legs of a merge: indices that still occur outside (count-based, like cotengra)
    sub = [Dict{Int,Int}(i => 1 for i in t) for t in ixs]
    for (a, b) in ssa
        d = mergewith(+, sub[a+1], sub[b+1])
        o = sort([i for (i, c) in d if c < cnt[i]])
        push!(nodes, OMEinsumContractionOrders.NestedEinsum(Any[nodes[a+1], nodes[b+1]], OMEinsumContractionOrders.EinCode([legs[a+1], legs[b+1]], o)))
        push!(legs, o); push!(sub, d)
    end
    nodes[end]
end

opt0 = optimize_code(code, sizes, TreeSA(ntrials = 1, niters = 1, score = score))   # compile
println(ARGS[1], " threads=", Threads.nthreads(), " ntrials=", ntrials, " sc_target=", sct, " seed=", seedf)
for ni in nis
    init = isempty(seedf) ? code : fromssa(read_tree(seedf)[2], net.nt)
    alg = TreeSA(ntrials = ntrials, niters = ni, score = score, initializer = isempty(seedf) ? :greedy : :specified)
    t = @elapsed (opt = optimize_code(init, sizes, alg))
    sl = Int[]
    if isfinite(sct)   # TreeSASlicer to width sc_target (Kalachev et al.), same ntrials / niters
        sopt = TreeSASlicer(score = ScoreFunction(sc_target = sct), ntrials = ntrials, niters = ni)
        t += @elapsed (sc = slice_code(opt, sizes, sopt))
        sl = collect(Int, sc.slicing); opt = sc.eins
    end
    cc = contraction_complexity(opt, sizes)
    ssa = tossa(opt, net.nt)
    write_tree("$out.ni$ni.tree", net.nt, ssa, sl; meta = Dict("t" => @sprintf("%.3f", t), "treesa_tc" => @sprintf("%.6f", cc.tc),
        "treesa_sc" => @sprintf("%.3f", cc.sc), "niters" => string(ni), "ntrials" => string(ntrials), "threads" => string(Threads.nthreads())))
    @printf("niters %d: %.2f s, tc %.4f sc %.1f nsliced %d\n", ni, t, cc.tc, cc.sc, length(sl))
    flush(stdout)
end
