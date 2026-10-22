# Tree decompositions of one TN at several quality levels, and the contraction trees they induce.
#
#   julia -t2 --project=<env> tdgen.jl <mode A|B> <queue file> <outdir> [texact=300]
#
# Workers share the queue: a network is claimed by mkdir <outdir>/claims/<name> (atomic), so any
# number of processes can run on one queue. Output: <outdir>/<name>.td (written via .tmp + mv).
#
# Mode A (exactness vs cost): exact (HybridTW, CPU, certified) + up to 7 exact solves on randomly
#   relabeled copies (other optimal TDs), MF/MF1/AMF/MMD/MCS/bw, MF and MF1 on 4 random relabelings
#   (random tie-breaking), METIS-ND (3 seeds + best of 8), HBT (SafeRules+SafeSeparators) at
#   0.1/1/10 s x 3 seeds.
# Mode B (orders for the GPU runs): ours = hybridcost(HybridTW(time=T), refine = branchrefine)
#   (pipeline ordering + MF/MF1/AMF/MMD portfolio, cheapest by exact total cost) at T = 10, 60 s;
#   the TD ladder MF1, HBT 1 s, exact (300 s).
#
# Per ordering (line-graph vertex permutation) the .td file has
#   ORDER tag seconds width lcost bt_cost bt_width bt_peak br_cost br_width br_peak br_seconds
#   <order, 1-based line-graph vertices = tnc labels>
#   <ssa path of branchtree (Richard's clique-tree binarization), flattened pairs>
#   <ssa path of branchrefine(branchtree)>
# width = max bag weight (log2), lcost = log2 sum over maximal bags of 2^w(bag); bt_/br_ costs are
# branchstats (log2 C, cotengra units), widths log2 of the largest intermediate, peak = log2 of the
# largest pairwise contraction. SSA paths index the non-empty tensors in .tnc order (= tnc.py load).
using CliqueTrees, Graphs, Metis, Random, SparseArrays, Printf
const CT = CliqueTrees
include(joinpath(ENV["TNCORPUS_DIR"], "load.jl")); using .TNCorpus

bwalg() = BestWidth(MF(), MF(; strategy = 1), AMF(), MMD())
nd1(s; alg = bwalg()) = ND(alg, METISND(; seed = s); width = 16, level = 12, scale = 100, imbalance = 300)
nd8(; alg = bwalg()) = BestWidth(ntuple(s -> nd1(s; alg), 8)...)
minalgs() = (MinimalChordal(MF()), MinimalChordal(MF(; strategy = 1)), MinimalChordal(AMF()), MinimalChordal(MMD()))
pipeline(alg) = SafeRules(SafeSeparators(alg, minalgs(); maxmissing = 50))

# pointed hypergraph without the empty tensors (the cotengra loader drops them)
function pointed(inst)
    nt = length(inst.hptr) - 1; ptr = [1]; idx = Int[]
    for t in 1:nt
        r = inst.hidx[inst.hptr[t]:inst.hptr[t + 1] - 1]
        isempty(r) && continue
        append!(idx, r); push!(ptr, length(idx) + 1)
    end
    append!(idx, inst.open); push!(ptr, length(idx) + 1)
    m = length(ptr) - 1
    return BipartiteGraph{Int, Int}(length(inst.weights), m, length(idx), ptr, idx), m
end

function lcost(w, g, o)
    label, tree = cliquetree(w, g; alg = o)
    s = 0.0
    for clique in tree
        s += exp2(sum(v -> w[label[v]], clique; init = 0.0))
    end
    return log2(s)
end

# a relabeled copy: vertex i of the copy is vertex perm[i] of g
function relabel(inst, perm)
    n = length(perm); ip = invperm(perm); g2 = SimpleGraph(n)
    for e in edges(inst.graph)
        add_edge!(g2, ip[src(e)], ip[dst(e)])
    end
    return inst.weights[perm], BipartiteGraph(g2)
end

flat(ssa) = join((string(x) for p in ssa for x in p), ' ')
f4(x) = @sprintf("%.6f", x)

function emit(io, ctx, tag, secs, o)
    (; w, g, H, point) = ctx
    width = treewidth(w, g, o); lc = lcost(w, g, o)
    bt = branchtree(w, H, point; alg = o); sb = branchstats(w, H, point, bt)
    t0 = time(); br = branchrefine(w, H, point, bt); tr = time() - t0
    sr = branchstats(w, H, point, br)
    println(io, join(("ORDER", tag, f4(secs), f4(width), f4(lc), f4(sb.cost), f4(sb.width), f4(sb.peak),
        f4(sr.cost), f4(sr.width), f4(sr.peak), f4(tr)), '\t'))
    println(io, join(o, ' '))
    println(io, flat(branchssa(bt, point)))
    println(io, flat(branchssa(br, point)))
    flush(io)
    return width
end

function timed(w, g, alg)
    GC.gc(false)
    t = @elapsed (o, _) = permutation(w, g, alg)
    return o, t
end

function exact!(io, ctx, inst, texact)
    (; w, g) = ctx
    t = @elapsed r = hybridtw(w, g, HybridTW(; time = texact, threads = Threads.nthreads(), gpu = false, seed = 1))
    println(io, join(("EXACT", f4(r.width), f4(r.lower), Int(r.certified), f4(t)), '\t'))
    emit(io, ctx, r.certified ? "exact" : "hyb$(round(Int, texact))", t, r.order)
    return r, t
end

function runA(io, ctx, inst, texact)
    (; w, g) = ctx; n = length(w)
    r, tex = exact!(io, ctx, inst, texact)
    if r.certified && tex < min(30.0, texact / 4)
        for k in 1:7
            perm = randperm(Xoshiro(100 + k), n); w2, g2 = relabel(inst, perm)
            t = @elapsed r2 = hybridtw(w2, g2, HybridTW(; time = min(texact, max(20.0, 4tex)), threads = Threads.nthreads(), gpu = false, seed = k))
            r2.certified && emit(io, ctx, "exact_r$k", t, perm[r2.order])
        end
    end
    for (tag, a) in (("MF", MF()), ("MF1", MF(; strategy = 1)), ("AMF", AMF()), ("MMD", MMD()), ("MCS", MCS()), ("bw", bwalg()))
        o, t = timed(w, g, a); emit(io, ctx, tag, t, o)
    end
    for k in 1:4
        perm = randperm(Xoshiro(k), n); w2, g2 = relabel(inst, perm)
        for (tag, a) in (("MF", MF()), ("MF1", MF(; strategy = 1)))
            o, t = timed(w2, g2, a); emit(io, ctx, "$(tag)_r$k", t, perm[o])
        end
    end
    for s in 1:3
        o, t = timed(w, g, nd1(s)); emit(io, ctx, "nd_s$s", t, o)
    end
    o, t = timed(w, g, nd8()); emit(io, ctx, "nd8", t, o)
    for T in (0.1, 1.0, 10.0), s in 1:3
        o, t = timed(w, g, pipeline(HBT(; time = T, threads = 1, seed = s))); emit(io, ctx, "hbt$(T)_s$s", t, o)
    end
end

function runB(io, ctx, inst, texact)
    (; w, H, point, g) = ctx
    ref = (w, H, p, par) -> branchrefine(w, H, p, par)
    for T in (10.0, 60.0)
        GC.gc(false)
        t = @elapsed r = hybridcost(w, H, point, HybridTW(; time = T, threads = 1, gpu = false, seed = 1, refine = ref))
        # r.order is over all indices (zero-weight last); the corpus has none of weight 0
        println(io, join(("OURS", "ours$(round(Int, T))", f4(t), string(r.source), f4(r.cost), f4(r.width), f4(r.peak),
            f4(r.result.width), f4(r.result.lower), Int(r.result.certified)), '\t'))
        println(io, flat(branchssa(r.parent, point))); flush(io)
    end
    o, t = timed(w, g, MF(; strategy = 1)); emit(io, ctx, "MF1", t, o)
    o, t = timed(w, g, pipeline(HBT(; time = 1.0, threads = 1, seed = 1))); emit(io, ctx, "hbt1.0_s1", t, o)
    exact!(io, ctx, inst, texact)
end

function warm()
    inst = TNCorpus.load("qaoazz_N30_p3_s0"); w = inst.weights; g = TNCorpus.bipartite(inst); H, point = pointed(inst)
    ctx = (; w, g, H, point)
    devnull_io = IOBuffer()
    runA(devnull_io, ctx, inst, 2.0)
    ref = (w, H, p, par) -> branchrefine(w, H, p, par)
    hybridcost(w, H, point, HybridTW(; time = 0.5, threads = 1, gpu = false, refine = ref))
end

function main()
    mode = ARGS[1]; queue = ARGS[2]; outdir = ARGS[3]; texact = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 300.0
    mkpath(joinpath(outdir, "claims"))
    warm()
    for name in filter(!isempty, strip.(readlines(queue)))
        isfile(joinpath(outdir, name * ".td")) && continue
        try
            mkdir(joinpath(outdir, "claims", name))
        catch
            continue
        end
        println(stderr, "[$(Dates_now())] start $name"); flush(stderr)
        inst = TNCorpus.load(name); w = inst.weights; g = TNCorpus.bipartite(inst); H, point = pointed(inst)
        ctx = (; w, g, H, point)
        f = joinpath(outdir, name * ".td")
        try
            open(f * ".tmp", "w") do io
                println(io, join(("NET", name, inst.n, nv(H) - 1, length(inst.open)), '\t'))
                mode == "A" ? runA(io, ctx, inst, texact) : runB(io, ctx, inst, texact)
                println(io, "END")
            end
            mv(f * ".tmp", f; force = true)
        catch e
            println(stderr, "FAILED $name: ", sprint(showerror, e, catch_backtrace()))
        end
        println(stderr, "[$(Dates_now())] done $name"); flush(stderr)
    end
end
Dates_now() = round(Int, time())
main()
