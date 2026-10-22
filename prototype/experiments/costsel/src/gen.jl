# CPU side of cost-aware selection: per network, many (near-)minimum-width TDs.
#
#   julia -t1 --project=<env> gen.jl <queue> <K.tsv> <outdir> [NU=1000] [NN=1000] [tmode=150]
#
# K.tsv: net, K (width to select at: the certified optimum, else the best known), certified.
# Per network (claimed by mkdir <outdir>/claims/<net>), output <outdir>/<net>.pool:
#   NET name n K certified natoms nmax t_front t_pool
#   ATOM i n m ncands npmc septw_found enum0 enum1 enum2 t      (one per atom with n > 2; enumK = nsep/npmc_new/s or -)
#   TD tag mode delta j t_cum width lcost br_log2C bucket_log2C
#   <ordering with the open indices last (Host.open_last; = the PEO cotengra's bucket stage uses), 1-based>
#   END
# Modes (whole-graph TDs = one TD per atom, assembled through the safe separators and the rules):
#   ref  delta 0/1/2 by width: the study's own TDs of the network (exactness resA/resB .td files), unchanged
#   uni  delta 0: each atom a uniformly random candidate of width <= K among the raw candidates (greedy,
#        relabeled greedy, septw at K on relabelings = the exact solver's random tie-breaking), minimalized
#   dp   delta d: pmccost (min sum 2^w(bag), bags <= K + d) over the atom's PMC pool (candidates' cliques,
#        plus all PMCs from separators lighter than K + d when listable in tenum s) -- one TD per delta
#   dpn  delta d: the same with random multiplicative noise (sigma 0.7) on the PMC costs: NN diverse TDs
# t_cum = seconds to produce TDs 1..j of that mode: the shared pool (front end + candidates) + the mode's
# own work (enumeration for dp/dpn, then each sample). br_log2C = branchstats cost of branchtree +
# branchrefine (our best conversion; = the study's "br" stage); bucket_log2C = Host.cost! of the bucket tree
# (= GPU kernel / cotengra "bucket" stage).
using CliqueTrees, Graphs, Random, Printf
const CT = CliqueTrees
include(joinpath(ENV["TNCORPUS_DIR"], "load.jl")); using .TNCorpus
include(joinpath(@__DIR__, "CostSel.jl")); using .CostSel
ENV["TNCORPUS_LOAD"] = joinpath(ENV["TNCORPUS_DIR"], "load.jl")
include(joinpath(ENV["GPUCOST_SRC"], "Host.jl")); using .Host

f6(x) = @sprintf("%.6f", x)

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

# orderings of the study's .td files (REF_TD = colon-separated dirs): [(tag, order, seconds)]
function readref(name)
    out = Tuple{String, Vector{Int}, Float64}[]
    for d in split(get(ENV, "REF_TD", ""), ':')
        f = joinpath(d, name * ".td"); isfile(f) || continue
        ls = readlines(f)
        for (k, l) in enumerate(ls)
            startswith(l, "ORDER\t") || continue
            x = split(l, '\t')
            push!(out, (String(x[2]), parse.(Int, split(ls[k + 1])), parse(Float64, x[3])))
        end
        break
    end
    return out
end

struct Ctx
    w::Vector{Float64}; g::Any; H::Any; point::Int; net::Any; fe::Any; K::Float64
end

const RECS = Any[]

function emit(io, ctx::Ctx, tag, mode, delta, j, tcum, o)
    width = treewidth(ctx.w, ctx.g, o)
    lc = lcostof(ctx.w, ctx.g, o)
    bt = branchtree(ctx.w, ctx.H, ctx.point; alg = o)
    br = branchrefine(ctx.w, ctx.H, ctx.point, bt)
    sr = branchstats(ctx.w, ctx.H, ctx.point, br)
    ol = Host.open_last(ctx.net, o)
    tr = Host.bucket_tree(ctx.net, ol); Host.cost!(tr, ctx.net)
    println(io, join(("TD", tag, mode, string(delta), string(j), f6(tcum), f6(width), f6(lc), f6(sr.cost), f6(tr.log2C)), '\t'))
    println(io, join(ol, ' '))
    push!(RECS, (; tag, mode, delta, j, tcum, width, lc, br = sr.cost, bk = tr.log2C, ssa = branchssa(br, ctx.point)))
    return width
end

flat(ssa) = join((string(x) for p in ssa for x in p), ' ')

# best of the first N TDs of each (mode, delta) group, by br cost and by bucket cost; the br ssa paths of the
# selected TDs (for the GPU polish and the cotengra recost)
function select!(io)
    groups = Dict{Tuple{String, Int}, Vector{Any}}()
    for r in RECS
        push!(get!(groups, (r.mode, r.delta), Any[]), r)
    end
    # pooled groups: every mode at delta 0 ("all0") and every mode at any delta ("all")
    for (key, f) in (("all0", r -> r.delta == 0), ("all", r -> true))
        rs = filter(f, RECS)
        isempty(rs) || (groups[(key, -1)] = sort(rs; by = r -> r.tcum))
    end
    want = Set{String}()
    for ((mode, d), rs) in sort!(collect(groups); by = first)
        for N in (1, 3, 10, 30, 100, 300, 1000, 3000, 10^9)
            N > length(rs) && N != 10^9 && continue
            sub = view(rs, 1:min(N, length(rs)))
            a = argmin([r.br for r in sub]); b = argmin([r.bk for r in sub])
            println(io, join(("SEL", mode, d, N == 10^9 ? length(rs) : N, sub[end].tcum, sub[a].tag, f6(sub[a].br), sub[b].tag, f6(sub[b].bk)), '\t'))
            push!(want, sub[a].tag); push!(want, sub[b].tag)
            N >= length(rs) && break
        end
    end
    for r in RECS
        r.tag in want || continue
        println(io, "SSA\t", r.tag); println(io, flat(r.ssa))
    end
end

# one ordering per atom -> full ordering; nothing if some atom has no choice
assemble1(ctx, choice) = any(isnothing, choice) ? nothing : assemble(ctx.fe, [c.order for c in choice])

function runnet(io, name, K, cert; NU = 1000, NN = 1000, tmode = 150.0, deltas = (0.0, 1.0, 2.0), tenum = 10.0, texact = 120.0, sigma = 0.7)
    inst = TNCorpus.load(name); w = inst.weights; g = TNCorpus.bipartite(inst); H, point = pointed(inst)
    net = Host.readnet(name)
    t0 = time()
    fe = frontend(w, g)
    tfront = time() - t0
    ctx = Ctx(w, g, H, point, net, fe, K)
    rng = Xoshiro(hash(name))
    pools = AtomPool[]; atominfo = String[]
    # the study's TDs of this network (exact solver + relabels, greedy, ND, HBT; or the B ladder), restricted
    # to every atom, are candidates too (reference pool; the exact one guarantees an atom TD of width K)
    refs = readref(name)
    maps = atommaps(fe)
    for (i, (bg, wgt)) in enumerate(fe.items)
        ta = time(); log = String[]
        P = AtomPool(wgt, bg, K)
        nref = 0
        if nv(bg) > 2
            for (tag, o, _) in refs
                pos = invperm(o)
                c = addcand!(P, restrict(maps[i], pos), "ref:" * tag)
                nref += !isnothing(c)
            end
        end
        hasK = any(c -> c.width <= K + 1.0e-9, P.cands)
        P = atompool(wgt, bg, K; P, nrel = 8, nexact = nv(bg) > 2 ? (hasK ? 2 : 4) : 0, texact = hasK ? 20.0 : texact, hbtT = 2.0, nhbt = 4, rng, log)
        push!(log, "nref=$nref")
        push!(pools, P)
        nv(bg) > 2 && push!(atominfo, join(("ATOM", i, nv(bg), ne(bg), length(P.cands), length(P.pmcsets),
            count(c -> c.src == "septw", P.cands), f6(time() - ta), join(log, ";")), '\t'))
    end
    tpool = time() - t0
    natoms = length(pools)
    println(io, join(("NET", name, length(w), f6(K), Int(cert), natoms, maximum(P -> nv(P.g), pools; init = 0), f6(tfront), f6(tpool)), '\t'))
    foreach(l -> println(io, l), atominfo); flush(io)
    tol = 1.0e-9
    # the study's whole-graph TDs as they are (mode ref; delta = 0, 1, 2 by width; t_cum = their own seconds)
    tr = 0.0; jr = 0
    for (tag, o, secs) in refs
        tr += secs
        wd = treewidth(w, g, o)
        d = wd <= K + tol ? 0 : wd <= K + 1 + tol ? 1 : wd <= K + 2 + tol ? 2 : -1
        d < 0 && continue
        jr += 1
        emit(io, ctx, "ref_" * tag, "ref", d, jr, tr, o)
    end
    if natoms == 0
        o = assemble(fe, Vector{Int}[])
        emit(io, ctx, "dp_0", "dp", 0, 0, tpool, o)
        return
    end
    # uni: raw candidates of width <= K
    raw = [filter(c -> c.width <= K + tol, P.cands) for P in pools]
    if all(!isempty, raw)
        seen = Set{UInt64}(); j = 0; tm = time(); miss = 0
        while j < NU && time() - tm < tmode && miss < 200
            choice = [rand(rng, r) for r in raw]
            h = hash([c.key for c in choice])
            if h in seen
                miss += 1; continue
            end
            push!(seen, h); miss = 0; j += 1
            o = assemble1(ctx, choice)
            emit(io, ctx, "uni_0_$j", "uni", 0, j, tpool + time() - tm, o)
        end
        flush(io)
    else
        println(io, "#NOUNI\t", join(findall(isempty, raw), ','))
    end
    # dp / dpn per delta
    for d in deltas
        cap = K + d; tm = time(); einfo = String[]
        for (i, P) in enumerate(pools)
            nv(P.g) > 2 || continue
            r = atomenum!(P, cap; mt = tenum)
            push!(einfo, isnothing(r) ? "$i:-" : "$i:$(r[1])/$(r[2])/$(round(r[3]; digits = 2))")
        end
        println(io, "#ENUM\t", f6(d), "\t", join(einfo, ' '), "\t", f6(time() - tm))
        choice = [atomdp(P, cap, "dp+$(Int(d))") for P in pools]
        o = assemble1(ctx, choice)
        if isnothing(o)
            println(io, "#NODP\t", f6(d)); continue
        end
        emit(io, ctx, "dp_$(Int(d))", "dp", Int(d), 1, tpool + time() - tm, o)
        seen = Set{UInt64}([hash([c.key for c in choice])]); j = 0; miss = 0
        while j < NN && time() - tm < tmode && miss < 200
            choice = [atomdp(P, cap, "dpn+$(Int(d))"; sigma, rng) for P in pools]
            h = hash([c.key for c in choice])
            if h in seen
                miss += 1; continue
            end
            push!(seen, h); miss = 0; j += 1
            emit(io, ctx, "dpn_$(Int(d))_$j", "dpn", Int(d), j, tpool + time() - tm, assemble1(ctx, choice))
        end
        flush(io)
    end
end

function main()
    queue, kfile, outdir = ARGS[1], ARGS[2], ARGS[3]
    NU = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1000
    NN = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1000
    tmode = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 150.0
    Ks = Dict{String, Tuple{Float64, Bool}}()
    for l in eachline(kfile)
        f = split(l, '\t'); length(f) >= 3 || continue
        f[1] == "net" && continue
        Ks[f[1]] = (parse(Float64, f[2]) + 1.0e-5, f[3] == "1")   # K is printed with 6 decimals
    end
    mkpath(joinpath(outdir, "claims"))
    # warm-up (compilation)
    runnet(IOBuffer(), "gm_1fas", 34.86918, true; NU = 5, NN = 5, tmode = 5.0); empty!(RECS)
    for name in filter(!isempty, strip.(readlines(queue)))
        isfile(joinpath(outdir, name * ".pool")) && continue
        haskey(Ks, name) || (println(stderr, "no K for $name"); continue)
        try
            mkdir(joinpath(outdir, "claims", name))
        catch
            continue
        end
        println(stderr, "[$(round(Int, time()))] start $name"); flush(stderr)
        f = joinpath(outdir, name * ".pool")
        try
            open(f * ".tmp", "w") do io
                K, cert = Ks[name]
                empty!(RECS)
                runnet(io, name, K, cert; NU, NN, tmode)
                select!(io)
                println(io, "END")
            end
            mv(f * ".tmp", f; force = true)
        catch e
            println(stderr, "FAILED $name: ", sprint(showerror, e, catch_backtrace()))
        end
        println(stderr, "[$(round(Int, time()))] done $name"); flush(stderr)
    end
end
main()
