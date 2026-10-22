# Cost-aware selection among (near-)minimum-width tree decompositions: the CPU side.
#
#   include("CostSel.jl"); using .CostSel
#
# Front end = the one of HybridTW (safe rules, then safe separators -> atoms). The width of the graph is the
# largest width of an atom, so every atom may take ANY ordering of width <= K (K = the certified width, or
# K + delta) without changing the width of the graph; the pipeline instead keeps the first ordering that
# meets the bound (often a greedy one). Per atom we collect candidate MINIMAL triangulations (MinimalChordal
# of greedy orderings, of greedy orderings of random relabelings, of the exact solver septw at width K on
# random relabelings = its random tie-breaking, optionally HBT). Their maximal cliques are PMCs of the atom;
# the union of them (plus, when it is cheap, every PMC from all minimal separators lighter than K) feeds
# pmccost = the PMC dynamic program with (min, +) over the bag costs 2^w(bag) subject to bags <= K (the cost-
# aware tie-break inside the exact DP; on the candidate pool it recombines the candidates' pieces).
# Per atom: the DP's TD (exactly cheapest by lcost over the pool), DP with random multiplicative noise on
# the PMC costs (diverse cheap TDs), and the raw candidates. Whole-graph TDs are assemblies of one TD per
# atom (lcost is additive over the atoms).
module CostSel

using CliqueTrees, Graphs, Random, Printf
const CT = CliqueTrees

Base.include(CliqueTrees, joinpath(@__DIR__, "pmccost.jl"))

export lcostof, frontend, assemble, atommaps, restrict, addcand!, AtomPool, atompool, atomdp, atomenum!, relabeled

# log2 sum over maximal cliques of 2^w(clique)
function lcostof(w, g, o)
    label, tree = cliquetree(w, g; alg = o)
    s = 0.0
    for clique in tree
        s += exp2(sum(v -> w[label[v]], clique; init = 0.0))
    end
    return log2(s)
end

const MINS = (MinimalChordal(MF()), MinimalChordal(MF(; strategy = 1)), MinimalChordal(AMF()), MinimalChordal(MMD()))

"""Front end of HybridTW: (; rules, safe, graphs, items = [(bgraph, wgt)], lb, n10)."""
function frontend(weights::AbstractVector, graph; lbalg = CT.DEFAULT_LOWER_BOUND_ALGORITHM)
    W = Float64
    weights = convert(Vector{W}, collect(weights))
    lb0 = convert(W, CT.lowerbound(weights, graph, lbalg))
    weights10, graph10, rules = CT.compressreduce(5, weights, graph, lb0; minfrac = 1.0e-3)
    lb = max(lb0, convert(W, rules.width))
    if nv(graph10) > 1
        weights30, graphs30, safe = CT.safesolve(weights10, graph10, MINS, 50)
        lb = max(lb, convert(W, safe.width))
        items = [CT.hbt_graph(weights30[i], graphs30[i]) for i in eachindex(graphs30)]
    else
        safe = nothing; graphs30 = nothing; items = Any[]
    end
    return (; rules, safe, graphs = graphs30, items, lb, n10 = nv(graph10))
end

"""For each atom: the original vertices of each of its vertices (twin classes of the rules, classes of the
safe separators)."""
function atommaps(fe)
    maps = [Vector{Vector{Int}}() for _ in fe.items]
    isnothing(fe.safe) && return maps
    s = fe.safe; P = fe.rules.project
    for k in eachindex(s.lgraph)
        g = s.lgraph[k]; g == 0 && continue
        for x in 1:nv(fe.graphs[g])
            c = s.leafptr[k] + x - 1; orig = Int[]
            for p in s.prjptr[c]:s.prjptr[c + 1] - 1
                append!(orig, neighbors(P, s.prjtgt[p]))
            end
            push!(maps[g], orig)
        end
    end
    return maps
end

"""The ordering of an atom induced by an ordering of the whole graph (atom vertices by the first position of
their original vertices)."""
function restrict(amap, pos)
    return sortperm([minimum(v -> pos[v], orig) for orig in amap])
end

"""Full ordering from one ordering per atom."""
function assemble(fe, orders)
    order10 = isnothing(fe.safe) ? collect(1:fe.n10) : CT.safe_pull(fe.safe, fe.graphs, orders)
    return convert(Vector{Int}, CT.rule_pull(fe.rules, order10))
end

function relabeled(w, g::AbstractGraph, perm)
    n = length(perm); ip = invperm(perm); g2 = SimpleGraph(n)
    for v in 1:n, x in neighbors(g, v)
        v < x && add_edge!(g2, ip[v], ip[x])
    end
    return w[perm], CT.BipartiteGraph(g2)
end

# maximal cliques (sorted vertex lists) of the triangulation of ordering o, and their weights
function cliquesof(w, g, o)
    label, tree = cliquetree(w, g; alg = o)
    cl = Vector{Int}[]
    for clique in tree
        push!(cl, sort!([label[v] for v in clique]))
    end
    return cl
end

"""Per-atom state: candidate TDs (minimal triangulations) and the PMC pool."""
mutable struct AtomPool
    w::Vector{Float64}
    g::Any
    K::Float64
    cands::Vector{NamedTuple{(:order, :width, :lcost, :src, :key), Tuple{Vector{Int}, Float64, Float64, String, UInt64}}}
    keys::Set{UInt64}
    pmc::Dict{Vector{Int}, Int}       # PMC -> id
    pmcsets::Vector{Vector{Int}}
    pmcw::Vector{Float64}
    pmcgraph::Any                     # BipartiteGraph of pmcsets (rebuilt lazily)
    enumerated::Dict{Float64, Bool}   # cap -> all PMCs from the minimal separators lighter than cap are in the pool
    log::Vector{String}
end

AtomPool(w, g, K) = AtomPool(collect(Float64, w), g, K, [], Set{UInt64}(), Dict{Vector{Int}, Int}(), Vector{Int}[], Float64[], nothing, Dict{Float64, Bool}(), String[])

function addpmc!(P::AtomPool, c::Vector{Int})
    haskey(P.pmc, c) && return false
    push!(P.pmcsets, c); push!(P.pmcw, sum(v -> P.w[v], c; init = 0.0)); P.pmc[c] = length(P.pmcsets)
    P.pmcgraph = nothing
    return true
end

"""Add the minimal triangulation of ordering o (minimalized) as a candidate; its cliques go to the PMC pool.
Returns the candidate or nothing (duplicate TD)."""
function addcand!(P::AtomPool, o, src; minimalize = true)
    o = convert(Vector{Int}, collect(o))
    if minimalize
        o = convert(Vector{Int}, collect(first(permutation(P.w, P.g, MinimalChordal(o)))))
    end
    cl = cliquesof(P.w, P.g, o)
    key = hash(sort(cl))
    key in P.keys && return nothing
    push!(P.keys, key)
    foreach(c -> addpmc!(P, c), cl)
    width = maximum(c -> sum(v -> P.w[v], c; init = 0.0), cl; init = 0.0)
    lc = log2(sum(c -> exp2(sum(v -> P.w[v], c; init = 0.0)), cl; init = 0.0))
    c = (; order = o, width, lcost = lc, src = String(src), key)
    push!(P.cands, c)
    return c
end

function pmcbg(P::AtomPool)
    if isnothing(P.pmcgraph)
        n = nv(P.g); ptr = [1]; tgt = Int[]
        for c in P.pmcsets
            append!(tgt, c); push!(ptr, length(tgt) + 1)
        end
        P.pmcgraph = CT.BipartiteGraph{Int, Int}(n, length(P.pmcsets), length(tgt), ptr, tgt)
    end
    return P.pmcgraph
end

"""Candidate pool of one atom. texact: seconds of the first septw call at width K (unrelabeled); the
relabeled ones get min(texact, max(2, 3 t1))."""
function atompool(w, g, K; P = nothing, nrel = 8, nexact = 4, texact = 60.0, hbtT = 0.0, nhbt = 4, rng = Xoshiro(1), log = String[])
    isnothing(P) && (P = AtomPool(w, g, K))
    n = nv(g); P.log = log
    for (tag, a) in (("MF", MF()), ("MF1", MF(; strategy = 1)), ("AMF", AMF()), ("MMD", MMD()))
        o, _ = permutation(P.w, g, a); addcand!(P, o, tag)
    end
    n <= 2 && return P
    for k in 1:nrel
        perm = randperm(rng, n); w2, g2 = relabeled(P.w, g, perm)
        for (tag, a) in (("MF", MF()), ("MF1", MF(; strategy = 1)))
            o, _ = permutation(w2, g2, a); addcand!(P, perm[o], tag * "_r")
        end
    end
    tol = 1.0e-9
    # the exact solver at width K, only needed when the cheap candidates do not reach K
    if nexact > 0 && (minimum(c -> c.width, P.cands) > K + tol || nexact > 1)
        t1 = texact
        for k in 1:max(nexact, 1)
            perm = k == 1 ? collect(1:n) : randperm(rng, n); w2, g2 = relabeled(P.w, g, perm)
            tl = k == 1 ? texact : min(texact, max(2.0, 3t1))
            ts = time()
            o, _, kk, st = CT.septw(w2, g2, K, K; mt = tl, nt = 1)
            k == 1 && (t1 = time() - ts)
            st == CT.BT_FOUND ? addcand!(P, perm[o], "septw") : push!(log, "septw k=$k status=$st t=$(round(time() - ts; digits = 2))")
            k == 1 && st != CT.BT_FOUND && break   # the unrelabeled solve failed: the relabeled ones would too
        end
    end
    # the anytime search HBT (random seeds) on an atom with few candidates of width <= K
    if hbtT > 0 && nhbt > 0 && n >= 16 && count(c -> c.width <= K + tol, P.cands) < 8
        for k in 1:nhbt
            o, _ = permutation(P.w, g, HBT(; time = hbtT, threads = 1, seed = rand(rng, 1:10^6)))
            addcand!(P, o, "hbt")
        end
    end
    # RTW (contraction-recursive, as the pipeline's last exact stage) when still nothing reaches K
    if nexact > 0 && n > 2 && minimum(c -> c.width, P.cands) > K + tol
        ts = time()
        hb = HBT(; time = 2.0, threads = 1, seed = 1)
        o, _, kk, st, _ = CT.rtw(P.w, g, K, typemax(Float64), time() + 2texact, 1, (; alg = MF(; strategy = 1), hbt = hb, ticks = 100, verbose = false))
        kk <= K + tol ? addcand!(P, collect(o), "rtw") : push!(log, "rtw status=$st k=$kk t=$(round(time() - ts; digits = 2))")
    end
    return P
end

"""Add every PMC from the minimal separators lighter than cap (sepenum + pmcenum), if they can be listed
within mt seconds and maxsep separators. Returns (nsep, npmc_new, seconds) or nothing."""
function atomenum!(P::AtomPool, cap; mt = 10.0, maxsep = 20_000, maxpmc = 200_000)
    get(P.enumerated, cap, false) && return (0, 0, 0.0)
    W = Float64; dl = time() + mt; t0 = time()
    sep, st = CT.sepenum(P.w, P.g, convert(W, cap + 1.0e-9), dl, 1)
    if st == CT.BT_TIMEOUT
        push!(P.log, "enum cap=$cap: sepenum timeout"); return nothing
    end
    nsep = nv(sep)
    if nsep > maxsep
        push!(P.log, "enum cap=$cap: nsep=$nsep > $maxsep"); return nothing
    end
    pmc = CT.pmcenum(P.g, sep)
    if time() > dl || nv(pmc) > maxpmc
        push!(P.log, "enum cap=$cap: pmcenum $(nv(pmc)) PMCs, $(round(time() - t0; digits = 2)) s"); return nothing
    end
    nnew = 0
    for p in vertices(pmc)
        c = sort!(collect(Int, neighbors(pmc, p)))
        sum(v -> P.w[v], c; init = 0.0) <= cap + 1.0e-9 || continue
        nnew += addpmc!(P, c)
    end
    P.enumerated[cap] = true
    return (nsep, nnew, time() - t0)
end

"""pmccost over the pool at bag cap `cap`; returns the candidate (added to the pool) or nothing."""
function atomdp(P::AtomPool, cap, src; sigma = 0.0, rng = Xoshiro(1))
    isempty(P.pmcsets) && return nothing
    pg = pmcbg(P)
    mult = sigma > 0 ? exp.(sigma .* randn(rng, length(P.pmcsets))) : nothing
    perm, _, lc = CT.pmccost(P.w, P.g, pg, convert(Float64, cap); mult)
    isfinite(lc) || return nothing
    o = convert(Vector{Int}, collect(perm))
    c = addcand!(P, o, src)
    if isnothing(c)   # a TD already in the pool: find it
        o = convert(Vector{Int}, collect(first(permutation(P.w, P.g, MinimalChordal(o)))))
        key = hash(sort(cliquesof(P.w, P.g, o)))
        c = P.cands[findfirst(x -> x.key == key, P.cands)]
    end
    return c
end

end
