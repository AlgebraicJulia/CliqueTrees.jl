#
# The contraction-recursive algorithm RTW for treewidth (Tamaki 2023),
# with vertex weights: the width of a decomposition is the weight of
# its heaviest bag.
#
# A contraction of a graph is no wider than the graph, if a contracted
# vertex weighs as little as the lightest vertex of its part: the bags
# of a decomposition of the graph, mapped onto the contraction, are no
# heavier. So, to decide whether a connected graph G has a
# decomposition of width at most K, RTW recurses on its contractions:
#
#   - Each node of the recursion is a contraction H of G with a set of
#     PMCs of H, evaluated by the BT dynamic program of HBT (hbt_dp!).
#     If they admit a decomposition of width at most K, the answer is
#     yes.
#   - Otherwise, for each edge e of H, best first (rtw_edges), the PMCs
#     of H are mapped onto H/e, and H/e is decided. If the answer is
#     no, then it is no for H, and for G. If it is yes, the best
#     decompositions of H/e are mapped back onto H, made minimal, and
#     added to its PMCs; then a heuristic search (Tamaki's heuristic
#     PID, hpid.jl) improves them, for longer after each child.
#   - When every edge has been tried, an exact search (septw, behind
#     the safe reduction rules and safe separators) decides H. If the
#     answer is no, H is a certificate: a contraction of G with no
#     decomposition of width at most K, although each of its own
#     contractions by an edge has one.
#
# Starting from a heuristic decomposition (greedy, then HBT: rtw_root!),
# K is lowered below its width until the answer is no, or the width
# meets the target. HBT goes on in the background (rtw_bkg!): when it
# meets K first, the recursion stops, and K is lowered.
#
# Mapping PMCs between a graph G and a contraction H = G/γ is the BT
# dynamic program with the bags scored otherwise (Tamaki 2023, Sections
# 6 and 7): down, a bag X of G by the weight in H of γ(X); up, a bag Y
# of H by the weight in G of its preimage, a little less if the
# preimage is not a PMC of G (then making the decomposition minimal
# can make it lighter). The decompositions mapped either way are made
# minimal by MinimalChordal. (Tamaki makes those mapped up minimal
# optimally, Section 5: the lightest minimal triangulation inside their
# filled graph T, by sepdp on the minimal separators of G that are
# cliques of T. On the PACE instances, that found one of width at most
# K exactly when MinimalChordal did, and took up to 70% of the time.)
#
# Once H/e is decided yes, e is a reducer of H (Tamaki's suppressed
# edges, Section 10): a contraction of H/e has a decomposition of width
# at most K, so a later contraction by the image of e, of H or of a
# descendant, takes its PMCs from H/e, with no recursion.
#
# lo is a target: the search stops at the first decomposition of width
# at most lo (if lo is a lower bound, the decomposition is optimal). hi
# is a cap. The result is (perm, invp, k, status, cert), where (perm,
# invp) is the lightest ordering found, of width k, and
#
#   - BT_FOUND: k <= hi, and either k <= lo, or no ordering is lighter
#     than k;
#   - BT_ABOVE: every ordering has width greater than hi;
#   - BT_TIMEOUT: the search ran for more than mt seconds.
#
# Unlike the result of septw, the ordering is the lightest found
# whatever the status: the search is anytime. When the recursion proved
# that no ordering is lighter than k (or none is as light as hi), cert
# maps each vertex onto a vertex of a certificate: the contraction of
# the graph by the connected parts cert⁻¹(1), cert⁻¹(2), ..., whose
# vertices weigh as little as their parts, and which has no lighter
# decomposition. Otherwise cert is empty: in particular when the bound
# is the weight of the heaviest vertex or edge, which every bag meets.
#
# The graph is assumed connected (see permutation for RTW). The weights
# must be positive; with float weights, a lighter decomposition is one
# lighter by a relative margin (RTW_RTOL). The positional method takes
# the deadline dl, in the sense of time(), instead of mt, and the
# parameters par = (; alg, hbt, ticks, verbose): the elimination
# algorithm of the first decomposition, the first heuristic search (an
# HBT, or nothing), the ticks of heuristic PID per child, and whether to
# print progress. With nt > 1, the first heuristic search uses nt tasks.
#
#   - Tamaki, Hisao. "A contraction-recursive algorithm for treewidth."
#     *18th International Symposium on Parameterized and Exact
#     Computation (IPEC 2023)*. 2023.
#
function rtw(weights::AbstractVector, graph::AbstractGraph, lo, hi; mt::Real = Inf, nt::Integer = Threads.nthreads(),
        alg::PermutationOrAlgorithm = MF(; strategy = 1), hbt::Union{Nothing, HBT} = HBT(; time = 2.0), ticks::Integer = 100,
        verbose::Bool = false)
    return rtw(weights, graph, lo, hi, time() + mt, nt, (; alg, hbt, ticks = convert(Int, ticks), verbose))
end

function rtw(weights::AbstractVector, graph::AbstractGraph{V}, lo, hi, dl::Real, nt::Integer, par::NamedTuple) where {V}
    bgraph, wgt = hbt_graph(weights, graph); n = nv(bgraph); W = eltype(wgt)
    t0 = time(); hi = convert(W, hi); lo = min(convert(W, lo), hi)
    perm = FVector{V}(undef, n); invp = FVector{V}(undef, n); cert = V[]
    iszero(n) && return perm, invp, zero(W), BT_FOUND, cert

    # the root, and the state of the search
    rng = Xoshiro(isnothing(par.hbt) ? 1 : par.hbt.seed); hpar = rtw_hpar(par.hbt)
    root = rtw_node(bgraph, wgt, Int[], collect(oneto(n)), rng, hpar)

    run = (;
        clk = bt_clk(dl), par, hpar, rng, t0,
        nodes = typeof(root)[],                 # the nodes decided yes (reducers point here)
        cert = Ref(Int[]),                      # the map of the last certificate
        tsrh = Ref(0.0),                        # the time spent in heuristic searches
        nnode = Ref(0), nexa = Ref(0), nstep = Ref(0),
        # the first heuristic search, in the background (rtw_bkg!): the root,
        # the bound of the recursion, the time spent, and whether it met the
        # bound
        root, K = Ref(zero(W)), thbt = Ref(0.0), hit = Ref(false),
    )

    lo = max(lo, bt_bas(wgt, bgraph))

    # the first decomposition: greedy, then the heuristic search (rtw_root!),
    # until it meets the target. A graph that HBT would solve exactly (at
    # most base vertices) is left to the recursion.
    label, tree = rtw_min(bgraph, wgt, first(permutation(wgt, bgraph, par.alg)))
    hbt_cqs!(root.srh.ctx, root.ns.sol, label, tree); rtw_upd!(root)
    append!(root.avail, rtw_edges(bgraph))
    par.verbose && println("rtw: n = $n, greedy width $(hbt_str(rtw_width(root))), target $(hbt_str(lo))")

    if !isnothing(par.hbt) && rtw_width(root) > lo && n > par.hbt.base
        ts = time(); rtw_root!(run, root, lo, dl, nt, par.hbt); run.thbt[] += time() - ts
        par.verbose && println("rtw: hbt width $(hbt_str(rtw_width(root))) at $(round(time() - t0; digits = 2)) s")
    end

    # lower the width
    status = BT_FOUND

    while true
        k = rtw_width(root)

        # the target is met (or every decomposition is heavier than hi, if
        # the heaviest vertex or edge is)
        if k <= lo
            status = k <= hi ? BT_FOUND : BT_ABOVE; break
        end

        if time() > dl
            status = BT_TIMEOUT; break
        end

        K = k > hi ? hi : rtw_below(k)

        if K < lo
            status = k <= hi ? BT_FOUND : BT_ABOVE; break
        end
        run.K[] = K; run.hit[] = false
        yes = rtw_deep(() -> rtw_dec!(run, root, K), n)::Bool

        # the first heuristic search met the bound, and stopped the recursion
        if run.hit[]
            run.hit[] = false; run.clk.out[] = false
            par.verbose && println("rtw: hbt width $(hbt_str(rtw_width(root))) at $(round(time() - t0; digits = 2)) s ($(rtw_stats(run)))")
            continue
        end

        if run.clk.out[]
            status = BT_TIMEOUT
            par.verbose && println("rtw: out of time at $(round(time() - t0; digits = 2)) s")
            break
        end

        if !yes
            status = k > hi ? BT_ABOVE : BT_FOUND
            append!(cert, run.cert[])
            par.verbose && println("rtw: width $(hbt_str(k)) is optimal: certificate with $(maximum(cert)) vertices, at $(round(time() - t0; digits = 2)) s ($(rtw_stats(run)))")
            break
        end

        par.verbose && println("rtw: width $(hbt_str(rtw_width(root))) at $(round(time() - t0; digits = 2)) s ($(rtw_stats(run)))")
    end

    order = root.ns.order

    @inbounds for i in oneto(n)
        perm[i] = order[i]; invp[order[i]] = i
    end

    return perm, invp, rtw_width(root), status, cert
end

rtw_stats(run::NamedTuple) = "nodes = $(run.nnode[]), exact = $(run.nexa[]), steps = $(run.nstep[]), search = $(round(run.tsrh[]; digits = 2)) s"

# The bound for a decomposition lighter than one of width k. With float
# weights, the same bag summed in two orders can differ in its last
# bits, so the bound leaves a margin: a decomposition of width k never
# meets it.
const RTW_RTOL = 1.0e-10

rtw_below(k::Integer) = k - one(k)
rtw_below(k::AbstractFloat) = k - max(eps(k), abs(k) * RTW_RTOL)
rtw_below(k) = hbt_below(k)

# The heuristic searches get at most this share of the time: on a graph
# whose width is already optimal, every search fails.
const RTW_SHARE = 0.25

# The recursion goes one contraction deeper per level, and so can be
# nearly as deep as the graph has vertices: it runs on a task with a
# stack of RTW_STACK bytes per vertex, between RTW_STACK_MIN and
# RTW_STACK_MAX. Pages of the stack that are not used are not committed.
const RTW_STACK = 1 << 16
const RTW_STACK_MIN = 1 << 26
const RTW_STACK_MAX = 1 << 32

function rtw_deep(f::F, n::Integer) where {F}
    task = Task(f, clamp(RTW_STACK * n, RTW_STACK_MIN, RTW_STACK_MAX))
    schedule(task)
    return fetch(task)
end

# The first heuristic search gets at most this share of the time left,
# in slices of RTW_SLICE seconds (rtw_root!).
const RTW_ROOT = 0.5
const RTW_SLICE = 0.25

# The share of the time of the first heuristic search, when it goes on
# in the background (rtw_bkg!).
const RTW_BKG = 0.25

# The exact search (septw) that each graph of a batch gets first: this
# share of the time left, but at most RTW_PROBE_MAX seconds (rtw_batch).
const RTW_PROBE = 0.1
const RTW_PROBE_MAX = 10.0

# The exact search that each graph of a batch gets first (rtw_batch):
# septw (bt_tw with the rounds of septw_run), except that a round is not
# started when it would not end before the deadline, judged by the
# growth of the last two rounds. A round that runs out of time adds
# nothing to the lower bound, and the rounds grow fast.
function rtw_probe(weights::AbstractVector{W}, graph::AbstractGraph, lo, dl::Real, nt::Integer) where {W}
    return bt_tw(rtw_probe_run, weights, graph, convert(W, lo), typemax(W), dl, nt)
end

function rtw_probe_run(weights::AbstractVector{W}, graph::AbstractGraph{V}, k::W, lb::W, hi::W, hord::AbstractVector, hk::W, dl::Real, nt::Integer) where {W, V}
    wmin = minimum(view(weights, oneto(nv(graph)))); t0 = t1 = 0.0

    while true
        # every round below k failed, so the cheap ordering is optimal
        k >= hk && return hord, hk, hk <= hi ? BT_FOUND : BT_ABOVE
        k > hi && return hord, k, BT_ABOVE
        ts = time()
        ispositive(t1) && ts + max(2.0, ispositive(t0) ? t1 / t0 : 2.0) * t1 > dl && return hord, lb, BT_TIMEOUT
        sep, status = sepenum(weights, graph, k, dl, nt)
        status == BT_TIMEOUT && return hord, lb, BT_TIMEOUT
        perm, _, mink, status = sepdp(weights, graph, sep, k, dl, nt)
        status == BT_TIMEOUT && return hord, lb, BT_TIMEOUT
        status == BT_FOUND && return perm, mink, BT_FOUND
        lb = k = min(mink, k + wmin); t0 = t1; t1 = time() - ts
    end
end

# Has the search run out of time?
function rtw_out!(run::NamedTuple)
    time() > run.clk.deadline && (run.clk.out[] = true)
    return run.clk.out[]
end

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
#
# A node is a contraction of the root: an HBT search srh on its graph,
# with its own set of PMCs ns (a solution of HBT that only grows, with
# its best decomposition), the map pmap from the vertices of its parent
# onto its own (empty at the root), and the map rmap from the vertices
# of the root. At the root, the first heuristic search (HBT) keeps its
# own solution srh.sols[1], on the same graph context, so PMCs pass
# between the two as they are (hbt_cpy!); nsent is the number of PMCs of
# ns that it has. The heuristic PID of the node is made when it is first
# needed, for one K; it has the first npid PMCs of ns. So is bit, the
# adjacency of the node as packed sets, for the PMC tests of rtw_ispmc!,
# and clo, the closures of its blocks until it is lifted (rtw_clo).
#
# The edges of a node are untried (avail, best first) or reducers
# (u, v, r): the contraction by {u, v} is a contraction of the node
# run.nodes[r], which was decided yes. A child inherits the untried
# edges and reducers of its parent, mapped onto its vertices; a reducer
# whose ends merge (u == v) makes the child itself a contraction of
# run.nodes[r].
function rtw_node(graph::BipartiteGraph{Int, Int}, wgt::Vector{W}, pmap::Vector{Int}, rmap::Vector{Int}, rng::AbstractRNG, hpar::NamedTuple) where {W}
    srh = hbt_srh(hbt_ctx(graph, wgt), rng, hpar)
    return (;
        srh, ns = hbt_st(W, 0), pmap, rmap, avail = Tuple{Int, Int}[], reducers = Tuple{Int, Int, Int}[], nsent = Ref(0),
        pid = Ref{Any}(nothing), npid = Ref(0), bit = Ref{Any}(nothing), clo = Ref{Any}(nothing),
    )
end

rtw_graph(node::NamedTuple) = node.srh.ctx.graph
rtw_wgt(node::NamedTuple) = node.srh.ctx.wgt
rtw_width(node::NamedTuple) = node.ns.width[][1]

# The parameters of the heuristic searches, from those of an HBT (its
# defaults if there is none: then only the evaluation is used). Side
# solutions start from one greedy decomposition.
function rtw_hpar(hbt::Union{Nothing, HBT})
    alg = isnothing(hbt) ? HBT() : hbt

    return (; alg = alg.alg, ninit = 0, base = alg.base, ntry = alg.ntry, refined = alg.refined, merge = alg.merge,
        diversify = alg.diversify, dsize = alg.dsize, nsep = alg.nsep, patience = alg.patience, xtime = alg.xtime,
        xmin = alg.xmin, xstep = alg.xstep, near = alg.near, margin = alg.margin, pexact = alg.pexact,
        init = nothing, verbose = false)
end

# Evaluate the PMCs of the node, and record its decomposition if it is
# lighter. (Unlike hbt_upd!, this keeps every PMC.)
function rtw_upd!(node::NamedTuple)
    srh = node.srh; ns = node.ns
    width, root = hbt_dp!(srh.ctx, ns.sol)
    width < ns.width[] && hbt_rec!(srh, ns, width, root)
    return
end

# The contraction of the node by the edge {a, b}, without PMCs.
function rtw_chd(run::NamedTuple, node::NamedTuple, a::Int, b::Int)
    run.nnode[] += 1
    graph, wgt, pmap = rtw_contract(rtw_graph(node), rtw_wgt(node), a, b)
    return rtw_node(graph, wgt, pmap, pmap[node.rmap], run.rng, run.hpar)
end

# The untried edges and the reducers of the node, mapped onto the child.
function rtw_inh!(child::NamedTuple, node::NamedTuple)
    pmap = child.pmap; red = Set{Tuple{Int, Int}}(); keep = Set{Tuple{Int, Int}}()

    for (u, v, r) in node.reducers
        e = minmax(pmap[u], pmap[v])

        if e ∉ red
            push!(red, e); push!(child.reducers, (e..., r))
        end
    end

    for (u, v) in node.avail
        e = minmax(pmap[u], pmap[v])
        e[1] != e[2] && e ∉ red && push!(keep, e)
    end

    append!(child.avail, rtw_edges(rtw_graph(child), keep))
    return child
end

# The map from the vertices of src onto those of dst, where dst is a
# contraction of src (both are contractions of the root).
function rtw_rebase(src::NamedTuple, dst::NamedTuple)
    m = zeros(Int, nv(rtw_graph(src)))

    @inbounds for v in eachindex(src.rmap)
        m[src.rmap[v]] = dst.rmap[v]
    end

    return m
end

# ---------------------------------------------------------------------------
# Graphs and contractions
# ---------------------------------------------------------------------------

# The contraction of the edge {a, b}: b is merged into a, and the
# vertices after b are shifted down. The merged vertex weighs as little
# as the lighter of a and b. Returns the graph, its weights, and the map
# from the vertices of the graph onto those of the contraction.
function rtw_contract(graph::BipartiteGraph{Int, Int}, wgt::Vector{W}, a::Int, b::Int) where {W}
    n = nv(graph); pmap = Vector{Int}(undef, n)

    @inbounds for v in oneto(n)
        pmap[v] = v < b ? v : v - 1
    end

    pmap[b] = pmap[a]
    ptr = Vector{Int}(undef, n); tgt = Int[]; nwgt = Vector{W}(undef, n - 1); ptr[1] = 1

    @inbounds for v in oneto(n)
        v == b && continue
        x = pmap[v]; lo = length(tgt) + 1; nwgt[x] = wgt[v]

        for y in neighbors(graph, v)
            z = pmap[y]; z != x && push!(tgt, z)
        end

        if v == a
            nwgt[x] = min(wgt[a], wgt[b])

            for y in neighbors(graph, b)
                z = pmap[y]; z != x && push!(tgt, z)
            end
        end

        sort!(view(tgt, lo:length(tgt))); q = lo

        for i in lo:length(tgt)
            if q == lo || tgt[i] != tgt[q - 1]
                tgt[q] = tgt[i]; q += 1
            end
        end

        resize!(tgt, q - 1); ptr[x + 1] = q
    end

    return BipartiteGraph{Int, Int}(n - 1, n - 1, length(tgt), ptr, tgt), nwgt, pmap
end

# Is the graph connected?
function rtw_connected(graph::BipartiteGraph{Int, Int})
    n = nv(graph); iszero(n) && return true
    seen = falses(n); queue = [1]; seen[1] = true; head = 1

    @inbounds while head <= length(queue)
        u = queue[head]; head += 1

        for x in neighbors(graph, u)
            if !seen[x]
                seen[x] = true; push!(queue, x)
            end
        end
    end

    return length(queue) == n
end

# The parts of a contraction with m vertices: the vertices of the graph
# mapped onto each, as the lists pptr/ptgt.
function rtw_parts(pmap::Vector{Int}, m::Int)
    pptr = zeros(Int, m + 1)

    @inbounds for y in pmap
        pptr[y + 1] += 1
    end

    pptr[1] = 1

    @inbounds for y in oneto(m)
        pptr[y + 1] += pptr[y]
    end

    ptgt = Vector{Int}(undef, length(pmap)); next = pptr[1:m]

    @inbounds for (v, y) in enumerate(pmap)
        ptgt[next[y]] = v; next[y] += 1
    end

    return pptr, ptgt
end

# The bags (bptr, btgt) mapped onto a contraction with m vertices, by
# pmap (each bag sorted, without repeats).
function rtw_img(bptr::Vector{Int}, btgt::Vector{Int}, pmap::Vector{Int}, m::Int)
    ptr = Int[1]; tgt = Int[]; mark = zeros(Int, m)

    @inbounds for i in oneto(length(bptr) - 1)
        lo = length(tgt) + 1

        for v in view(btgt, bptr[i]:bptr[i + 1] - 1)
            y = pmap[v]

            if mark[y] != i
                mark[y] = i; push!(tgt, y)
            end
        end

        sort!(view(tgt, lo:length(tgt))); push!(ptr, length(tgt) + 1)
    end

    return ptr, tgt
end

# The preimage (sorted) of the set X of vertices of a contraction.
function rtw_pre!(P::Vector{Int}, X::AbstractVector{Int}, pptr::Vector{Int}, ptgt::Vector{Int})
    empty!(P)

    @inbounds for y in X
        append!(P, view(ptgt, pptr[y]:pptr[y + 1] - 1))
    end

    return sort!(P)
end

# The bags (bptr, btgt) of a contraction mapped back onto the graph.
function rtw_pre(bptr::Vector{Int}, btgt::Vector{Int}, pptr::Vector{Int}, ptgt::Vector{Int})
    ptr = Int[1]; tgt = Int[]; P = Int[]

    @inbounds for i in oneto(length(bptr) - 1)
        append!(tgt, rtw_pre!(P, view(btgt, bptr[i]:bptr[i + 1] - 1), pptr, ptgt))
        push!(ptr, length(tgt) + 1)
    end

    return ptr, tgt
end

# Is the set P (sorted) a PMC of the node? At once if it is one of its
# PMCs; otherwise by the test of the heuristic PID (hpid_ispmc), on packed
# sets, with the adjacency of the node made when it is first needed.
function rtw_ispmc!(node::NamedTuple, P::AbstractVector{Int})
    ispositive(bt_tab_find(node.ns.sol.pmcs, P)) && return true
    return rtw_bitpmc!(rtw_bit(node), P)
end

# The adjacency of the node as packed sets, with a pool.
function rtw_bit(node::NamedTuple)
    if isnothing(node.bit[])
        g = rtw_graph(node); n = nv(g); nwrd = cld(n, 64)
        adj = bt_mat(nwrd, n); bt_adj!(adj, g)
        node.bit[] = (; pool = UInt64[], adj, nwrd, n)
    end

    return node.bit[]
end

function rtw_bitpmc!(bit::NamedTuple, P::AbstractVector{Int})
    pool = bit.pool; empty!(pool); p = bt_new!(pool, bit.nwrd)

    @inbounds for v in P
        pool[p + bt_wrd(v)] |= bt_bit(v)
    end

    ok = hpid_ispmc(bit, p); empty!(pool)
    return ok
end

# The edges {u, v} (u < v), in the order of Tamaki's edge values: an
# edge comes early if contracting it into one of its ends v leaves the
# neighborhood of v nearly a clique (few missing edges for its degree),
# so that the contraction is likely to keep the width. With keep, only
# the edges in keep.
function rtw_edges(graph::BipartiteGraph{Int, Int}, keep::Union{Nothing, Set{Tuple{Int, Int}}} = nothing)
    n = nv(graph); mark = zeros(Int, n)
    deg = [length(neighbors(graph, v)) for v in oneto(n)]

    # the common neighbors of the ends of each arc, and the missing edges
    # in the neighborhood of each vertex
    ptr = pointers(graph); common = Vector{Int}(undef, de(graph)); miss = zeros(Int, n)

    @inbounds for u in oneto(n)
        for x in neighbors(graph, u)
            mark[x] = u
        end

        total = 0

        for (j, w) in enumerate(neighbors(graph, u))
            c = 0

            for x in neighbors(graph, w)
                mark[x] == u && (c += 1)
            end

            common[ptr[u] + j - 1] = c; total += c
        end

        miss[u] = (deg[u] * (deg[u] - 1) - total) ÷ 2
    end

    keys = Tuple{Rational{Int}, Int, Rational{Int}, Int}[]; edges = Tuple{Int, Int}[]

    @inbounds for u in oneto(n)
        for (j, v) in enumerate(neighbors(graph, u))
            u < v || continue
            (isnothing(keep) || (u, v) in keep) || continue
            c = common[ptr[u] + j - 1]

            # the missing edges of N(u) - v, and of N(v) - u
            mu = miss[u] - (deg[u] - 1 - c); mv = miss[v] - (deg[v] - 1 - c)
            a = (mu // deg[u], deg[u]); b = (mv // deg[v], deg[v])
            push!(keys, a <= b ? (a..., b...) : (b..., a...)); push!(edges, (u, v))
        end
    end

    return edges[sortperm(keys)]
end

# ---------------------------------------------------------------------------
# Decompositions
# ---------------------------------------------------------------------------

# The maximal cliques of a minimal triangulation inside the
# triangulation by order: they are PMCs. Returns (label, tree), a
# clique tree of it.
function rtw_min(graph::BipartiteGraph{Int, Int}, wgt::AbstractVector, order::AbstractVector)
    order, _ = permutation(wgt, graph, MinimalChordal(order))
    return cliquetree(wgt, graph, order)
end

# Add the PMCs of a minimal triangulation inside the triangulation by
# order to the node.
function rtw_add!(node::NamedTuple, order::AbstractVector)
    label, tree = rtw_min(rtw_graph(node), rtw_wgt(node), order)
    hbt_cqs!(node.srh.ctx, node.ns.sol, label, tree); rtw_upd!(node)
    return
end

# The closed neighborhood N[C] (sorted) of the block b: the component of
# G - S that holds its smallest vertex, and S, found on packed sets.
function rtw_closure!(X::Vector{Int}, bit::NamedTuple, ctx::NamedTuple, b::Int)
    pool = bit.pool; nwrd = bit.nwrd; empty!(pool); empty!(X)
    r = bt_new!(pool, nwrd); s = bt_new!(pool, nwrd)

    @inbounds for v in bt_tab_seq(ctx.seps, ctx.bsep[b])
        pool[s + bt_wrd(v)] |= bt_bit(v)
    end

    @inbounds for i in oneto(nwrd)
        pool[r + i] = ~pool[s + i]
    end

    bt_cap!(pool, bit.n, r, nwrd); c = bt_new!(pool, nwrd)
    bt_cmp!(pool, bit.adj, c, r, ctx.bmin[b], nwrd)

    @inbounds for i in oneto(nwrd)
        word = pool[c + i] | pool[s + i]

        while !iszero(word)
            push!(X, bt_vtx(Int, i, word)); word &= word - one(UInt64)
        end
    end

    empty!(pool)
    return X
end

# The BT dynamic program on the PMCs of the node, with every bag (a PMC,
# or the closed neighborhood N[C] of a block C) scored by score instead
# of its weight (the bags N[C] by bscore, if it is given). The weights
# of the context are kept apart and put back by rtw_rst!; until then the
# program's tables hold the scored values. Returns the weights kept, the
# best value, and its root.
#
# With cache, the closures of the blocks are kept on the node (rtw_clo):
# a node maps its PMCs onto each of its children, and is lifted onto its
# parent after it is decided. The parent then drops them.
function rtw_scr!(node::NamedTuple, score::F, cache::Bool = false, bscore::G = score) where {F, G}
    ctx = node.srh.ctx; sol = node.ns.sol
    kept = (copy(sol.pwgt), copy(ctx.bwgt), copy(ctx.swgt))

    @inbounds for p in oneto(hbt_len(sol))
        sol.pwgt[p] = score(hbt_pmc(sol, p))
    end

    rtw_bsc!(ctx, sol, bscore, rtw_bit(node), cache ? rtw_clo(node) : nothing)
    width, root = hbt_dp!(ctx, sol)
    return kept, width, root
end

# The bag N[C] of a block weighs bwgt + swgt in the program.
function rtw_bsc!(ctx::NamedTuple, sol::NamedTuple, score::F, bit::NamedTuple, clo::Union{Nothing, NamedTuple}) where {F}
    W = eltype(ctx.wgt); seen = falses(length(ctx.bmin)); X = Int[]

    @inbounds for blks in (sol.csub, sol.csup), b in blks
        if !seen[b]
            seen[b] = true; ctx.swgt[ctx.bsep[b]] = zero(W)
            ctx.bwgt[b] = score(isnothing(clo) ? rtw_closure!(X, bit, ctx, b) : rtw_clo!(clo, X, bit, ctx, b))
        end
    end

    return
end

# The closures of the blocks of the node, kept: the closure of block b is
# tgt[ptr[b]:ptr[b] + len[b] - 1], if ptr[b] is positive.
function rtw_clo(node::NamedTuple)
    isnothing(node.clo[]) && (node.clo[] = (; ptr = Int[], len = Int[], tgt = Int[]))
    return node.clo[]
end

function rtw_clo!(clo::NamedTuple, X::Vector{Int}, bit::NamedTuple, ctx::NamedTuple, b::Int)
    ptr = clo.ptr

    while length(ptr) < b
        push!(ptr, 0); push!(clo.len, 0)
    end

    @inbounds if iszero(ptr[b])
        rtw_closure!(X, bit, ctx, b); ptr[b] = length(clo.tgt) + 1; clo.len[b] = length(X); append!(clo.tgt, X)
    end

    return @inbounds view(clo.tgt, ptr[b]:ptr[b] + clo.len[b] - 1)
end

function rtw_rst!(node::NamedTuple, kept::Tuple)
    ctx = node.srh.ctx; sol = node.ns.sol
    copyto!(sol.pwgt, kept[1]); copyto!(ctx.bwgt, kept[2]); copyto!(ctx.swgt, kept[3])
    return
end

# Call f(bptr, btgt, parent) on the best decompositions of the scored
# program of the node: one for each root of the best value whose bags
# are not all in an earlier one (Tamaki's firstTD and anotherTD), until
# f returns true. Returns whether it did.
function rtw_each(f::F, node::NamedTuple, width) where {F}
    ctx = node.srh.ctx; sol = node.ns.sol; np = hbt_len(sol); used = falses(np)

    for p in oneto(np)
        (used[p] || sol.root[p] != width) && continue
        bptr, btgt, parent = hbt_td(ctx, sol, p)

        for i in eachindex(parent)
            q = bt_tab_find(sol.pmcs, view(btgt, bptr[i]:bptr[i + 1] - 1))
            ispositive(q) && (used[q] = true)
        end

        f(bptr, btgt, parent) && return true
    end

    return false
end

# The score of an uncontracted bag of weight w: a little less if it is
# not a PMC of the graph (with integer weights, the scores are doubled).
rtw_score(w::Integer, ispmc::Bool) = ispmc ? 2w : 2w - one(w)
rtw_score(w::AbstractFloat, ispmc::Bool) = ispmc ? w : prevfloat(w)
rtw_score(w, ispmc::Bool) = w

# The PMCs of src mapped onto dst, a contraction of src by m: the
# decomposition of src whose image is lightest, mapped and made minimal.
function rtw_dwn!(dst::NamedTuple, src::NamedTuple, m::Vector{Int}, cache::Bool = false)
    wgt = rtw_wgt(dst); W = eltype(wgt); nd = nv(rtw_graph(dst)); mark = zeros(Int, nd); stamp = Ref(0)

    # the weight in dst of the image of a bag
    score = X -> begin
        s = stamp[] += 1; w = zero(W)

        @inbounds for v in X
            y = m[v]

            if mark[y] != s
                mark[y] = s; w += wgt[y]
            end
        end

        w
    end

    kept, _, root = rtw_scr!(src, score, cache)
    bptr, btgt, parent = hbt_td(src.srh.ctx, src.ns.sol, root)
    rtw_rst!(src, kept)
    bptr, btgt = rtw_img(bptr, btgt, m, nd)
    rtw_add!(dst, hbt_ord(dst.srh.ctx, bptr, btgt, parent))
    return
end

# Map the best decompositions of the child back onto the node, made
# minimal (MinimalChordal), and add their PMCs to the node, until its
# width is at most K. Returns whether it is.
function rtw_lft!(run::NamedTuple, node::NamedTuple, child::NamedTuple, K)
    pptr, ptgt = rtw_parts(child.pmap, nv(rtw_graph(child))); P = Int[]
    wgt = rtw_wgt(node); W = eltype(wgt)

    # the weight of the preimage of a bag, less if it is not a PMC
    score = X -> begin
        rtw_pre!(P, X, pptr, ptgt); w = zero(W)

        @inbounds for v in P
            w += wgt[v]
        end

        rtw_score(w, rtw_ispmc!(node, P))
    end

    # the bag N[C] of a block: a PMC only if C is a clique, and too heavy
    # to matter when C is large, so it is scored as if it were not one,
    # from the weights of the parts
    pw = zeros(W, nv(rtw_graph(child)))

    @inbounds for (v, y) in enumerate(child.pmap)
        pw[y] += wgt[v]
    end

    bscore = X -> begin
        w = zero(W)

        @inbounds for y in X
            w += pw[y]
        end

        rtw_score(w, false)
    end

    kept, width, _ = rtw_scr!(child, score, true, bscore)

    found = rtw_each(child, width) do bptr, btgt, parent
        rtw_out!(run) && return true
        bptr, btgt = rtw_pre(bptr, btgt, pptr, ptgt)
        rtw_add!(node, hbt_ord(node.srh.ctx, bptr, btgt, parent))
        run.clk.out[] || rtw_width(node) <= K
    end

    rtw_rst!(child, kept)
    return found && !run.clk.out[]
end

# ---------------------------------------------------------------------------
# The recursion
# ---------------------------------------------------------------------------

# The first heuristic search, at the root: HBT for its time, and then
# for as long as it keeps improving, in slices of RTW_SLICE seconds: it
# stops when the time since its last improvement is longer than the time
# it took to make it (at least a slice), or at RTW_ROOT of the time left.
# On a hard graph, HBT can improve for many seconds, and its width is the
# one returned if the recursion does not finish.
function rtw_root!(run::NamedTuple, root::NamedTuple, lo, dl::Real, nt::Integer, hbt::HBT)
    ts = time(); cap = min(dl, ts + max(hbt.time, RTW_ROOT * (dl - ts)))
    tl = ts; w = rtw_width(root)

    while true
        rtw_hbt!(run, root, lo, min(cap, time() + RTW_SLICE), nt, typemax(Int)); now = time()

        if rtw_width(root) < w
            w = rtw_width(root); tl = now
        end

        (w <= lo || now >= cap || run.clk.out[]) && break
        now - ts >= hbt.time && now - tl > max(RTW_SLICE, tl - ts) && break
    end

    return
end

# The first heuristic search (HBT), at the root: given the PMCs of the
# node that it does not have yet, it runs until the deadline, until its
# width is at most K, or until it has not improved for patience steps
# (with nt > 1, on nt tasks, and to the deadline or K only); if it is
# lighter than the node, its PMCs are added to the node.
function rtw_hbt!(run::NamedTuple, node::NamedTuple, K, deadline::Float64, nt::Integer, patience::Integer)
    srh = node.srh; ns = node.ns; W = eltype(srh.ctx.wgt)
    isempty(srh.sols) && push!(srh.sols, hbt_st(W, 0))
    st = srh.sols[1]

    for p in node.nsent[] + 1:hbt_len(ns.sol)
        hbt_cpy!(st.sol, ns.sol, p)
    end

    node.nsent[] = hbt_len(ns.sol); hbt_upd!(srh, st); srh.deadline[] = deadline

    if nt > 1
        hbt_par!(srh, st, K, nothing, nt)
    else
        while st.width[][1] > K && !hbt_expired(srh) && st.stall[] < patience
            hbt_imp!(srh, st); run.nstep[] += 1
        end
    end

    rtw_out!(run)

    if st.width[] < ns.width[]
        for p in oneto(hbt_len(st.sol))
            hbt_cpy!(ns.sol, st.sol, p)
        end

        node.nsent[] = hbt_len(ns.sol); rtw_upd!(node)
    end

    return
end

# The first heuristic search goes on in the background: a slice of it
# (RTW_SLICE seconds) after each child of any node, within its share of
# the time (RTW_BKG, counting the time it had at first). When it finds a
# decomposition of the root of width at most the bound of the recursion,
# the recursion stops, as if it ran out of time, and the bound is
# lowered: on a hard graph, HBT keeps improving long after the first
# search, in bursts that no rule of patience foresees.
function rtw_bkg!(run::NamedTuple, ts::Float64)
    hbt = run.par.hbt; root = run.root
    (isnothing(hbt) || nv(rtw_graph(root)) <= hbt.base || run.thbt[] > RTW_BKG * (ts - run.t0)) && return
    K = run.K[]; rtw_hbt!(run, root, K, min(run.clk.deadline, ts + RTW_SLICE), 1, typemax(Int))
    run.thbt[] += time() - ts

    if rtw_width(root) <= K
        run.hit[] = true; run.clk.out[] = true
    end

    return
end

# The heuristic PID of the node (Tamaki's improver): given the PMCs of
# the node that it does not have yet, it searches for ticks ticks. If it
# finds a decomposition of width at most K, its PMCs of width at most K
# (usefulPMCs) are added to the node. (Tamaki adds them whether or not:
# then the node's PMCs grow faster than they help, and so does the cost
# of mapping them onto its children.)
function rtw_pid!(run::NamedTuple, node::NamedTuple, K, ticks::Int)
    hp = node.pid[]

    if isnothing(hp) || hp.K != K
        hp = hpid(node.srh.ctx, K); node.pid[] = hp; node.npid[] = 0
    end

    rtw_pid!(run, node, hp, ticks)
    return
end

function rtw_pid!(run::NamedTuple, node::NamedTuple, hp::NamedTuple, ticks::Int)
    ctx = node.srh.ctx; sol = node.ns.sol
    hpid_import!(hp, (hbt_pmc(sol, p) for p in node.npid[] + 1:hbt_len(sol)))
    hpid_improve!(hp, ticks, run.clk.deadline); run.nstep[] += hp.tick[]
    rtw_out!(run); out = hpid_width(hp) <= hp.K ? hpid_export!(hp) : Vector{Int}[]

    for X in out
        hbt_add!(ctx, sol, X)
    end

    node.npid[] = hbt_len(sol)
    isempty(out) || rtw_upd!(node)
    return
end

# The heuristic search after the nchild-th child: the heuristic PID for
# nchild times its ticks, within the share of the time that the searches
# get. Returns whether the width is at most K.
function rtw_srh!(run::NamedTuple, node::NamedTuple, K, nchild::Int)
    ts = time(); rtw_bkg!(run, ts)
    run.clk.out[] && return false

    if run.tsrh[] <= RTW_SHARE * (ts - run.t0)
        rtw_pid!(run, node, K, run.par.ticks * nchild)
        run.tsrh[] += time() - ts
    end

    return rtw_width(node) <= K
end

# Decide the node by an exact search. Returns whether it has a
# decomposition of width at most K; if not, and the search did not run
# out of time, it is the certificate.
function rtw_fin!(run::NamedTuple, node::NamedTuple, K::W) where {W}
    run.nexa[] += 1; graph = rtw_graph(node); wgt = rtw_wgt(node); dl = run.clk.deadline
    lb = convert(W, lowerbound(wgt, graph, DEFAULT_LOWER_BOUND_ALGORITHM))

    if lb <= K
        order, status = hbt_safe(wgt, graph, lb, K, K, dl)

        if status == BT_TIMEOUT
            run.clk.out[] = true; return false
        elseif status == BT_FOUND
            rtw_add!(node, order)
            rtw_width(node) <= K && return true

            # The pieces of the kernel have decompositions of width at most
            # K, but the reduction rules may force a heavier bag: with
            # integer weights, a lower bound on the width (see hbt_exa).
            # With float weights, the bag may be heavier by rounding alone,
            # so the question is settled without the rules.
            if !(W <: Integer)
                perm, _, _, status = septw(wgt, graph, K, K, dl, 1)

                if status == BT_TIMEOUT
                    run.clk.out[] = true; return false
                elseif status == BT_FOUND
                    rtw_add!(node, perm); return true
                end
            end
        end
    end

    run.cert[] = copy(node.rmap)
    return false
end

# Decide whether the node has a decomposition of width at most K. If it
# has, its PMCs admit one when this returns true. If this returns false,
# either the search ran out of time (run.clk.out[]), or the node has no
# such decomposition and the certificate is recorded.
function rtw_dec!(run::NamedTuple, node::NamedTuple, K)
    rtw_width(node) <= K && return true
    rtw_out!(run) && return false

    # every bag weighs at most K
    if sum(rtw_wgt(node)) <= K
        rtw_add!(node, first(permutation(rtw_wgt(node), rtw_graph(node), MF(; strategy = 1))))
        return true
    end

    # a reducer whose ends have merged: the node is a contraction of a
    # node decided yes
    for (u, v, r) in node.reducers
        if u == v
            R = run.nodes[r]; rtw_dwn!(node, R, rtw_rebase(R, node))
            rtw_width(node) <= K && return true
        end
    end

    # the contractions by the other reducers take their PMCs from them
    for (u, v, r) in node.reducers
        u == v && continue
        rtw_out!(run) && return false
        R = run.nodes[r]; child = rtw_chd(run, node, u, v)
        rtw_dwn!(child, R, rtw_rebase(R, child))
        rtw_lft!(run, node, child, K) && return true
        run.clk.out[] && return false
    end

    # the untried edges: a recursive call each, after which the edge is a
    # reducer, and then a heuristic search, longer after each child
    nchild = 0

    while !isempty(node.avail)
        a, b = popfirst!(node.avail)
        child = rtw_chd(run, node, a, b)
        rtw_dwn!(child, node, child.pmap, true)
        rtw_inh!(child, node)
        rtw_dec!(run, child, K) || return false
        push!(run.nodes, child); push!(node.reducers, (a, b, length(run.nodes))); nchild += 1
        found = rtw_lft!(run, node, child, K); child.clo[] = nothing
        found && return true
        run.clk.out[] && return false
        rtw_srh!(run, node, K, nchild) && return true
        run.clk.out[] && return false
    end

    return rtw_fin!(run, node, K)
end
