# Heuristic upper bounds on treewidth by BT dynamic programming over sets of
# potential maximal cliques.
#
#   - Tamaki, Hisao. "A heuristic use of dynamic programming to upperbound
#     treewidth." arXiv:1909.07647 (2019).
#   - Tamaki, Hisao. "Heuristic computation of exact treewidth." arXiv:2202.07793
#     (2022). Reference implementation: https://github.com/twalgor/tw
#     (io.github.twalgor.upper.HBTMerge).
#
# A *solution* is a set Π of potential maximal cliques (PMCs). Its value is the
# smallest (refined) width of a tree decomposition all of whose bags lie in Π,
# computed by the Bouchitté-Todinca recurrence. Solutions are improved by
# adding PMCs taken from minimal triangulations of local graphs:
#
#   - Diversification (2019, Section 5.1): take a largest bag X₀ of the best
#     decomposition and a random subtree around it, and re-triangulate the
#     local graph on the subtree's vertices with a minimal separator crossing
#     X₀ filled in, so that X₀ cannot reappear.
#   - Merging (2022): pairs of PMCs X ∈ Π, Y ∈ Ω, where Ω is an independently
#     generated solution of no greater width, pick out small "focus" vertex
#     sets U; the local graph on each U is triangulated and its maximal
#     cliques are added to Π ∪ Ω.
#
# By default a solution is diversified, and merged only when diversification
# has stalled. After an improvement, Π is cut down to the PMCs that root a
# decomposition of the new width.
#
# Departures from the papers and the reference implementation:
#
#   - The 2019 paper also describes "connection" strategies, which are not
#     implemented. Its diversified subtrees have unspecified size; here the
#     size is drawn log-uniformly between |X₀| and `dsize`. Diversification
#     also re-triangulates around bags slightly smaller than the largest
#     (see `near`), which helps on instances where it otherwise stalls.
#   - Exact local triangulations ask PIDBT for the target width directly
#     rather than for the optimum, and give up after a time limit that adapts
#     to the progress of the search (see `hbt_cutoff!`). They
#     are used for local graphs of up to max(`base`, |X₀| + `margin`)
#     vertices, and a fraction `pexact` of the diversified regions are drawn
#     from just above the size of X₀. With a fixed limit of 60 (the
#     reference's), no region of a graph of width ~100 is ever solved
#     exactly; yet such a region, slightly larger than one bag, has a nearly
#     complete local graph that PIDBT usually solves in milliseconds, often
#     below the current width. This was the largest single improvement on
#     wide instances. A region that has no decomposition below the current
#     width k has treewidth k, and that bounds the treewidth of each of its
#     diversified variants below, so PIDBT starts at k for them. Merges cap
#     PIDBT at k, since a wider triangulation is of no use there.
#   - The component structure of a PMC (its components, their separators, and
#     the superblocks it caps) is computed once, when the PMC is created, and
#     shared by every solution that contains it; the reference recomputes it
#     on every evaluation. Blocks are identified by (smallest vertex,
#     separator), so no vertex set of size Θ(n) is ever stored. PMCs from a
#     local graph get their structure from the local graph in linear time
#     (with word operations on a bit matrix when the local graph is dense).
#   - The focus for Y is the component of G - Y containing the smallest vertex
#     outside the scope, rather than the first component that leaves the scope.
#   - Greedy triangulations use `alg` (default MF(strategy = 1), minimum
#     average fill) on a random relabeling, followed by MinimalChordal; the
#     reference uses MMAF, a minimal variant of minimum average fill. The
#     choice matters a great deal: with AMF, the widths on wide PACE
#     instances were 10-20% larger. Exact local triangulations use PIDBT.
#   - Widths are measured by bag weight, so integer vertex weights are
#     supported.
#
# All vertex sets are sorted vectors of vertex indices.

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

# Build a simple graph (sorted neighborhoods, no self loops, no parallel
# edges) from `graph`, together with integer vertex weights. HBT works on
# this `BipartiteGraph` directly and assumes its neighborhoods are sorted.
function hbt_graph(weights::AbstractVector, graph::AbstractGraph)
    n = nv(graph)

    # count the non-self-loop arcs: an upper bound on the number of
    # arcs after duplicates are removed
    m = 0

    for v in 1:n, w in neighbors(graph, v)
        w == v || (m += 1)
    end

    ptr = Vector{Int}(undef, n + 1)
    tgt = Vector{Int}(undef, m)
    wgt = Vector{Int}(undef, n)
    ptr[1] = 1

    # `p` is the next free slot in `tgt`
    p = 1

    for v in 1:n
        wgt[v] = trunc(Int, weights[v])
        start = p

        for w in neighbors(graph, v)
            if w != v
                tgt[p] = w; p += 1
            end
        end

        # sort and deduplicate the neighborhood of `v`
        sort!(view(tgt, start:p - 1))
        q = start

        for i in start:p - 1
            if q == start || tgt[i] != tgt[q - 1]
                tgt[q] = tgt[i]; q += 1
            end
        end

        p = q
        ptr[v + 1] = p
    end

    resize!(tgt, p - 1)
    return BipartiteGraph{Int, Int}(n, n, p - 1, ptr, tgt), wgt
end

# ---------------------------------------------------------------------------
# Potential maximal cliques
# ---------------------------------------------------------------------------

# A PMC X together with the components C₁, …, Cₜ of G - X. A *block* is a
# component C whose neighborhood N(C) is a minimal separator; it is
# identified by its smallest vertex and its separator, and stored once in the
# context's block table. For each component Cⱼ we store its block and the
# block of its superblock: the full component of N(Cⱼ) that contains
# X - N(Cⱼ). X is a cap of each superblock. The superblock of Cⱼ consists of
# X - N(Cⱼ) and the components Cᵢ with N(Cᵢ) ⊈ N(Cⱼ); these are the *inner*
# components of j, stored in itgt[iptr[j]:iptr[j + 1] - 1].
struct HBTPMC
    verts::Vector{Int}
    wgt::Int
    csub::Vector{Int}
    csup::Vector{Int}
    iptr::Vector{Int}
    itgt::Vector{Int}
end

@inline hbt_ncomponents(X::HBTPMC) = length(X.csub)

# ---------------------------------------------------------------------------
# Context: the graph, parameters, and scratch space shared by all solutions
# ---------------------------------------------------------------------------

mutable struct HBTContext{A, R <: AbstractRNG}
    const graph::BipartiteGraph{Int, Int, Vector{Int}, Vector{Int}}
    const wgt::Vector{Int}
    const alg::A
    const rng::R
    const base::Int
    const ntry::Int
    const ninit::Int
    const deadline::Float64
    const t0::Float64
    const verbose::Bool
    const refined::Bool
    const merge::Bool
    const diversify::Bool
    const dsize::Int
    const nsep::Int
    const patience::Int
    const xtime::Float64
    const xmin::Float64
    const xstep::Int
    const near::Int
    const margin::Int
    const pexact::Float64

    # interned separators
    const sepindex::Dict{Vector{Int}, Int}
    const seps::Vector{Vector{Int}}
    const sepwgt::Vector{Int}

    # time limit of the exact triangulations in the current step
    cutoff::Float64

    # markers: `mark*[v] == s` for the current stamp s
    stamp::Int
    const mark1::Vector{Int}
    const mark2::Vector{Int}
    const mark3::Vector{Int}
    const loc::Vector{Int}
    const queue::Vector{Int}
    const buf::Vector{Int}
    const bitbuf::Vector{UInt64}

    # blocks, keyed by (smallest vertex, separator)
    const blockindex::Dict{Tuple{Int, Int}, Int}
    const bmin::Vector{Int}
    const bsize::Vector{Int}
    const bwgt::Vector{Int}
    const bsep::Vector{Int}

    # dynamic programming scratch space, indexed by block
    const bstamp::Vector{Int}
    const bloc::Vector{Int}
    const bwidth::Vector{Int}
    const bcapp::Vector{Int}
    const bcapj::Vector{Int}
    const active::Vector{Int}
    const cptr::Vector{Int}
    const capb::Vector{Int}
    const capp::Vector{Int}
    const capj::Vector{Int}
    const capp2::Vector{Int}
    const capj2::Vector{Int}
    const rootval::Vector{Int}
    const cnt::Vector{Int}
    const sorted::Vector{Int}

    # statistics
    nstep::Int
    nlocal::Int
    nexact::Int
    ntimeout::Int

    # contexts for parallel diversification (see `hbt_workers!`)
    const workers::Vector{HBTContext{A, Xoshiro}}
end

function HBTContext(graph::BipartiteGraph{Int, Int, Vector{Int}, Vector{Int}}, wgt::Vector{Int}, alg::A, rng::R, base::Int, ntry::Int, ninit::Int, deadline::Float64;
        t0::Float64=time(), verbose::Bool=false, refined::Bool=true, merge::Bool=true, diversify::Bool=true,
        dsize::Int=typemax(Int), nsep::Int=4, patience::Int=100, xtime::Float64=1.0, xmin::Float64=0.05, xstep::Int=8,
        near::Int=3, margin::Int=40, pexact::Float64=0.5) where {A, R}
    n = nv(graph)

    return HBTContext{A, R}(
        graph, wgt, alg, rng, base, ntry, ninit, deadline, t0, verbose,
        refined, merge, diversify, dsize, nsep, patience, xtime, xmin, xstep, near, margin, pexact,
        Dict{Vector{Int}, Int}(), Vector{Int}[], Int[],
        xtime, 0, zeros(Int, n), zeros(Int, n), zeros(Int, n), zeros(Int, n), Int[], Int[], UInt64[],
        Dict{Tuple{Int, Int}, Int}(), Int[], Int[], Int[], Int[],
        Int[], Int[], Int[], Int[], Int[], Int[],
        Int[], Int[], Int[], Int[], Int[], Int[], Int[], Int[], Int[],
        0, 0, 0, 0, HBTContext{A, Xoshiro}[])
end

# Give the context `nworkers` workers: contexts that share its graph and
# parameters, but have their own scratch space and random number generator
# (seeded from the context's), so that they can diversify a solution in
# parallel (see `hbt_parallel!`).
function hbt_workers!(ctx::HBTContext{A}, nworkers::Int) where {A}
    for _ in 1:nworkers
        push!(ctx.workers, HBTContext(ctx.graph, ctx.wgt, ctx.alg, Xoshiro(rand(ctx.rng, UInt64)), ctx.base, ctx.ntry, ctx.ninit, ctx.deadline;
            t0 = ctx.t0, refined = ctx.refined, merge = ctx.merge, diversify = ctx.diversify, dsize = ctx.dsize, nsep = ctx.nsep,
            patience = ctx.patience, xtime = ctx.xtime, xmin = ctx.xmin, xstep = ctx.xstep, near = ctx.near, margin = ctx.margin,
            pexact = ctx.pexact))
    end

    return ctx
end

# statistics summed over the context and its workers
function hbt_stats(ctx::HBTContext)
    nstep = ctx.nstep; nlocal = ctx.nlocal; nexact = ctx.nexact; ntimeout = ctx.ntimeout

    for w in ctx.workers
        nstep += w.nstep; nlocal += w.nlocal; nexact += w.nexact; ntimeout += w.ntimeout
    end

    return nstep, nlocal, nexact, ntimeout
end

@inline function hbt_stamp!(ctx::HBTContext)
    return ctx.stamp += 1
end

@inline hbt_expired(ctx::HBTContext) = time() > ctx.deadline

# The time limit of the exact triangulations made while improving a solution
# whose largest bag has not shrunk for `stall` steps: `xmin`, doubled every
# `xstep` steps, up to `xtime`. Most exact triangulations that succeed take a
# few milliseconds, but some take much longer; these are worth the wait only
# once the cheap improvements have run out. (Steps that only reduce the number
# of largest bags do not reset the count: there can be many of them.)
function hbt_cutoff!(ctx::HBTContext, stall::Int)
    e = stall ÷ ctx.xstep
    ctx.cutoff = e >= 30 ? ctx.xtime : min(ctx.xtime, ctx.xmin * 2.0^e)
    return
end

function hbt_intern_sep!(ctx::HBTContext, sep::AbstractVector{Int})
    id = get(ctx.sepindex, sep, 0)

    if iszero(id)
        key = collect(sep)
        push!(ctx.seps, key)
        w = 0

        for v in key
            w += ctx.wgt[v]
        end

        push!(ctx.sepwgt, w)
        id = ctx.sepindex[key] = length(ctx.seps)
    end

    return id
end

# Fill in the superblocks and inner components of a PMC, given its
# components, and register its blocks.
function hbt_pmc(ctx::HBTContext, X::Vector{Int}, cmin::Vector{Int}, csize::Vector{Int}, cwgt::Vector{Int}, csep::Vector{Int})
    g = ctx.graph; mark = ctx.mark3; t = length(cmin)
    csub = Vector{Int}(undef, t)
    csup = Vector{Int}(undef, t)
    iptr = Vector{Int}(undef, t + 1)
    itgt = Int[]
    xw = 0

    for x in X
        xw += ctx.wgt[x]
    end

    for j in 1:t
        s = hbt_stamp!(ctx)

        for v in ctx.seps[csep[j]]
            mark[v] = s
        end

        m = typemax(Int); sz = 0; w = 0

        for x in X
            if mark[x] != s
                m = min(m, x); sz += 1; w += ctx.wgt[x]
            end
        end

        iptr[j] = length(itgt) + 1

        for i in 1:t
            i == j && continue
            csep[i] == csep[j] && continue
            inner = false

            for v in ctx.seps[csep[i]]
                if mark[v] != s
                    inner = true
                    break
                end
            end

            if inner
                push!(itgt, i)
                m = min(m, cmin[i]); sz += csize[i]; w += cwgt[i]
            end
        end

        csub[j] = hbt_block!(ctx, cmin[j], csep[j], csize[j], cwgt[j])
        csup[j] = hbt_block!(ctx, m, csep[j], sz, w)
    end

    iptr[t + 1] = length(itgt) + 1
    return HBTPMC(X, xw, csub, csup, iptr, itgt)
end

function hbt_block!(ctx::HBTContext, m::Int, sep::Int, size::Int, wgt::Int)
    key = (m, sep)
    b = get(ctx.blockindex, key, 0)

    if iszero(b)
        push!(ctx.bmin, m); push!(ctx.bsize, size); push!(ctx.bwgt, wgt); push!(ctx.bsep, sep)
        b = ctx.blockindex[key] = length(ctx.bmin)
    end

    return b
end

# Construct a PMC by searching the components of G - X. O(n + m).
function hbt_pmc(ctx::HBTContext, X::Vector{Int})
    g = ctx.graph; n = nv(g)
    mark1 = ctx.mark1; mark2 = ctx.mark2; mark3 = ctx.mark3; queue = ctx.queue; buf = ctx.buf
    sx = hbt_stamp!(ctx)

    for x in X
        mark1[x] = sx
    end

    cmin = Int[]; csize = Int[]; cwgt = Int[]; csep = Int[]
    sc = hbt_stamp!(ctx)

    for v0 in 1:n
        (mark1[v0] == sx || mark2[v0] == sc) && continue
        sn = hbt_stamp!(ctx)
        empty!(buf); empty!(queue)
        push!(queue, v0); mark2[v0] = sc
        sz = 0; w = 0; head = 1

        while head <= length(queue)
            u = queue[head]; head += 1
            sz += 1; w += ctx.wgt[u]

            for x in neighbors(g, u)
                if mark1[x] == sx
                    if mark3[x] != sn
                        mark3[x] = sn; push!(buf, x)
                    end
                elseif mark2[x] != sc
                    mark2[x] = sc; push!(queue, x)
                end
            end
        end

        sort!(buf)
        push!(cmin, v0); push!(csize, sz); push!(cwgt, w)
        push!(csep, hbt_intern_sep!(ctx, buf))
    end

    return hbt_pmc(ctx, X, cmin, csize, cwgt, csep)
end

# The maximal cliques of a minimal triangulation. `label` maps the clique
# tree's vertex labels to vertices of G.
#
# (The clique tree does not determine the components of G - X directly: one
# tree edge can carry several components, e.g. a star whose clique tree is a
# path. So each component structure is found by search.)
function hbt_pmcs(ctx::HBTContext, label::AbstractVector, tree::CliqueTree)
    pmcs = HBTPMC[]

    for clique in tree
        X = Int[label[v] for v in clique]
        sort!(X)
        push!(pmcs, hbt_pmc(ctx, X))
    end

    return pmcs
end

# ---------------------------------------------------------------------------
# Solutions
# ---------------------------------------------------------------------------

mutable struct HBTState
    const pmcs::Vector{HBTPMC}
    const index::Dict{Vector{Int}, Int}
    width::Int                      # refined width of the best decomposition
    bags::Vector{Vector{Int}}       # the best decomposition: its bags...
    parent::Vector{Int}             # ... and their parents
    order::Vector{Int}              # elimination order of the best decomposition
    maxbag::Int                     # number of vertices in its largest bag
    side::Union{Nothing, HBTState}
    const depth::Int
    stall::Int                      # steps since the last improvement
    kstall::Int                     # steps since the last improvement of the largest bag
end

# What a diversification reads from a solution: its best decomposition and
# how long it has been stuck. Workers diversify a snapshot while the solution
# changes (see `hbt_parallel!`).
struct HBTSnapshot
    bags::Vector{Vector{Int}}
    parent::Vector{Int}
    width::Int
    kstall::Int
end

HBTSnapshot(st::HBTState) = HBTSnapshot(st.bags, st.parent, st.width, st.kstall)

function hbt_add!(st::HBTState, X::HBTPMC)
    if !haskey(st.index, X.verts)
        push!(st.pmcs, X)
        st.index[X.verts] = length(st.pmcs)
    end

    return
end

# ---------------------------------------------------------------------------
# Refined widths
#
# The width of a decomposition is the pair (k, f), where k is the largest bag
# weight and f is the number of bags of weight k, ordered lexicographically
# (Tamaki 2019, Section 2). Lowering f is progress even when k stays put. A
# pair is packed into one integer as k << 32 | f.
# ---------------------------------------------------------------------------

@inline hbt_bag(k::Int) = (k << 32) | 1
@inline hbt_k(w::Int) = w >> 32
@inline hbt_f(w::Int) = w & 0xffffffff

@inline function hbt_plus(a::Int, b::Int)
    ka = a >> 32; kb = b >> 32
    return ka == kb ? a + (b & 0xffffffff) : max(a, b)
end

hbt_string(w::Int) = "$(hbt_k(w) - 1) ($(hbt_f(w)))"

# ---------------------------------------------------------------------------
# BT dynamic programming over a set of PMCs
# ---------------------------------------------------------------------------

# Evaluate the solution: compute the width of every block and the root value
# of every PMC. Returns (width, root). The block data stay valid until the
# next call.
function hbt_dp!(ctx::HBTContext, st::HBTState)
    P = st.pmcs; np = length(P); nb = length(ctx.bmin)
    bstamp = ctx.bstamp; bloc = ctx.bloc; active = ctx.active

    if length(bstamp) < nb
        m = max(nb, 2length(bstamp))
        resize!(bstamp, m); fill!(bstamp, 0)
        resize!(bloc, m); resize!(ctx.bwidth, m); resize!(ctx.bcapp, m); resize!(ctx.bcapj, m)
    end

    s = hbt_stamp!(ctx)
    empty!(active); empty!(ctx.capb); empty!(ctx.capp); empty!(ctx.capj)

    @inbounds for p in 1:np
        X = P[p]

        for j in 1:hbt_ncomponents(X)
            b = X.csub[j]

            if bstamp[b] != s
                bstamp[b] = s; push!(active, b)
            end

            b = X.csup[j]

            if bstamp[b] != s
                bstamp[b] = s; push!(active, b)
            end

            push!(ctx.capb, b); push!(ctx.capp, p); push!(ctx.capj, j)
        end
    end

    # evaluate the blocks in order of increasing size (a stable counting sort:
    # sizes are at most n)
    bsize = ctx.bsize; na = length(active); n = nv(ctx.graph)
    cnt = ctx.cnt; resize!(cnt, n + 1); fill!(cnt, 0)
    sorted = ctx.sorted; resize!(sorted, na)

    @inbounds for b in active
        cnt[bsize[b] + 1] += 1
    end

    @inbounds for i in 2:n + 1
        cnt[i] += cnt[i - 1]
    end

    # now cnt[s] is the number of blocks smaller than s
    @inbounds for b in active
        sz = bsize[b]; cnt[sz] += 1; sorted[cnt[sz]] = b
    end

    copyto!(active, sorted)

    @inbounds for (i, b) in enumerate(active)
        bloc[b] = i
    end

    # bucket the caps by block
    cptr = ctx.cptr; resize!(cptr, na + 2); fill!(cptr, 0)

    @inbounds for b in ctx.capb
        cptr[bloc[b] + 2] += 1
    end

    cptr[1] = 1; cptr[2] = 1

    @inbounds for i in 2:na + 1
        cptr[i + 1] += cptr[i]
    end

    nc = length(ctx.capb)
    resize!(ctx.capp2, nc); resize!(ctx.capj2, nc)

    @inbounds for c in 1:nc
        i = bloc[ctx.capb[c]] + 1
        k = cptr[i]; cptr[i] += 1
        ctx.capp2[k] = ctx.capp[c]; ctx.capj2[k] = ctx.capj[c]
    end

    # now cptr[i]:cptr[i + 1] - 1 are the caps of active[i]
    @inbounds for (i, b) in enumerate(active)
        w = hbt_bag(ctx.bwgt[b] + ctx.sepwgt[ctx.bsep[b]])
        bp = 0; bj = 0

        for c in cptr[i]:cptr[i + 1] - 1
            p = ctx.capp2[c]; j = ctx.capj2[c]; X = P[p]
            v = hbt_bag(X.wgt)
            v < w || continue

            for q in X.iptr[j]:X.iptr[j + 1] - 1
                v = hbt_plus(v, ctx.bwidth[X.csub[X.itgt[q]]])
                v < w || break
            end

            if v < w
                w = v; bp = p; bj = j
            end
        end

        ctx.bwidth[b] = w; ctx.bcapp[b] = bp; ctx.bcapj[b] = bj
    end

    # root values
    rootval = ctx.rootval; resize!(rootval, np)
    width = typemax(Int); root = 0

    @inbounds for p in 1:np
        X = P[p]; v = hbt_bag(X.wgt)

        for j in 1:hbt_ncomponents(X)
            v = hbt_plus(v, ctx.bwidth[X.csub[j]])
        end

        rootval[p] = v

        if v < width
            width = v; root = p
        end
    end

    return width, root
end

# Read a tree decomposition off the DP of `hbt_dp!`. Returns the bags and
# the parent of each bag (0 for the root).
function hbt_td(ctx::HBTContext, st::HBTState, root::Int)
    g = ctx.graph; P = st.pmcs
    mark1 = ctx.mark1; mark2 = ctx.mark2; queue = ctx.queue
    bags = Vector{Int}[P[root].verts]; parent = Int[0]
    stack = Tuple{Int, Int}[]

    for b in P[root].csub
        push!(stack, (b, 1))
    end

    while !isempty(stack)
        b, pn = pop!(stack)
        p = ctx.bcapp[b]

        if iszero(p)                    # no cap: the bag N[C]
            s = hbt_stamp!(ctx)
            sep = ctx.seps[ctx.bsep[b]]

            for v in sep
                mark1[v] = s
            end

            v0 = ctx.bmin[b]
            empty!(queue); push!(queue, v0); mark2[v0] = s; head = 1

            while head <= length(queue)
                u = queue[head]; head += 1

                for x in neighbors(g, u)
                    if mark1[x] != s && mark2[x] != s
                        mark2[x] = s; push!(queue, x)
                    end
                end
            end

            bag = sort!(vcat(queue, sep))
            push!(bags, bag); push!(parent, pn)
        else
            push!(bags, P[p].verts); push!(parent, pn)
            node = length(bags)
            X = P[p]; j = ctx.bcapj[b]

            for q in X.iptr[j]:X.iptr[j + 1] - 1
                push!(stack, (X.csub[X.itgt[q]], node))
            end
        end
    end

    return bags, parent
end

# An elimination order whose width is that of a tree decomposition: visit
# the bags children-first and eliminate the vertices of each bag that are not
# in its parent. Bags are listed parents-first.
function hbt_tdorder(ctx::HBTContext, bags::Vector{Vector{Int}}, parent::Vector{Int})
    n = nv(ctx.graph); mark = ctx.mark1
    order = Int[]; sizehint!(order, n)

    for i in length(bags):-1:1
        p = parent[i]

        if iszero(p)
            append!(order, bags[i])
        else
            s = hbt_stamp!(ctx)

            for v in bags[p]
                mark[v] = s
            end

            for v in bags[i]
                mark[v] == s || push!(order, v)
            end
        end
    end

    @assert length(order) == n
    return order
end

function hbt_record!(ctx::HBTContext, st::HBTState, width::Int, root::Int)
    st.width = width
    st.bags, st.parent = hbt_td(ctx, st, root)
    st.order = hbt_tdorder(ctx, st.bags, st.parent)
    st.maxbag = maximum(length, st.bags)
    return
end

# Evaluate a solution and, if it improved, record the decomposition and drop
# every PMC that is not the root of a decomposition of the new width.
function hbt_update!(ctx::HBTContext, st::HBTState)
    width, root = hbt_dp!(ctx, st)

    # refined widths: progress is any decrease of (k, f); otherwise only a
    # decrease of k counts
    better = ctx.refined ? width < st.width : hbt_k(width) < hbt_k(st.width)

    if better
        report = ctx.verbose && iszero(st.depth) && hbt_k(width) < hbt_k(st.width)
        hbt_record!(ctx, st, width, root)
        hbt_filter!(ctx, st, ctx.refined ? width : (hbt_k(width) << 32) | 0xffffffff)

        if report
            println("hbt: width $(hbt_k(width) - 1) at $(round(time() - ctx.t0; digits = 2)) s (steps = $(hbt_stats(ctx)[1]), pmcs = $(length(st.pmcs)))")
        end

        return true
    end

    return false
end

function hbt_filter!(ctx::HBTContext, st::HBTState, width::Int)
    rootval = ctx.rootval; P = st.pmcs; k = 0

    for p in eachindex(P)
        if rootval[p] <= width
            k += 1; P[k] = P[p]
        end
    end

    resize!(P, k)
    empty!(st.index)

    for p in 1:k
        st.index[P[p].verts] = p
    end

    return
end

# ---------------------------------------------------------------------------
# Greedy minimal triangulations
# ---------------------------------------------------------------------------

# The largest bag weight of a clique tree whose vertices are labeled by
# `label`.
function hbt_treewidth(weights::AbstractVector{Int}, label::AbstractVector, tree::CliqueTree)
    width = 0

    for clique in tree
        w = 0

        for v in clique
            w += weights[label[v]]
        end

        width = max(width, w)
    end

    return width
end

# Run `alg` on a random relabeling of a graph (greedy algorithms break ties
# by label), then make the ordering minimal.
function hbt_greedy(ctx::HBTContext, weights::AbstractVector{Int}, graph::BipartiteGraph)
    n = nv(graph)
    perm = randperm(ctx.rng, n)         # perm[new] = old
    invp = invperm(perm)
    ptr = Vector{Int}(undef, n + 1); tgt = Vector{Int}(undef, de(graph))
    ptr[1] = 1

    for i in 1:n
        k = ptr[i]

        for w in neighbors(graph, perm[i])
            tgt[k] = invp[w]; k += 1
        end

        sort!(view(tgt, ptr[i]:k - 1))
        ptr[i + 1] = k
    end

    pgraph = BipartiteGraph{Int, Int}(n, n, ptr[n + 1] - 1, ptr, tgt)
    order, _ = permutation(weights[perm], pgraph, ctx.alg)
    order = perm[order]
    order, _ = permutation(weights, graph, MinimalChordal(order))
    return order
end

hbt_greedy(ctx::HBTContext) = hbt_greedy(ctx, ctx.wgt, ctx.graph)

function hbt_state(ctx::HBTContext, depth::Int)
    g = ctx.graph
    best = typemax(Int); bestlabel = nothing; besttree = nothing

    for i in 1:ctx.ninit
        (i > 1 && hbt_expired(ctx)) && break
        order = hbt_greedy(ctx)
        label, tree = cliquetree(ctx.wgt, ctx.graph, order)
        width = hbt_treewidth(ctx.wgt, label, tree)

        if width < best
            best = width; bestlabel = label; besttree = tree
        end
    end

    st = HBTState(HBTPMC[], Dict{Vector{Int}, Int}(), typemax(Int), Vector{Int}[], Int[], Int[], 0, nothing, depth, 0, 0)

    for X in hbt_pmcs(ctx, bestlabel, besttree)
        hbt_add!(st, X)
    end

    width, root = hbt_dp!(ctx, st)
    @assert hbt_k(width) <= best
    hbt_record!(ctx, st, width, root)
    return st
end

# Triangulate a (connected) local graph with PIDBT, giving up after the time
# limit of the current step (see `hbt_cutoff!`). Returns a minimal ordering,
# or `nothing` if the time runs out.
#
# PIDBT tries the widths k = k₀, k₀ + 1, ... in turn, and every width below the
# treewidth costs an exhaustive search. A triangulation is useful only if its
# bags weigh at most `target`, so k₀ is chosen to match: the result is optimal
# if no triangulation meets the target, and meets the target otherwise. `lb`
# is a known lower bound on the bag weight. If the caller has no use for bags
# heavier than `cap`, then no width above `cap` is tried, and the result is
# `missing` if the graph has no triangulation whose bags weigh at most `cap`.
function hbt_exact(ctx::HBTContext, weights::Vector{Int}, H::BipartiteGraph, target::Int; lb::Int=0, cap::Int=typemax(Int))
    start = max(target, lb, lowerbound(weights, H, DEFAULT_LOWER_BOUND_ALGORITHM))
    start > cap && return missing
    deadline = min(ctx.deadline, time() + ctx.cutoff)
    order = nothing

    try
        order = pidbt(weights, H, start; deadline, max_k = cap)
    catch err
        err isa ArgumentError || rethrow()    # disconnected
    end

    ismissing(order) && return missing

    if isnothing(order)
        ctx.ntimeout += 1
        return nothing
    end

    ctx.nexact += 1
    order, _ = permutation(weights, H, MinimalChordal(order))
    return order
end

# Exact if the local graph has at most `limit` vertices and PIDBT finishes
# in time, greedy otherwise. `missing` if the exact triangulation exceeds
# `cap` (see `hbt_exact`).
function hbt_triangulate(ctx::HBTContext, weights::Vector{Int}, H::BipartiteGraph, target::Int, limit::Int=ctx.base; lb::Int=0, cap::Int=typemax(Int))
    order = nv(H) <= limit ? hbt_exact(ctx, weights, H, target; lb, cap) : nothing
    ismissing(order) && return missing
    return isnothing(order) ? hbt_greedy(ctx, weights, H) : order
end

# The size limit for exact triangulations of a region around bags with up to
# `size` vertices. A local graph only slightly larger than one bag is nearly
# complete, and PIDBT solves it quickly even when the bag is wide.
@inline hbt_limit(ctx::HBTContext, size::Int) = max(ctx.base, size + ctx.margin)

# ---------------------------------------------------------------------------
# Local graphs
# ---------------------------------------------------------------------------

# The local graph on U: the graph obtained from G[U] by making N(B) a clique
# for every component B of G - U. Its vertices are numbered by their position
# in U. For each component B we keep its smallest vertex, size, weight, and
# neighborhood (in local numbering).
#
# Local graphs are usually dense (the neighborhoods N(B) are separators of
# the size of a bag), so the graph is also kept as a bit matrix when that is
# no larger than its adjacency lists: row i, in words (i - 1)W + 1 : iW,
# is the neighborhood of vertex i. The components of H - K are then found
# with word operations (see `hbt_pmc`). Otherwise `W` is zero.
struct HBTLocal
    U::Vector{Int}
    graph::BipartiteGraph{Int, Int, Vector{Int}, Vector{Int}}
    wgt::Vector{Int}
    bmin::Vector{Int}
    bsize::Vector{Int}
    bwgt::Vector{Int}
    bnbr::Vector{Vector{Int}}
    nbrs::Set{Vector{Int}}
    W::Int
    bits::Vector{UInt64}
end

# Local graphs with more vertices than this are built from adjacency lists.
const HBT_MAX_BITS = 1024

@inline hbt_setbit!(bits::Vector{UInt64}, offset::Int, i::Int) =
    @inbounds bits[offset + ((i - 1) >> 6) + 1] |= one(UInt64) << ((i - 1) & 63)

function hbt_local(ctx::HBTContext, U::Vector{Int})
    g = ctx.graph; n = nv(g); nu = length(U)
    mark1 = ctx.mark1; mark2 = ctx.mark2; mark3 = ctx.mark3; loc = ctx.loc
    queue = ctx.queue; buf = ctx.buf
    su = hbt_stamp!(ctx)

    for (i, u) in enumerate(U)
        mark1[u] = su; loc[u] = i
    end

    usebits = nu <= HBT_MAX_BITS
    W = usebits ? cld(nu, 64) : 0
    bits = usebits ? zeros(UInt64, W * nu) : UInt64[]
    adj = usebits ? Vector{Int}[] : [Int[] for _ in 1:nu]

    @inbounds for (i, u) in enumerate(U)
        for v in neighbors(g, u)
            if mark1[v] == su
                if usebits
                    hbt_setbit!(bits, (i - 1) * W, loc[v])
                else
                    push!(adj[i], loc[v])
                end
            end
        end
    end

    bmin = Int[]; bsize = Int[]; bwgt = Int[]; bnbr = Vector{Int}[]
    nbrs = Set{Vector{Int}}()
    row = zeros(UInt64, W)
    sc = hbt_stamp!(ctx)

    @inbounds for v0 in 1:n
        (mark1[v0] == su || mark2[v0] == sc) && continue
        sn = hbt_stamp!(ctx)
        empty!(buf); empty!(queue)
        push!(queue, v0); mark2[v0] = sc; head = 1
        sz = 0; w = 0

        while head <= length(queue)
            u = queue[head]; head += 1
            sz += 1; w += ctx.wgt[u]

            for x in neighbors(g, u)
                if mark1[x] == su
                    if mark3[x] != sn
                        mark3[x] = sn; push!(buf, loc[x])
                    end
                elseif mark2[x] != sc
                    mark2[x] = sc; push!(queue, x)
                end
            end
        end

        sort!(buf)
        nb = copy(buf)
        push!(bmin, v0); push!(bsize, sz); push!(bwgt, w); push!(bnbr, nb)

        if nb ∉ nbrs
            push!(nbrs, nb)

            if usebits
                # make nb a clique: OR its bit set into the rows of its vertices
                fill!(row, zero(UInt64))

                for a in nb
                    hbt_setbit!(row, 0, a)
                end

                for a in nb
                    offset = (a - 1) * W

                    for j in 1:W
                        bits[offset + j] |= row[j]
                    end
                end
            else
                for a in nb, b in nb
                    a == b || push!(adj[a], b)
                end
            end
        end
    end

    ptr = Vector{Int}(undef, nu + 1); tgt = Int[]; ptr[1] = 1

    if usebits
        @inbounds for i in 1:nu
            offset = (i - 1) * W

            for j in 1:W
                word = bits[offset + j]

                while !iszero(word)
                    v = ((j - 1) << 6) + trailing_zeros(word) + 1
                    v == i || push!(tgt, v)
                    word &= word - one(UInt64)
                end
            end

            ptr[i + 1] = length(tgt) + 1
        end

        # (the diagonal may have been set by the cliques)
        @inbounds for i in 1:nu
            bits[(i - 1) * W + ((i - 1) >> 6) + 1] &= ~(one(UInt64) << ((i - 1) & 63))
        end

        # keep the bit matrix only if the BFS over it is no slower than over
        # the adjacency lists: W words per vertex against its degree
        if W * nu > length(tgt)
            W = 0; bits = UInt64[]
        end
    else
        for i in 1:nu
            list = adj[i]; sort!(list); unique!(list)
            append!(tgt, list)
            ptr[i + 1] = length(tgt) + 1
        end
    end

    H = BipartiteGraph{Int, Int}(nu, nu, length(tgt), ptr, tgt)
    return HBTLocal(U, H, ctx.wgt[U], bmin, bsize, bwgt, bnbr, nbrs, W, bits)
end

# Construct the PMC U[K] of G from a PMC K of the local graph. The components
# of G - U[K] are the components Q of H - K, each joined with the components
# B of G - U with ∅ ≠ N(B) - K ⊆ Q, together with the components B of G - U
# with N(B) ⊆ K. The separator of the former is N_H(Q) ∩ K. Linear in the
# size of the local graph.
function hbt_pmc(ctx::HBTContext, L::HBTLocal, K::Vector{Int}, lmark::Vector{Int}, lcomp::Vector{Int})
    U = L.U; buf = ctx.buf
    sk = hbt_stamp!(ctx)

    for k in K
        lmark[k] = sk
    end

    cmin = Int[]; csize = Int[]; cwgt = Int[]; csep = Int[]
    seps = Vector{Int}[]

    if L.W > 0
        hbt_components_bits!(ctx, L, K, lcomp, cmin, csize, cwgt, seps)
    else
        hbt_components_lists!(ctx, L, sk, lmark, lcomp, cmin, csize, cwgt, seps)
    end

    # attach the components of G - U
    for b in eachindex(L.bmin)
        nb = L.bnbr[b]
        q = 0

        for v in nb
            if lmark[v] != sk
                q = lcomp[v]
                break
            end
        end

        if iszero(q)
            push!(cmin, L.bmin[b]); push!(csize, L.bsize[b]); push!(cwgt, L.bwgt[b])
            push!(seps, nb)
        else
            cmin[q] = min(cmin[q], L.bmin[b]); csize[q] += L.bsize[b]; cwgt[q] += L.bwgt[b]
        end
    end

    for sep in seps
        empty!(buf)

        for v in sep
            push!(buf, U[v])
        end

        push!(csep, hbt_intern_sep!(ctx, buf))
    end

    return hbt_pmc(ctx, U[K], cmin, csize, cwgt, csep)
end

# Add the PMC U[K] of G, for a PMC K of the local graph, to a solution,
# unless it is there already. It is taken from the solution `side` if it is
# there; otherwise it is built.
function hbt_addpmc!(ctx::HBTContext, st::HBTState, L::HBTLocal, K::Vector{Int}, lmark::Vector{Int}, lcomp::Vector{Int}, side::Union{Nothing, HBTState}=nothing)
    X = L.U[K]
    haskey(st.index, X) && return
    j = isnothing(side) ? 0 : get(side.index, X, 0)
    hbt_add!(st, iszero(j) ? hbt_pmc(ctx, L, K, lmark, lcomp) : side.pmcs[j])
    return
end

# The components Q of H - K, by search over the adjacency lists: their
# smallest vertices (in G), sizes, weights, and separators N_H(Q) ∩ K (in local
# numbering). Sets lcomp[v] to the component of every v ∉ K. The vertices of K
# are marked with `sk` in `lmark`.
function hbt_components_lists!(ctx::HBTContext, L::HBTLocal, sk::Int, lmark::Vector{Int}, lcomp::Vector{Int}, cmin::Vector{Int}, csize::Vector{Int}, cwgt::Vector{Int}, seps::Vector{Vector{Int}})
    U = L.U; H = L.graph; nu = length(U); wgt = L.wgt
    queue = ctx.queue; mark3 = ctx.mark3
    sc = hbt_stamp!(ctx)

    # components of H - K; lcomp[v] = component index (valid when lmark[v] == sc)
    for v0 in 1:nu
        (lmark[v0] == sk || lmark[v0] == sc) && continue
        q = length(cmin) + 1
        empty!(queue); push!(queue, v0); lmark[v0] = sc; lcomp[v0] = q; head = 1
        sep = Int[]; m = U[v0]; sz = 0; w = 0
        ss = hbt_stamp!(ctx)    # marks the separator, in mark3 (indexed in G)

        while head <= length(queue)
            u = queue[head]; head += 1
            sz += 1; w += wgt[u]

            for x in neighbors(H, u)
                if lmark[x] == sk
                    if mark3[U[x]] != ss
                        mark3[U[x]] = ss; push!(sep, x)
                    end
                elseif lmark[x] != sc
                    lmark[x] = sc; lcomp[x] = q; push!(queue, x)
                end
            end
        end

        sort!(sep)
        push!(cmin, m); push!(csize, sz); push!(cwgt, w); push!(seps, sep)
    end

    return
end

# The same, by search over the bit matrix of the local graph: a component is
# grown a layer at a time, the next layer being the neighbors of the current
# one (the OR of their rows) that are neither in K nor seen. Its separator is
# the OR of the rows of all its vertices, restricted to K.
function hbt_components_bits!(ctx::HBTContext, L::HBTLocal, K::Vector{Int}, lcomp::Vector{Int}, cmin::Vector{Int}, csize::Vector{Int}, cwgt::Vector{Int}, seps::Vector{Vector{Int}})
    U = L.U; nu = length(U); wgt = L.wgt; W = L.W; bits = L.bits
    scratch = ctx.bitbuf
    length(scratch) < 4W && resize!(scratch, 4W)

    # scratch[0W + j]: K; [1W + j]: vertices not yet in a component;
    # [2W + j]: the current layer; [3W + j]: the OR of the rows of the component
    @inbounds begin
        for j in 1:W
            scratch[j] = zero(UInt64)
            scratch[W + j] = typemax(UInt64)
        end

        r = nu & 63
        iszero(r) || (scratch[2W] = (one(UInt64) << r) - one(UInt64))

        for k in K
            hbt_setbit!(scratch, 0, k)
        end

        for j in 1:W
            scratch[W + j] &= ~scratch[j]
        end

        j0 = 1

        while true
            # the smallest vertex not yet in a component
            while j0 <= W && iszero(scratch[W + j0])
                j0 += 1
            end

            j0 > W && break
            v0 = ((j0 - 1) << 6) + trailing_zeros(scratch[W + j0]) + 1
            q = length(cmin) + 1; sz = 0; w = 0

            for j in 1:W
                scratch[2W + j] = zero(UInt64); scratch[3W + j] = zero(UInt64)
            end

            hbt_setbit!(scratch, 2W, v0)
            scratch[W + j0] &= ~(one(UInt64) << ((v0 - 1) & 63))
            grow = true

            while grow
                # visit the layer
                for j in 1:W
                    word = scratch[2W + j]

                    while !iszero(word)
                        v = ((j - 1) << 6) + trailing_zeros(word) + 1
                        word &= word - one(UInt64)
                        lcomp[v] = q; sz += 1; w += wgt[v]
                        offset = (v - 1) * W

                        for i in 1:W
                            scratch[3W + i] |= bits[offset + i]
                        end
                    end
                end

                # the next layer: unseen neighbors outside K
                grow = false

                for j in 1:W
                    word = scratch[3W + j] & scratch[W + j]
                    scratch[2W + j] = word
                    scratch[W + j] &= ~word
                    grow |= !iszero(word)
                end
            end

            sep = Int[]

            for j in 1:W
                word = scratch[3W + j] & scratch[j]

                while !iszero(word)
                    push!(sep, ((j - 1) << 6) + trailing_zeros(word) + 1)
                    word &= word - one(UInt64)
                end
            end

            # U is sorted, so the smallest local vertex is the smallest in G
            push!(cmin, U[v0]); push!(csize, sz); push!(cwgt, w); push!(seps, sep)
        end
    end

    return
end

# Triangulate the local graph on U and add the resulting PMCs of G to `st` if
# the triangulation is no wider than the current solution.
function hbt_local!(ctx::HBTContext, st::HBTState, side::HBTState, U::Vector{Int})
    L = hbt_local(ctx, U)
    H = L.graph; weights = L.wgt; nu = length(U)
    ctx.nlocal += 1

    # a triangulation is used only if no bag is heavier than the current width
    k = hbt_k(st.width)
    order = hbt_triangulate(ctx, weights, H, k, hbt_limit(ctx, st.maxbag); cap = k)
    ismissing(order) && return

    label, tree = cliquetree(weights, H, order)
    hbt_treewidth(weights, label, tree) <= hbt_k(st.width) || return

    # Every PMC of the local graph is a PMC of G, unless it is the
    # neighborhood of a component of G - U.
    lmark = zeros(Int, nu); lcomp = zeros(Int, nu)

    for clique in tree
        K = Int[label[v] for v in clique]
        sort!(K)
        K in L.nbrs && continue
        hbt_addpmc!(ctx, st, L, K, lmark, lcomp, side)
    end

    return
end

# ---------------------------------------------------------------------------
# Merge and improve
# ---------------------------------------------------------------------------

function hbt_merge!(ctx::HBTContext, st::HBTState, side::HBTState)
    hbt_cutoff!(ctx, st.kstall)
    g = ctx.graph; n = nv(g)
    mark1 = ctx.mark1; mark2 = ctx.mark2; mark3 = ctx.mark3; queue = ctx.queue
    ctx.nstep += 1

    # scope: a random PMC X and the largest component C of G - X
    X = rand(ctx.rng, st.pmcs)
    focuses = Set{Vector{Int}}()

    if hbt_ncomponents(X) > 0
        j = argmax(j -> ctx.bsize[X.csub[j]], 1:hbt_ncomponents(X))
        ss = hbt_stamp!(ctx)

        for x in X.verts
            mark1[x] = ss
        end

        v0 = ctx.bmin[X.csub[j]]
        empty!(queue); push!(queue, v0); mark1[v0] = ss; head = 1

        while head <= length(queue)
            u = queue[head]; head += 1

            for x in neighbors(g, u)
                if mark1[x] != ss
                    mark1[x] = ss; push!(queue, x)
                end
            end
        end

        # smallest vertex outside the scope
        o = 0

        for v in 1:n
            if mark1[v] != ss
                o = v
                break
            end
        end

        if !iszero(o)
            for Y in side.pmcs
                Y.wgt < hbt_k(st.width) || continue
                all(y -> mark1[y] == ss, Y.verts) || continue

                # focus: N[D] ∩ scope for the component D of G - Y containing o
                sy = hbt_stamp!(ctx)

                for y in Y.verts
                    mark2[y] = sy
                end

                U = Int[]
                empty!(queue); push!(queue, o); mark3[o] = sy; head = 1

                while head <= length(queue)
                    u = queue[head]; head += 1
                    mark1[u] == ss && push!(U, u)

                    for x in neighbors(g, u)
                        if mark2[x] == sy
                            if mark3[x] != sy
                                mark3[x] = sy; push!(U, x)
                            end
                        elseif mark3[x] != sy
                            mark3[x] = sy; push!(queue, x)
                        end
                    end
                end

                sort!(U)
                push!(focuses, U)
            end
        end
    end

    list = collect(focuses)
    sort!(list; by = length)

    for i in 1:min(length(list), ctx.ntry)
        hbt_expired(ctx) && break
        hbt_local!(ctx, st, side, list[i])
    end

    for Y in side.pmcs
        hbt_add!(st, Y)
    end

    return hbt_update!(ctx, st)
end

# Tamaki (2022): merge with an independent solution of no greater width,
# improving that solution first if necessary (recursively, with the same
# strategies).
function hbt_improve_merge!(ctx::HBTContext, st::HBTState)
    if isnothing(st.side)
        st.side = hbt_state(ctx, st.depth + 1)
    end

    side = st.side

    if hbt_k(side.width) > hbt_k(st.width)
        hbt_improve!(ctx, side)
        return false
    else
        return hbt_merge!(ctx, st, side)
    end
end

# One improvement step. With both strategies enabled, a solution is improved
# by diversification, and merged with an independent solution only once it
# has gone `patience` steps without progress.
function hbt_improve!(ctx::HBTContext, st::HBTState)
    k = hbt_k(st.width)

    if ctx.diversify
        hbt_diversify!(ctx, st)
        improved = hbt_update!(ctx, st)
    else
        improved = false
    end

    return hbt_endstep!(ctx, st, k, improved)
end

# Merge if the solution has stalled, and count the steps without progress.
function hbt_endstep!(ctx::HBTContext, st::HBTState, k::Int, improved::Bool)
    if ctx.merge && !hbt_expired(ctx) && (!ctx.diversify || st.stall >= ctx.patience)
        improved |= hbt_improve_merge!(ctx, st)
    end

    st.stall = improved ? 0 : st.stall + 1
    st.kstall = hbt_k(st.width) < k ? 0 : st.kstall + 1
    return improved
end

mutable struct HBTShared
    @atomic snapshot::HBTSnapshot
    @atomic stop::Bool
end

# The main loop with workers on other threads. Each worker diversifies the
# latest snapshot of the solution, again and again, and passes the results
# to this thread, which adds them to the solution, evaluates it, and merges
# when it stalls, as `hbt_improve!` does. When no result is waiting, this
# thread diversifies the solution itself. (The time of a diversification
# varies a great deal, so workers that waited for each other would mostly
# wait.)
function hbt_parallel!(ctx::HBTContext, st::HBTState, lb::Int)
    shared = HBTShared(HBTSnapshot(st), false)
    results = Channel{Tuple{HBTLocal, Vector{Vector{Int}}}}(typemax(Int))

    tasks = map(ctx.workers) do w
        Threads.@spawn hbt_work(w, shared, results)
    end

    try
        while !hbt_expired(ctx) && hbt_k(st.width) > lb
            k = hbt_k(st.width)

            if isready(results)
                L, cliques = take!(results)
            else
                L, cliques = hbt_diversify(ctx, HBTSnapshot(st))
            end

            hbt_addcliques!(ctx, st, L, cliques)
            improved = hbt_update!(ctx, st)
            hbt_endstep!(ctx, st, k, improved)
            @atomic shared.snapshot = HBTSnapshot(st)
        end
    finally
        @atomic shared.stop = true
        foreach(wait, tasks)
    end

    return
end

function hbt_work(w::HBTContext, shared::HBTShared, results::Channel)
    while !(@atomic shared.stop) && !hbt_expired(w)
        put!(results, hbt_diversify(w, @atomic shared.snapshot))
    end

    return
end

# ---------------------------------------------------------------------------
# Diversification (Tamaki 2019, Section 5.1)
#
# Pick a largest bag X₀ of the current best decomposition T and a random
# subtree R of T containing it. Let U be the union of the bags of R and H the
# local graph on U; R is a decomposition of H, so a better decomposition of H
# would give a better decomposition of G. To steer away from R, fill a
# minimal separator S of H that crosses X₀ (so X₀ is not a maximal clique of
# any minimal triangulation of H + S) and triangulate H + S; a minimal
# triangulation of H + S is a minimal triangulation of H (Proposition 3.2), so
# its maximal cliques are PMCs of H, hence of G (Proposition 3.6). This is
# repeated for several separators.
# ---------------------------------------------------------------------------

# One diversification of a solution, by the context `ctx`, which may be a
# worker (see `hbt_workers!`). Reads a snapshot of the solution: returns the
# local graph and the PMCs of it to be added to the solution (see
# `hbt_addcliques!`).
function hbt_diversify(ctx::HBTContext, st::HBTSnapshot)
    hbt_cutoff!(ctx, st.kstall)
    g = ctx.graph; rng = ctx.rng; bags = st.bags; parent = st.parent; nt = length(bags)
    mark1 = ctx.mark1; mark2 = ctx.mark2; loc = ctx.loc
    ctx.nstep += 1

    # a random bag among the largest ones: of the largest weight k with
    # probability 1/2, of weight k - 1 with probability 1/4, and so on, down
    # to k - near (re-triangulating near the largest bags makes room for
    # removing them later)
    k = hbt_k(st.width)
    delta = 0

    while delta < ctx.near && rand(rng, Bool)
        delta += 1
    end

    top = Int[]

    while isempty(top)
        for i in 1:nt
            w = 0

            for v in bags[i]
                w += ctx.wgt[v]
            end

            w == k - delta && push!(top, i)
        end

        delta -= 1
    end

    i0 = rand(rng, top)
    X0 = bags[i0]

    # a random subtree containing it, grown until its vertex set reaches a
    # random target size
    nbrs = [Int[] for _ in 1:nt]

    for i in 1:nt
        p = parent[i]

        if !iszero(p)
            push!(nbrs[i], p); push!(nbrs[p], i)
        end
    end

    # region size: with probability `pexact`, just above |X₀| (up to `margin`
    # more vertices), where exact triangulation is fast even for wide bags;
    # otherwise log-uniform between |X₀| and `dsize`, so that every scale
    # gets tried
    lo = length(X0)

    if rand(rng) < ctx.pexact
        target = lo + rand(rng, 1:max(1, ctx.margin))
    else
        hi = max(lo, min(ctx.dsize, nv(g)))
        target = round(Int, exp(log(lo) + rand(rng) * (log(hi) - log(lo))))
    end
    su = hbt_stamp!(ctx); sn = hbt_stamp!(ctx)
    U = Int[]

    for v in X0
        mark1[v] = su; push!(U, v)
    end

    mark2[i0] = sn   # (mark2 is indexed by bag here)
    frontier = copy(nbrs[i0])

    for i in frontier
        mark2[i] = sn
    end

    while !isempty(frontier) && length(U) < target
        r = rand(rng, eachindex(frontier))
        i = frontier[r]; frontier[r] = frontier[end]; pop!(frontier)
        grow = 0

        for v in bags[i]
            mark1[v] == su || (grow += 1)
        end

        length(U) + grow > target && continue

        for v in bags[i]
            if mark1[v] != su
                mark1[v] = su; push!(U, v)
            end
        end

        for j in nbrs[i]
            if mark2[j] != sn
                mark2[j] = sn; push!(frontier, j)
            end
        end
    end

    sort!(U)
    L = hbt_local(ctx, U)
    H = L.graph; nu = length(U); weights = L.wgt
    x0 = Int[loc[v] for v in X0]     # X₀ in local numbering (set by hbt_local)
    lmark = zeros(Int, nu)
    limit = hbt_limit(ctx, length(X0))
    cliques = Vector{Int}[]

    # a region small enough to be solved exactly is also triangulated as it
    # is: PIDBT finds a decomposition of H below the current width if there
    # is one. If there is none, it finds the treewidth of H, which bounds
    # that of H + S below: then PIDBT need not try the widths below it again
    # for H + S (where they cost an exhaustive search each, and fail).
    lbS = 0

    if nu <= limit
        order = hbt_exact(ctx, weights, H, k - 1)

        if isnothing(order)
            order = hbt_greedy(ctx, weights, H)
            hbt_cliques!(cliques, ctx, L, H, order, k)
        else
            w = hbt_cliques!(cliques, ctx, L, H, order, k)
            w >= k && (lbS = w)
        end
    end

    # minimal separators crossing X₀: for nonadjacent a, b ∈ X₀, the
    # neighborhood of the component of H - N[a] containing b
    tried = Set{Vector{Int}}()

    for _ in 1:ctx.nsep
        hbt_expired(ctx) && break
        a = 0; b = 0

        for _ in 1:20
            a1 = rand(rng, x0); b1 = rand(rng, x0)
            if a1 != b1 && !insorted(b1, neighbors(H, a1))
                a = a1; b = b1
                break
            end
        end

        iszero(a) && break
        S = hbt_minsep(ctx, H, a, b, lmark)
        S in tried && continue
        push!(tried, S)

        # H + S
        ptr = Vector{Int}(undef, nu + 1); tgt = Int[]; ptr[1] = 1
        sm = hbt_stamp!(ctx)

        for v in S
            lmark[v] = sm
        end

        for v in 1:nu
            start = length(tgt) + 1
            append!(tgt, neighbors(H, v))

            if lmark[v] == sm
                for w in S
                    w == v || push!(tgt, w)
                end

                list = view(tgt, start:length(tgt)); sort!(list)
                m = start - 1

                for i in start:length(tgt)
                    if m < start || tgt[i] != tgt[m]
                        m += 1; tgt[m] = tgt[i]
                    end
                end

                resize!(tgt, m)
            end

            ptr[v + 1] = length(tgt) + 1
        end

        HS = BipartiteGraph{Int, Int}(nu, nu, length(tgt), ptr, tgt)

        order = hbt_triangulate(ctx, weights, HS, k - 1, limit; lb = lbS)
        hbt_cliques!(cliques, ctx, L, HS, order, k)
    end

    return L, cliques
end

# Diversify the solution once (in this thread).
function hbt_diversify!(ctx::HBTContext, st::HBTState)
    hbt_addcliques!(ctx, st, hbt_diversify(ctx, HBTSnapshot(st))...)
    return
end

# Collect the maximal cliques of the triangulation of HS (a minimal
# triangulation of the local graph L) given by `order`, skipping cliques
# heavier than k: they cannot be bags of a decomposition of width at most k.
# Returns the weight of the heaviest clique.
function hbt_cliques!(cliques::Vector{Vector{Int}}, ctx::HBTContext, L::HBTLocal, HS::BipartiteGraph, order::Vector{Int}, k::Int)
    weights = L.wgt
    ctx.nlocal += 1
    label, tree = cliquetree(weights, HS, order)
    width = 0

    for clique in tree
        w = 0

        for v in clique
            w += weights[label[v]]
        end

        width = max(width, w)
        w > k && continue
        K = Int[label[v] for v in clique]
        sort!(K)
        K in L.nbrs || push!(cliques, K)
    end

    return width
end

# Add the PMCs U[K] of G, for the PMCs K of the local graph, to the solution.
function hbt_addcliques!(ctx::HBTContext, st::HBTState, L::HBTLocal, cliques::Vector{Vector{Int}})
    nu = length(L.U); lmark = zeros(Int, nu); lcomp = zeros(Int, nu)

    for K in cliques
        hbt_addpmc!(ctx, st, L, K, lmark, lcomp)
    end

    return
end

# A minimal separator of H separating nonadjacent vertices a and b: the
# neighborhood of the component of H - N[a] that contains b.
function hbt_minsep(ctx::HBTContext, H::BipartiteGraph, a::Int, b::Int, lmark::Vector{Int})
    queue = ctx.queue
    sa = hbt_stamp!(ctx)
    lmark[a] = sa

    for v in neighbors(H, a)
        lmark[v] = sa
    end

    sb = hbt_stamp!(ctx)
    S = Int[]
    empty!(queue); push!(queue, b); lmark[b] = sb; head = 1

    while head <= length(queue)
        u = queue[head]; head += 1

        for x in neighbors(H, u)
            if lmark[x] == sa
                lmark[x] = sb; push!(S, x)          # in N(a): on the separator
            elseif lmark[x] != sb
                lmark[x] = sb; push!(queue, x)
            end
        end
    end

    return sort!(S)
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

# Solve each connected component of a graph. Small components are solved
# exactly; the others are improved, always working on the widest one, until
# the time runs out or the width matches the lower bound.
function hbt(weights::AbstractVector, graph::AbstractGraph{V}, alg::HBT) where {V}
    t0 = time()
    deadline = t0 + alg.time
    n = nv(graph)
    order = V[]
    n == 0 && return order, invperm(order)
    bgraph, wgt = hbt_graph(weights, graph)
    lb = convert(Int, lowerbound(wgt, graph, alg.lb))
    rng = Xoshiro(alg.seed)

    # the graph is assumed connected; disconnected inputs are handled by the
    # `ConnectedComponents` wrapper, as for `PIDBT` and `BT`

    # small graphs are solved exactly if that takes at most half the time budget
    if n <= alg.base
        suborder = pidbt(wgt, bgraph, lowerbound(wgt, bgraph, alg.lb); deadline = t0 + alg.time / 2)

        if !isnothing(suborder)
            order = convert(Vector{V}, suborder)
            return order, invperm(order)
        end
    end

    ctx = HBTContext(bgraph, wgt, alg.alg, rng, alg.base, alg.ntry, alg.ninit, deadline; t0, verbose = alg.verbose, refined = alg.refined, merge = alg.merge, diversify = alg.diversify, dsize = alg.dsize, nsep = alg.nsep, patience = alg.patience, xtime = alg.xtime, xmin = alg.xmin, xstep = alg.xstep, near = alg.near, margin = alg.margin, pexact = alg.pexact)
    st = hbt_state(ctx, 0)
    alg.verbose && println("hbt: n = $n, initial width $(hbt_string(st.width)) at $(round(time() - t0; digits = 2)) s")

    if alg.threads > 1 && ctx.diversify
        hbt_workers!(ctx, alg.threads - 1)
        hbt_parallel!(ctx, st, lb)
    else
        while !hbt_expired(ctx) && hbt_k(st.width) > lb
            hbt_improve!(ctx, st)
        end
    end

    if alg.verbose
        nstep, nlocal, nexact, ntimeout = hbt_stats(ctx)
        println("hbt: n = $n, width = $(hbt_string(st.width)), steps = $nstep, local = $nlocal, exact = $nexact, timeouts = $ntimeout, pmcs = $(length(st.pmcs)), time = $(round(time() - t0; digits = 1))")
    end

    order = convert(Vector{V}, st.order)
    return order, invperm(order)
end
