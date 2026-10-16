# The minor test of `SafeSeparators(...; maxmissing)`.
#
# Let H be a minimal triangulation of G, given by a postordered clique tree,
# and let S = sep(j) be the separator of a node j. S is safe for treewidth if,
# for every component C of G - S, contracting parts of G - C turns N(C) into a
# clique, in which every vertex keeps at least its weight (Bodlaender and
# Koster, "Safe separators for treewidth"): then every graph G[C ∪ S] with S
# filled is a minor of G. The contractions are searched for greedily.
#
# Components. Every other minimal separator of H is parallel to S, so lies
# inside C ∪ N(C) for a single component C of G - S; filling it changes
# neither the components of G - S nor their neighborhoods. So these are read
# off the clique tree of H:
#
#   - The vertices below j (an interval of the elimination order) form a full
#     component of S: a subtree of the elimination tree is connected in H.
#   - Every other component C has a top node q: either the root of j's tree,
#     or a node with sep(q) ⊆ S. Since the least vertex of sep(q) is the parent
#     of the last vertex of q in the elimination tree, the candidates for q are
#     the nodes whose separators start at a vertex of S.
#   - If q is not an ancestor of j, then C is the subtree below q, and N(C) =
#     sep(q). Otherwise C is a segment of the path from j to the root, with
#     everything hanging off it, and N(C) is the separator of the next node
#     below on the path that cuts it off: j, or an ancestor q' with sep(q') ⊆ S.
#
# A vertex belongs to the component whose top is the first such node above
# its own node; this is looked up lazily, by walks up the tree that are
# memoized for each separator.
#
# Search. For a component C with N = N(C), the search works in the region R
# = V - C - N. It keeps *nodes*: disjoint connected sets D ⊆ R, each with the
# set of vertices of N it touches (a bit set), its least weight, and a label
# in N (or none).
#
#   1. Seed: every vertex of R that is adjacent to an endpoint of a missing
#      edge of N, has another neighbor, and is no lighter than that endpoint.
#   2. Phase 1: while some missing edge {a, b} is touched at both ends by no
#      node, merge a node touching a (and not b) with a node touching b (and
#      not a) along a path through free vertices of R, none lighter than the
#      lighter of a and b. If there is no such pair, fail.
#   3. Phase 2: while there are more than k/2 nodes (k = |N|), merge a pair
#      for the missing edge with the fewest nodes that could cover it. Stop at
#      the first edge with no pair. Then drop the nodes that cover nothing.
#   4. Phase 3: label greedily. Each round labels one node D with a vertex v
#      it touches (no vertex of D lighter than v), maximizing (n, c): c is the
#      number of missing edges at v that D covers, and n the least number of
#      other unlabelled nodes that could still cover an edge left uncovered.
#      If n = 0, fail.
#
# Every labelled node is then contracted into its label. The search always
# ends: every merge removes a node, and every round of phase 3 labels one.
# Its cost grows quickly with the number of missing edges of S, so only
# separators with at most `maxmissing` missing edges are tested. The pairs are found
# by one breadth-first search from all the candidate first nodes at once,
# which takes the candidate second node it reaches last (the one furthest
# from the first nodes): taking the nearest one instead found fewer safe
# separators.


# the clique tree, and the components of G - S for the current separator
struct ACSDTree{V}
    parent::Vector{V}   # parent node (0 at a root)
    fdesc::Vector{V}    # first descendant
    home::Vector{V}     # index -> node whose residual contains it
    rfirst::Vector{V}   # first index of a residual
    rlast::Vector{V}    # last index of a residual
    bptr::Vector{Int}   # nodes whose separator starts at index i:
    btgt::Vector{V}     #   btgt[bptr[i]:bptr[i + 1] - 1]
    smark::Vector{Int}  # (index) in S
    istop::Vector{Int}  # (node) top of a component
    ctag::Vector{Int}   # (node) top of its component known...
    ctop::Vector{V}     # ... and equal to this
    path::Vector{V}
    tag::Base.RefValue{Int}
end

function ACSDTree(tree::CliqueTree{V}, n::Integer) where {V}
    t = length(tree)
    parent = zeros(V, t); fdesc = Vector{V}(undef, t)
    home = Vector{V}(undef, n); rfirst = Vector{V}(undef, t); rlast = Vector{V}(undef, t)

    for q in oneto(t)
        p = parentindex(tree, q)
        isnothing(p) || (parent[q] = p)
        fdesc[q] = q
    end

    # the nodes are in postorder
    for q in oneto(t)
        p = parent[q]
        ispositive(p) && (fdesc[p] = min(fdesc[p], fdesc[q]))
        r = residual(tree, q)
        rfirst[q] = first(r); rlast[q] = last(r)

        for i in r
            home[i] = q
        end
    end

    bptr = zeros(Int, n + 1)

    for q in oneto(t)
        sep = separator(tree, q)
        isempty(sep) || (bptr[first(sep) + 1] += 1)
    end

    bptr[1] = 1

    for i in oneto(n)
        bptr[i + 1] += bptr[i]
    end

    btgt = Vector{V}(undef, bptr[n + 1] - 1); pos = copy(bptr)

    for q in oneto(t)
        sep = separator(tree, q)

        if !isempty(sep)
            i = first(sep); btgt[pos[i]] = q; pos[i] += 1
        end
    end

    return ACSDTree{V}(parent, fdesc, home, rfirst, rlast, bptr, btgt,
        zeros(Int, n), zeros(Int, t), zeros(Int, t), zeros(V, t), V[], Ref(0))
end

# The components of G - S for S = sep(j), as pairs (top, q) with N(C) =
# sep(q), largest subtree first (their regions are the smallest: they are
# the cheapest to search, and the likeliest to fail).
function acsd_components!(comps::Vector{Tuple{V, V}}, ancs::Vector{V}, at::ACSDTree{V}, tree::CliqueTree{V}, j::V) where {V}
    at.tag[] += 1; tag = at.tag[]
    S = separator(tree, j)

    for i in S
        at.smark[i] = tag
    end

    empty!(comps); empty!(ancs)
    push!(comps, (j, j)); at.istop[j] = tag

    for s in S, k in at.bptr[s]:at.bptr[s + 1] - 1
        q = at.btgt[k]
        q == j && continue
        sep = separator(tree, q)
        length(sep) <= length(S) || continue
        all(i -> at.smark[i] == tag, sep) || continue
        at.istop[q] = tag

        if at.fdesc[q] <= j < q     # an ancestor of j
            push!(ancs, q)
        else
            push!(comps, (q, q))
        end
    end

    sort!(ancs); lower = j

    for a in ancs
        push!(comps, (a, lower)); lower = a
    end

    r = j

    while ispositive(at.parent[r])
        r = at.parent[r]
    end

    at.istop[r] = tag
    push!(comps, (r, lower))
    sort!(comps; by = ((top, _),) -> at.rlast[top] - at.rfirst[at.fdesc[top]], rev = true)
    return comps
end

# the top of the component of node x
function acsd_top!(at::ACSDTree{V}, x::V) where {V}
    tag = at.tag[]; path = at.path; empty!(path); y = x

    while true
        if at.ctag[y] == tag
            y = at.ctop[y]; break
        elseif at.istop[y] == tag
            break
        end

        push!(path, y)
        p = at.parent[y]
        iszero(p) && break      # the root of another tree
        y = p
    end

    for z in path
        at.ctag[z] = tag; at.ctop[z] = y
    end

    return y
end

# the search for one component
mutable struct ACSDMinor{V}
    # per vertex
    status::Vector{Int}     # 0: free; -1: out of the region; > 0: node
    sidx::Vector{Int}       # position in N (0 if not in N)
    stamp::Vector{Int}
    parent::Vector{V}
    src::Vector{Int}
    queue::Vector{V}
    touched::Vector{V}
    tag::Int
    # per node
    nn::Int
    W::Int                  # words per bit set over N
    verts::Vector{Vector{V}}
    bits::Vector{UInt64}
    label::Vector{Int}
    mw::Vector{Int}         # least weight
    nstamp::Vector{Int}
    nvia::Vector{V}
    # per missing edge
    edges::Vector{Tuple{Int, Int}}
    count::Vector{Int}
    val::Vector{Int}
    order::Vector{Int}
    bucket::Vector{Int}
    rem::Vector{UInt64}
    isend::Vector{Bool}
    Nl::Vector{V}
end

function ACSDMinor{V}(n::Integer) where {V}
    return ACSDMinor{V}(zeros(Int, n), zeros(Int, n), zeros(Int, n), zeros(V, n), zeros(Int, n), V[], V[], 0,
        0, 0, Vector{V}[], UInt64[], Int[], Int[], Int[], V[],
        Tuple{Int, Int}[], Int[], Int[], Int[], Int[], UInt64[], Bool[], V[])
end

@inline function acsd_hasbit(bits::Vector{UInt64}, off::Int, s::Int)
    return isone((bits[off + ((s - 1) >> 6) + 1] >> ((s - 1) & 63)) & one(UInt64))
end

@inline function acsd_setbit!(bits::Vector{UInt64}, off::Int, s::Int)
    bits[off + ((s - 1) >> 6) + 1] |= one(UInt64) << ((s - 1) & 63)
    return
end

@inline acsd_off(ms::ACSDMinor, i::Int) = (i - 1) * ms.W

@inline function acsd_covers(ms::ACSDMinor, i::Int, a::Int, b::Int)
    off = acsd_off(ms, i)
    return acsd_hasbit(ms.bits, off, a) & acsd_hasbit(ms.bits, off, b)
end

# is vertex `x` free and in the region (not in the component `top`)?
@inline function acsd_free(ms::ACSDMinor{V}, at::ACSDTree{V}, index::AbstractVector{V}, top::V, x::V) where {V}
    iszero(ms.status[x]) || return false
    i = index[x]
    at.smark[i] == at.tag[] && return true      # in S - N
    return acsd_top!(at, at.home[i]) != top
end

function acsd_newnode!(ms::ACSDMinor{V}, graph::AbstractGraph{V}, weights::Vector{Int}, v::V) where {V}
    ms.nn += 1; i = ms.nn

    if i > length(ms.verts)
        push!(ms.verts, V[]); push!(ms.label, 0); push!(ms.mw, 0); push!(ms.nstamp, 0); push!(ms.nvia, zero(V))
    end

    length(ms.bits) < i * ms.W && resize!(ms.bits, i * ms.W)
    empty!(ms.verts[i]); push!(ms.verts[i], v)
    ms.label[i] = 0; ms.mw[i] = weights[v]; ms.nstamp[i] = 0
    off = acsd_off(ms, i)

    for k in 1:ms.W
        ms.bits[off + k] = zero(UInt64)
    end

    for x in neighbors(graph, v)
        s = ms.sidx[x]
        ispositive(s) && acsd_setbit!(ms.bits, off, s)
    end

    ms.status[v] = i; push!(ms.touched, v)
    return
end

# move node j into slot i
function acsd_movenode!(ms::ACSDMinor, i::Int, j::Int)
    i == j && return
    ms.verts[i], ms.verts[j] = ms.verts[j], ms.verts[i]
    ms.label[i] = ms.label[j]; ms.mw[i] = ms.mw[j]
    oi = acsd_off(ms, i); oj = acsd_off(ms, j)

    for k in 1:ms.W
        ms.bits[oi + k] = ms.bits[oj + k]
    end

    for v in ms.verts[i]
        ms.status[v] = i
    end

    return
end

function acsd_deletenode!(ms::ACSDMinor, i::Int)
    for v in ms.verts[i]
        ms.status[v] = -1
    end

    acsd_movenode!(ms, i, ms.nn); ms.nn -= 1
    return
end

@inline acsd_wt(ms::ACSDMinor, weights::Vector{Int}, s::Int) = weights[ms.Nl[s]]

# does node i cover a missing edge?
function acsd_coversany(ms::ACSDMinor, i::Int)
    for (a, b) in ms.edges
        acsd_covers(ms, i, a, b) && return true
    end

    return false
end

# a missing edge that no unlabelled node can cover, or 0
function acsd_uncovered(ms::ACSDMinor, weights::Vector{Int})
    for (e, (a, b)) in enumerate(ms.edges)
        wmin = min(acsd_wt(ms, weights, a), acsd_wt(ms, weights, b))
        covered = false

        for i in 1:ms.nn
            if iszero(ms.label[i]) && ms.mw[i] >= wmin && acsd_covers(ms, i, a, b)
                covered = true; break
            end
        end

        covered || return e
    end

    return 0
end

# the missing edge that the fewest unlabelled nodes can cover
function acsd_leastcovered(ms::ACSDMinor, weights::Vector{Int})
    nmin = emin = 0

    for (e, (a, b)) in enumerate(ms.edges)
        wmin = min(acsd_wt(ms, weights, a), acsd_wt(ms, weights, b)); n = 0

        for i in 1:ms.nn
            n += iszero(ms.label[i]) & (ms.mw[i] >= wmin) & acsd_covers(ms, i, a, b)
        end

        if iszero(emin) || n < nmin
            nmin = n; emin = e
        end
    end

    return emin
end

# A pair of nodes (D₁, D₂), D₁ touching a and not b, D₂ touching b and not a,
# joined by a path through free vertices of the region that are no lighter
# than the lighter of a and b: one breadth-first search from every candidate
# D₁, taking the last candidate D₂ that it reaches. Returns (D₁, D₂), or
# (0, 0) if there is none; the path is left in `parent` and `nvia`.
function acsd_pair!(ms::ACSDMinor{V}, at::ACSDTree{V}, index::AbstractVector{V}, top::V, graph::AbstractGraph{V}, weights::Vector{Int}, a::Int, b::Int) where {V}
    wmin = min(acsd_wt(ms, weights, a), acsd_wt(ms, weights, b))
    ms.tag += 1; tag = ms.tag
    queue = ms.queue; empty!(queue)
    n2 = 0

    for i in 1:ms.nn
        ms.mw[i] >= wmin || continue
        off = acsd_off(ms, i); ha = acsd_hasbit(ms.bits, off, a); hb = acsd_hasbit(ms.bits, off, b)

        if hb && !ha
            n2 += 1
        elseif ha && !hb
            for v in ms.verts[i]
                ms.stamp[v] = tag; ms.parent[v] = zero(V); ms.src[v] = i; push!(queue, v)
            end
        end
    end

    last = 0; nfound = 0; head = 1

    while head <= length(queue)
        u = queue[head]; head += 1

        for x in neighbors(graph, u)
            s = ms.status[x]

            if ispositive(s)
                ms.nstamp[s] == tag && continue
                off = acsd_off(ms, s)

                if ms.mw[s] >= wmin && acsd_hasbit(ms.bits, off, b) && !acsd_hasbit(ms.bits, off, a)
                    ms.nstamp[s] = tag; ms.nvia[s] = u; last = s; nfound += 1
                    # every candidate has been reached: none is further
                    nfound == n2 && return ms.src[u], last
                end
            elseif ms.stamp[x] != tag && weights[x] >= wmin && acsd_free(ms, at, index, top, x)
                ms.stamp[x] = tag; ms.parent[x] = u; ms.src[x] = ms.src[u]; push!(queue, x)
            end
        end
    end

    ispositive(last) && return ms.src[ms.nvia[last]], last
    return 0, 0
end

# merge node i2 into node i1 along the path of the last search
function acsd_merge!(ms::ACSDMinor{V}, graph::AbstractGraph{V}, weights::Vector{Int}, i1::Int, i2::Int) where {V}
    off1 = acsd_off(ms, i1); off2 = acsd_off(ms, i2)
    mw = min(ms.mw[i1], ms.mw[i2]); u = ms.nvia[i2]

    while ms.status[u] != i1
        # `u` is a free vertex on the path
        push!(ms.verts[i1], u); ms.status[u] = i1; push!(ms.touched, u)
        mw = min(mw, weights[u])

        for x in neighbors(graph, u)
            s = ms.sidx[x]
            ispositive(s) && acsd_setbit!(ms.bits, off1, s)
        end

        u = ms.parent[u]
    end

    for v in ms.verts[i2]
        push!(ms.verts[i1], v); ms.status[v] = i1
    end

    for k in 1:ms.W
        ms.bits[off1 + k] |= ms.bits[off2 + k]
    end

    ms.mw[i1] = mw; empty!(ms.verts[i2])
    acsd_movenode!(ms, i2, ms.nn); ms.nn -= 1
    return
end

# Is N(C) = `Nl` (sorted) a labelled clique minor of G - C? The vertices of
# `Nl` are marked out of the region by the caller.
function acsd_cliqueminor!(ms::ACSDMinor{V}, at::ACSDTree{V}, index::AbstractVector{V}, top::V, graph::AbstractGraph{V}, weights::Vector{Int}) where {V}
    Nl = ms.Nl; k = length(Nl); W = ms.W = cld(k, 64); ms.nn = 0
    edges = empty!(ms.edges)

    for (s, v) in enumerate(Nl)
        ms.sidx[v] = s
    end

    # the missing edges of N
    for (s, v) in enumerate(Nl)
        ms.tag += 1

        for x in neighbors(graph, v)
            ms.stamp[x] = ms.tag
        end

        for r in s + 1:k
            ms.stamp[Nl[r]] == ms.tag || push!(edges, (s, r))
        end
    end

    isempty(edges) && return true
    isend = ms.isend; resize!(isend, k); fill!(isend, false)

    for (a, b) in edges
        isend[a] = isend[b] = true
    end

    # seeds
    for s in 1:k
        isend[s] || continue
        w = Nl[s]

        for v in neighbors(graph, w)
            if eltypedegree(graph, v) > 1 && weights[v] >= weights[w] && acsd_free(ms, at, index, top, v)
                acsd_newnode!(ms, graph, weights, v)
            end
        end
    end

    # phase 1
    e = acsd_uncovered(ms, weights)

    while ispositive(e)
        i1, i2 = acsd_pair!(ms, at, index, top, graph, weights, edges[e]...)
        iszero(i1) && return false
        acsd_merge!(ms, graph, weights, i1, i2)
        e = acsd_uncovered(ms, weights)
    end

    # phase 2
    e = acsd_leastcovered(ms, weights)

    while 2ms.nn > k && ispositive(e)
        i1, i2 = acsd_pair!(ms, at, index, top, graph, weights, edges[e]...)
        iszero(i1) && break
        acsd_merge!(ms, graph, weights, i1, i2)
        e = acsd_leastcovered(ms, weights)
    end

    # drop the nodes that cover no missing edge
    i = 1

    while i <= ms.nn
        if acsd_coversany(ms, i)
            i += 1
        else
            acsd_deletenode!(ms, i)
        end
    end

    # phase 3
    count = ms.count; val = ms.val; order = ms.order; bucket = ms.bucket; rem = ms.rem
    resize!(rem, k * W)

    while !isempty(edges)
        vmax = imax = nmax = cmax = 0
        m = length(edges)
        resize!(count, m); resize!(val, m); resize!(order, m)

        for (e, (a, b)) in enumerate(edges)
            wmin = min(acsd_wt(ms, weights, a), acsd_wt(ms, weights, b)); n = 0

            for i in 1:ms.nn
                n += iszero(ms.label[i]) & (ms.mw[i] >= wmin) & acsd_covers(ms, i, a, b)
            end

            count[e] = n
        end

        fill!(rem, zero(UInt64))

        for (a, b) in edges
            acsd_setbit!(rem, (a - 1) * W, b); acsd_setbit!(rem, (b - 1) * W, a)
        end

        for i in 1:ms.nn
            iszero(ms.label[i]) || continue
            off = acsd_off(ms, i); mw = ms.mw[i]

            # the cover counts of the edges without this node, and the edges
            # sorted by them (a counting sort)
            top2 = 0

            for (e, (a, b)) in enumerate(edges)
                x = val[e] = count[e] - (acsd_covers(ms, i, a, b) & (mw >= min(acsd_wt(ms, weights, a), acsd_wt(ms, weights, b))))
                top2 = max(top2, x)
            end

            resize!(bucket, top2 + 2); fill!(bucket, 0); bucket[1] = 1

            for e in 1:m
                bucket[val[e] + 2] += 1
            end

            for x in 2:top2 + 2
                bucket[x] += bucket[x - 1]
            end

            for e in 1:m
                x = val[e] + 1; order[bucket[x]] = e; bucket[x] += 1
            end

            for v in 1:k
                acsd_hasbit(ms.bits, off, v) || continue
                mw >= acsd_wt(ms, weights, v) || continue
                c = 0

                for r in 1:W
                    c += count_ones(rem[(v - 1) * W + r] & ms.bits[off + r])
                end

                n = typemax(Int)

                for r in 1:m
                    e = order[r]; a, b = edges[e]

                    if !(((v == a) & acsd_hasbit(ms.bits, off, b)) | ((v == b) & acsd_hasbit(ms.bits, off, a)))
                        n = val[e]; break
                    end
                end

                if (n, c) > (nmax, cmax) || ((n, c) == (nmax, cmax) && v < vmax)
                    vmax = v; imax = i; nmax = n; cmax = c
                end
            end
        end

        iszero(nmax) && return false
        ms.label[imax] = vmax
        e = 1

        while e <= length(edges)
            a, b = edges[e]; covered = false

            for i in 1:ms.nn
                v = ms.label[i]; off = acsd_off(ms, i)

                if ((v == a) & acsd_hasbit(ms.bits, off, b)) | ((v == b) & acsd_hasbit(ms.bits, off, a))
                    covered = true; break
                end
            end

            if covered
                edges[e] = edges[end]; pop!(edges)
            else
                e += 1
            end
        end
    end

    return true
end

# does S = sep(j), with at most `maxmissing` missing edges, pass the test?
function acsd_minorsafe!(ms::ACSDMinor{V}, at::ACSDTree{V}, comps::Vector{Tuple{V, V}}, ancs::Vector{V}, index::AbstractVector{V}, graph::AbstractGraph{V}, weights::Vector{Int}, order::AbstractVector{V}, tree::CliqueTree{V}, j::V, maxmissing::Int) where {V}
    S = separator(tree, j)

    # the missing edges of S
    ms.tag += 1; tag = ms.tag

    for i in S
        ms.stamp[order[i]] = tag
    end

    missing = 0

    for i in S
        v = order[i]; c = 0

        for x in neighbors(graph, v)
            c += (x != v) & (ms.stamp[x] == tag)
        end

        missing += length(S) - 1 - c
    end

    # (each missing edge is counted twice)
    (iszero(missing) || missing > 2maxmissing) && return false
    acsd_components!(comps, ancs, at, tree, j)

    for (top, q) in comps
        N = separator(tree, q)
        length(N) < 2 && continue
        Nl = ms.Nl; empty!(Nl)

        for i in N
            push!(Nl, order[i])
        end

        sort!(Nl); empty!(ms.touched)

        for x in Nl
            ms.status[x] = -1
        end

        ok = acsd_cliqueminor!(ms, at, index, top, graph, weights)

        for x in ms.touched
            ms.status[x] = 0
        end

        for x in Nl
            ms.status[x] = 0; ms.sidx[x] = 0
        end

        ok || return false
    end

    return true
end

# Fill the separators of the clique tree `tree` (of a minimal triangulation
# of `graph`) that are not cliques, have at most `maxmissing` missing edges,
# and pass the minor test. The separators are all tested against the same
# graph and filled together: they are pairwise parallel, so filling one keeps
# the certificates of the others.
function acsd_minor(weights::AbstractVector, graph::AbstractGraph{V}, order::AbstractVector{V}, tree::CliqueTree{V}, maxmissing::Int) where {V}
    n = nv(graph)
    n < 4 && return graph, false

    # the test only compares weights, so their ranks will do
    rank = acsd_ranks(weights, n)
    index = invperm(order)
    at = ACSDTree(tree, n)
    ms = ACSDMinor{V}(n)
    comps = Tuple{V, V}[]; ancs = V[]; fills = V[]

    for j in oneto(convert(V, length(tree)))
        length(separator(tree, j)) < 3 && continue

        if acsd_minorsafe!(ms, at, comps, ancs, index, graph, rank, order, tree, j, maxmissing)
            push!(fills, j)
        end
    end

    isempty(fills) && return graph, false
    return acsd_fill(graph, order, tree, fills), true
end

# the graph with the separators of the nodes `fills` filled
function acsd_fill(graph::AbstractGraph{V}, order::AbstractVector{V}, tree::CliqueTree{V, E}, fills::Vector{V}) where {V, E}
    n = nv(graph)

    # the filled separators that contain each vertex
    fptr = zeros(Int, n + 1)

    for j in fills, i in separator(tree, j)
        fptr[order[i] + 1] += 1
    end

    fptr[1] = 1

    for v in oneto(n)
        fptr[v + 1] += fptr[v]
    end

    ftgt = Vector{V}(undef, fptr[n + 1] - 1); pos = copy(fptr)

    for j in fills, i in separator(tree, j)
        v = order[i]; ftgt[pos[v]] = j; pos[v] += 1
    end

    mark = zeros(V, n); adj = V[]
    pointer = FVector{E}(undef, n + 1); target = V[]
    pointer[begin] = p = one(E)

    for v in oneto(n)
        empty!(adj)

        for w in neighbors(graph, v)
            if w != v && mark[w] != v
                mark[w] = v; push!(adj, w)
            end
        end

        for k in fptr[v]:fptr[v + 1] - 1, i in separator(tree, ftgt[k])
            w = order[i]

            if w != v && mark[w] != v
                mark[w] = v; push!(adj, w)
            end
        end

        sort!(adj); append!(target, adj)
        pointer[v + 1] = p += convert(E, length(adj))
    end

    m = p - one(E)
    return BipartiteGraph(n, n, m, pointer, FVector{V}(target))
end
