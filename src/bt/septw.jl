#
# The width of a connected graph: an elimination ordering whose bags
# weigh at most hi, as light as the search can make it.
#
#   - septw tries the widths k = lo, ... in turn, each with sepenum
#     (the minimal separators lighter than k) and sepdp; after a
#     failed round, the next width is min(k', k + w), where k' is
#     the lower bound of sepdp and w the smallest weight (see
#     sepdp);
#   - pmctw lists every minimal separator (sepenum), every potential
#     maximal clique (pmcenum), and finds the best ordering (pmcdp).
#
# The graph is assumed connected: a disconnected graph is solved as a
# whole, correctly but slowly, and ConnectedComponents splits it.
#
# lo is the first width to try. If it is a lower bound on the width,
# then the ordering is optimal; if not, its bags weigh at most lo.
# The search starts at no less than a bound that every decomposition
# meets: the weight of the heaviest vertex, or of the heaviest edge
# (every edge lies in some bag). A width at which a cheap ordering
# (MF) already has bags light enough is not searched: the ordering
# is used instead, at the start (the bound is met) or after failed
# rounds (the ordering is optimal). This matters, as a large k leaves
# many separators lighter than k.
#
# The result is (perm, invp, k, status):
#
#   - BT_FOUND: (perm, invp) is an ordering of width k <= hi;
#   - BT_ABOVE: every ordering has width at least k > hi;
#   - BT_TIMEOUT: the search ran for more than mt seconds, and every
#     ordering has width at least k.
#
# If the status is not BT_FOUND, (perm, invp) is the identity. The
# weights must be positive, and only weights[1:nv(graph)] belong
# to the graph. The positional methods take the deadline dl, in the
# sense of time(), instead of mt. With nt > 1, sepenum and sepdp use
# nt tasks, and the result is the same; pmcenum and pmcdp use one,
# and pmctw sees the deadline only between them.
#
function septw(weights::AbstractVector, graph::AbstractGraph, lo, hi; mt::Real = Inf, nt::Integer = Threads.nthreads())
    return septw(weights, graph, lo, hi, time() + mt, nt)
end

function septw(weights::AbstractVector{W}, graph::AbstractGraph{V}, lo, hi, dl::Real, nt::Integer) where {W, V}
    return bt_tw(septw_run, weights, graph, convert(W, lo), convert(W, hi), dl, nt)
end

function pmctw(weights::AbstractVector, graph::AbstractGraph, lo, hi; mt::Real = Inf, nt::Integer = Threads.nthreads())
    return pmctw(weights, graph, lo, hi, time() + mt, nt)
end

function pmctw(weights::AbstractVector{W}, graph::AbstractGraph{V}, lo, hi, dl::Real, nt::Integer) where {W, V}
    return bt_tw(pmctw_run, weights, graph, convert(W, lo), convert(W, hi), dl, nt)
end

# The bound, the cheap ordering, and the search f. The search runs on
# a graph with Int vertices: the lists of separators, blocks, and caps
# are indexed by the vertex type, and can be long.
function bt_tw(f::F, weights::AbstractVector{W}, graph00::AbstractGraph{V}, lo::W, hi::W, dl::Real, nt::Integer) where {F, W, V}
    graph = bt_int(graph00); nvtx = nv(graph)
    perm = FVector{V}(undef, nvtx)
    invp = FVector{V}(undef, nvtx)

    base = bt_bas(weights, graph)

    if base > hi
        order = oneto(nvtx); k = base; status = BT_ABOVE
    else
        k = max(base, min(lo, hi))

        # the cheap ordering
        hord, _ = permutation(weights, graph, MF())
        hk = treewidth(weights, graph, hord)

        if hk <= k
            order = hord; k = hk; status = BT_FOUND
        else
            order, k, status = f(weights, graph, k, base, hi, hord, hk, dl, nt)
        end
    end

    if status == BT_FOUND
        for i in oneto(nvtx)
            perm[i] = order[i]; invp[order[i]] = i
        end
    else
        for v in oneto(nvtx)
            perm[v] = invp[v] = v
        end
    end

    return perm, invp, k, status
end

# The rounds of septw, from the width k, where lb <= k is a lower
# bound and the cheap ordering hord has width hk > k.
# Each round runs on the backend (see bt_round!).
function septw_run(weights::AbstractVector{W}, graph::AbstractGraph{V}, k::W, lb::W, hi::W, hord::AbstractVector, hk::W, dl::Real, nt::Integer) where {W, V}
    esc = bt_esc(SEPTW[], weights, graph, dl)

    while true
        # every round below k failed, so the cheap ordering is optimal
        if k >= hk
            return hord, hk, hk <= hi ? BT_FOUND : BT_ABOVE
        end

        k > hi && return hord, k, BT_ABOVE

        perm, next, status = bt_round!(esc, weights, graph, k, dl, nt)
        status == BT_TIMEOUT && return hord, lb, BT_TIMEOUT
        status == BT_FOUND && return perm, next, BT_FOUND

        lb = k = next
    end
end

# The search of pmctw: every potential maximal clique.
function pmctw_run(weights::AbstractVector{W}, graph::AbstractGraph{V}, k::W, lb::W, hi::W, hord::AbstractVector, hk::W, dl::Real, nt::Integer) where {W, V}
    sep, status = sepenum(weights, graph, typemax(W), dl, nt)
    status == BT_TIMEOUT && return hord, lb, BT_TIMEOUT
    pmc = pmcenum(graph, sep)
    time() > dl && return hord, lb, BT_TIMEOUT
    perm, _, mink = pmcdp(weights, graph, pmc)

    if mink <= hi
        return perm, mink, BT_FOUND
    else
        return hord, mink, BT_ABOVE
    end
end

# A bound that every decomposition meets: the weight of the
# heaviest vertex, or of the heaviest edge.
function bt_bas(weights::AbstractVector{W}, graph::AbstractGraph{V}) where {W, V}
    k = zero(W)

    @inbounds for v in vertices(graph)
        wv = weights[v]; k = max(k, wv)

        for x in neighbors(graph, v)
            x != v && (k = max(k, wv + weights[x]))
        end
    end

    return k
end

bt_int(graph::AbstractGraph{Int}) = graph
bt_int(graph::AbstractGraph) = BipartiteGraph{Int, Int}(graph)
