# The connectivity test of `SafeSeparators(...; maxmissing)`, tried before the
# minor test (Medema and Lazovik, "A safeness condition for minimal separators
# based on vertex connectivity", Theorem 16).
#
# Let S be a minimal separator of G and K ⊆ S a clique. If, for every
# component C of G - S, every missing edge {a, b} of S has |S - K|
# vertex-disjoint a-b paths in G - ((C ∪ K) - {a, b}), then S is safe for
# treewidth. With weights, blow every vertex v up into a clique of w(v) twins:
# the paths become a flow with vertex capacities w (a and b have none), of
# value w(S - K). Take K = S - X for a light vertex cover X of the missing
# edges, found greedily and then made minimal: adding a vertex x to K lowers
# the requirement by w(x) and the flow by at most w(x).
#
# For each missing edge {a, b}, with r = w(X):
#
#   1. Bound. Every unit of flow leaves a through a neighbor, so the neighbors
#      of a must have capacity r outside of every component C; likewise for
#      b. This is checked once for each end, before any flow, and rejects
#      most of the separators that fail.
#   2. Flow. Augment along shortest paths in G - K, up to 2r. Removing a
#      component C takes away at most the flow that enters C from S, so only
#      the components that take more than F - r are removed, one at a time,
#      and the flow found again, up to r. (The flow on an edge goes only one
#      way, so the flow entering C is the flow on the edges from S to C.)
#
# The components of G - S are read off the clique tree (see acsd_minor.jl).
mutable struct ACSDFlow{V, E, W}
    # the graph, with sorted neighbors, and the reverse of each arc
    ptr::FVector{E}
    tgt::FVector{V}
    rev::FVector{E}
    # the flow on each arc (fe[rev[p]] = -fe[p]) and through each vertex
    fe::FVector{W}
    fv::FVector{W}
    tarcs::Vector{E}
    tverts::Vector{V}
    # the search: each vertex v has the states in(v) and out(v); `ain[v]` is
    # the arc from the previous state to in(v), and `aout[v]` is the arc
    # between out(v) and the previous state (0: the other state of v). In the
    # queue, in(v) is -v and out(v) is v.
    sin::FVector{Int}
    sout::FVector{Int}
    ain::FVector{E}
    aout::FVector{E}
    queue::Vector{V}
    # the separator: `stamp[v] == stag` if v ∈ S, and `snode[v]` is the index
    # of v in `ends` (the ends of the missing edges) or 0
    stamp::FVector{Int}
    snode::FVector{Int}
    mark::FVector{Int}
    stag::Int
    tag::Int
    ends::Vector{V}
    inX::Vector{Bool}
    udeg::Vector{Int}
    perm::Vector{Int}
    mptr::Vector{Int}
    mtgt::Vector{Int}
    # the components, by top node
    inflow::FVector{W}
    itag::FVector{Int}
    tops::Vector{V}
    # the current missing edge, and the removed component (0: none)
    va::V
    vb::V
    drop::V
end

function ACSDFlow(weights::AbstractVector{W}, graph::AbstractGraph{V}, t::Integer) where {W, V}
    E = etype(graph); n = nv(graph); m = zero(E)
    ptr = FVector{E}(undef, n + one(V))
    ptr[one(V)] = one(E)

    @inbounds for v in vertices(graph)
        for x in neighbors(graph, v)
            x != v && (m += one(E))
        end

        ptr[v + one(V)] = m + one(E)
    end

    tgt = FVector{V}(undef, m); rev = FVector{E}(undef, m)
    aout = FVector{E}(undef, n)

    @inbounds for v in vertices(graph)
        p = ptr[v]

        for x in neighbors(graph, v)
            if x != v
                tgt[p] = x; p += one(E)
            end
        end

        sort!(view(tgt, ptr[v]:p - one(E)))
        aout[v] = ptr[v]
    end

    # the neighbors are sorted, so the arcs into u, in the order of their
    # tails, are matched with the arcs out of u, in order (`aout` counts)
    @inbounds for v in vertices(graph), p in ptr[v]:ptr[v + one(V)] - one(E)
        u = tgt[p]; rev[p] = aout[u]; aout[u] += one(E)
    end

    fe = FVector{W}(undef, m); fv = FVector{W}(undef, n)
    sin = FVector{Int}(undef, n); sout = FVector{Int}(undef, n)
    stamp = FVector{Int}(undef, n); snode = FVector{Int}(undef, n); mark = FVector{Int}(undef, n)
    inflow = FVector{W}(undef, t); itag = FVector{Int}(undef, t)
    fill!(fe, zero(W)); fill!(fv, zero(W)); fill!(sin, 0); fill!(sout, 0)
    fill!(stamp, 0); fill!(snode, 0); fill!(mark, 0); fill!(itag, 0)

    return ACSDFlow{V, E, W}(ptr, tgt, rev, fe, fv, E[], V[], sin, sout, FVector{E}(undef, n), aout, V[],
        stamp, snode, mark, 0, 0, V[], Bool[], Int[], Int[], Int[], Int[], inflow, itag, V[],
        zero(V), zero(V), zero(V))
end

# Mark S and find the ends of its missing edges. Returns twice the number of
# missing edges.
function acsd_missing!(fs::ACSDFlow{V, E}, order::AbstractVector{V}, S::AbstractVector{V}) where {V, E}
    fs.tag += 1; fs.stag = tag = fs.tag; k = length(S); missing = 0
    empty!(fs.ends)

    @inbounds for i in S
        v = order[i]; fs.stamp[v] = tag; fs.snode[v] = 0
    end

    @inbounds for i in S
        v = order[i]; c = 0

        for p in fs.ptr[v]:fs.ptr[v + one(V)] - one(E)
            c += fs.stamp[fs.tgt[p]] == tag
        end

        if c < k - 1
            push!(fs.ends, v); fs.snode[v] = length(fs.ends); missing += k - 1 - c
        end
    end

    return missing
end

# the capacity of vertex v for the current missing edge
@inline function acsd_cap(fs::ACSDFlow{V, E, W}, at::ACSDTree{V}, index::AbstractVector{V}, weights::AbstractVector{W}, v::V) where {V, E, W}
    (v == fs.va || v == fs.vb) && return typemax(W)

    if fs.stamp[v] == fs.stag
        s = fs.snode[v]
        return ispositive(s) && fs.inX[s] ? weights[v] : zero(W)
    end

    ispositive(fs.drop) && acsd_top!(at, at.home[index[v]]) == fs.drop && return zero(W)
    return weights[v]
end

# Is S, marked by `acsd_missing!`, safe by the connectivity test? The
# components of G - S have been marked by `acsd_components!`.
function acsd_flowsafe!(fs::ACSDFlow{V, E, W}, at::ACSDTree{V}, index::AbstractVector{V}, weights::AbstractVector{W}, order::AbstractVector{V}, S::AbstractVector{V}) where {V, E, W}
    tol = tolerance(W); ends = fs.ends; ns = length(ends)
    resize!(fs.mptr, ns + 1); resize!(fs.inX, ns); resize!(fs.udeg, ns)
    empty!(fs.mtgt); fs.mptr[1] = 1

    # the missing edges at each end
    @inbounds for a in oneto(ns)
        v = ends[a]; fs.tag += 1; tag = fs.tag

        for p in fs.ptr[v]:fs.ptr[v + one(V)] - one(E)
            fs.mark[fs.tgt[p]] = tag
        end

        for i in S
            x = order[i]
            x != v && fs.mark[x] != tag && push!(fs.mtgt, fs.snode[x])
        end

        fs.mptr[a + 1] = length(fs.mtgt) + 1
        fs.inX[a] = false; fs.udeg[a] = fs.mptr[a + 1] - fs.mptr[a]
    end

    # a light vertex cover X: greedy, by uncovered edges per weight ...
    while true
        best = 0

        @inbounds for a in oneto(ns)
            ispositive(fs.udeg[a]) || continue

            if iszero(best) || fs.udeg[a] * weights[ends[best]] > fs.udeg[best] * weights[ends[a]]
                best = a
            end
        end

        iszero(best) && break
        fs.inX[best] = true; fs.udeg[best] = 0

        @inbounds for q in fs.mptr[best]:fs.mptr[best + 1] - 1
            c = fs.mtgt[q]; fs.inX[c] || (fs.udeg[c] -= 1)
        end
    end

    # ... made minimal, heaviest first
    perm = fs.perm; resize!(perm, ns)
    copyto!(perm, oneto(ns)); sort!(perm; by = a -> weights[ends[a]], rev = true)
    r = zero(W)

    @inbounds for a in perm
        fs.inX[a] || continue
        keep = false

        for q in fs.mptr[a]:fs.mptr[a + 1] - 1
            fs.inX[fs.mtgt[q]] || (keep = true; break)
        end

        if keep
            r += weights[ends[a]]
        else
            fs.inX[a] = false
        end
    end

    # the bound at each end
    for v in ends
        acsd_bound(fs, at, index, weights, v) + tol > r || return false
    end

    # the flows
    for a in oneto(ns), q in fs.mptr[a]:fs.mptr[a + 1] - 1
        b = fs.mtgt[q]
        a < b && !acsd_flowpair!(fs, at, index, weights, ends[a], ends[b], r) && return false
    end

    return true
end

# the capacity of the neighbors of v, minus their largest part in one component
function acsd_bound(fs::ACSDFlow{V, E, W}, at::ACSDTree{V}, index::AbstractVector{V}, weights::AbstractVector{W}, v::V) where {V, E, W}
    fs.tag += 1; tag = fs.tag; empty!(fs.tops); tot = big = zero(W)

    @inbounds for p in fs.ptr[v]:fs.ptr[v + one(V)] - one(E)
        u = fs.tgt[p]; c = acsd_cap(fs, at, index, weights, u); tot += c
        fs.stamp[u] == fs.stag && continue
        top = acsd_top!(at, at.home[index[u]])

        if fs.itag[top] != tag
            fs.itag[top] = tag; fs.inflow[top] = zero(W); push!(fs.tops, top)
        end

        fs.inflow[top] += c
        big = max(big, fs.inflow[top])
    end

    return tot - big
end

# Does the missing edge {va, vb} have flow r outside of every component?
function acsd_flowpair!(fs::ACSDFlow{V, E, W}, at::ACSDTree{V}, index::AbstractVector{V}, weights::AbstractVector{W}, va::V, vb::V, r::W) where {V, E, W}
    tol = tolerance(W); fs.va = va; fs.vb = vb; fs.drop = zero(V)
    F = acsd_maxflow!(fs, at, index, weights, r + r)
    ok = F + tol > r

    # the flow entering each component from S (only the ends carry flow)
    fs.tag += 1; tag = fs.tag; empty!(fs.tops)

    @inbounds if ok
        for v in fs.ends, p in fs.ptr[v]:fs.ptr[v + one(V)] - one(E)
            ispositive(fs.fe[p]) || continue
            u = fs.tgt[p]; fs.stamp[u] == fs.stag && continue
            top = acsd_top!(at, at.home[index[u]])

            if fs.itag[top] != tag
                fs.itag[top] = tag; fs.inflow[top] = zero(W); push!(fs.tops, top)
            end

            fs.inflow[top] += fs.fe[p]
        end
    end

    acsd_flowreset!(fs)

    @inbounds for top in fs.tops
        ok || break
        F - fs.inflow[top] + tol > r && continue
        fs.drop = top
        ok = acsd_maxflow!(fs, at, index, weights, r) + tol > r
        acsd_flowreset!(fs)
    end

    fs.va = fs.vb = fs.drop = zero(V)
    return ok
end

function acsd_flowreset!(fs::ACSDFlow{V, E, W}) where {V, E, W}
    @inbounds for p in fs.tarcs
        fs.fe[p] = zero(W)
    end

    @inbounds for v in fs.tverts
        fs.fv[v] = zero(W)
    end

    empty!(fs.tarcs); empty!(fs.tverts)
    return
end

# Augment from out(va) to in(vb) along shortest paths, up to `target`.
# Returns the value of the flow.
function acsd_maxflow!(fs::ACSDFlow{V, E, W}, at::ACSDTree{V}, index::AbstractVector{V}, weights::AbstractVector{W}, target::W) where {V, E, W}
    va = fs.va; vb = fs.vb; queue = fs.queue; F = zero(W)

    @inbounds while F < target
        fs.tag += 1; tag = fs.tag
        empty!(queue); push!(queue, va); fs.sout[va] = tag; h = 1

        while h <= length(queue) && fs.sin[vb] != tag
            x = queue[h]; h += 1

            if isnegative(x)
                # in(v): to out(v), or back along an arc with flow into v
                v = -x

                if fs.sout[v] != tag && fs.fv[v] < acsd_cap(fs, at, index, weights, v)
                    fs.sout[v] = tag; fs.aout[v] = zero(E); push!(queue, v)
                end

                for p in fs.ptr[v]:fs.ptr[v + one(V)] - one(E)
                    u = fs.tgt[p]

                    if fs.fe[p] < zero(W) && fs.sout[u] != tag && u != vb
                        fs.sout[u] = tag; fs.aout[u] = fs.rev[p]; push!(queue, u)
                    end
                end
            else
                # out(v): back to in(v), or along an arc with no flow into v
                v = x

                if v != va && fs.sin[v] != tag && ispositive(fs.fv[v])
                    fs.sin[v] = tag; fs.ain[v] = zero(E); push!(queue, -v)
                end

                for p in fs.ptr[v]:fs.ptr[v + one(V)] - one(E)
                    u = fs.tgt[p]

                    if fs.fe[p] >= zero(W) && fs.sin[u] != tag && u != va && ispositive(acsd_cap(fs, at, index, weights, u))
                        fs.sin[u] = tag; fs.ain[u] = p; push!(queue, -u)
                    end
                end
            end
        end

        fs.sin[vb] == tag || break

        # the bottleneck ...
        d = target - F; x = -vb

        while x != va
            if isnegative(x)
                v = -x; p = fs.ain[v]

                if iszero(p)
                    d = min(d, fs.fv[v]); x = v
                else
                    x = fs.tgt[fs.rev[p]]
                end
            else
                v = x; p = fs.aout[v]

                if iszero(p)
                    d = min(d, acsd_cap(fs, at, index, weights, v) - fs.fv[v]); x = -v
                else
                    d = min(d, fs.fe[p]); x = -fs.tgt[p]
                end
            end
        end

        # ... and the augmentation
        x = -vb

        while x != va
            if isnegative(x)
                v = -x; p = fs.ain[v]

                if iszero(p)
                    fs.fv[v] -= d; push!(fs.tverts, v); x = v
                else
                    fs.fe[p] += d; fs.fe[fs.rev[p]] -= d; push!(fs.tarcs, p, fs.rev[p])
                    x = fs.tgt[fs.rev[p]]
                end
            else
                v = x; p = fs.aout[v]

                if iszero(p)
                    fs.fv[v] += d; push!(fs.tverts, v); x = -v
                else
                    fs.fe[p] -= d; fs.fe[fs.rev[p]] += d; push!(fs.tarcs, p, fs.rev[p])
                    x = -fs.tgt[p]
                end
            end
        end

        F += d
    end

    return F
end
