#
# Branch decompositions (contraction trees) of a pointed hypergraph, built by
# binarizing a clique tree of its line graph one clique at a time. The hypergraph
# H has one vertex per index (weight = log2 of its dimension), one hyperedge per
# tensor, and one hyperedge (the point) for the output indices. A branch tree is a
# binary tree whose leaves are the tensors and whose root is the point; `branchtree`
# returns it as a parent vector of length 2m - 2.
#
# Costs of a branch tree (`branchstats` returns all three in one pass):
#   - branchwidth: max over contractions of w(legs)               -- largest intermediate (not the inputs); the branchwidth
#   - branchpeak:  max over contractions of w(legs(l) u legs(r))  -- most expensive step; induced treewidth of the line graph
#   - branchcost:  log2 sum over contractions of 2^w(legs(l) u legs(r)) -- total flops
# These match cotengra's contraction_width, max log2 get_flops, and contraction_cost.
#
# branchrefine is an optional post-pass (subtree reconfiguration; never wider). It is
# stoppable: a BranchSolver holds the tree, branchrefine_step! runs one sweep, and
# branchrefine_solve! runs sweeps to a fixed point. branchanneal is a second post-pass:
# simulated annealing by tree rotations, as TreeSA (never wider by default). branchssa
# converts the parent vector to an SSA contraction path.
#
# Files: branchtree.jl (branchtree, branchrefine), anneal.jl (branchanneal), and this one
# (the costs, conversions and hypergraph helpers they share).
#
# A merge's binary tree is passed around as the tuple
#   (k, left, right, U, L, out, order, stack, stamp)
# over items 1:k (leaves) and internal nodes k+1:2k-1 (root 2k-1). A store of masks is an
# FVector{UInt64} (one word per mask, the fast path) or an FMatrix{UInt64} (one column of
# words per mask, for >64 vertices), the bt primitives bt_nwd/bt_get/bt_put! over it.
#
const EPS = 1e-9

include("branchtree.jl")
include("anneal.jl")

# ----------------------------------------------------------------------------------------
# costs and conversions
# ----------------------------------------------------------------------------------------

"""
    branchstats(weights, hypergraph, point, parent) -> (; width, peak, cost)

Width, contraction complexity and log2 of the total cost of any contraction tree given as
a parent vector (see the module docstring for the definitions; the width is that of the
intermediates, so an input tensor wider than all of them does not count). Computed in one pass by
the Gilbert-Ng-Peyton column-count scheme (union-find for the LCAs), with no leg sets:
`cross[x] = w(legs[x])` from deltas (+w(v) at each occurrence, -w(v) at each
consecutive-pair LCA, -w(v) at L(v) = LCA of v's occurrences), and
`tc[x] = cross[x] + closedwt[x]` where `closedwt[x] = w(indices whose LCA is x)`.
"""
function branchstats(weights::AbstractVector, hypergraph::BipartiteGraph, point::Integer, parent::AbstractVector{<:Integer})
    H = simplehypergraph(hypergraph)
    m = nv(H); n = nov(H); nn = length(parent)
    @assert nn == 2m - 2

    deg      = FVector{Int}(undef, n)
    par      = FVector{Int}(undef, nn)
    left     = FVector{Int}(undef, nn)
    right    = FVector{Int}(undef, nn)
    pre      = FVector{Int}(undef, nn)
    preidx   = FVector{Int}(undef, nn)
    stack    = FVector{Int}(undef, nn)
    uf       = FVector{Int}(undef, nn)
    prev     = FVector{Int}(undef, n)
    Lv       = FVector{Int}(undef, n)
    delta    = FVector{Float64}(undef, nn)
    closedwt = FVector{Float64}(undef, nn)

    fill!(deg, 0)
    fill!(left, 0)
    fill!(right, 0)
    fill!(preidx, 0)
    fill!(prev, 0)
    fill!(Lv, 0)
    fill!(delta, 0.0)
    fill!(closedwt, 0.0)

    @inbounds for i in arcs(H)
        deg[targets(H)[i]] += 1
    end

    @inbounds for x in oneto(nn)
        par[x] = parent[x]
    end

    @inbounds for x in oneto(nn)
        p = par[x]

        if p != 0
            if left[p] == 0
                left[p] = x
            else
                right[p] = x
            end
        end
    end

    @assert par[point] == 0 && left[point] != 0 && right[point] == 0 "the point must have exactly one child"

    # preorder from the point; preidx[x] = preorder position (ancestors come earlier)
    npre = 0; sp = 1; stack[1] = point

    @inbounds while sp > 0
        x = stack[sp]; sp -= 1
        npre += 1; pre[npre] = x; preidx[x] = npre

        if left[x] != 0
            sp += 1; stack[sp] = left[x]
        end

        if right[x] != 0
            sp += 1; stack[sp] = right[x]
        end
    end

    # union-find (each node under its parent as it finishes, so find gives the LCA node)
    @inbounds for x in oneto(nn)
        uf[x] = x
    end

    function find(x)
        @inbounds while uf[x] != x
            uf[x] = uf[uf[x]]; x = uf[x]
        end

        return x
    end

    # single pass in children-before-parents order (reverse of preorder)
    @inbounds for p in nn:-1:1
        x = pre[p]

        if x <= m
            for v in neighbors(H, x)
                if deg[v] > 1
                    delta[x] += weights[v]

                    if prev[v] != 0
                        q = find(prev[v])
                        delta[q] -= weights[v]

                        if Lv[v] == 0 || preidx[q] < preidx[Lv[v]]
                            Lv[v] = q
                        end
                    end

                    prev[v] = x
                end
            end
        end

        if par[x] != 0
            uf[x] = par[x]
        end
    end

    @inbounds for v in oneto(n)
        if deg[v] > 1 && prev[v] != 0
            delta[Lv[v]] -= weights[v]
            closedwt[Lv[v]] += weights[v]
        end
    end

    # cross[x] = subtree sum of delta (children before parents = reverse preorder)
    cross = delta

    @inbounds for p in nn:-1:1
        x = pre[p]

        if par[x] != 0
            cross[par[x]] += cross[x]
        end
    end

    width = 0.0; peak = 0.0; mx = -Inf

    @inbounds for x in m + 1:nn
        @assert left[x] != 0 && right[x] != 0 "node $x is not binary"
        tc = cross[x] + closedwt[x]
        width = max(width, cross[x])
        peak = max(peak, tc)
        mx = max(mx, tc)
    end

    if isinf(mx)
        cost = -Inf
    else
        total = 0.0

        @inbounds for x in m + 1:nn
            total += exp2((cross[x] + closedwt[x]) - mx)
        end

        cost = mx + log2(total)
    end

    return (; width, peak, cost)
end

"""
    branchssa(parent, point) -> Vector{Tuple{Int, Int}}

The contraction as a 0-based SSA path over the tensors (the hyperedges other than the
point, in order), e.g. for cotengra's `ContractionTree.from_path(..., ssa_path=...)`.
The contractions are listed in the order of their nodes, except that a node always comes
after its children: any binary tree gives a valid path, whatever its numbering (that of
`branchtree`, `branchrefine` and `branchanneal` already puts children first).
"""
function branchssa(parent::AbstractVector{<:Integer}, point::Integer)
    nn = length(parent); m = (nn + 2) ÷ 2

    id    = FVector{Int}(undef, nn)
    left  = FVector{Int}(undef, nn)
    right = FVector{Int}(undef, nn)
    stack = FVector{Int}(undef, nn)
    path  = Vector{Tuple{Int, Int}}(undef, max(nn - m, 0))

    fill!(id, -1)
    fill!(left, 0)
    fill!(right, 0)

    t = 0

    for e in oneto(m)
        if e != point
            id[e] = t; t += 1
        end
    end

    for x in oneto(nn)
        p = parent[x]

        if p != 0 && p != point
            if left[p] == 0
                left[p] = x
            else
                right[p] = x
            end
        end
    end

    # nodes in index order, each after its children (a postorder of the subtree below
    # a node that is not yet numbered)
    k = 0

    for x in m + 1:nn
        if id[x] < 0
            sp = 1; stack[1] = x

            while sp > 0
                y = stack[sp]; l = left[y]; r = right[y]

                if id[l] < 0
                    sp += 1; stack[sp] = l
                elseif id[r] < 0
                    sp += 1; stack[sp] = r
                else
                    sp -= 1; id[y] = t; t += 1
                    k += 1; path[k] = (id[l], id[r])
                end
            end
        end
    end

    return path
end

# thin accessors over branchstats (one pass each)
function branchwidth(weights, hypergraph, point, parent)
    return branchstats(weights, hypergraph, point, parent).width
end

function branchpeak(weights, hypergraph, point, parent)
    return branchstats(weights, hypergraph, point, parent).peak
end

function branchcost(weights, hypergraph, point, parent)
    return branchstats(weights, hypergraph, point, parent).cost
end

# ----------------------------------------------------------------------------------------
# hypergraphs
# ----------------------------------------------------------------------------------------

# a copy of the hypergraph without repeated vertices in a hyperedge
function simplehypergraph(H::BipartiteGraph)
    m = nv(H); n = nov(H)

    mark = FVector{Int}(undef, n)
    ptr  = FVector{Int}(undef, m + 1)
    tgt  = FVector{Int}(undef, de(H))      # overallocated: distinct ≤ total targets

    fill!(mark, 0)
    ptr[1] = 1

    @inbounds for e in oneto(m)
        c = 0

        for v in neighbors(H, e)
            if mark[v] != e
                mark[v] = e; c += 1
            end
        end

        ptr[e + 1] = ptr[e] + c
    end

    fill!(mark, 0)

    @inbounds for e in oneto(m)
        p = ptr[e]

        for v in neighbors(H, e)
            if mark[v] != e
                mark[v] = e; tgt[p] = v; p += 1
            end
        end
    end

    return BipartiteGraph{Int, Int}(n, m, ptr[m + 1] - 1, ptr, tgt)
end

# the hypergraph on the vertices v with new[v] > 0, renumbered to new[v]
function restrict(H::BipartiteGraph, new::AbstractVector{Int}, n2::Int)
    m = nv(H)

    ptr = FVector{Int}(undef, m + 1)
    tgt = FVector{Int}(undef, de(H))       # overallocated: kept ≤ total targets

    ptr[1] = 1

    @inbounds for e in oneto(m)
        c = 0

        for v in neighbors(H, e)
            if ispositive(new[v])
                c += 1
            end
        end

        ptr[e + 1] = ptr[e] + c
    end

    @inbounds for e in oneto(m)
        p = ptr[e]

        for v in neighbors(H, e)
            if ispositive(new[v])
                tgt[p] = new[v]; p += 1
            end
        end
    end

    return BipartiteGraph{Int, Int}(n2, m, ptr[m + 1] - 1, ptr, tgt)
end
