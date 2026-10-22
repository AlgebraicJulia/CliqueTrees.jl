# ----------------------------------------------------------------------------------------
# structs
# ----------------------------------------------------------------------------------------

# A stoppable solver over a whole contraction tree. The tree is (par, left, right), its
# per-node legs are a mutable node->legs BipartiteGraph kept in a flat arena (legtgt
# values, legcnt counts, legptr/leglen rows, legwt row weights) compacted each sweep; the
# caps Wcap, TCcap bound every intermediate's width and every step's tc; the DP scratch is
# reused across windows. branchrefine_step! runs one sweep, branchrefine_solve! runs
# sweeps to a fixed point.
mutable struct BranchSolver
    # hypergraph and tree
    H       ::FBipartiteGraph{Int, Int}
    par     ::FVector{Int}
    left    ::FVector{Int}
    right   ::FVector{Int}
    root    ::Int
    m       ::Int
    point   ::Int
    nn      ::Int
    # weights, degrees, caps
    wv      ::FVector{Float64}
    deg     ::FVector{Int}
    Wcap    ::Float64
    TCcap   ::Float64
    window  ::Int
    # leg arena
    legtgt  ::Vector{Int}      # flat index values
    legcnt  ::Vector{Int}      # parallel occurrence counts (valid only within compactlegs!)
    legptr  ::FVector{Int}     # row start                  (length nn)
    leglen  ::FVector{Int}     # row length                 (length nn)
    legwt   ::FVector{Float64} # row weight                 (length nn)
    acc     ::FVector{Int}     # occurrence accumulator (leg rebuild)
    # sweep bookkeeping
    nstamp  ::FVector{Int}     # when a node last changed
    solved  ::FVector{Int}     # when the window at a node was last found optimal
    clock   ::Int
    # DP scratch
    wid     ::FVector{Float64}
    bw      ::FVector{Float64}
    fl      ::FVector{Float64}
    split   ::FVector{Int}
    u1      ::FVector{UInt64}
    l1      ::FVector{UInt64}
    tab     ::FMatrix{Float64}
    etab    ::FMatrix{Float64}
    wloc    ::FVector{Float64}
    ploc    ::FVector{Float64}
    localid ::FVector{Int}
    vstamp  ::FVector{Int}     # dedup the vertices of a window's pieces / a leg merge
    touched ::FVector{Int}     # distinct vertices of the current window / merge
    tick    ::Int
end

# ----------------------------------------------------------------------------------------
# constructors
# ----------------------------------------------------------------------------------------

function BranchSolver(
        weights::AbstractVector,
        hypergraph::BipartiteGraph,
        point::Integer,
        parent::AbstractVector{<:Integer};
        window::Integer = 8,
        W = nothing,
        TC = nothing,
    )
    H = simplehypergraph(hypergraph)
    m = nv(H); n = nov(H); nn = length(parent)
    @assert nn == 2m - 2 && parent[point] == 0
    window = clamp(window, 3, 16)
    s = 1 << window; ncol = max(cld(n, 8), 1)

    # all arrays up front (sizes known from the arguments)
    par     = FVector{Int}(undef, nn)
    left    = FVector{Int}(undef, nn)
    right   = FVector{Int}(undef, nn)
    wv      = FVector{Float64}(undef, n)
    deg     = FVector{Int}(undef, n)
    legtgt  = Int[]
    legcnt  = Int[]
    legptr  = FVector{Int}(undef, nn)
    leglen  = FVector{Int}(undef, nn)
    legwt   = FVector{Float64}(undef, nn)
    acc     = FVector{Int}(undef, n)
    nstamp  = FVector{Int}(undef, nn)
    solved  = FVector{Int}(undef, nn)
    wid     = FVector{Float64}(undef, s)
    bw      = FVector{Float64}(undef, s)
    fl      = FVector{Float64}(undef, s)
    split   = FVector{Int}(undef, s)
    u1      = FVector{UInt64}(undef, s)
    l1      = FVector{UInt64}(undef, s)
    tab     = FMatrix{Float64}(undef, 256, ncol)
    etab    = FMatrix{Float64}(undef, 256, ncol)
    wloc    = FVector{Float64}(undef, n)
    ploc    = FVector{Float64}(undef, n)
    localid = FVector{Int}(undef, n)
    vstamp  = FVector{Int}(undef, n)
    touched = FVector{Int}(undef, n)

    fill!(deg, 0)
    fill!(left, 0)
    fill!(right, 0)
    fill!(acc, 0)
    fill!(nstamp, 0)
    fill!(solved, 0)
    fill!(localid, 0)
    fill!(vstamp, 0)

    @inbounds for i in arcs(H)
        deg[targets(H)[i]] += 1
    end

    root = 0

    @inbounds for x in oneto(nn)
        par[x] = parent[x]
    end

    @inbounds for x in oneto(nn)
        p = par[x]

        if p != 0
            if p == point
                root = x
            elseif left[p] == 0
                left[p] = x
            else
                right[p] = x
            end
        end
    end

    @inbounds for v in oneto(n)
        wv[v] = weights[v]
    end

    sol = BranchSolver(
        H, par, left, right, root, m, Int(point), nn,
        wv, deg, 0.0, 0.0, window,
        legtgt, legcnt, legptr, leglen, legwt, acc,
        nstamp, solved, 0,
        wid, bw, fl, split, u1, l1, tab, etab, wloc, ploc, localid, vstamp, touched, 0,
    )

    compactlegs!(sol)      # initial legs

    # caps (default: the tree's own width / tc, so neither can grow)
    if W === nothing
        Wcap = 0.0

        for x in m + 1:nn
            if x != point
                Wcap = max(Wcap, sol.legwt[x])
            end
        end
    else
        Wcap = Float64(W)
    end

    sol.Wcap = Wcap

    if TC === nothing
        tcmax = 0.0

        for x in m + 1:nn
            if x != point
                sol.tick += 1; tk = sol.tick; tcx = 0.0

                for c in (left[x], right[x])
                    for i in sol.legptr[c]:(sol.legptr[c] + sol.leglen[c] - 1)
                        v = sol.legtgt[i]

                        if sol.vstamp[v] != tk
                            sol.vstamp[v] = tk; tcx += wv[v]
                        end
                    end
                end

                tcmax = max(tcmax, tcx)
            end
        end

        TCcap = tcmax
    else
        TCcap = Float64(TC)
    end

    sol.TCcap = TCcap

    return sol
end

# ----------------------------------------------------------------------------------------
# entry points
# ----------------------------------------------------------------------------------------

"""
    branchtree(weights, hypergraph, point;
        alg=DEFAULT_ELIMINATION_ALGORITHM, snd=DEFAULT_SUPERNODE_TYPE, kexact=10, window=8)

Contraction tree of a pointed hypergraph, as a parent vector of length 2m - 2 (see the
module docstring).

  - `weights`: vertex weights (log2 of the index dimensions), length n. Vertices of weight
    0 (dimension 1) cost nothing and are dropped before ordering.
  - `hypergraph`: a `BipartiteGraph` with `nv(hypergraph) == m` hyperedges (the tensors
    and the point) and `nov(hypergraph) == n` vertices (the indices):
    `neighbors(hypergraph, e)` lists the vertices of hyperedge e
  - `point`: the hyperedge of the output indices (it may have no vertices)
  - `alg`: elimination algorithm, or permutation of 1:n, for the line graph
  - `snd`: supernode type of the clique tree
  - `kexact`: merges of at most this many items are binarized exactly (≈ 3^k work)
  - `window`: larger merges are refined by re-solving subtrees of up to this many pieces
"""
function branchtree(
        weights::AbstractVector,
        hypergraph::BipartiteGraph,
        point::Integer;
        alg = DEFAULT_ELIMINATION_ALGORITHM,
        snd = DEFAULT_SUPERNODE_TYPE,
        kexact::Integer = 10,
        window::Integer = 8,
    )
    H = simplehypergraph(hypergraph)
    n = nov(H)
    @assert length(weights) == n
    @assert all(>=(0), weights) "weights must be nonnegative"

    anyzero = false

    @inbounds for v in oneto(n)
        if iszero(weights[v])
            anyzero = true
        end
    end

    if anyzero       # drop dimension-1 indices (they cost nothing)
        new = FVector{Int}(undef, n)
        fill!(new, 0)
        nkeep = 0

        @inbounds for v in oneto(n)
            if !iszero(weights[v])
                nkeep += 1; new[v] = nkeep
            end
        end

        keptw = FVector{Float64}(undef, nkeep)

        @inbounds for v in oneto(n)
            if !iszero(new[v])
                keptw[new[v]] = weights[v]
            end
        end

        H = restrict(H, new, nkeep)
        weights = keptw

        if alg isa AbstractVector
            keptalg = FVector{Int}(undef, length(alg))
            nkept = 0

            for v in alg
                if new[v] > 0
                    nkept += 1; keptalg[nkept] = new[v]
                end
            end

            alg = view(keptalg, oneto(nkept))
        end
    end

    if iszero(nov(H))
        return scalartree(nv(H), point)
    end

    label, tree = cliquetree(weights, linegraph(reverse(H), H); alg, snd)

    return branchtree(weights, H, point, label, tree; kexact, window)
end

"""
    branchtree(weights, hypergraph, point, label, tree; kexact=10, window=8)

As above, from a clique tree `(label, tree)` of the line graph `linegraph(reverse(H), H)`,
e.g. from `cliquetree`.
"""
function branchtree(
        weights::AbstractVector,
        hypergraph::BipartiteGraph,
        point::Integer,
        label::AbstractVector,
        tree::CliqueTree;
        kexact::Integer = 10,
        window::Integer = 8,
    )
    H = simplehypergraph(hypergraph)
    m = nv(H); n = nov(H)
    @assert 1 <= point <= m
    @assert m >= 2 "a contraction needs at least one tensor besides the point"
    @assert 2 <= kexact <= 20
    @assert length(weights) == n
    window = min(max(window, 3), kexact)

    if length(tree) == 0
        return scalartree(m, point)
    end

    Hr = reverse(H)
    label = collect(Int, label)

    # re-root at the clique holding the point's vertices
    pv = collect(neighbors(H, point))

    if !isempty(pv)
        invl = invperm(label)
        perm, tree = cliquetree(tree, invl[pv])
        label = label[perm]
    end

    nc, root, chg, asg, scalarroots, free = cliques(tree, label, H, Hr, point, n, m)

    # upper bound on the items of any merge (children + tensors of a clique)
    kmax = 1

    @inbounds for i in oneto(nc)
        kmax = max(kmax, length(neighbors(chg, i)) + length(neighbors(asg, i)))
    end

    # all scratch up front (sizes from n, nc, kexact, kmax; tab/etab bounded by bmax ≤ n)
    s = 1 << kexact
    acc     = FVector{Int}(undef, n)
    stamp   = FVector{Int}(undef, n)
    localid = FVector{Int}(undef, n)
    deg     = FVector{Int}(undef, n)
    wv      = FVector{Float64}(undef, n)
    wloc    = FVector{Float64}(undef, n)
    ploc    = FVector{Float64}(undef, n)
    parent  = FVector{Int}(undef, 2m - 2)
    rs      = FVector{Int}(undef, nc)      # result legs of clique i: rv[rs[i]:re[i]-1]
    re      = FVector{Int}(undef, nc)
    nitems  = FVector{Int}(undef, nc)
    tw      = FVector{Float64}(undef, nc)
    resnode = FVector{Int}(undef, nc)
    big     = FVector{Int}(undef, nc)      # the k ≥ 3 cliques, heaviest first (pass 1b)
    cache   = Vector{Any}(undef, nc)
    wid     = FVector{Float64}(undef, s)
    bw      = FVector{Float64}(undef, s)
    fl      = FVector{Float64}(undef, s)
    split   = FVector{Int}(undef, s)
    u1      = FVector{UInt64}(undef, s)
    l1      = FVector{UInt64}(undef, s)
    tab     = FMatrix{Float64}(undef, 256, max(cld(n, 8), 1))
    etab    = FMatrix{Float64}(undef, 256, max(cld(n, 8), 1))
    P       = FMatrix{Float64}(undef, kmax, kmax)
    Q       = FMatrix{Float64}(undef, kmax, kmax)
    touched = FVector{Int}(undef, n)
    rv      = Int[]
    rc      = Int[]

    fill!(acc, 0)
    fill!(stamp, 0)
    fill!(localid, 0)
    fill!(deg, 0)
    fill!(parent, 0)
    fill!(rs, 0)
    fill!(re, 0)
    fill!(nitems, 0)
    fill!(tw, 0.0)
    fill!(resnode, 0)
    fill!(cache, nothing)
    tick = 0
    next = m + 1

    @inbounds for v in oneto(n)
        wv[v] = weights[v]; deg[v] = length(neighbors(Hr, v))
    end

    # pass 1a, bottom-up: the legs of every clique's merge; W starts at the widest result
    W = 0.0

    @inbounds for i in oneto(nc)
        tick += 1
        k, nt = gather!(chg, asg, i, H, rs, re, rv, rc, acc, stamp, deg, touched, tick)
        nitems[i] = k

        if k != 0
            tv = view(touched, oneto(nt))
            rs[i] = length(rv) + 1; sacc = 0.0

            for v in tv
                # a single item passes up unchanged
                if k == 1 || acc[v] < deg[v]
                    push!(rv, v); push!(rc, acc[v]); sacc += wv[v]
                end
            end

            re[i] = length(rv) + 1

            if k >= 2
                W = max(W, sacc)
            end

            tw[i] = touchedweight(wv, tv)
        end
    end

    # pass 1b: the smallest width of each merge of k ≥ 3 items, heaviest first; a merge
    # whose vertices weigh at most W in total cannot raise W
    nbig = 0

    for i in oneto(nc)
        if nitems[i] >= 3
            nbig += 1; big[nbig] = i
        end
    end

    bigv = view(big, oneto(nbig))
    sort!(bigv; by = i -> tw[i], rev = true)

    @inbounds for i in bigv
        if tw[i] <= W + EPS
            break
        end

        tick += 1
        _, nt = gather!(chg, asg, i, H, rs, re, rv, rc, acc, stamp, deg, touched, tick)
        tv = view(touched, oneto(nt))
        wi = clique_width(storetype(nt), chg, asg, i, nitems[i], kexact, window, H, rs, re, rv, resnode, cache,
            localid, tv, wv, acc, deg, wloc, ploc, tab, etab, wid, bw, fl, split, u1, l1, P, Q)
        W = max(W, wi)
    end

    # pass 2, bottom-up: the cheapest binarization of each clique within W
    @inbounds for i in oneto(nc)
        k = nitems[i]

        if k != 0
            tick += 1
            _, nt = gather!(chg, asg, i, H, rs, re, rv, rc, acc, stamp, deg, touched, tick)

            if k == 1
                resnode[i] = firstitem(chg, asg, i, rs, resnode)
            elseif k == 2
                a, c = twoitems(chg, asg, i, rs, resnode)
                x = next; next += 1; parent[a] = x; parent[c] = x; resnode[i] = x
            else
                tv = view(touched, oneto(nt))
                resnode[i], next = clique_build(storetype(nt), chg, asg, i, k, kexact, window, W, H, rs, re, rv, resnode, cache, parent, next,
                    localid, tv, wv, acc, deg, wloc, ploc, tab, etab, wid, bw, fl, split, u1, l1, P, Q)
            end
        end
    end

    # the scalars (other components, hyperedges without vertices) are multiplied together,
    # then into the root's result
    if nitems[root] >= 1
        top = resnode[root]
    else
        top = 0
    end

    sc = 0

    for c in scalarroots
        if nitems[c] >= 1
            t = resnode[c]

            if sc == 0
                sc = t
            else
                x = next; next += 1; parent[sc] = x; parent[t] = x; sc = x
            end
        end
    end

    for t in free
        if sc == 0
            sc = t
        else
            x = next; next += 1; parent[sc] = x; parent[t] = x; sc = x
        end
    end

    if top != 0 && sc != 0
        x = next; next += 1; parent[top] = x; parent[sc] = x; top = x
    elseif top == 0
        top = sc
    end

    @assert top != 0 "nothing to contract"
    parent[top] = point
    parent[point] = 0
    @assert next == 2m - 1 "expected $(m - 2) contractions, made $(next - m - 1)"

    return parent
end

"""
    branchrefine(weights, hypergraph, point, parent; window=8, maxsweeps=8, widthfirst=false,
        W=nothing, TC=nothing) -> parent

Subtree reconfiguration over a whole contraction tree, ignoring clique boundaries: for
every contraction x (top-down) its subtree is cut into at most `window` pieces (the widest
piece is split first) and the merge of the pieces is re-solved exactly, with every
intermediate of width ≤ W and every step of tc ≤ TC (by default the tree's own width and
tc, so neither can grow). A window is replaced when it gets cheaper. With
`widthfirst = true`, a first round replaces windows that get narrower (within TC), and the
cost round then keeps the new width. Returns a new parent vector (same convention as
`branchtree`). A window only needs the vertices of its pieces' legs: a vertex summed
inside a piece occurs nowhere else.
"""
function branchrefine(
        weights::AbstractVector,
        hypergraph::BipartiteGraph,
        point::Integer,
        parent::AbstractVector{<:Integer};
        window::Integer = 8,
        maxsweeps::Integer = 8,
        widthfirst::Bool = false,
        W = nothing,
        TC = nothing,
    )
    if nv(hypergraph) <= 3
        return collect(Int, parent)
    end

    s = BranchSolver(weights, hypergraph, point, parent; window, W, TC)

    if widthfirst
        branchrefine_solve!(s, true; maxsweeps)
        compactlegs!(s)
        Wcap = 0.0

        for x in s.m + 1:s.nn
            if x != s.point
                Wcap = max(Wcap, s.legwt[x])
            end
        end

        s.Wcap = Wcap
    end

    branchrefine_solve!(s, false; maxsweeps)

    return branchrefine_parent(s)
end

# sweeps of one mode to a fixed point (or `maxsweeps`); the arena is compacted each sweep.
# widthmode = true: shrink width; widthmode = false: cut flops within the width cap.
function branchrefine_solve!(s::BranchSolver, widthmode::Bool; maxsweeps::Integer = 8)
    fill!(s.solved, 0)

    for _ in 1:maxsweeps
        compactlegs!(s)

        if !branchrefine_step!(s, widthmode)
            break
        end
    end

    return s
end

# one top-down sweep; returns whether any window improved
function branchrefine_step!(s::BranchSolver, widthmode::Bool)
    improved, s.tick, s.clock = branchrefine_step_impl!(
        s.par, s.left, s.right, s.legtgt, s.legptr, s.leglen, s.legwt,
        s.touched, s.localid, s.vstamp, s.wv, s.wloc, s.ploc, s.tab, s.etab,
        s.wid, s.bw, s.fl, s.split, s.u1, s.l1, s.solved, s.nstamp,
        s.Wcap, s.TCcap, s.m, s.window, s.nn, s.root, widthmode, s.tick, s.clock)
    return improved
end

function branchrefine_step_impl!(
        par::AbstractVector{Int},
        left::AbstractVector{Int},
        right::AbstractVector{Int},
        legtgt::AbstractVector{Int},
        legptr::AbstractVector{Int},
        leglen::AbstractVector{Int},
        legwt::AbstractVector{Float64},
        touched::AbstractVector{Int},
        localid::AbstractVector{Int},
        vstamp::AbstractVector{Int},
        wv::AbstractVector{Float64},
        wloc::AbstractVector{Float64},
        ploc::AbstractVector{Float64},
        tab::AbstractMatrix{Float64},
        etab::AbstractMatrix{Float64},
        wid::AbstractVector{Float64},
        bw::AbstractVector{Float64},
        fl::AbstractVector{Float64},
        split::AbstractVector{Int},
        u1::AbstractVector{UInt64},
        l1::AbstractVector{UInt64},
        solved::AbstractVector{Int},
        nstamp::AbstractVector{Int},
        Wcap::Float64,
        TCcap::Float64,
        m::Int,
        window::Int,
        nn::Int,
        root::Int,
        widthmode::Bool,
        tick::Int,
        clock::Int,
    )
    order  = FVector{Int}(undef, nn)
    stack  = FVector{Int}(undef, nn)
    pieces = FVector{Int}(undef, window)
    win    = FVector{Int}(undef, window)
    norder = 0; nstack = 0

    nstack += 1; stack[nstack] = root

    while nstack > 0                            # preorder: top-down
        x = stack[nstack]; nstack -= 1

        if x > m
            norder += 1; order[norder] = x
            nstack += 1; stack[nstack] = right[x]
            nstack += 1; stack[nstack] = left[x]
        end
    end

    improved = false

    for oi in oneto(norder)
        x = order[oi]
        npieces = 0; nwin = 0
        npieces += 1; pieces[npieces] = left[x]
        npieces += 1; pieces[npieces] = right[x]
        nwin += 1; win[nwin] = x

        while npieces < window
            best = 0; bv = -Inf

            for j in oneto(npieces)
                p = pieces[j]

                if p > m
                    v = legwt[p]

                    if v > bv
                        bv = v; best = j
                    end
                end
            end

            if best == 0
                break
            end

            p = pieces[best]; pieces[best] = left[p]
            npieces += 1; pieces[npieces] = right[p]
            nwin += 1; win[nwin] = p
        end

        if npieces > 2
            stale = false

            if solved[x] > 0
                newest = 0

                for j in oneto(nwin)
                    newest = max(newest, nstamp[win[j]])
                end

                for j in oneto(npieces)
                    newest = max(newest, nstamp[pieces[j]])
                end

                stale = newest < solved[x]
            end

            if !stale
                # local bits: the vertices of the pieces' legs
                tick += 1; tk = tick; nt = 0

                for j in oneto(npieces)
                    p = pieces[j]

                    for i in legptr[p]:(legptr[p] + leglen[p] - 1)
                        v = legtgt[i]

                        if vstamp[v] != tk
                            vstamp[v] = tk; nt += 1; touched[nt] = v
                        end
                    end
                end

                pv = view(pieces, oneto(npieces))
                wnv = view(win, oneto(nwin))

                if solve_window_impl!(storetype(nt), par, left, right, legtgt, legptr, leglen, legwt,
                        touched, localid, wv, wloc, ploc, tab, etab, wid, bw, fl, split,
                        u1, l1, Wcap, TCcap, x, pv, wnv, widthmode, nt)
                    clock += 1

                    for j in oneto(nwin)
                        nstamp[win[j]] = clock
                    end

                    improved = true
                else
                    clock += 1; solved[x] = clock
                end
            end
        end
    end

    return improved, tick, clock
end

# ----------------------------------------------------------------------------------------
# cliques
# ----------------------------------------------------------------------------------------

# Children and assigned hyperedges of each clique, as two BipartiteGraphs (chg maps a
# clique to its children, asg to its tensors). The roots of the other components of a
# forest (their results are scalars) and the hyperedges without vertices are multiplied in
# at the end, outside the root's merge. The point is assigned nowhere: the root's merge
# is the output.
function cliques(tree::CliqueTree, label::AbstractVector{Int}, H::BipartiteGraph, Hr::BipartiteGraph, point::Int, n::Int, m::Int)
    nc = length(tree)

    prnt        = FVector{Int}(undef, nc)
    cliqueof    = FVector{Int}(undef, n)
    scalarroots = FVector{Int}(undef, nc)      # overallocated: the scalar roots ⊆ cliques
    chptr       = FVector{Int}(undef, nc + 1)
    ch          = FVector{Int}(undef, nc)      # overallocated: |children| ≤ nc
    owner       = FVector{Int}(undef, m)
    free        = FVector{Int}(undef, m)       # overallocated: |free| ≤ m
    asptr       = FVector{Int}(undef, nc + 1)
    as          = FVector{Int}(undef, m)       # overallocated: |assigned| ≤ m
    fillp       = FVector{Int}(undef, nc + 1)  # reused fill cursor (chptr, then asptr)

    fill!(prnt, 0)
    fill!(cliqueof, 0)
    fill!(owner, 0)

    @inbounds for i in oneto(nc)
        p = parentindex(tree, i)

        if isnothing(p)
            prnt[i] = 0
        else
            prnt[i] = p
        end
    end

    @inbounds for i in oneto(nc)
        for q in residual(tree, i)
            cliqueof[label[q]] = i
        end
    end

    pv = neighbors(H, point)

    if isempty(pv)
        root = 0

        for i in oneto(nc)
            if iszero(prnt[i])
                root = i
            end
        end
    else
        root = cliqueof[first(pv)]
    end

    @assert prnt[root] == 0 "the point's clique is not a root"
    @assert root == nc "the root clique must come last in postorder"

    nscalar = 0

    @inbounds for i in oneto(nc)
        if prnt[i] == 0 && i != root
            nscalar += 1; scalarroots[nscalar] = i
        end

        if prnt[i] != 0
            @assert prnt[i] > i "the clique tree is not in postorder"
        end
    end

    chptr[1] = 1

    @inbounds for i in oneto(nc)
        chptr[i + 1] = 0
    end

    @inbounds for i in oneto(nc)
        if prnt[i] != 0
            chptr[prnt[i] + 1] += 1
        end
    end

    @inbounds for i in oneto(nc)
        chptr[i + 1] += chptr[i]
    end

    copyto!(fillp, chptr)

    @inbounds for i in oneto(nc)
        p = prnt[i]

        if p != 0
            ch[fillp[p]] = i; fillp[p] += 1
        end
    end

    # assignment: the first clique (in postorder) whose residual holds a vertex of e
    @inbounds for i in oneto(nc)
        for q in residual(tree, i)
            v = label[q]

            for e in neighbors(Hr, v)
                if e != point && owner[e] == 0
                    owner[e] = i
                end
            end
        end
    end

    nfree = 0

    @inbounds for e in oneto(m)
        if e != point && owner[e] == 0
            nfree += 1; free[nfree] = e
        end
    end

    asptr[1] = 1

    @inbounds for i in oneto(nc)
        asptr[i + 1] = 0
    end

    @inbounds for e in oneto(m)
        if owner[e] != 0
            asptr[owner[e] + 1] += 1
        end
    end

    @inbounds for i in oneto(nc)
        asptr[i + 1] += asptr[i]
    end

    copyto!(fillp, asptr)

    @inbounds for e in oneto(m)
        i = owner[e]

        if i != 0
            as[fillp[i]] = e; fillp[i] += 1
        end
    end

    # the two CSRs as graphs: chg clique -> children, asg clique -> tensors
    chg = BipartiteGraph{Int, Int}(nc, nc, chptr[nc + 1] - 1, chptr, ch)
    asg = BipartiteGraph{Int, Int}(m, nc, asptr[nc + 1] - 1, asptr, as)

    return nc, root, chg, asg, view(scalarroots, oneto(nscalar)), view(free, oneto(nfree))
end

# Legs of the items of clique i, with occurrence counts in acc and the distinct vertices in
# touched; returns the number of items. Items: the children with a result (legs
# rv[rs[c]:re[c]-1], counts rc) and the assigned hyperedges (their vertices held by another
# hyperedge too, count 1). chg maps a clique to its children, asg to its tensors.
function gather!(
        chg::BipartiteGraph,
        asg::BipartiteGraph,
        i::Int,
        H::BipartiteGraph,
        rs::AbstractVector{Int},
        re::AbstractVector{Int},
        rv::AbstractVector{Int},
        rc::AbstractVector{Int},
        acc::AbstractVector{Int},
        stamp::AbstractVector{Int},
        deg::AbstractVector{Int},
        touched::AbstractVector{Int},
        tk::Int,
    )
    k = 0; nt = 0

    @inbounds for c in neighbors(chg, i)
        if rs[c] != 0
            k += 1

            for q in rs[c]:re[c] - 1
                v = rv[q]

                if stamp[v] != tk
                    stamp[v] = tk; acc[v] = 0; nt += 1; touched[nt] = v
                end

                acc[v] += rc[q]
            end
        end
    end

    @inbounds for e in neighbors(asg, i)
        k += 1

        for v in neighbors(H, e)
            if deg[v] != 1         # deg 1: summed inside the tensor (cotengra: preprocessing)
                if stamp[v] != tk
                    stamp[v] = tk; acc[v] = 0; nt += 1; touched[nt] = v
                end

                acc[v] += 1
            end
        end
    end

    return k, nt
end

function touchedweight(wv::AbstractVector{Float64}, touched::AbstractVector{Int})
    wt = 0.0

    @inbounds for v in touched
        wt += wv[v]
    end

    return wt
end

# mask store type for a merge of `nt` vertices: one word if at most 64
function storetype(nt::Int)
    if nt <= 64
        return FVector{UInt64}
    else
        return FMatrix{UInt64}
    end
end

# local masks of the items of clique i (after gather!): M (k columns), the merge's legs E
# (1 column), the global ids of the items, and the byte tables of the local weights
function localmasks(
        ::Type{S},
        chg::BipartiteGraph,
        asg::BipartiteGraph,
        i::Int,
        k::Int,
        H::BipartiteGraph,
        rs::AbstractVector{Int},
        re::AbstractVector{Int},
        rv::AbstractVector{Int},
        resnode::AbstractVector{Int},
        localid::AbstractVector{Int},
        touched::AbstractVector{Int},
        wv::AbstractVector{Float64},
        acc::AbstractVector{Int},
        deg::AbstractVector{Int},
        wloc::AbstractVector{Float64},
        ploc::AbstractVector{Float64},
        tab::AbstractMatrix{Float64},
        etab::AbstractMatrix{Float64},
    ) where {S <: AbstractVecOrMat{UInt64}}
    b = length(touched); nw = max(cld(b, 64), 1)

    @inbounds for (j, v) in enumerate(touched)
        localid[v] = j - 1; wloc[j] = wv[v]
    end

    nbytes, shift = buildtables!(tab, etab, ploc, wloc, b)

    M   = newstore(S, nw, k)
    E   = newstore(S, nw, 1)
    ids = FVector{Int}(undef, k)

    fill!(M, zero(UInt64))      # msetbit! ORs bits into zeroed words
    fill!(E, zero(UInt64))

    j = 0

    @inbounds for c in neighbors(chg, i)
        if rs[c] != 0
            j += 1

            for q in rs[c]:re[c] - 1
                msetbit!(M, j, localid[rv[q]])
            end

            ids[j] = resnode[c]
        end
    end

    @inbounds for e in neighbors(asg, i)
        j += 1

        for v in neighbors(H, e)
            if deg[v] != 1
                msetbit!(M, j, localid[v])
            end
        end

        ids[j] = e
    end

    @inbounds for v in touched
        if acc[v] < deg[v]
            msetbit!(E, 1, localid[v])
        end
    end

    return M, E, ids, b, nbytes, shift
end

# pass 1: the smallest width of clique i's merge (large merges keep their tree in cache)
function clique_width(
        ::Type{S},
        chg::BipartiteGraph,
        asg::BipartiteGraph,
        i::Int,
        k::Int,
        kexact::Int,
        window::Int,
        H::BipartiteGraph,
        rs::AbstractVector{Int},
        re::AbstractVector{Int},
        rv::AbstractVector{Int},
        resnode::AbstractVector{Int},
        cache::AbstractVector,
        localid::AbstractVector{Int},
        touched::AbstractVector{Int},
        wv::AbstractVector{Float64},
        acc::AbstractVector{Int},
        deg::AbstractVector{Int},
        wloc::AbstractVector{Float64},
        ploc::AbstractVector{Float64},
        tab::AbstractMatrix{Float64},
        etab::AbstractMatrix{Float64},
        wid::AbstractVector{Float64},
        bw::AbstractVector{Float64},
        fl::AbstractVector{Float64},
        split::AbstractVector{Int},
        u1::AbstractVector{UInt64},
        l1::AbstractVector{UInt64},
        P::AbstractMatrix{Float64},
        Q::AbstractMatrix{Float64},
    ) where {S <: AbstractVecOrMat{UInt64}}
    M, E, ids, b, nbytes, shift = localmasks(S, chg, asg, i, k, H, rs, re, rv, resnode, localid, touched, wv, acc, deg, wloc, ploc, tab, etab)

    if k <= kexact
        if S <: AbstractVector
            U = u1
            L = l1
        else
            U = newstore(S, bt_nwd(M), 1 << k)
            L = newstore(S, bt_nwd(M), 1 << k)
        end

        result = width_dp!(wid, bw, U, L, tab, etab, nbytes, shift, M, M, E, 1, k)
    else
        t = widthtree(S, tab, etab, nbytes, shift, P, Q, wid, bw, fl, split, M, E, k, b, window)
        cache[i] = t
        result = maxwidth(tab, nbytes, t)
    end

    return result
end

# pass 2: the cheapest binarization of clique i's merge within width W
function clique_build(
        ::Type{S},
        chg::BipartiteGraph,
        asg::BipartiteGraph,
        i::Int,
        k::Int,
        kexact::Int,
        window::Int,
        W::Float64,
        H::BipartiteGraph,
        rs::AbstractVector{Int},
        re::AbstractVector{Int},
        rv::AbstractVector{Int},
        resnode::AbstractVector{Int},
        cache::AbstractVector,
        parent::AbstractVector{Int},
        next::Int,
        localid::AbstractVector{Int},
        touched::AbstractVector{Int},
        wv::AbstractVector{Float64},
        acc::AbstractVector{Int},
        deg::AbstractVector{Int},
        wloc::AbstractVector{Float64},
        ploc::AbstractVector{Float64},
        tab::AbstractMatrix{Float64},
        etab::AbstractMatrix{Float64},
        wid::AbstractVector{Float64},
        bw::AbstractVector{Float64},
        fl::AbstractVector{Float64},
        split::AbstractVector{Int},
        u1::AbstractVector{UInt64},
        l1::AbstractVector{UInt64},
        P::AbstractMatrix{Float64},
        Q::AbstractMatrix{Float64},
    ) where {S <: AbstractVecOrMat{UInt64}}
    M, E, ids, b, nbytes, shift = localmasks(S, chg, asg, i, k, H, rs, re, rv, resnode, localid, touched, wv, acc, deg, wloc, ploc, tab, etab)

    if k <= kexact
        if S <: AbstractVector
            U = u1
            L = l1
        else
            U = newstore(S, bt_nwd(M), 1 << k)
            L = newstore(S, bt_nwd(M), 1 << k)
        end

        c = flops_dp!(wid, fl, split, U, L, tab, etab, nbytes, shift, M, M, E, 1, k, W)
        @assert c < Inf "no binarization within width $W"
        result = emit_dp!(parent, next, split, (1 << k) - 1, ids)
    else
        # starts: the capped greedy, and (if pass 1 built it) the narrowest tree, which
        # fits in W by construction; each refined for flops within W; keep the cheaper
        best = nothing

        if cache[i] !== nothing
            best = reconfigure!(cache[i], M, E, 1, W, false, window, wid, bw, fl, split, tab, etab, nbytes, shift)
            @assert maxwidth(tab, nbytes, best) <= W + EPS
        end

        t = greedy_tree(S, tab, etab, nbytes, shift, P, Q, M, E, 1, k, b, W, true)

        if maxwidth(tab, nbytes, t) <= W + EPS
            reconfigure!(t, M, E, 1, W, false, window, wid, bw, fl, split, tab, etab, nbytes, shift)

            if best === nothing || treecost(tab, etab, nbytes, shift, t) < treecost(tab, etab, nbytes, shift, best)
                best = t
            end
        end

        @assert best !== nothing "no binarization within width $W"
        result = emit_local!(parent, next, best, ids)
    end

    return result
end

# the narrowest tree the heuristics find for a large merge
function widthtree(
        ::Type{S},
        tab::AbstractMatrix{Float64},
        etab::AbstractMatrix{Float64},
        nbytes::Int,
        shift::Float64,
        P::AbstractMatrix{Float64},
        Q::AbstractMatrix{Float64},
        wid::AbstractVector{Float64},
        bw::AbstractVector{Float64},
        fl::AbstractVector{Float64},
        split::AbstractVector{Int},
        M::S,
        E::S,
        k::Int,
        b::Int,
        window::Int,
    ) where {S <: AbstractVecOrMat{UInt64}}
    t = greedy_tree(S, tab, etab, nbytes, shift, P, Q, M, E, 1, k, b, Inf, false)
    return reconfigure!(t, M, E, 1, Inf, true, window, wid, bw, fl, split, tab, etab, nbytes, shift)
end

function firstitem(chg::BipartiteGraph, asg::BipartiteGraph, i::Int, rs::AbstractVector{Int}, resnode::AbstractVector{Int})
    @inbounds for c in neighbors(chg, i)
        if rs[c] != 0
            return resnode[c]
        end
    end

    return @inbounds first(neighbors(asg, i))
end

function twoitems(chg::BipartiteGraph, asg::BipartiteGraph, i::Int, rs::AbstractVector{Int}, resnode::AbstractVector{Int})
    a = 0

    @inbounds for c in neighbors(chg, i)
        if rs[c] != 0
            if a == 0
                a = resnode[c]
            else
                return a, resnode[c]
            end
        end
    end

    @inbounds for e in neighbors(asg, i)
        if a == 0
            a = e
        else
            return a, e
        end
    end

    error("clique $i has fewer than two items")
end

# every tensor is a scalar: multiply them in a chain
function scalartree(m::Int, point::Int)
    @assert m >= 2 "a contraction needs at least one tensor besides the point"
    parent = FVector{Int}(undef, 2m - 2)
    fill!(parent, 0)
    next = m + 1; s = 0

    for e in oneto(m)
        if e != point
            if s == 0
                s = e
            else
                parent[s] = next; parent[e] = next; s = next; next += 1
            end
        end
    end

    parent[s] = point

    return parent
end

# ----------------------------------------------------------------------------------------
# refinement helpers: the leg arena, a window re-solve, and the result
# ----------------------------------------------------------------------------------------

# leaf x's legs: the hyperedge's indices with deg > 1, counts 1
function leafrow!(s::BranchSolver, x::Int)
    return leafrow_impl!(s.legtgt, s.legcnt, s.legptr, s.leglen, s.legwt, s.H, s.deg, s.wv, x)
end

function leafrow_impl!(
        legtgt::AbstractVector{Int},
        legcnt::AbstractVector{Int},
        legptr::AbstractVector{Int},
        leglen::AbstractVector{Int},
        legwt::AbstractVector{Float64},
        H::BipartiteGraph,
        deg::AbstractVector{Int},
        wv::AbstractVector{Float64},
        x::Int,
    )
    start = length(legtgt) + 1
    wt = 0.0

    @inbounds for v in neighbors(H, x)
        if deg[v] > 1
            push!(legtgt, v); push!(legcnt, 1); wt += wv[v]
        end
    end

    legptr[x] = start
    leglen[x] = length(legtgt) - start + 1
    legwt[x] = wt
    return
end

# internal x's legs: merge children l, r; keep indices still open (acc < deg)
function mergerow!(s::BranchSolver, x::Int, l::Int, r::Int)
    s.tick = mergerow_impl!(s.legtgt, s.legcnt, s.legptr, s.leglen, s.legwt, s.acc,
        s.touched, s.vstamp, s.deg, s.wv, x, l, r, s.tick)
    return
end

function mergerow_impl!(
        legtgt::AbstractVector{Int},
        legcnt::AbstractVector{Int},
        legptr::AbstractVector{Int},
        leglen::AbstractVector{Int},
        legwt::AbstractVector{Float64},
        acc::AbstractVector{Int},
        touched::AbstractVector{Int},
        vstamp::AbstractVector{Int},
        deg::AbstractVector{Int},
        wv::AbstractVector{Float64},
        x::Int,
        l::Int,
        r::Int,
        tick::Int,
    )
    tick += 1; tk = tick; nt = 0

    @inbounds for c in (l, r)
        for i in legptr[c]:(legptr[c] + leglen[c] - 1)
            v = legtgt[i]

            if vstamp[v] != tk
                vstamp[v] = tk; acc[v] = 0; nt += 1; touched[nt] = v
            end

            acc[v] += legcnt[i]
        end
    end

    start = length(legtgt) + 1
    wt = 0.0

    @inbounds for j in oneto(nt)
        v = touched[j]

        if acc[v] < deg[v]
            push!(legtgt, v); push!(legcnt, acc[v]); wt += wv[v]
        end
    end

    legptr[x] = start
    leglen[x] = length(legtgt) - start + 1
    legwt[x] = wt
    return tick
end

# rebuild the whole arena bottom-up from the current tree (initial build + compaction)
function compactlegs!(s::BranchSolver)
    s.tick = compactlegs_impl!(s.legtgt, s.legcnt, s.legptr, s.leglen, s.legwt, s.acc,
        s.touched, s.vstamp, s.H, s.deg, s.wv, s.left, s.right, s.root, s.m, s.nn, s.tick)
    return
end

function compactlegs_impl!(
        legtgt::AbstractVector{Int},
        legcnt::AbstractVector{Int},
        legptr::AbstractVector{Int},
        leglen::AbstractVector{Int},
        legwt::AbstractVector{Float64},
        acc::AbstractVector{Int},
        touched::AbstractVector{Int},
        vstamp::AbstractVector{Int},
        H::BipartiteGraph,
        deg::AbstractVector{Int},
        wv::AbstractVector{Float64},
        left::AbstractVector{Int},
        right::AbstractVector{Int},
        root::Int,
        m::Int,
        nn::Int,
        tick::Int,
    )
    empty!(legtgt); empty!(legcnt)

    # preorder, then reverse for children-before-parents
    pre   = FVector{Int}(undef, nn)
    stack = FVector{Int}(undef, nn)
    npre = 0; nstack = 1; stack[1] = root

    while nstack > 0
        x = stack[nstack]; nstack -= 1
        npre += 1; pre[npre] = x

        if x > m
            nstack += 1; stack[nstack] = left[x]
            nstack += 1; stack[nstack] = right[x]
        end
    end

    @inbounds for p in npre:-1:1
        x = pre[p]

        if x <= m
            leafrow_impl!(legtgt, legcnt, legptr, leglen, legwt, H, deg, wv, x)
        else
            tick = mergerow_impl!(legtgt, legcnt, legptr, leglen, legwt, acc, touched, vstamp, deg, wv, x, left[x], right[x], tick)
        end
    end

    return tick
end

# window rewrite: node y's legs read out of the DP result mask L (column col); append.
# legcnt is not maintained here (recomputed by the next compactlegs!)
function setrow!(s::BranchSolver, y::Int, L::AbstractVecOrMat{UInt64}, col::Int, nw::Int)
    return setrow_impl!(s.legtgt, s.legptr, s.leglen, s.legwt, s.touched, s.wv, y, L, col, nw)
end

function setrow_impl!(
        legtgt::AbstractVector{Int},
        legptr::AbstractVector{Int},
        leglen::AbstractVector{Int},
        legwt::AbstractVector{Float64},
        touched::AbstractVector{Int},
        wv::AbstractVector{Float64},
        y::Int,
        L::AbstractVecOrMat{UInt64},
        col::Int,
        nw::Int,
    )
    start = length(legtgt) + 1
    wt = 0.0

    @inbounds for i in oneto(nw)
        word = bt_get(L, i, col)

        while word != 0
            v = touched[64 * (i - 1) + trailing_zeros(word) + 1]
            push!(legtgt, v); wt += wv[v]; word &= word - 1
        end
    end

    legptr[y] = start
    leglen[y] = length(legtgt) - start + 1
    legwt[y] = wt
    return
end

# re-solve the window (pieces under x); returns true if it was replaced.
# widthmode = true: replace when the largest intermediate shrinks; false: cheaper within W.
function solve_window!(s::BranchSolver, ::Type{S}, x::Int, pieces::AbstractVector{Int}, win::AbstractVector{Int}, widthmode::Bool, nt::Int) where {S <: AbstractVecOrMat{UInt64}}
    return solve_window_impl!(S, s.par, s.left, s.right, s.legtgt, s.legptr, s.leglen, s.legwt,
        s.touched, s.localid, s.wv, s.wloc, s.ploc, s.tab, s.etab, s.wid, s.bw, s.fl, s.split,
        s.u1, s.l1, s.Wcap, s.TCcap, x, pieces, win, widthmode, nt)
end

function solve_window_impl!(
        ::Type{S},
        par::AbstractVector{Int},
        left::AbstractVector{Int},
        right::AbstractVector{Int},
        legtgt::AbstractVector{Int},
        legptr::AbstractVector{Int},
        leglen::AbstractVector{Int},
        legwt::AbstractVector{Float64},
        touched::AbstractVector{Int},
        localid::AbstractVector{Int},
        wv::AbstractVector{Float64},
        wloc::AbstractVector{Float64},
        ploc::AbstractVector{Float64},
        tab::AbstractMatrix{Float64},
        etab::AbstractMatrix{Float64},
        wid::AbstractVector{Float64},
        bw::AbstractVector{Float64},
        fl::AbstractVector{Float64},
        split::AbstractVector{Int},
        u1::AbstractVector{UInt64},
        l1::AbstractVector{UInt64},
        Wcap::Float64,
        TCcap::Float64,
        x::Int,
        pieces::AbstractVector{Int},
        win::AbstractVector{Int},
        widthmode::Bool,
        nt::Int,
    ) where {S <: AbstractVecOrMat{UInt64}}
    b = nt; nw = max(cld(b, 64), 1); np = length(pieces)

    @inbounds for j in oneto(nt)
        v = touched[j]
        localid[v] = j - 1; wloc[j] = wv[v]
    end

    nbytes, shift = buildtables!(tab, etab, ploc, wloc, b)

    # columns 1:np the pieces, np+1:np+|win| the window's nodes (their legs ⊆ the pieces')
    nwin = length(win)
    X = newstore(S, nw, np + nwin)
    fill!(X, zero(UInt64))      # msetbit! ORs bits into zeroed words

    # (a name of its own: a `j` here would be the `j` of the loops below, captured and boxed)
    function col(y)
        hit = findfirst(==(y), pieces)

        if hit === nothing
            return np + something(findfirst(==(y), win))
        else
            return hit
        end
    end

    j = 0

    for y in pieces
        j += 1

        for i in legptr[y]:(legptr[y] + leglen[y] - 1)
            msetbit!(X, j, localid[legtgt[i]])
        end
    end

    for y in win
        j += 1

        for i in legptr[y]:(legptr[y] + leglen[y] - 1)
            msetbit!(X, j, localid[legtgt[i]])
        end
    end

    tcf = mexp2(shift, TCcap) * (1 + 1e-12)

    if S <: AbstractVector
        U = u1
        L = l1
    else
        U = newstore(S, nw, 1 << np)
        L = newstore(S, nw, 1 << np)
    end

    if widthmode
        oldw = 0.0

        for y in win
            oldw = max(oldw, mwt(tab, nbytes, X, col(y)))
        end

        bwin = width_dp!(wid, bw, U, L, tab, etab, nbytes, shift, X, X, X, np + 1, np; tcf)     # E = legs of x (column np + 1)

        if !(bwin < oldw - EPS)
            return false
        end

        c = flops_dp!(wid, fl, split, U, L, tab, etab, nbytes, shift, X, X, X, np + 1, np, bwin; tcf)

        if !(c < Inf)
            return false
        end
    else
        oldc = 0.0

        for y in win
            oldc += mflopsor(tab, etab, nbytes, shift, X, col(left[y]), X, col(right[y]))
        end

        c = flops_dp!(wid, fl, split, U, L, tab, etab, nbytes, shift, X, X, X, np + 1, np, Wcap; tcf)

        if !(c < oldc * (1 - 1e-12))
            return false
        end
    end

    # rebuild: x stays on top, the other window nodes are reused; legs from the DP's masks
    full = (1 << np) - 1

    function build(T, slot)
        if T & (T - 1) == 0
            y = pieces[trailing_zeros(T) + 1]
        else
            A = split[T + 1]

            if T == full
                y = x
            else
                slot += 1
                y = win[slot]
            end

            l, slot = build(A, slot)
            r, slot = build(T ⊻ A, slot)
            left[y] = l; right[y] = r; par[l] = y; par[r] = y

            if y != x
                setrow_impl!(legtgt, legptr, leglen, legwt, touched, wv, y, L, T + 1, nw)
            end
        end

        return y, slot
    end

    build(full, 1)
    return true
end

# the refined tree as a new parent vector, contractions renumbered children first
function branchrefine_parent(s::BranchSolver)
    return branchrefine_parent_impl!(s.par, s.left, s.right, s.root, s.m, s.point, s.nn)
end

function branchrefine_parent_impl!(
        par::AbstractVector{Int},
        left::AbstractVector{Int},
        right::AbstractVector{Int},
        root::Int,
        m::Int,
        point::Int,
        nn::Int,
    )
    newid     = FVector{Int}(undef, nn)
    out       = FVector{Int}(undef, nn)
    stacknode = FVector{Int}(undef, 2nn + 1)
    stackdone = FVector{Bool}(undef, 2nn + 1)

    fill!(newid, 0)
    fill!(out, 0)

    next = m
    nstack = 1; stacknode[1] = root; stackdone[1] = false

    while nstack > 0
        x = stacknode[nstack]; done = stackdone[nstack]; nstack -= 1

        if x <= m
            newid[x] = x
        elseif done
            next += 1; newid[x] = next
        else
            stacknode[nstack + 1] = x;        stackdone[nstack + 1] = true
            stacknode[nstack + 2] = right[x]; stackdone[nstack + 2] = false
            stacknode[nstack + 3] = left[x];  stackdone[nstack + 3] = false
            nstack += 3
        end
    end

    @inbounds for x in oneto(nn)
        if x != point
            p = par[x]

            if p == point
                out[newid[x]] = point
            else
                out[newid[x]] = newid[p]
            end
        end
    end

    return out
end

# ----------------------------------------------------------------------------------------
# large merges (k > kexact items): greedy, then subtree reconfiguration
# ----------------------------------------------------------------------------------------

# agglomerative binarization: repeatedly merge the best-scoring pair (P, Q hold the pair
# scores at [c, a] and are sized to the largest merge). Returns the merge's tree tuple.
function greedy_tree(
        ::Type{S},
        tab::AbstractMatrix{Float64},
        etab::AbstractMatrix{Float64},
        nbytes::Int,
        shift::Float64,
        P::AbstractMatrix{Float64},
        Q::AbstractMatrix{Float64},
        M::S,
        E::S,
        e::Int,
        k::Int,
        b::Int,
        W::Float64,
        capped::Bool,
    ) where {S <: AbstractVecOrMat{UInt64}}
    nw = bt_nwd(M)
    nn = 2k - 1

    left  = FVector{Int}(undef, nn)
    right = FVector{Int}(undef, nn)
    stamp = FVector{Int}(undef, nn)
    node  = FVector{Int}(undef, k)
    ew    = FVector{Float64}(undef, k)
    alive = FVector{Bool}(undef, k)
    cnt   = FVector{Int}(undef, b)
    U     = newstore(S, nw, nn)
    L     = newstore(S, nw, nn)
    out   = newstore(S, nw, nn)
    G     = newstore(S, nw, 3)
    order = FVector{Int}(undef, k - 1)   # the k-1 internal nodes, in recompute! order
    stack = FVector{Int}(undef, nn)      # DFS scratch for recompute!
    UA    = newstore(S, nw, k)           # live unions / legs, seeded from M
    LG    = newstore(S, nw, k)

    fill!(left, 0)
    fill!(right, 0)
    fill!(stamp, 0)
    fill!(ew, 0.0)
    fill!(alive, true)
    fill!(cnt, 0)
    fill!(G, zero(UInt64))      # msetbit!/mclearbit! update zeroed words
    copyto!(UA, M)
    copyto!(LG, M)

    @inbounds for i in oneto(k)
        node[i] = i
    end

    @inbounds for i in oneto(k)
        for w in oneto(nw)
            x = bt_get(M, w, i)

            while x != 0
                cnt[64 * (w - 1) + trailing_zeros(x) + 1] += 1; x &= x - 1
            end
        end
    end

    @inbounds for j in 0:b - 1
        c = cnt[j + 1]

        if c >= 2
            msetbit!(G, 1, j)
        end

        if c >= 3
            msetbit!(G, 2, j)
        end
    end

    @inbounds for i in oneto(k)
        ew[i] = mexp2(shift, mwt(tab, nbytes, M, i))
    end

    @inbounds for a in oneto(k)
        for c in a + 1:k
            P[c, a], Q[c, a] = pairscore(tab, etab, nbytes, shift, G, UA, LG, ew, a, c, E, e, W, capped)
        end
    end

    @inbounds for step in 1:k - 1
        ba = 0; bc = 0; bp = Inf; bq = Inf

        for a in oneto(k)
            if alive[a]
                for c in a + 1:k
                    if alive[c]
                        p = P[c, a]; q = Q[c, a]

                        if ba == 0 || p < bp || (p == bp && q < bq)
                            ba = a; bc = c; bp = p; bq = q
                        end
                    end
                end
            end
        end

        pairlegs!(G, UA, ba, bc, E, e)

        for w in oneto(nw)                   # holders of the shared vertices: two become one
            x = bt_get(UA, w, ba) & bt_get(UA, w, bc)

            while x != 0
                j = 64 * (w - 1) + trailing_zeros(x); x &= x - 1
                cnt[j + 1] -= 1; cj = cnt[j + 1]

                if cj < 3
                    mclearbit!(G, 2, j)
                end

                if cj < 2
                    mclearbit!(G, 1, j)
                end
            end
        end

        mor!(UA, ba, UA, ba, UA, bc); mcopy!(LG, ba, G, 3)
        ew[ba] = mexp2(shift, mwt(tab, nbytes, G, 3)); alive[bc] = false
        x = k + step; left[x] = node[ba]; right[x] = node[bc]; node[ba] = x

        for c in oneto(k)
            if alive[c] && c != ba
                a1, c1 = minmax(ba, c)
                P[c1, a1], Q[c1, a1] = pairscore(tab, etab, nbytes, shift, G, UA, LG, ew, a1, c1, E, e, W, capped)
            end
        end
    end

    t = (k, left, right, U, L, out, order, stack, stamp)
    return recompute!(t, M, E, e)
end

# Subtree reconfiguration: for every internal node x, cut its subtree into at most `window`
# pieces (expanding the widest piece first) and re-solve the merge of the pieces exactly.
# widthmode = true: replace when the window's largest intermediate shrinks (then the
# cheapest such tree); false: replace when the window gets cheaper within width W.
# A window is re-solved only if one of its nodes changed since it was last found optimal.
function reconfigure!(
        t,
        M::S,
        E::S,
        e::Int,
        W::Float64,
        widthmode::Bool,
        window::Int,
        wid::AbstractVector{Float64},
        bw::AbstractVector{Float64},
        fl::AbstractVector{Float64},
        split::AbstractVector{Int},
        tab::AbstractMatrix{Float64},
        etab::AbstractMatrix{Float64},
        nbytes::Int,
        shift::Float64;
        maxsweeps::Int = 16,
    ) where {S <: AbstractVecOrMat{UInt64}}
    k, left, right, U, L, out, order, stack, stamp = t
    nw = bt_nwd(M)
    sz = length(wid)

    pieces = FVector{Int}(undef, window)
    win    = FVector{Int}(undef, window)
    no     = length(order)               # the k-1 internal nodes
    todo   = FVector{Int}(undef, no)
    Mu     = newstore(S, nw, window)
    Ml     = newstore(S, nw, window)
    Udp    = newstore(S, nw, sz)
    Ldp    = newstore(S, nw, sz)
    solved = FVector{Int}(undef, 2k - 1)
    fill!(solved, 0)
    clock  = 0

    fill!(stamp, 0)

    for _ in 1:maxsweeps
        improved = false

        for i in oneto(no)               # top-down = reverse of recompute! order
            todo[i] = order[no - i + 1]
        end

        for ti in oneto(no)
            x = todo[ti]
            npieces = 0; nwin = 0
            npieces += 1; pieces[npieces] = left[x]
            npieces += 1; pieces[npieces] = right[x]
            nwin += 1; win[nwin] = x

            while npieces < window
                best = 0; bv = -Inf

                for j in oneto(npieces)
                    p = pieces[j]

                    if p > k
                        v = mwt(tab, nbytes, L, p)

                        if v > bv
                            bv = v; best = j
                        end
                    end
                end

                if best == 0
                    break
                end

                p = pieces[best]; pieces[best] = left[p]
                npieces += 1; pieces[npieces] = right[p]
                nwin += 1; win[nwin] = p
            end

            np = npieces

            if np > 2
                stale = false

                if solved[x] > 0
                    newest = 0

                    for j in oneto(nwin)
                        newest = max(newest, stamp[win[j]])
                    end

                    for j in oneto(npieces)
                        newest = max(newest, stamp[pieces[j]])
                    end

                    stale = newest < solved[x]      # unchanged since found optimal
                end

                if !stale
                    for j in oneto(npieces)
                        p = pieces[j]
                        mcopy!(Mu, j, U, p); mcopy!(Ml, j, L, p)
                    end

                    replace = false

                    if widthmode
                        oldw = 0.0

                        for j in oneto(nwin)
                            oldw = max(oldw, mwt(tab, nbytes, L, win[j]))
                        end

                        bwin = width_dp!(wid, bw, Udp, Ldp, tab, etab, nbytes, shift, Mu, Ml, L, x, np)

                        if bwin < oldw - EPS
                            flops_dp!(wid, fl, split, Udp, Ldp, tab, etab, nbytes, shift, Mu, Ml, L, x, np, bwin)
                            replace = true
                        end
                    else
                        oldc = 0.0

                        for j in oneto(nwin)
                            y = win[j]
                            oldc += mflopsor(tab, etab, nbytes, shift, L, left[y], L, right[y])
                        end

                        c = flops_dp!(wid, fl, split, Udp, Ldp, tab, etab, nbytes, shift, Mu, Ml, L, x, np, W)

                        if c < oldc * (1 - 1e-12)
                            replace = true
                        end
                    end

                    if replace
                        pv = view(pieces, oneto(npieces))
                        wv = view(win, oneto(nwin))
                        rebuild!(t, split, pv, wv, (1 << np) - 1, (1 << np) - 1, 1)
                        recompute!(t, M, E, e)
                        clock += 1

                        for j in oneto(nwin)
                            stamp[win[j]] = clock
                        end

                        improved = true
                    else
                        clock += 1; solved[x] = clock
                    end
                end
            end
        end

        if !improved
            break
        end
    end

    return t
end

# rebuild a window from the splits of the last DP: x = win[1] stays on top, the other
# window nodes are reused; slot counts the slots used (returned alongside the subtree root)
function rebuild!(t, split::AbstractVector{Int}, pieces::AbstractVector{Int}, win::AbstractVector{Int}, T::Int, full::Int, slot::Int)
    k, left, right, U, L, out, order, stack, stamp = t

    if T & (T - 1) == 0
        y = @inbounds pieces[trailing_zeros(T) + 1]
    else
        A = @inbounds split[T + 1]

        if T == full
            y = win[1]
        else
            slot += 1
            y = @inbounds win[slot]
        end

        l, slot = rebuild!(t, split, pieces, win, A, full, slot)
        r, slot = rebuild!(t, split, pieces, win, T ⊻ A, full, slot)
        @inbounds left[y] = l
        @inbounds right[y] = r
    end

    return y, slot
end

# recompute U, out and L of every node after the tree's shape changed
function recompute!(t, M::S, E::S, e::Int) where {S <: AbstractVecOrMat{UInt64}}
    k, left, right, U, L, out, order, stack, stamp = t
    root = 2k - 1
    norder = 0; nstack = 0

    nstack += 1; stack[nstack] = root

    @inbounds while nstack > 0                  # preorder, reversed below
        x = stack[nstack]; nstack -= 1

        if x > k
            norder += 1; order[norder] = x
            nstack += 1; stack[nstack] = left[x]
            nstack += 1; stack[nstack] = right[x]
        end
    end

    reverse!(order)      # order holds exactly the k-1 internal nodes

    @inbounds begin
        for i in oneto(k)
            mcopy!(U, i, M, i); mcopy!(L, i, M, i)
        end

        for x in order
            mor!(U, x, U, left[x], U, right[x])
        end

        mclear!(out, root)

        for x in Iterators.reverse(order)
            l = left[x]; r = right[x]
            mor!(out, l, out, x, U, r); mor!(out, r, out, x, U, l)
            mlegs!(L, x, U, x, out, x, E, e)
        end
    end

    return t
end

function maxwidth(tab::AbstractMatrix{Float64}, nbytes::Int, t)
    k, left, right, U, L, out, order, stack, stamp = t
    w = 0.0

    for x in order
        w = max(w, mwt(tab, nbytes, L, x))
    end

    return w
end

function treecost(tab::AbstractMatrix{Float64}, etab::AbstractMatrix{Float64}, nbytes::Int, shift::Float64, t)
    k, left, right, U, L, out, order, stack, stamp = t
    c = 0.0

    for x in order
        c += mflopsor(tab, etab, nbytes, shift, L, left[x], L, right[x])
    end

    return c
end

# emit a local tree as global nodes (children first); returns the root and the next id
function emit_local!(parent::AbstractVector{Int}, next::Int, t, ids::AbstractVector{Int})
    k, left, right, U, L, out, order, stack, stamp = t
    gid = FVector{Int}(undef, 2k - 1)
    fill!(gid, 0)

    @inbounds for i in oneto(k)
        gid[i] = ids[i]
    end

    @inbounds for x in order
        y = next; next += 1; gid[x] = y
        parent[gid[left[x]]] = y
        parent[gid[right[x]]] = y
    end

    return gid[2k - 1], next
end

# legs of the merge of nodes a and c, written to G[3]: the vertices of a ∪ c that occur in
# another node or in E. A vertex in exactly one of a, c needs one other holder (two in
# all); in both, a third. G[1] and G[2] mark the vertices with at least 2 and 3 holders.
@inline function pairlegs!(G::S, UA::S, a::Int, c::Int, E::S, e::Int) where {S <: AbstractVecOrMat{UInt64}}
    @inbounds for i in oneto(bt_nwd(G))
        ua = bt_get(UA, i, a); uc = bt_get(UA, i, c)
        others = (bt_get(G, i, 1) & (ua ⊻ uc)) | (bt_get(G, i, 2) & ua & uc)
        bt_put!(G, (ua | uc) & (others | bt_get(E, i, e)), i, 3)
    end

    return
end

# score of merging a and c: (primary, secondary), smaller is better.
# capped = false: (result width, flops). capped = true: (0, memory change 2^out - 2^a - 2^c)
# when the result fits in W, else (result width, flops).
@inline function pairscore(
        tab::AbstractMatrix{Float64},
        etab::AbstractMatrix{Float64},
        nbytes::Int,
        shift::Float64,
        G::S,
        UA::S,
        LG::S,
        ew::AbstractVector{Float64},
        a::Int,
        c::Int,
        E::S,
        e::Int,
        W::Float64,
        capped::Bool,
    ) where {S <: AbstractVecOrMat{UInt64}}
    pairlegs!(G, UA, a, c, E, e)
    wo = mwt(tab, nbytes, G, 3)

    if capped && wo <= W + EPS
        return 0.0, mexp2(shift, wo) - (@inbounds ew[a]) - (@inbounds ew[c])
    end

    return wo, mflopsor(tab, etab, nbytes, shift, LG, a, LG, c)
end

# ----------------------------------------------------------------------------------------
# subset dynamic programming (k ≤ kexact items)
# ----------------------------------------------------------------------------------------

# Items i = 1:k with union masks Mu[i] (the vertices below them that still matter) and legs
# Ml[i]; E[e] = legs of the whole merge. U[S] = union of S, L[S] = legs of the merge of S
# (Ml[i] for a singleton), wid[S] = w(L[S]) (0 for singletons: items are counted where
# they are made). For a whole clique Mu = Ml.
function prepare_dp!(
        wid::AbstractVector{Float64},
        U::S,
        L::S,
        tab::AbstractMatrix{Float64},
        nbytes::Int,
        Mu::S,
        Ml::S,
        E::S,
        e::Int,
        k::Int,
    ) where {S <: AbstractVecOrMat{UInt64}}
    full = (1 << k) - 1

    @inbounds begin
        mclear!(U, 1)

        for T in 1:full
            mor!(U, T + 1, U, (T & (T - 1)) + 1, Mu, trailing_zeros(T) + 1)
        end

        for T in 1:full - 1
            if T & (T - 1) == 0
                mcopy!(L, T + 1, Ml, trailing_zeros(T) + 1)
                wid[T + 1] = 0.0
            else
                mlegs!(L, T + 1, U, T + 1, U, (full ⊻ T) + 1, E, e)
                wid[T + 1] = mwt(tab, nbytes, L, T + 1)
            end
        end

        mcopy!(L, full + 1, E, e)
        wid[full + 1] = mwt(tab, nbytes, L, full + 1)
    end

    return
end

# smallest achievable width of the merge (largest intermediate, the merge's result included)
function width_dp!(
        wid::AbstractVector{Float64},
        bw::AbstractVector{Float64},
        U::S,
        L::S,
        tab::AbstractMatrix{Float64},
        etab::AbstractMatrix{Float64},
        nbytes::Int,
        shift::Float64,
        Mu::S,
        Ml::S,
        E::S,
        e::Int,
        k::Int;
        tcf::Float64 = Inf,
    ) where {S <: AbstractVecOrMat{UInt64}}
    prepare_dp!(wid, U, L, tab, nbytes, Mu, Ml, E, e, k)
    full = (1 << k) - 1

    @inbounds for T in 1:full
        if T & (T - 1) == 0
            bw[T + 1] = 0.0
        else
            low = T & -T; rest = T ⊻ low; best = Inf; wt = wid[T + 1]
            sub = (rest - 1) & rest

            while true
                A = low | sub
                c = max(bw[A + 1], bw[(T ⊻ A) + 1])

                # a step above the tc cap (given as a flops threshold) is not allowed
                if tcf < Inf && c < best && mflopsor(tab, etab, nbytes, shift, L, A + 1, L, (T ⊻ A) + 1) > tcf
                    c = Inf
                end

                best = min(best, c)

                if best <= wt || sub == 0      # cannot get below the node itself
                    break
                end

                sub = (sub - 1) & rest
            end

            bw[T + 1] = max(wt, best)
        end
    end

    return @inbounds bw[full + 1]
end

# fewest flops with every intermediate below the top of width ≤ W; returns the (scaled)
# cost, Inf if infeasible, and leaves the best splits in split
function flops_dp!(
        wid::AbstractVector{Float64},
        fl::AbstractVector{Float64},
        split::AbstractVector{Int},
        U::S,
        L::S,
        tab::AbstractMatrix{Float64},
        etab::AbstractMatrix{Float64},
        nbytes::Int,
        shift::Float64,
        Mu::S,
        Ml::S,
        E::S,
        e::Int,
        k::Int,
        W::Float64;
        tcf::Float64 = Inf,
    ) where {S <: AbstractVecOrMat{UInt64}}
    prepare_dp!(wid, U, L, tab, nbytes, Mu, Ml, E, e, k)
    full = (1 << k) - 1

    @inbounds for T in 1:full
        if T & (T - 1) == 0
            fl[T + 1] = 0.0
        else
            split[T + 1] = 0

            if T != full && wid[T + 1] > W + EPS
                fl[T + 1] = Inf
            else
                # a step's indices include the result's, so it costs at least 2^wid[T]
                low = T & -T; rest = T ⊻ low; best = Inf; bestA = 0
                lb = mexp2(shift, wid[T + 1]) * (1 - 1e-12)
                sub = (rest - 1) & rest

                while true
                    A = low | sub; B = T ⊻ A
                    fa = fl[A + 1]; fb = fl[B + 1]

                    if fa + fb + lb < best
                        step = mflopsor(tab, etab, nbytes, shift, L, A + 1, L, B + 1)

                        if step > tcf
                            c = Inf
                        else
                            c = fa + fb + step
                        end

                        if c < best
                            best = c; bestA = A
                        end
                    end

                    if sub == 0
                        break
                    end

                    sub = (sub - 1) & rest
                end

                fl[T + 1] = best; split[T + 1] = bestA
            end
        end
    end

    return @inbounds fl[full + 1]
end

# emit the splits of the last flops_dp! as global nodes; returns the root and the next id
function emit_dp!(
        parent::AbstractVector{Int},
        next::Int,
        split::AbstractVector{Int},
        T::Int,
        ids::AbstractVector{Int},
    )
    if T & (T - 1) == 0
        x = @inbounds ids[trailing_zeros(T) + 1]
    else
        A = @inbounds split[T + 1]
        a, next = emit_dp!(parent, next, split, A, ids)
        b, next = emit_dp!(parent, next, split, T ⊻ A, ids)
        x = next; next += 1
        @inbounds parent[a] = x
        @inbounds parent[b] = x
    end

    return x, next
end

# ----------------------------------------------------------------------------------------
# byte tables and weights
# ----------------------------------------------------------------------------------------

# byte tables of the local weights: the weight of a mask is a sum of nbytes lookups and its
# flops a product of nbytes lookups (both tables built by recurrences, without exp2). Fills
# tab, etab and ploc; returns the byte count and the flops shift.
function buildtables!(
        tab::AbstractMatrix{Float64},
        etab::AbstractMatrix{Float64},
        ploc::AbstractVector{Float64},
        wloc::AbstractVector{Float64},
        b::Int,
    )
    nbytes = max(cld(b, 8), 1)

    total = 0.0

    @inbounds for j in oneto(b)
        total += wloc[j]
    end

    # flops up to 2^total: exact products of byte factors when total ≤ 900, else
    # 2^(weight - shift) through one exp2 (mflopsor)
    shift = max(0.0, total - 900.0)

    if iszero(shift)
        @inbounds for j in oneto(b)
            ploc[j] = exp2(wloc[j])
        end
    end

    @inbounds for j in oneto(nbytes)
        tab[1, j] = 0.0
        etab[1, j] = 1.0
        base = 8 * (j - 1)

        for x in 1:255
            bit = base + trailing_zeros(x)
            prev = (x & (x - 1)) + 1

            if bit < b
                tab[x + 1, j] = tab[prev, j] + wloc[bit + 1]

                if iszero(shift)
                    etab[x + 1, j] = etab[prev, j] * ploc[bit + 1]
                end
            else
                tab[x + 1, j] = tab[prev, j]

                if iszero(shift)
                    etab[x + 1, j] = etab[prev, j]
                end
            end
        end
    end

    return nbytes, shift
end

# weight of the mask X[v]
@inline function mwt(tab::AbstractMatrix{Float64}, nbytes::Int, X::AbstractVecOrMat{UInt64}, v::Int)
    s = 0.0

    @inbounds for jj in 0:nbytes - 1
        s += tab[((bt_get(X, (jj >> 3) + 1, v) >> ((jj & 7) << 3)) & 0xff) + 1, jj + 1]
    end

    return s
end

# flops 2^(w(X[a] | Z[b]) - shift)
@inline function mflopsor(
        tab::AbstractMatrix{Float64},
        etab::AbstractMatrix{Float64},
        nbytes::Int,
        shift::Float64,
        X::S,
        a::Int,
        Z::S,
        b::Int,
    ) where {S <: AbstractVecOrMat{UInt64}}
    if shift > 0.0
        s = 0.0

        @inbounds for jj in 0:nbytes - 1
            w = (jj >> 3) + 1
            s += tab[(((bt_get(X, w, a) | bt_get(Z, w, b)) >> ((jj & 7) << 3)) & 0xff) + 1, jj + 1]
        end

        return exp2(s - shift)
    end

    p = 1.0

    @inbounds for jj in 0:nbytes - 1
        w = (jj >> 3) + 1
        p *= etab[(((bt_get(X, w, a) | bt_get(Z, w, b)) >> ((jj & 7) << 3)) & 0xff) + 1, jj + 1]
    end

    return p
end

@inline function mexp2(shift::Float64, w::Float64)
    return exp2(w - shift)
end

# ----------------------------------------------------------------------------------------
# bit masks over the vertices of one clique (the bt primitives bt_nwd/bt_get/bt_put!)
# ----------------------------------------------------------------------------------------

function newstore(::Type{FVector{UInt64}}, nw::Int, n::Int)
    return FVector{UInt64}(undef, n)
end

function newstore(::Type{FMatrix{UInt64}}, nw::Int, n::Int)
    return FMatrix{UInt64}(undef, nw, n)
end

@inline function mclear!(X::AbstractVecOrMat{UInt64}, v::Int)
    @inbounds for i in oneto(bt_nwd(X))
        bt_put!(X, zero(UInt64), i, v)
    end

    return
end

# Y[v] = X[u]
@inline function mcopy!(Y::S, v::Int, X::S, u::Int) where {S <: AbstractVecOrMat{UInt64}}
    @inbounds for i in oneto(bt_nwd(Y))
        bt_put!(Y, bt_get(X, i, u), i, v)
    end

    return
end

@inline function msetbit!(X::AbstractVecOrMat{UInt64}, v::Int, j::Int)
    i = (j >> 6) + 1
    @inbounds bt_put!(X, bt_get(X, i, v) | (UInt64(1) << (j & 63)), i, v)

    return
end

@inline function mclearbit!(X::AbstractVecOrMat{UInt64}, v::Int, j::Int)
    i = (j >> 6) + 1
    @inbounds bt_put!(X, bt_get(X, i, v) & ~(UInt64(1) << (j & 63)), i, v)

    return
end

# Y[v] = X[a] | Z[b]
@inline function mor!(Y::S, v::Int, X::S, a::Int, Z::S, b::Int) where {S <: AbstractVecOrMat{UInt64}}
    @inbounds for i in oneto(bt_nwd(Y))
        bt_put!(Y, bt_get(X, i, a) | bt_get(Z, i, b), i, v)
    end

    return
end

# Y[v] = X[a] & (Z[b] | F[c]): the vertices of X[a] that also occur in Z[b] or F[c]
@inline function mlegs!(Y::S, v::Int, X::S, a::Int, Z::S, b::Int, F::S, c::Int) where {S <: AbstractVecOrMat{UInt64}}
    @inbounds for i in oneto(bt_nwd(Y))
        bt_put!(Y, bt_get(X, i, a) & (bt_get(Z, i, b) | bt_get(F, i, c)), i, v)
    end

    return
end
