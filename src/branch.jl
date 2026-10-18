#
# Branch decompositions (contraction trees) of a pointed hypergraph, built by
# binarizing a clique tree of its line graph one clique at a time. The hypergraph
# H has one vertex per index (weight = log2 of its dimension), one hyperedge per
# tensor, and one hyperedge (the point) for the output indices. A branch tree is a
# binary tree whose leaves are the tensors and whose root is the point; `branchtree`
# returns it as a parent vector of length 2m - 2.
#
# Costs of a branch tree (`branchstats` returns all three in one pass):
#   - branchwidth: max over contractions of w(legs)               -- largest intermediate; the branchwidth
#   - branchpeak:  max over contractions of w(legs(l) u legs(r))  -- most expensive step; induced treewidth of the line graph
#   - branchcost:  log2 sum over contractions of 2^w(legs(l) u legs(r)) -- total flops
# These match cotengra's contraction_width, max log2 get_flops, and contraction_cost.
#
# branchrefine is an optional post-pass (subtree reconfiguration; never wider). It is
# stoppable: a BranchSolver holds the tree, branchrefine_step! runs one sweep, and
# branchrefine_solve! runs sweeps to a fixed point. branchssa converts the parent vector
# to an SSA contraction path.
#
const EPS = 1e-9

# ----------------------------------------------------------------------------------------
# bit masks over the vertices of one clique, as in CliqueTrees' bt: a store of masks is an
# FVector{UInt64} (one word per mask, the fast path) or an FMatrix{UInt64} (one column of
# words per mask, for cliques with more than 64 vertices). The word count and word access
# are the bt primitives bt_nwd/bt_get/bt_put!.
# ----------------------------------------------------------------------------------------

const Store = AbstractVecOrMat{UInt64}

function newstore(::Type{FVector{UInt64}}, nw::Int, n::Int)
    return fill!(FVector{UInt64}(undef, n), zero(UInt64))
end

function newstore(::Type{FMatrix{UInt64}}, nw::Int, n::Int)
    return fill!(FMatrix{UInt64}(undef, nw, n), zero(UInt64))
end

@inline function mclear!(X::Store, v::Int)
    @inbounds for i in oneto(bt_nwd(X))
        bt_put!(X, zero(UInt64), i, v)
    end

    return
end

# Y[v] = X[u]
@inline function mcopy!(Y::S, v::Int, X::S, u::Int) where {S <: Store}
    @inbounds for i in oneto(bt_nwd(Y))
        bt_put!(Y, bt_get(X, i, u), i, v)
    end

    return
end

@inline function msetbit!(X::Store, v::Int, j::Int)
    i = (j >> 6) + 1
    @inbounds bt_put!(X, bt_get(X, i, v) | (UInt64(1) << (j & 63)), i, v)

    return
end

@inline function mclearbit!(X::Store, v::Int, j::Int)
    i = (j >> 6) + 1
    @inbounds bt_put!(X, bt_get(X, i, v) & ~(UInt64(1) << (j & 63)), i, v)

    return
end

# Y[v] = X[a] | Z[b]
@inline function mor!(Y::S, v::Int, X::S, a::Int, Z::S, b::Int) where {S <: Store}
    @inbounds for i in oneto(bt_nwd(Y))
        bt_put!(Y, bt_get(X, i, a) | bt_get(Z, i, b), i, v)
    end

    return
end

# Y[v] = X[a] & (Z[b] | F[c]): the vertices of X[a] that also occur in Z[b] or F[c]
@inline function mlegs!(Y::S, v::Int, X::S, a::Int, Z::S, b::Int, F::S, c::Int) where {S <: Store}
    @inbounds for i in oneto(bt_nwd(Y))
        bt_put!(Y, bt_get(X, i, a) & (bt_get(Z, i, b) | bt_get(F, i, c)), i, v)
    end

    return
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
                iszero(shift) && (etab[x + 1, j] = etab[prev, j] * ploc[bit + 1])
            else
                tab[x + 1, j] = tab[prev, j]
                iszero(shift) && (etab[x + 1, j] = etab[prev, j])
            end
        end
    end

    return nbytes, shift
end

# weight of the mask X[v]
@inline function mwt(tab::AbstractMatrix{Float64}, nbytes::Int, X::Store, v::Int)
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
    ) where {S <: Store}
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
    ) where {S <: Store}
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
    ) where {S <: Store}
    prepare_dp!(wid, U, L, tab, nbytes, Mu, Ml, E, e, k)
    full = (1 << k) - 1

    @inbounds for T in 1:full
        if T & (T - 1) == 0
            bw[T + 1] = 0.0
            continue
        end

        low = T & -T; rest = T ⊻ low; best = Inf; wt = wid[T + 1]
        sub = (rest - 1) & rest

        while true
            A = low | sub
            c = max(bw[A + 1], bw[(T ⊻ A) + 1])

            # a step above the tc cap (given as a flops threshold) is not allowed
            (tcf < Inf && c < best && mflopsor(tab, etab, nbytes, shift, L, A + 1, L, (T ⊻ A) + 1) > tcf) && (c = Inf)
            c < best && (best = c)
            (best <= wt || sub == 0) && break      # cannot get below the node itself
            sub = (sub - 1) & rest
        end

        bw[T + 1] = max(wt, best)
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
    ) where {S <: Store}
    prepare_dp!(wid, U, L, tab, nbytes, Mu, Ml, E, e, k)
    full = (1 << k) - 1

    @inbounds for T in 1:full
        if T & (T - 1) == 0
            fl[T + 1] = 0.0
            continue
        end

        split[T + 1] = 0

        if T != full && wid[T + 1] > W + EPS
            fl[T + 1] = Inf
            continue
        end

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

            sub == 0 && break
            sub = (sub - 1) & rest
        end

        fl[T + 1] = best; split[T + 1] = bestA
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
        return (@inbounds ids[trailing_zeros(T) + 1]), next
    end

    A = @inbounds split[T + 1]
    a, next = emit_dp!(parent, next, split, A, ids)
    b, next = emit_dp!(parent, next, split, T ⊻ A, ids)

    x = next; next += 1
    @inbounds parent[a] = x
    @inbounds parent[b] = x

    return x, next
end

# ----------------------------------------------------------------------------------------
# large merges (k > kexact items): greedy, then subtree reconfiguration
# ----------------------------------------------------------------------------------------
#
# A merge's binary tree is the tuple
#
#   t = (k, left, right, U, L, out, order, stack, stamp)
#
# over items 1:k (leaves) and internal nodes k+1:2k-1 (root 2k-1). left/right are the
# children; U/L/out are per-node masks (union below / legs / union outside); order/stack
# are traversal scratch; stamp records when a node last changed (reconfigure!).

# recompute U, out and L of every node after the tree's shape changed
function recompute!(t, M::S, E::S, e::Int) where {S <: Store}
    k, left, right, U, L, out, order, stack, stamp = t
    root = 2k - 1

    empty!(order); empty!(stack); push!(stack, root)

    @inbounds while !isempty(stack)            # preorder, reversed below
        x = pop!(stack)

        if x > k
            push!(order, x); push!(stack, left[x]); push!(stack, right[x])
        end
    end

    reverse!(order)

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

# legs of the merge of nodes a and c, written to G[3]: the vertices of a ∪ c that occur in
# another node or in E. A vertex in exactly one of a, c needs one other holder (two in
# all); in both, a third. G[1] and G[2] mark the vertices with at least 2 and 3 holders.
@inline function pairlegs!(G::S, UA::S, a::Int, c::Int, E::S, e::Int) where {S <: Store}
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
    ) where {S <: Store}
    pairlegs!(G, UA, a, c, E, e)
    wo = mwt(tab, nbytes, G, 3)

    if capped && wo <= W + EPS
        return 0.0, mexp2(shift, wo) - (@inbounds ew[a]) - (@inbounds ew[c])
    end

    return wo, mflopsor(tab, etab, nbytes, shift, LG, a, LG, c)
end

# agglomerative binarization: repeatedly merge the best-scoring pair (P, Q hold the pair
# scores at [c, a] and are sized to the largest merge)
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
    ) where {S <: Store}
    nw = bt_nwd(M)
    nn = 2k - 1

    left  = fill!(FVector{Int}(undef, nn), 0)
    right = fill!(FVector{Int}(undef, nn), 0)
    U     = newstore(S, nw, nn)
    L     = newstore(S, nw, nn)
    out   = newstore(S, nw, nn)
    order = Int[]
    stack = Int[]
    stamp = fill!(FVector{Int}(undef, nn), 0)

    UA    = copy(M)
    LG    = copy(M)
    G     = newstore(S, nw, 3)
    ew    = fill!(FVector{Float64}(undef, k), 0.0)
    alive = fill!(FVector{Bool}(undef, k), true)
    node  = FVector{Int}(undef, k)
    cnt   = fill!(FVector{Int}(undef, b), 0)

    @inbounds for i in oneto(k)
        node[i] = i
    end

    @inbounds for i in oneto(k), w in oneto(nw)
        x = bt_get(M, w, i)

        while x != 0
            cnt[64 * (w - 1) + trailing_zeros(x) + 1] += 1; x &= x - 1
        end
    end

    @inbounds for j in 0:b - 1
        c = cnt[j + 1]
        c >= 2 && msetbit!(G, 1, j); c >= 3 && msetbit!(G, 2, j)
    end

    @inbounds for i in oneto(k)
        ew[i] = mexp2(shift, mwt(tab, nbytes, M, i))
    end

    @inbounds for a in oneto(k), c in a + 1:k
        P[c, a], Q[c, a] = pairscore(tab, etab, nbytes, shift, G, UA, LG, ew, a, c, E, e, W, capped)
    end

    @inbounds for step in 1:k - 1
        ba = 0; bc = 0; bp = Inf; bq = Inf

        for a in oneto(k)
            alive[a] || continue

            for c in a + 1:k
                alive[c] || continue
                p = P[c, a]; q = Q[c, a]

                if ba == 0 || p < bp || (p == bp && q < bq)
                    ba = a; bc = c; bp = p; bq = q
                end
            end
        end

        pairlegs!(G, UA, ba, bc, E, e)

        for w in oneto(nw)                   # holders of the shared vertices: two become one
            x = bt_get(UA, w, ba) & bt_get(UA, w, bc)

            while x != 0
                j = 64 * (w - 1) + trailing_zeros(x); x &= x - 1
                cnt[j + 1] -= 1; cj = cnt[j + 1]
                cj < 3 && mclearbit!(G, 2, j); cj < 2 && mclearbit!(G, 1, j)
            end
        end

        mor!(UA, ba, UA, ba, UA, bc); mcopy!(LG, ba, G, 3)
        ew[ba] = mexp2(shift, mwt(tab, nbytes, G, 3)); alive[bc] = false
        x = k + step; left[x] = node[ba]; right[x] = node[bc]; node[ba] = x

        for c in oneto(k)
            (alive[c] && c != ba) || continue
            a1, c1 = minmax(ba, c)
            P[c1, a1], Q[c1, a1] = pairscore(tab, etab, nbytes, shift, G, UA, LG, ew, a1, c1, E, e, W, capped)
        end
    end

    t = (k, left, right, U, L, out, order, stack, stamp)
    return recompute!(t, M, E, e)
end

# rebuild a window from the splits of the last DP: x = win[1] stays on top, the other window
# nodes are reused; slot[] counts the slots used
function rebuild!(t, split::AbstractVector{Int}, pieces::Vector{Int}, win::Vector{Int}, T::Int, full::Int, slot::Base.RefValue{Int})
    k, left, right, U, L, out, order, stack, stamp = t

    if T & (T - 1) == 0
        return @inbounds pieces[trailing_zeros(T) + 1]
    end

    A = @inbounds split[T + 1]

    if T == full
        y = win[1]
    else
        slot[] += 1
        y = @inbounds win[slot[]]
    end

    l = rebuild!(t, split, pieces, win, A, full, slot)
    r = rebuild!(t, split, pieces, win, T ⊻ A, full, slot)
    @inbounds left[y] = l
    @inbounds right[y] = r

    return y
end

# Subtree reconfiguration: for every internal node x, cut its subtree into at most `window`
# pieces (expanding the widest piece first) and re-solve the merge of the pieces exactly.
# mode = :width: replace when the window's largest intermediate shrinks (then the cheapest
# such tree); mode = :flops: replace when the window gets cheaper within width W.
# A window is re-solved only if one of its nodes changed since it was last found optimal.
function reconfigure!(
        t,
        M::S,
        E::S,
        e::Int,
        W::Float64,
        mode::Symbol,
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
    ) where {S <: Store}
    k, left, right, U, L, out, order, stack, stamp = t
    nw = bt_nwd(M)
    s = length(wid)

    pieces = Int[]
    win    = Int[]
    todo   = Int[]
    Mu     = newstore(S, nw, window)
    Ml     = newstore(S, nw, window)
    Udp    = newstore(S, nw, s)
    Ldp    = newstore(S, nw, s)
    solved = fill!(FVector{Int}(undef, 2k - 1), 0)
    clock  = 0
    slot   = Ref(1)

    fill!(stamp, 0)

    for _ in 1:maxsweeps
        improved = false
        append!(empty!(todo), order); reverse!(todo)      # top-down

        for x in todo
            empty!(pieces); empty!(win)
            push!(pieces, left[x], right[x]); push!(win, x)

            while length(pieces) < window
                best = 0; bv = -Inf

                for (j, p) in enumerate(pieces)
                    if p > k
                        v = mwt(tab, nbytes, L, p)
                        v > bv && (bv = v; best = j)
                    end
                end

                best == 0 && break
                p = pieces[best]; pieces[best] = left[p]; push!(pieces, right[p]); push!(win, p)
            end

            np = length(pieces)
            np <= 2 && continue

            if solved[x] > 0
                newest = 0

                for y in win
                    newest = max(newest, stamp[y])
                end

                for y in pieces
                    newest = max(newest, stamp[y])
                end

                newest < solved[x] && continue          # unchanged since found optimal
            end

            for (j, p) in enumerate(pieces)
                mcopy!(Mu, j, U, p); mcopy!(Ml, j, L, p)
            end

            if mode === :width
                oldw = 0.0

                for y in win
                    oldw = max(oldw, mwt(tab, nbytes, L, y))
                end

                bwin = width_dp!(wid, bw, Udp, Ldp, tab, etab, nbytes, shift, Mu, Ml, L, x, np)

                if !(bwin < oldw - EPS)
                    clock += 1; solved[x] = clock; continue
                end

                flops_dp!(wid, fl, split, Udp, Ldp, tab, etab, nbytes, shift, Mu, Ml, L, x, np, bwin)
            else
                oldc = 0.0

                for y in win
                    oldc += mflopsor(tab, etab, nbytes, shift, L, left[y], L, right[y])
                end

                c = flops_dp!(wid, fl, split, Udp, Ldp, tab, etab, nbytes, shift, Mu, Ml, L, x, np, W)

                if !(c < oldc * (1 - 1e-12))
                    clock += 1; solved[x] = clock; continue
                end
            end

            slot[] = 1
            rebuild!(t, split, pieces, win, (1 << np) - 1, (1 << np) - 1, slot)
            recompute!(t, M, E, e)
            clock += 1

            for y in win
                stamp[y] = clock
            end

            improved = true
        end

        improved || break
    end

    return t
end

# emit a local tree as global nodes (children first); returns the root and the next id
function emit_local!(parent::AbstractVector{Int}, next::Int, t, ids::AbstractVector{Int})
    k, left, right, U, L, out, order, stack, stamp = t
    gid = fill!(FVector{Int}(undef, 2k - 1), 0)

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

# ----------------------------------------------------------------------------------------
# cliques
# ----------------------------------------------------------------------------------------

# Legs of the items of clique i, with occurrence counts in acc and the distinct vertices in
# touched; returns the number of items. Items: the children with a result (legs
# rv[rs[c]:re[c]-1], counts rc) and the assigned hyperedges (their vertices held by another
# hyperedge too, count 1).
function gather!(
        chptr::AbstractVector{Int},
        ch::AbstractVector{Int},
        asptr::AbstractVector{Int},
        as::AbstractVector{Int},
        i::Int,
        H::BipartiteGraph,
        rs::AbstractVector{Int},
        re::AbstractVector{Int},
        rv::Vector{Int},
        rc::Vector{Int},
        acc::AbstractVector{Int},
        stamp::AbstractVector{Int},
        deg::AbstractVector{Int},
        touched::Vector{Int},
        tk::Int,
    )
    empty!(touched); k = 0

    @inbounds for p in chptr[i]:chptr[i + 1] - 1
        c = ch[p]
        rs[c] == 0 && continue
        k += 1

        for q in rs[c]:re[c] - 1
            v = rv[q]

            if stamp[v] != tk
                stamp[v] = tk; acc[v] = 0; push!(touched, v)
            end

            acc[v] += rc[q]
        end
    end

    @inbounds for p in asptr[i]:asptr[i + 1] - 1
        k += 1

        for v in neighbors(H, as[p])
            deg[v] == 1 && continue          # summed inside the tensor (cotengra: preprocessing)

            if stamp[v] != tk
                stamp[v] = tk; acc[v] = 0; push!(touched, v)
            end

            acc[v] += 1
        end
    end

    return k
end

function touchedweight(wv::AbstractVector{Float64}, touched::Vector{Int})
    return sum((@inbounds wv[v] for v in touched); init = 0.0)
end

# mask store type for the current merge: one word if it has at most 64 vertices
function storetype(touched::Vector{Int})
    if length(touched) <= 64
        return FVector{UInt64}
    else
        return FMatrix{UInt64}
    end
end

# local masks of the items of clique i (after gather!): M (k columns), the merge's legs E
# (1 column), the global ids of the items, and the byte tables of the local weights
function localmasks(
        ::Type{S},
        chptr::AbstractVector{Int},
        ch::AbstractVector{Int},
        asptr::AbstractVector{Int},
        as::AbstractVector{Int},
        i::Int,
        k::Int,
        H::BipartiteGraph,
        rs::AbstractVector{Int},
        re::AbstractVector{Int},
        rv::Vector{Int},
        resnode::AbstractVector{Int},
        localid::AbstractVector{Int},
        touched::Vector{Int},
        wv::AbstractVector{Float64},
        acc::AbstractVector{Int},
        deg::AbstractVector{Int},
        wloc::AbstractVector{Float64},
        ploc::AbstractVector{Float64},
        tab::AbstractMatrix{Float64},
        etab::AbstractMatrix{Float64},
    ) where {S <: Store}
    b = length(touched); nw = max(cld(b, 64), 1)

    @inbounds for (j, v) in enumerate(touched)
        localid[v] = j - 1; wloc[j] = wv[v]
    end

    nbytes, shift = buildtables!(tab, etab, ploc, wloc, b)

    M   = newstore(S, nw, k)
    E   = newstore(S, nw, 1)
    ids = FVector{Int}(undef, k)
    j   = 0

    @inbounds for p in chptr[i]:chptr[i + 1] - 1
        c = ch[p]
        rs[c] == 0 && continue
        j += 1

        for q in rs[c]:re[c] - 1
            msetbit!(M, j, localid[rv[q]])
        end

        ids[j] = resnode[c]
    end

    @inbounds for p in asptr[i]:asptr[i + 1] - 1
        e = as[p]; j += 1

        for v in neighbors(H, e)
            deg[v] == 1 || msetbit!(M, j, localid[v])
        end

        ids[j] = e
    end

    @inbounds for v in touched
        acc[v] < deg[v] && msetbit!(E, 1, localid[v])
    end

    return M, E, ids, b, nbytes, shift
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
    ) where {S <: Store}
    t = greedy_tree(S, tab, etab, nbytes, shift, P, Q, M, E, 1, k, b, Inf, false)
    return reconfigure!(t, M, E, 1, Inf, :width, window, wid, bw, fl, split, tab, etab, nbytes, shift)
end

# pass 1: the smallest width of clique i's merge (large merges keep their tree in cache)
function clique_width(
        ::Type{S},
        chptr::AbstractVector{Int},
        ch::AbstractVector{Int},
        asptr::AbstractVector{Int},
        as::AbstractVector{Int},
        i::Int,
        k::Int,
        kexact::Int,
        window::Int,
        H::BipartiteGraph,
        rs::AbstractVector{Int},
        re::AbstractVector{Int},
        rv::Vector{Int},
        resnode::AbstractVector{Int},
        cache::Vector{Any},
        localid::AbstractVector{Int},
        touched::Vector{Int},
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
    ) where {S <: Store}
    M, E, ids, b, nbytes, shift = localmasks(S, chptr, ch, asptr, as, i, k, H, rs, re, rv, resnode, localid, touched, wv, acc, deg, wloc, ploc, tab, etab)

    if k <= kexact
        if S === FVector{UInt64}
            U = u1
            L = l1
        else
            U = newstore(S, bt_nwd(M), length(wid))
            L = newstore(S, bt_nwd(M), length(wid))
        end

        return width_dp!(wid, bw, U, L, tab, etab, nbytes, shift, M, M, E, 1, k)
    end

    t = widthtree(S, tab, etab, nbytes, shift, P, Q, wid, bw, fl, split, M, E, k, b, window)
    cache[i] = t

    return maxwidth(tab, nbytes, t)
end

# pass 2: the cheapest binarization of clique i's merge within width W
function clique_build(
        ::Type{S},
        chptr::AbstractVector{Int},
        ch::AbstractVector{Int},
        asptr::AbstractVector{Int},
        as::AbstractVector{Int},
        i::Int,
        k::Int,
        kexact::Int,
        window::Int,
        W::Float64,
        H::BipartiteGraph,
        rs::AbstractVector{Int},
        re::AbstractVector{Int},
        rv::Vector{Int},
        resnode::AbstractVector{Int},
        cache::Vector{Any},
        parent::AbstractVector{Int},
        next::Int,
        localid::AbstractVector{Int},
        touched::Vector{Int},
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
    ) where {S <: Store}
    M, E, ids, b, nbytes, shift = localmasks(S, chptr, ch, asptr, as, i, k, H, rs, re, rv, resnode, localid, touched, wv, acc, deg, wloc, ploc, tab, etab)

    if k <= kexact
        if S === FVector{UInt64}
            U = u1
            L = l1
        else
            U = newstore(S, bt_nwd(M), length(wid))
            L = newstore(S, bt_nwd(M), length(wid))
        end

        c = flops_dp!(wid, fl, split, U, L, tab, etab, nbytes, shift, M, M, E, 1, k, W)
        @assert c < Inf "no binarization within width $W"

        return emit_dp!(parent, next, split, (1 << k) - 1, ids)
    end

    # starts: the capped greedy, and (if pass 1 built it) the narrowest tree, which fits in W
    # by construction; each refined for flops within W; keep the cheaper
    best = nothing

    if cache[i] !== nothing
        best = reconfigure!(cache[i], M, E, 1, W, :flops, window, wid, bw, fl, split, tab, etab, nbytes, shift)
        @assert maxwidth(tab, nbytes, best) <= W + EPS
    end

    t = greedy_tree(S, tab, etab, nbytes, shift, P, Q, M, E, 1, k, b, W, true)

    if maxwidth(tab, nbytes, t) <= W + EPS
        reconfigure!(t, M, E, 1, W, :flops, window, wid, bw, fl, split, tab, etab, nbytes, shift)

        if best === nothing || treecost(tab, etab, nbytes, shift, t) < treecost(tab, etab, nbytes, shift, best)
            best = t
        end
    end

    @assert best !== nothing "no binarization within width $W"

    return emit_local!(parent, next, best, ids)
end

# ----------------------------------------------------------------------------------------
# global refinement: subtree reconfiguration across cliques, within width and tc caps
# ----------------------------------------------------------------------------------------

# A stoppable solver over a whole contraction tree. The tree is (par, left, right, legs,
# root); the caps Wcap, TCcap bound every intermediate's width and every step's tc; the DP
# scratch is reused across windows. branchrefine_step! runs one sweep, branchrefine_solve!
# runs sweeps to a fixed point.
mutable struct BranchSolver
    # tree
    par     ::FVector{Int}
    left    ::FVector{Int}
    right   ::FVector{Int}
    legs    ::Vector{Vector{Int}}
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
    # sweep bookkeeping
    nstamp  ::FVector{Int}      # when a node last changed
    solved  ::FVector{Int}      # when the window at a node was last found optimal
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
    vstamp  ::FVector{Int}      # dedup the vertices of a window's pieces
    touched ::Vector{Int}
    tick    ::Int
end

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

    deg = fill!(FVector{Int}(undef, n), 0)

    @inbounds for t in targets(H)
        deg[t] += 1
    end

    par   = FVector{Int}(undef, nn)
    left  = fill!(FVector{Int}(undef, nn), 0)
    right = fill!(FVector{Int}(undef, nn), 0)
    root  = 0

    @inbounds for x in oneto(nn)
        par[x] = parent[x]
    end

    @inbounds for x in oneto(nn)
        p = par[x]
        p == 0 && continue

        if p == point
            root = x
        elseif left[p] == 0
            left[p] = x
        else
            right[p] = x
        end
    end

    # legs of every node, bottom-up (contractions are numbered after their children)
    legs = Vector{Vector{Int}}(undef, nn)
    cnts = Vector{Vector{Int}}(undef, nn)
    acc  = fill!(FVector{Int}(undef, n), 0)
    st   = fill!(FVector{Int}(undef, n), 0)

    @inbounds for e in oneto(m)
        legs[e] = [v for v in neighbors(H, e) if deg[v] > 1]
        cnts[e] = ones(Int, length(legs[e]))
    end

    @inbounds for x in m + 1:nn
        touched = Int[]

        for c in (left[x], right[x]), (v, kk) in zip(legs[c], cnts[c])
            if st[v] != x
                st[v] = x; acc[v] = 0; push!(touched, v)
            end

            acc[v] += kk
        end

        l = Int[]; lc = Int[]

        for v in touched
            if acc[v] < deg[v]
                push!(l, v); push!(lc, acc[v])
            end
        end

        legs[x] = l; cnts[x] = lc
    end

    wv = FVector{Float64}(undef, n)

    @inbounds for v in oneto(n)
        wv[v] = weights[v]
    end

    function wleg(x)
        return sum((wv[v] for v in legs[x]); init = 0.0)
    end

    if W === nothing
        Wcap = maximum(wleg(x) for x in m + 1:nn if x != point)
    else
        Wcap = Float64(W)
    end

    # tc of a contraction: the total weight of the vertices of its two children's legs
    function tcof(x)
        s = 0.0; tick = -x

        for c in (left[x], right[x]), v in legs[c]
            if st[v] != tick
                st[v] = tick; s += wv[v]
            end
        end

        return s
    end

    if TC === nothing
        TCcap = maximum(tcof(x) for x in m + 1:nn if x != point)
    else
        TCcap = Float64(TC)
    end

    s = 1 << window; ncol = max(cld(n, 8), 1)

    nstamp  = fill!(FVector{Int}(undef, nn), 0)
    solved  = fill!(FVector{Int}(undef, nn), 0)
    clock   = 0
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
    localid = fill!(FVector{Int}(undef, n), 0)
    vstamp  = fill!(FVector{Int}(undef, n), 0)
    touched = Int[]
    tick    = 0

    return BranchSolver(
        par, left, right, legs, root, m, Int(point), nn,
        wv, deg, Wcap, TCcap, window,
        nstamp, solved, clock,
        wid, bw, fl, split, u1, l1, tab, etab, wloc, ploc, localid, vstamp, touched, tick,
    )
end

# one top-down sweep; returns whether any window improved
function branchrefine_step!(s::BranchSolver, mode::Symbol)
    m = s.m; window = s.window

    order  = Int[]
    stack  = Int[]
    pieces = Int[]
    win    = Int[]

    push!(stack, s.root)

    while !isempty(stack)                       # preorder: top-down
        x = pop!(stack)

        if x > m
            push!(order, x); push!(stack, s.right[x]); push!(stack, s.left[x])
        end
    end

    improved = false

    for x in order
        empty!(pieces); empty!(win); push!(pieces, s.left[x], s.right[x]); push!(win, x)

        while length(pieces) < window
            best = 0; bv = -Inf

            for (j, p) in enumerate(pieces)
                if p > m
                    v = sum((s.wv[u] for u in s.legs[p]); init = 0.0)
                    v > bv && (bv = v; best = j)
                end
            end

            best == 0 && break
            p = pieces[best]; pieces[best] = s.left[p]; push!(pieces, s.right[p]); push!(win, p)
        end

        np = length(pieces)
        np <= 2 && continue

        if s.solved[x] > 0
            newest = 0

            for y in Iterators.flatten((win, pieces))
                newest = max(newest, s.nstamp[y])
            end

            newest < s.solved[x] && continue
        end

        # local bits: the vertices of the pieces' legs
        s.tick += 1; tk = s.tick; touched = s.touched; empty!(touched)

        for p in pieces, v in s.legs[p]
            if s.vstamp[v] != tk
                s.vstamp[v] = tk; push!(touched, v)
            end
        end

        if solve_window!(s, storetype(touched), x, pieces, win, mode)
            s.clock += 1

            for y in win
                s.nstamp[y] = s.clock
            end

            improved = true
        else
            s.clock += 1; s.solved[x] = s.clock
        end
    end

    return improved
end

# sweeps of one mode to a fixed point (or `maxsweeps`)
function branchrefine_solve!(s::BranchSolver, mode::Symbol; maxsweeps::Integer = 8)
    fill!(s.solved, 0)

    for _ in 1:maxsweeps
        branchrefine_step!(s, mode) || break
    end

    return s
end

# re-solve the window (pieces under x); returns true if it was replaced
function solve_window!(s::BranchSolver, ::Type{S}, x::Int, pieces::Vector{Int}, win::Vector{Int}, mode::Symbol) where {S <: Store}
    touched = s.touched; b = length(touched); nw = max(cld(b, 64), 1); np = length(pieces)

    @inbounds for (j, v) in enumerate(touched)
        s.localid[v] = j - 1; s.wloc[j] = s.wv[v]
    end

    nbytes, shift = buildtables!(s.tab, s.etab, s.ploc, s.wloc, b)

    # columns 1:np the pieces, np+1:np+|win| the window's nodes (their legs ⊆ the pieces')
    nwin = length(win)
    X = newstore(S, nw, np + nwin)

    function col(y)
        j = findfirst(==(y), pieces)

        if j === nothing
            return np + findfirst(==(y), win)
        else
            return j
        end
    end

    for (j, y) in enumerate(Iterators.flatten((pieces, win)))
        for v in s.legs[y]
            msetbit!(X, j, s.localid[v])
        end
    end

    tcf = mexp2(shift, s.TCcap) * (1 + 1e-12)

    if S === FVector{UInt64}
        U = s.u1
        L = s.l1
    else
        U = newstore(S, nw, length(s.wid))
        L = newstore(S, nw, length(s.wid))
    end

    if mode === :width
        oldw = 0.0

        for y in win
            oldw = max(oldw, mwt(s.tab, nbytes, X, col(y)))
        end

        bwin = width_dp!(s.wid, s.bw, U, L, s.tab, s.etab, nbytes, shift, X, X, X, np + 1, np; tcf)     # E = legs of x (column np + 1)
        bwin < oldw - EPS || return false

        c = flops_dp!(s.wid, s.fl, s.split, U, L, s.tab, s.etab, nbytes, shift, X, X, X, np + 1, np, bwin; tcf)
        c < Inf || return false
    else
        oldc = 0.0

        for y in win
            oldc += mflopsor(s.tab, s.etab, nbytes, shift, X, col(s.left[y]), X, col(s.right[y]))
        end

        c = flops_dp!(s.wid, s.fl, s.split, U, L, s.tab, s.etab, nbytes, shift, X, X, X, np + 1, np, s.Wcap; tcf)
        c < oldc * (1 - 1e-12) || return false
    end

    # rebuild: x stays on top, the other window nodes are reused; legs from the DP's masks
    full = (1 << np) - 1; slot = Ref(1)

    function build(T)
        if T & (T - 1) == 0
            return pieces[trailing_zeros(T) + 1]
        end

        A = s.split[T + 1]

        if T == full
            y = x
        else
            slot[] += 1
            y = win[slot[]]
        end

        l = build(A); r = build(T ⊻ A)
        s.left[y] = l; s.right[y] = r; s.par[l] = y; s.par[r] = y

        if y != x
            lg = Int[]

            for i in oneto(nw)
                word = bt_get(L, i, T + 1)

                while word != 0
                    push!(lg, touched[64 * (i - 1) + trailing_zeros(word) + 1]); word &= word - 1
                end
            end

            s.legs[y] = lg
        end

        return y
    end

    build(full)

    return true
end

# the refined tree as a new parent vector, contractions renumbered children first
function branchrefine_parent(s::BranchSolver)
    nn = s.nn; m = s.m; point = s.point

    newid = fill!(FVector{Int}(undef, nn), 0)
    next = m
    stack = [(s.root, false)]

    while !isempty(stack)
        x, done = pop!(stack)

        if x <= m
            newid[x] = x
        elseif done
            next += 1; newid[x] = next
        else
            push!(stack, (x, true)); push!(stack, (s.right[x], false)); push!(stack, (s.left[x], false))
        end
    end

    out = fill!(FVector{Int}(undef, nn), 0)

    @inbounds for x in oneto(nn)
        x == point && continue
        p = s.par[x]

        if p == point
            out[newid[x]] = point
        else
            out[newid[x]] = newid[p]
        end
    end

    return out
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
        branchrefine_solve!(s, :width; maxsweeps)
        s.Wcap = maximum(sum((s.wv[v] for v in s.legs[x]); init = 0.0) for x in s.m + 1:s.nn if x != s.point)
    end

    branchrefine_solve!(s, :flops; maxsweeps)

    return branchrefine_parent(s)
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

    if any(iszero, weights)       # drop dimension-1 indices (they cost nothing)
        keep = findall(!iszero, weights)
        new = fill!(FVector{Int}(undef, n), 0)

        for (j, v) in enumerate(keep)
            new[v] = j
        end

        H = restrict(H, new, length(keep))
        weights = weights[keep]
        alg isa AbstractVector && (alg = [new[v] for v in alg if new[v] > 0])
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

    nc, root, chptr, ch, asptr, as, scalarroots, free = cliques(tree, label, H, Hr, point, n, m)

    # per-vertex scratch and the output
    acc     = fill!(FVector{Int}(undef, n), 0)
    stamp   = fill!(FVector{Int}(undef, n), 0)
    localid = fill!(FVector{Int}(undef, n), 0)
    deg     = fill!(FVector{Int}(undef, n), 0)
    wv      = FVector{Float64}(undef, n)
    touched = Int[]
    parent  = fill!(FVector{Int}(undef, 2m - 2), 0)
    tick    = 0
    next    = m + 1

    @inbounds for v in oneto(n)
        wv[v] = weights[v]; deg[v] = length(neighbors(Hr, v))
    end

    rs      = fill!(FVector{Int}(undef, nc), 0)      # result legs of clique i: rv[rs[i]:re[i]-1]
    re      = fill!(FVector{Int}(undef, nc), 0)
    rv      = Int[]
    rc      = Int[]
    nitems  = fill!(FVector{Int}(undef, nc), 0)
    tw      = fill!(FVector{Float64}(undef, nc), 0.0)
    resnode = fill!(FVector{Int}(undef, nc), 0)
    cache   = fill!(Vector{Any}(undef, nc), nothing)

    # pass 1a, bottom-up: the legs of every clique's merge; W starts at the widest result.
    # Record the largest merge (bmax vertices, kmax items) to size the DP scratch.
    W = 0.0; bmax = 0; kmax = 0

    @inbounds for i in oneto(nc)
        tick += 1
        k = gather!(chptr, ch, asptr, as, i, H, rs, re, rv, rc, acc, stamp, deg, touched, tick)
        nitems[i] = k; bmax = max(bmax, length(touched)); kmax = max(kmax, k)
        k == 0 && continue
        rs[i] = length(rv) + 1; sacc = 0.0

        for v in touched
            # a single item passes up unchanged
            if k == 1 || acc[v] < deg[v]
                push!(rv, v); push!(rc, acc[v]); sacc += wv[v]
            end
        end

        re[i] = length(rv) + 1
        k >= 2 && (W = max(W, sacc))
        tw[i] = touchedweight(wv, touched)
    end

    # DP / byte-table / greedy scratch, sized from the observed maxima
    s = 1 << kexact; ncol = max(cld(bmax, 8), 1); kpq = max(kmax, 1)
    wid   = FVector{Float64}(undef, s)
    bw    = FVector{Float64}(undef, s)
    fl    = FVector{Float64}(undef, s)
    split = FVector{Int}(undef, s)
    u1    = FVector{UInt64}(undef, s)
    l1    = FVector{UInt64}(undef, s)
    wloc  = FVector{Float64}(undef, n)
    ploc  = FVector{Float64}(undef, n)
    tab   = FMatrix{Float64}(undef, 256, ncol)
    etab  = FMatrix{Float64}(undef, 256, ncol)
    P     = FMatrix{Float64}(undef, kpq, kpq)
    Q     = FMatrix{Float64}(undef, kpq, kpq)

    # pass 1b: the smallest width of each merge of k ≥ 3 items, heaviest first; a merge
    # whose vertices weigh at most W in total cannot raise W
    big = [i for i in 1:nc if nitems[i] >= 3]
    sort!(big; by = i -> tw[i], rev = true)

    @inbounds for i in big
        tw[i] <= W + EPS && break
        tick += 1
        gather!(chptr, ch, asptr, as, i, H, rs, re, rv, rc, acc, stamp, deg, touched, tick)
        wi = clique_width(storetype(touched), chptr, ch, asptr, as, i, nitems[i], kexact, window, H, rs, re, rv, resnode, cache,
            localid, touched, wv, acc, deg, wloc, ploc, tab, etab, wid, bw, fl, split, u1, l1, P, Q)
        W = max(W, wi)
    end

    # pass 2, bottom-up: the cheapest binarization of each clique within W
    @inbounds for i in oneto(nc)
        k = nitems[i]
        k == 0 && continue
        tick += 1
        gather!(chptr, ch, asptr, as, i, H, rs, re, rv, rc, acc, stamp, deg, touched, tick)

        if k == 1
            resnode[i] = firstitem(chptr, ch, asptr, as, i, rs, resnode)
        elseif k == 2
            a, c = twoitems(chptr, ch, asptr, as, i, rs, resnode)
            x = next; next += 1; parent[a] = x; parent[c] = x; resnode[i] = x
        else
            resnode[i], next = clique_build(storetype(touched), chptr, ch, asptr, as, i, k, kexact, window, W, H, rs, re, rv, resnode, cache, parent, next,
                localid, touched, wv, acc, deg, wloc, ploc, tab, etab, wid, bw, fl, split, u1, l1, P, Q)
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

    for t in Iterators.flatten(((resnode[c] for c in scalarroots if nitems[c] >= 1), free))
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

# every tensor is a scalar: multiply them in a chain
function scalartree(m::Int, point::Int)
    @assert m >= 2 "a contraction needs at least one tensor besides the point"
    parent = fill!(FVector{Int}(undef, 2m - 2), 0)
    next = m + 1; s = 0

    for e in oneto(m)
        e == point && continue

        if s == 0
            s = e
        else
            parent[s] = next; parent[e] = next; s = next; next += 1
        end
    end

    parent[s] = point

    return parent
end

function firstitem(
        chptr::AbstractVector{Int},
        ch::AbstractVector{Int},
        asptr::AbstractVector{Int},
        as::AbstractVector{Int},
        i::Int,
        rs::AbstractVector{Int},
        resnode::AbstractVector{Int},
    )
    @inbounds for p in chptr[i]:chptr[i + 1] - 1
        c = ch[p]

        if rs[c] != 0
            return resnode[c]
        end
    end

    return @inbounds as[asptr[i]]
end

function twoitems(
        chptr::AbstractVector{Int},
        ch::AbstractVector{Int},
        asptr::AbstractVector{Int},
        as::AbstractVector{Int},
        i::Int,
        rs::AbstractVector{Int},
        resnode::AbstractVector{Int},
    )
    a = 0

    @inbounds for p in chptr[i]:chptr[i + 1] - 1
        c = ch[p]
        rs[c] == 0 && continue

        if a == 0
            a = resnode[c]
        else
            return a, resnode[c]
        end
    end

    @inbounds for p in asptr[i]:asptr[i + 1] - 1
        e = as[p]

        if a == 0
            a = e
        else
            return a, e
        end
    end

    error("clique $i has fewer than two items")
end

# Children and assigned hyperedges of each clique, as CSRs (chptr/ch, asptr/as). The roots
# of the other components of a forest (their results are scalars) and the hyperedges
# without vertices are multiplied in at the end, outside the root's merge. The point is
# assigned nowhere: the root's merge is the output.
function cliques(tree::CliqueTree, label::Vector{Int}, H::BipartiteGraph, Hr::BipartiteGraph, point::Int, n::Int, m::Int)
    nc = length(tree)

    prnt = fill!(FVector{Int}(undef, nc), 0)

    @inbounds for i in oneto(nc)
        p = AbstractTrees.parentindex(tree, i)

        if isnothing(p)
            prnt[i] = 0
        else
            prnt[i] = p
        end
    end

    cliqueof = fill!(FVector{Int}(undef, n), 0)

    @inbounds for i in oneto(nc), q in residual(tree, i)
        cliqueof[label[q]] = i
    end

    pv = neighbors(H, point)

    if isempty(pv)
        root = findlast(iszero, prnt)
    else
        root = cliqueof[first(pv)]
    end

    @assert prnt[root] == 0 "the point's clique is not a root"
    @assert root == nc "the root clique must come last in postorder"

    nscalar = 0

    @inbounds for i in oneto(nc)
        (prnt[i] == 0 && i != root) && (nscalar += 1)
    end

    scalarroots = FVector{Int}(undef, nscalar)
    j = 0

    @inbounds for i in oneto(nc)
        if prnt[i] == 0 && i != root
            j += 1; scalarroots[j] = i
        end

        prnt[i] == 0 || @assert prnt[i] > i "the clique tree is not in postorder"
    end

    chptr = fill!(FVector{Int}(undef, nc + 1), 0)
    chptr[1] = 1

    @inbounds for i in oneto(nc)
        prnt[i] != 0 && (chptr[prnt[i] + 1] += 1)
    end

    @inbounds for i in oneto(nc)
        chptr[i + 1] += chptr[i]
    end

    ch = FVector{Int}(undef, chptr[nc + 1] - 1)
    fillp = copy(chptr)

    @inbounds for i in oneto(nc)
        p = prnt[i]

        if p != 0
            ch[fillp[p]] = i; fillp[p] += 1
        end
    end

    # assignment: the first clique (in postorder) whose residual holds a vertex of e
    owner = fill!(FVector{Int}(undef, m), 0)

    @inbounds for i in oneto(nc), q in residual(tree, i)
        v = label[q]

        for e in neighbors(Hr, v)
            (e != point && owner[e] == 0) && (owner[e] = i)
        end
    end

    nfree = 0

    @inbounds for e in oneto(m)
        (e != point && owner[e] == 0) && (nfree += 1)
    end

    free = FVector{Int}(undef, nfree)
    j = 0

    @inbounds for e in oneto(m)
        if e != point && owner[e] == 0
            j += 1; free[j] = e
        end
    end

    asptr = fill!(FVector{Int}(undef, nc + 1), 0)
    asptr[1] = 1

    @inbounds for e in oneto(m)
        owner[e] != 0 && (asptr[owner[e] + 1] += 1)
    end

    @inbounds for i in oneto(nc)
        asptr[i + 1] += asptr[i]
    end

    as = FVector{Int}(undef, asptr[nc + 1] - 1)
    fillp = copy(asptr)

    @inbounds for e in oneto(m)
        i = owner[e]

        if i != 0
            as[fillp[i]] = e; fillp[i] += 1
        end
    end

    return nc, root, chptr, ch, asptr, as, scalarroots, free
end

# a copy of the hypergraph without repeated vertices in a hyperedge
function simplehypergraph(H::BipartiteGraph)
    m = nv(H); n = nov(H)

    mark = fill!(FVector{Int}(undef, n), 0)
    ptr = FVector{Int}(undef, m + 1)
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

    tgt = FVector{Int}(undef, ptr[m + 1] - 1)
    fill!(mark, 0)

    @inbounds for e in oneto(m)
        p = ptr[e]

        for v in neighbors(H, e)
            if mark[v] != e
                mark[v] = e; tgt[p] = v; p += 1
            end
        end
    end

    return BipartiteGraph{Int, Int}(n, m, length(tgt), ptr, tgt)
end

# the hypergraph on the vertices v with new[v] > 0, renumbered to new[v]
function restrict(H::BipartiteGraph, new::AbstractVector{Int}, n2::Int)
    m = nv(H)

    ptr = FVector{Int}(undef, m + 1)
    ptr[1] = 1

    @inbounds for e in oneto(m)
        c = 0

        for v in neighbors(H, e)
            ispositive(new[v]) && (c += 1)
        end

        ptr[e + 1] = ptr[e] + c
    end

    tgt = FVector{Int}(undef, ptr[m + 1] - 1)

    @inbounds for e in oneto(m)
        p = ptr[e]

        for v in neighbors(H, e)
            if ispositive(new[v])
                tgt[p] = new[v]; p += 1
            end
        end
    end

    return BipartiteGraph{Int, Int}(n2, m, length(tgt), ptr, tgt)
end

# ----------------------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------------------

"""
    branchstats(weights, hypergraph, point, parent) -> (; width, peak, cost)

Width, contraction complexity and log2 of the total cost of any contraction tree given as
a parent vector (see the module docstring for the definitions).
"""
function branchstats(weights::AbstractVector, hypergraph::BipartiteGraph, point::Integer, parent::AbstractVector{<:Integer})
    H = simplehypergraph(hypergraph)
    m = nv(H); n = nov(H)
    nn = length(parent); @assert nn == 2m - 2

    deg = fill!(FVector{Int}(undef, n), 0)

    @inbounds for t in targets(H)
        deg[t] += 1
    end

    kids = [Int[] for _ in 1:nn]

    @inbounds for x in oneto(nn)
        parent[x] == 0 || push!(kids[parent[x]], x)
    end

    @assert parent[point] == 0 && length(kids[point]) == 1

    legs = Vector{Vector{Int}}(undef, nn)
    cnts = Vector{Vector{Int}}(undef, nn)

    @inbounds for e in oneto(m)
        legs[e] = [v for v in neighbors(H, e) if deg[v] > 1]
        cnts[e] = ones(Int, length(legs[e]))
    end

    acc   = fill!(FVector{Int}(undef, n), 0)
    stamp = fill!(FVector{Int}(undef, n), 0)
    width = 0.0; tc = 0.0; costs = Float64[]

    @inbounds for x in m + 1:nn
        @assert length(kids[x]) == 2 "node $x is not binary"
        touched = Int[]

        for c in kids[x], (v, k) in zip(legs[c], cnts[c])
            if stamp[v] != x
                stamp[v] = x; acc[v] = 0; push!(touched, v)
            end

            acc[v] += k
        end

        l = Int[]; lc = Int[]

        for v in touched
            if acc[v] < deg[v]
                push!(l, v); push!(lc, acc[v])
            end
        end

        legs[x] = l; cnts[x] = lc
        involved = sum(weights[v] for v in touched; init = 0.0)
        width = max(width, sum(weights[v] for v in l; init = 0.0))
        tc = max(tc, involved); push!(costs, involved)
    end

    if isempty(costs)
        mx = 0.0
        log2cost = -Inf
    else
        mx = maximum(costs)
        log2cost = mx + log2(sum(exp2(c - mx) for c in costs))
    end

    return (; width, peak = tc, cost = log2cost)
end

"""
    branchssa(parent, point) -> Vector{Tuple{Int, Int}}

The contraction as a 0-based SSA path over the tensors (the hyperedges other than the
point, in order), e.g. for cotengra's `ContractionTree.from_path(..., ssa_path=...)`.
"""
function branchssa(parent::AbstractVector{<:Integer}, point::Integer)
    nn = length(parent); m = (nn + 2) ÷ 2

    id = fill!(FVector{Int}(undef, nn), 0)
    t = 0

    for e in oneto(m)
        e == point && continue
        id[e] = t; t += 1
    end

    for x in m + 1:nn
        id[x] = t; t += 1
    end

    kids = [Int[] for _ in 1:nn]

    for x in oneto(nn)
        (parent[x] == 0 || parent[x] == point) || push!(kids[parent[x]], x)
    end

    return [(id[kids[x][1]], id[kids[x][2]]) for x in m + 1:nn]
end

# thin accessors over branchstats (one bottom-up pass each)
function branchwidth(weights, hypergraph, point, parent)
    return branchstats(weights, hypergraph, point, parent).width
end

function branchpeak(weights, hypergraph, point, parent)
    return branchstats(weights, hypergraph, point, parent).peak
end

function branchcost(weights, hypergraph, point, parent)
    return branchstats(weights, hypergraph, point, parent).cost
end
