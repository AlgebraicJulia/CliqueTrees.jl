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
# branchrefine is an optional post-pass (subtree reconfiguration; never wider).
# branchssa converts the parent vector to an SSA contraction path.
#
const EPS = 1e-9

# ----------------------------------------------------------------------------------------
# bit masks over the vertices of one clique, as in CliqueTrees' bt: a store of masks is a
# Vector{UInt64} (one word per mask, the fast path) or a Matrix{UInt64} (one column of
# words per mask, for cliques with more than 64 vertices)
# ----------------------------------------------------------------------------------------

const Store = Union{Vector{UInt64}, Matrix{UInt64}}

@inline nwd(::Vector{UInt64}) = 1
@inline nwd(X::Matrix{UInt64}) = size(X, 1)
@inline getw(X::Vector{UInt64}, i::Int, v::Int) = @inbounds X[v]
@inline getw(X::Matrix{UInt64}, i::Int, v::Int) = @inbounds X[i, v]
@inline putw!(X::Vector{UInt64}, w::UInt64, i::Int, v::Int) = (@inbounds X[v] = w; nothing)
@inline putw!(X::Matrix{UInt64}, w::UInt64, i::Int, v::Int) = (@inbounds X[i, v] = w; nothing)

newstore(::Type{Vector{UInt64}}, nw::Int, n::Int) = zeros(UInt64, n)
newstore(::Type{Matrix{UInt64}}, nw::Int, n::Int) = zeros(UInt64, nw, n)

@inline function mclear!(X::Store, v::Int)
    for i in 1:nwd(X)
        putw!(X, zero(UInt64), i, v)
    end
end
# Y[v] = X[u]
@inline function mcopy!(Y::S, v::Int, X::S, u::Int) where {S <: Store}
    for i in 1:nwd(Y)
        putw!(Y, getw(X, i, u), i, v)
    end
end
@inline function msetbit!(X::Store, v::Int, j::Int)
    i = (j >> 6) + 1
    putw!(X, getw(X, i, v) | (UInt64(1) << (j & 63)), i, v)
end
@inline function mclearbit!(X::Store, v::Int, j::Int)
    i = (j >> 6) + 1
    putw!(X, getw(X, i, v) & ~(UInt64(1) << (j & 63)), i, v)
end
# Y[v] = X[a] | Z[b]
@inline function mor!(Y::S, v::Int, X::S, a::Int, Z::S, b::Int) where {S <: Store}
    for i in 1:nwd(Y)
        putw!(Y, getw(X, i, a) | getw(Z, i, b), i, v)
    end
end
# Y[v] = X[a] & (Z[b] | F[c]): the vertices of X[a] that also occur in Z[b] or F[c]
@inline function mlegs!(Y::S, v::Int, X::S, a::Int, Z::S, b::Int, F::S, c::Int) where {S <: Store}
    for i in 1:nwd(Y)
        putw!(Y, getw(X, i, a) & (getw(Z, i, b) | getw(F, i, c)), i, v)
    end
end

# ----------------------------------------------------------------------------------------
# workspace
# ----------------------------------------------------------------------------------------

mutable struct Workspace
    # per vertex
    acc::Vector{Int}        # occurrence counts in the current merge
    stamp::Vector{Int}
    localid::Vector{Int}    # bit of a vertex in the current clique (0-based)
    deg::Vector{Int}        # number of hyperedges holding the vertex (point included)
    weights::Vector{Float64}
    # current clique
    touched::Vector{Int}    # distinct vertices of the current merge
    wloc::Vector{Float64}   # weights of the local bits
    ploc::Vector{Float64}   # 2^wloc
    nbytes::Int
    shift::Float64          # flops are 2^(weight - shift); shift > 0 only for huge cliques
    tab::Matrix{Float64}    # tab[x + 1, j]: weight of byte value x in byte position j
    etab::Matrix{Float64}   # 2^tab (used when shift == 0)
    # subset dynamic programming (length 2^kexact)
    wid::Vector{Float64}
    bw::Vector{Float64}
    fl::Vector{Float64}
    split::Vector{Int}
    U1::Vector{UInt64}      # DP masks, one word
    L1::Vector{UInt64}
    UW::Matrix{UInt64}      # DP masks, several words
    LW::Matrix{UInt64}
    # greedy
    P::Matrix{Float64}
    Q::Matrix{Float64}
    # output
    parent::Vector{Int}
    next::Int
    tick::Int
end

function Workspace(n::Int, m::Int, kexact::Int)
    s = 1 << kexact
    return Workspace(zeros(Int, n), zeros(Int, n), zeros(Int, n), zeros(Int, n), zeros(Float64, n),
        Int[], Float64[], Float64[], 1, 0.0, zeros(Float64, 256, 8), ones(Float64, 256, 8),
        zeros(Float64, s), zeros(Float64, s), zeros(Float64, s), zeros(Int, s),
        zeros(UInt64, s), zeros(UInt64, s), zeros(UInt64, 0, s), zeros(UInt64, 0, s),
        zeros(Float64, 0, 0), zeros(Float64, 0, 0), zeros(Int, 2m - 2), m + 1, 0)
end

dpbufs(ws::Workspace, ::Type{Vector{UInt64}}, nw::Int) = (ws.U1, ws.L1)
function dpbufs(ws::Workspace, ::Type{Matrix{UInt64}}, nw::Int)
    if size(ws.UW, 1) != nw
        ws.UW = zeros(UInt64, nw, length(ws.wid)); ws.LW = zeros(UInt64, nw, length(ws.wid))
    end
    return (ws.UW, ws.LW)
end

# byte tables of the local weights: the weight of a mask is a sum of nbytes lookups and its
# flops a product of nbytes lookups (both tables built by recurrences, without exp2)
function buildtables!(ws::Workspace, b::Int)
    nbytes = max(cld(b, 8), 1)
    if size(ws.tab, 2) < nbytes
        ws.tab = zeros(Float64, 256, nbytes); ws.etab = ones(Float64, 256, nbytes)
    end
    tab = ws.tab; etab = ws.etab; wloc = ws.wloc; ploc = ws.ploc
    total = 0.0
    @inbounds for j in 1:b
        total += wloc[j]
    end
    # flops up to 2^total: exact products of byte factors when total ≤ 900, else
    # 2^(weight - shift) through one exp2 (mflopsor)
    shift = max(0.0, total - 900.0)
    if shift == 0.0
        resize!(ploc, b)
        @inbounds for j in 1:b
            ploc[j] = exp2(wloc[j])
        end
    end
    @inbounds for j in 1:nbytes
        tab[1, j] = 0.0; etab[1, j] = 1.0; base = 8 * (j - 1)
        for x in 1:255
            bit = base + trailing_zeros(x); prev = (x & (x - 1)) + 1
            inside = bit < b
            tab[x + 1, j] = tab[prev, j] + (inside ? wloc[bit + 1] : 0.0)
            shift == 0.0 && (etab[x + 1, j] = etab[prev, j] * (inside ? ploc[bit + 1] : 1.0))
        end
    end
    ws.nbytes = nbytes; ws.shift = shift
    return
end

# weight of the mask X[v]
@inline function mwt(ws::Workspace, X::Store, v::Int)
    s = 0.0; tab = ws.tab
    @inbounds for jj in 0:ws.nbytes - 1
        s += tab[((getw(X, (jj >> 3) + 1, v) >> ((jj & 7) << 3)) & 0xff) + 1, jj + 1]
    end
    return s
end

# flops 2^(w(X[a] | Z[b]) - shift)
@inline function mflopsor(ws::Workspace, X::S, a::Int, Z::S, b::Int) where {S <: Store}
    nb = ws.nbytes
    if ws.shift > 0.0
        s = 0.0; tab = ws.tab
        @inbounds for jj in 0:nb - 1
            w = (jj >> 3) + 1
            s += tab[(((getw(X, w, a) | getw(Z, w, b)) >> ((jj & 7) << 3)) & 0xff) + 1, jj + 1]
        end
        return exp2(s - ws.shift)
    end
    p = 1.0; etab = ws.etab
    @inbounds for jj in 0:nb - 1
        w = (jj >> 3) + 1
        p *= etab[(((getw(X, w, a) | getw(Z, w, b)) >> ((jj & 7) << 3)) & 0xff) + 1, jj + 1]
    end
    return p
end

@inline mexp2(ws::Workspace, w::Float64) = exp2(w - ws.shift)

# ----------------------------------------------------------------------------------------
# subset dynamic programming (k ≤ kexact items)
# ----------------------------------------------------------------------------------------

# Items i = 1:k with union masks Mu[i] (the vertices below them that still matter) and legs
# Ml[i]; E[e] = legs of the whole merge. U[S] = union of S, L[S] = legs of the merge of S
# (Ml[i] for a singleton), wid[S] = w(L[S]) (0 for singletons: items are counted where
# they are made). For a whole clique Mu = Ml.
function prepare_dp!(ws::Workspace, Mu::S, Ml::S, E::S, e::Int, k::Int) where {S <: Store}
    U, L = dpbufs(ws, S, nwd(Mu))
    full = (1 << k) - 1; wid = ws.wid
    @inbounds begin
        mclear!(U, 1)
        for T in 1:full
            mor!(U, T + 1, U, (T & (T - 1)) + 1, Mu, trailing_zeros(T) + 1)
        end
        for T in 1:full - 1
            if T & (T - 1) == 0
                mcopy!(L, T + 1, Ml, trailing_zeros(T) + 1); wid[T + 1] = 0.0
            else
                mlegs!(L, T + 1, U, T + 1, U, (full ⊻ T) + 1, E, e); wid[T + 1] = mwt(ws, L, T + 1)
            end
        end
        mcopy!(L, full + 1, E, e); wid[full + 1] = mwt(ws, L, full + 1)
    end
    return U, L
end

# smallest achievable width of the merge (largest intermediate, the merge's result included)
function width_dp!(ws::Workspace, Mu::S, Ml::S, E::S, e::Int, k::Int; tcf::Float64 = Inf) where {S <: Store}
    _, L = prepare_dp!(ws, Mu, Ml, E, e, k)
    full = (1 << k) - 1; wid = ws.wid; bw = ws.bw
    @inbounds for T in 1:full
        if T & (T - 1) == 0
            bw[T + 1] = 0.0; continue
        end
        low = T & -T; rest = T ⊻ low; best = Inf; wt = wid[T + 1]
        sub = (rest - 1) & rest
        while true
            A = low | sub
            c = max(bw[A + 1], bw[(T ⊻ A) + 1])
            # a step above the tc cap (given as a flops threshold) is not allowed
            (tcf < Inf && c < best && mflopsor(ws, L, A + 1, L, (T ⊻ A) + 1) > tcf) && (c = Inf)
            c < best && (best = c)
            (best <= wt || sub == 0) && break      # cannot get below the node itself
            sub = (sub - 1) & rest
        end
        bw[T + 1] = max(wt, best)
    end
    return @inbounds bw[full + 1]
end

# fewest flops with every intermediate below the top of width ≤ W; returns the (scaled)
# cost, Inf if infeasible, and leaves the best splits in ws.split
function flops_dp!(ws::Workspace, Mu::S, Ml::S, E::S, e::Int, k::Int, W::Float64; tcf::Float64 = Inf) where {S <: Store}
    U, L = prepare_dp!(ws, Mu, Ml, E, e, k)
    full = (1 << k) - 1; wid = ws.wid; fl = ws.fl; split = ws.split
    @inbounds for T in 1:full
        if T & (T - 1) == 0
            fl[T + 1] = 0.0; continue
        end
        split[T + 1] = 0
        if T != full && wid[T + 1] > W + EPS
            fl[T + 1] = Inf; continue
        end
        # a step's indices include the result's, so it costs at least 2^wid[T]
        low = T & -T; rest = T ⊻ low; best = Inf; bestA = 0
        lb = mexp2(ws, wid[T + 1]) * (1 - 1e-12)
        sub = (rest - 1) & rest
        while true
            A = low | sub; B = T ⊻ A
            fa = fl[A + 1]; fb = fl[B + 1]
            if fa + fb + lb < best
                step = mflopsor(ws, L, A + 1, L, B + 1)
                c = step > tcf ? Inf : fa + fb + step
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

# emit the splits of the last flops_dp! as global nodes; returns the root
function emit_dp!(ws::Workspace, T::Int, ids::Vector{Int})
    T & (T - 1) == 0 && return @inbounds ids[trailing_zeros(T) + 1]
    A = @inbounds ws.split[T + 1]
    a = emit_dp!(ws, A, ids); b = emit_dp!(ws, T ⊻ A, ids)
    x = ws.next; ws.next += 1
    @inbounds ws.parent[a] = x; ws.parent[b] = x
    return x
end

# ----------------------------------------------------------------------------------------
# large merges (k > kexact items): greedy, then subtree reconfiguration
# ----------------------------------------------------------------------------------------

# A binary tree over the k items of a merge: leaves 1:k, internal nodes k+1:2k-1.
mutable struct LocalTree{S <: Store}
    k::Int
    left::Vector{Int}
    right::Vector{Int}
    U::S                    # union of the items below
    L::S                    # legs
    out::S                  # union of the items not below
    root::Int
    order::Vector{Int}      # internal nodes, children before parents
    stack::Vector{Int}
    stamp::Vector{Int}      # when the node last changed (reconfigure!)
end

function LocalTree(::Type{S}, nw::Int, k::Int) where {S <: Store}
    nn = 2k - 1
    return LocalTree{S}(k, zeros(Int, nn), zeros(Int, nn), newstore(S, nw, nn), newstore(S, nw, nn),
        newstore(S, nw, nn), 0, Int[], Int[], zeros(Int, nn))
end

# recompute U, out and L of every node after the tree's shape changed
function recompute!(t::LocalTree{S}, M::S, E::S, e::Int) where {S <: Store}
    k = t.k; order = t.order; stack = t.stack
    empty!(order); empty!(stack); push!(stack, t.root)
    @inbounds while !isempty(stack)            # preorder, reversed below
        x = pop!(stack)
        if x > k
            push!(order, x); push!(stack, t.left[x]); push!(stack, t.right[x])
        end
    end
    reverse!(order)
    @inbounds begin
        for i in 1:k
            mcopy!(t.U, i, M, i); mcopy!(t.L, i, M, i)
        end
        for x in order
            mor!(t.U, x, t.U, t.left[x], t.U, t.right[x])
        end
        mclear!(t.out, t.root)
        for x in Iterators.reverse(order)
            l = t.left[x]; r = t.right[x]
            mor!(t.out, l, t.out, x, t.U, r); mor!(t.out, r, t.out, x, t.U, l)
            mlegs!(t.L, x, t.U, x, t.out, x, E, e)
        end
    end
    return t
end

function maxwidth(ws::Workspace, t::LocalTree)
    w = 0.0
    for x in t.order
        w = max(w, mwt(ws, t.L, x))
    end
    return w
end

function treecost(ws::Workspace, t::LocalTree)
    c = 0.0
    for x in t.order
        c += mflopsor(ws, t.L, t.left[x], t.L, t.right[x])
    end
    return c
end

# legs of the merge of nodes a and c, written to G[3]: the vertices of a ∪ c that occur in
# another node or in E. A vertex in exactly one of a, c needs one other holder (two in
# all); in both, a third. G[1] and G[2] mark the vertices with at least 2 and 3 holders.
@inline function pairlegs!(G::S, UA::S, a::Int, c::Int, E::S, e::Int) where {S <: Store}
    for i in 1:nwd(G)
        ua = getw(UA, i, a); uc = getw(UA, i, c)
        others = (getw(G, i, 1) & (ua ⊻ uc)) | (getw(G, i, 2) & ua & uc)
        putw!(G, (ua | uc) & (others | getw(E, i, e)), i, 3)
    end
end

# score of merging a and c: (primary, secondary), smaller is better.
# capped = false: (result width, flops). capped = true: (0, memory change 2^out - 2^a - 2^c)
# when the result fits in W, else (result width, flops).
@inline function pairscore(ws::Workspace, G::S, UA::S, LG::S, ew, a, c, E::S, e, W, capped) where {S <: Store}
    pairlegs!(G, UA, a, c, E, e); wo = mwt(ws, G, 3)
    if capped && wo <= W + EPS
        return 0.0, mexp2(ws, wo) - (@inbounds ew[a]) - (@inbounds ew[c])
    end
    return wo, mflopsor(ws, LG, a, LG, c)
end

# agglomerative binarization: repeatedly merge the best-scoring pair
function greedy_tree(ws::Workspace, M::S, E::S, e::Int, k::Int, b::Int, W::Float64, capped::Bool) where {S <: Store}
    nw = nwd(M)
    t = LocalTree(S, nw, k)
    UA = copy(M); LG = copy(M); G = newstore(S, nw, 3)
    ew = zeros(Float64, k); alive = trues(k); node = collect(1:k)
    cnt = zeros(Int, b)
    @inbounds for i in 1:k, w in 1:nw
        x = getw(M, w, i)
        while x != 0
            cnt[64 * (w - 1) + trailing_zeros(x) + 1] += 1; x &= x - 1
        end
    end
    @inbounds for j in 0:b - 1
        c = cnt[j + 1]
        c >= 2 && msetbit!(G, 1, j); c >= 3 && msetbit!(G, 2, j)
    end
    @inbounds for i in 1:k
        ew[i] = mexp2(ws, mwt(ws, M, i))
    end
    if size(ws.P, 1) < k
        ws.P = fill(Inf, k, k); ws.Q = fill(Inf, k, k)
    end
    P = ws.P; Q = ws.Q                       # scores of pairs (a < c) at [c, a]
    @inbounds for a in 1:k, c in a + 1:k
        P[c, a], Q[c, a] = pairscore(ws, G, UA, LG, ew, a, c, E, e, W, capped)
    end
    @inbounds for step in 1:k - 1
        ba = 0; bc = 0; bp = Inf; bq = Inf
        for a in 1:k
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
        for w in 1:nw                        # holders of the shared vertices: two become one
            x = getw(UA, w, ba) & getw(UA, w, bc)
            while x != 0
                j = 64 * (w - 1) + trailing_zeros(x); x &= x - 1
                cnt[j + 1] -= 1; cj = cnt[j + 1]
                cj < 3 && mclearbit!(G, 2, j); cj < 2 && mclearbit!(G, 1, j)
            end
        end
        mor!(UA, ba, UA, ba, UA, bc); mcopy!(LG, ba, G, 3)
        ew[ba] = mexp2(ws, mwt(ws, G, 3)); alive[bc] = false
        x = k + step; t.left[x] = node[ba]; t.right[x] = node[bc]; node[ba] = x
        for c in 1:k
            (alive[c] && c != ba) || continue
            a1, c1 = minmax(ba, c)
            P[c1, a1], Q[c1, a1] = pairscore(ws, G, UA, LG, ew, a1, c1, E, e, W, capped)
        end
    end
    t.root = 2k - 1
    return recompute!(t, M, E, e)
end

# rebuild a window from the splits of the last DP: x = win[1] stays on top, the other window
# nodes are reused; next[] counts the slots used
function rebuild!(t::LocalTree, ws::Workspace, pieces::Vector{Int}, win::Vector{Int}, T::Int, full::Int, next::Base.RefValue{Int})
    T & (T - 1) == 0 && return @inbounds pieces[trailing_zeros(T) + 1]
    A = @inbounds ws.split[T + 1]
    y = T == full ? win[1] : (next[] += 1; @inbounds win[next[]])
    l = rebuild!(t, ws, pieces, win, A, full, next)
    r = rebuild!(t, ws, pieces, win, T ⊻ A, full, next)
    @inbounds t.left[y] = l; t.right[y] = r
    return y
end

# Subtree reconfiguration: for every internal node x, cut its subtree into at most `window`
# pieces (expanding the widest piece first) and re-solve the merge of the pieces exactly.
# mode = :width: replace when the window's largest intermediate shrinks (then the cheapest
# such tree); mode = :flops: replace when the window gets cheaper within width W.
# A window is re-solved only if one of its nodes changed since it was last found optimal.
function reconfigure!(ws::Workspace, t::LocalTree{S}, M::S, E::S, e::Int, W::Float64,
        mode::Symbol, window::Int; maxsweeps::Int = 16) where {S <: Store}
    k = t.k; nw = nwd(M); nn = 2k - 1
    pieces = Int[]; win = Int[]; todo = Int[]
    Mu = newstore(S, nw, window); Ml = newstore(S, nw, window)
    stamp = t.stamp; fill!(stamp, 0); solved = zeros(Int, nn); clock = 0
    next = Ref(1)
    for _ in 1:maxsweeps
        improved = false
        append!(empty!(todo), t.order); reverse!(todo)      # top-down
        for x in todo
            empty!(pieces); empty!(win)
            push!(pieces, t.left[x], t.right[x]); push!(win, x)
            while length(pieces) < window
                best = 0; bv = -Inf
                for (j, p) in enumerate(pieces)
                    if p > k
                        v = mwt(ws, t.L, p)
                        v > bv && (bv = v; best = j)
                    end
                end
                best == 0 && break
                p = pieces[best]; pieces[best] = t.left[p]; push!(pieces, t.right[p]); push!(win, p)
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
                mcopy!(Mu, j, t.U, p); mcopy!(Ml, j, t.L, p)
            end
            if mode === :width
                oldw = 0.0
                for y in win
                    oldw = max(oldw, mwt(ws, t.L, y))
                end
                bwin = width_dp!(ws, Mu, Ml, t.L, x, np)
                if !(bwin < oldw - EPS)
                    clock += 1; solved[x] = clock; continue
                end
                flops_dp!(ws, Mu, Ml, t.L, x, np, bwin)
            else
                oldc = 0.0
                for y in win
                    oldc += mflopsor(ws, t.L, t.left[y], t.L, t.right[y])
                end
                c = flops_dp!(ws, Mu, Ml, t.L, x, np, W)
                if !(c < oldc * (1 - 1e-12))
                    clock += 1; solved[x] = clock; continue
                end
            end
            next[] = 1
            rebuild!(t, ws, pieces, win, (1 << np) - 1, (1 << np) - 1, next)
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

# emit a local tree as global nodes (children first); returns the root
function emit_local!(ws::Workspace, t::LocalTree, ids::Vector{Int})
    gid = zeros(Int, 2t.k - 1)
    @inbounds for i in 1:t.k
        gid[i] = ids[i]
    end
    @inbounds for x in t.order
        y = ws.next; ws.next += 1; gid[x] = y
        ws.parent[gid[t.left[x]]] = y; ws.parent[gid[t.right[x]]] = y
    end
    return gid[t.root]
end

# ----------------------------------------------------------------------------------------
# cliques
# ----------------------------------------------------------------------------------------

struct Cliques
    nc::Int
    root::Int                 # the clique whose merge gives the output (= nc)
    chptr::Vector{Int}        # children of each clique (CSR)
    ch::Vector{Int}
    asptr::Vector{Int}        # hyperedges assigned to each clique (CSR)
    as::Vector{Int}
    scalarroots::Vector{Int}  # roots of the other components (their results are scalars)
    free::Vector{Int}         # hyperedges without vertices
end

@inline hedge(hptr, htgt, e) = @inbounds view(htgt, hptr[e]:hptr[e + 1] - 1)

# Legs of the items of clique i, with occurrence counts in ws.acc and the distinct
# vertices in ws.touched; returns the number of items. Items: the children with a result
# (legs rv[rs[c]:re[c]-1], counts rc) and the assigned hyperedges (their vertices held
# by another hyperedge too, count 1).
function gather!(ws::Workspace, C::Cliques, i::Int, hptr, htgt, rs, re, rv, rc)
    ws.tick += 1; tick = ws.tick; acc = ws.acc; stamp = ws.stamp; touched = ws.touched
    empty!(touched); k = 0
    @inbounds for p in C.chptr[i]:C.chptr[i + 1] - 1
        c = C.ch[p]
        rs[c] == 0 && continue
        k += 1
        for q in rs[c]:re[c] - 1
            v = rv[q]
            if stamp[v] != tick
                stamp[v] = tick; acc[v] = 0; push!(touched, v)
            end
            acc[v] += rc[q]
        end
    end
    @inbounds for p in C.asptr[i]:C.asptr[i + 1] - 1
        k += 1
        for v in hedge(hptr, htgt, C.as[p])
            ws.deg[v] == 1 && continue       # summed inside the tensor (cotengra: preprocessing)
            if stamp[v] != tick
                stamp[v] = tick; acc[v] = 0; push!(touched, v)
            end
            acc[v] += 1
        end
    end
    return k
end

touchedweight(ws::Workspace) = sum((@inbounds ws.weights[v] for v in ws.touched); init = 0.0)

# mask store type for the current merge: one word if it has at most 64 vertices
storetype(ws::Workspace) = length(ws.touched) <= 64 ? Vector{UInt64} : Matrix{UInt64}

# local masks of the items of clique i (after gather!): M (k columns) and the merge's legs E
# (1 column)
function localmasks(ws::Workspace, ::Type{S}, C::Cliques, i::Int, k::Int, hptr, htgt, rs, re, rv, resnode) where {S <: Store}
    b = length(ws.touched); nw = max(cld(b, 64), 1)
    resize!(ws.wloc, b)
    @inbounds for (j, v) in enumerate(ws.touched)
        ws.localid[v] = j - 1; ws.wloc[j] = ws.weights[v]
    end
    buildtables!(ws, b)
    M = newstore(S, nw, k); E = newstore(S, nw, 1); ids = Vector{Int}(undef, k); j = 0
    @inbounds for p in C.chptr[i]:C.chptr[i + 1] - 1
        c = C.ch[p]
        rs[c] == 0 && continue
        j += 1
        for q in rs[c]:re[c] - 1
            msetbit!(M, j, ws.localid[rv[q]])
        end
        ids[j] = resnode[c]
    end
    @inbounds for p in C.asptr[i]:C.asptr[i + 1] - 1
        e = C.as[p]; j += 1
        for v in hedge(hptr, htgt, e)
            ws.deg[v] == 1 || msetbit!(M, j, ws.localid[v])
        end
        ids[j] = e
    end
    @inbounds for v in ws.touched
        ws.acc[v] < ws.deg[v] && msetbit!(E, 1, ws.localid[v])
    end
    return M, E, ids, b
end

# the narrowest tree the heuristics find for a large merge
function widthtree(ws::Workspace, M::S, E::S, k, b, window) where {S <: Store}
    t = greedy_tree(ws, M, E, 1, k, b, Inf, false)
    return reconfigure!(ws, t, M, E, 1, Inf, :width, window)
end

# pass 1: the smallest width of clique i's merge (large merges keep their tree in cache)
function clique_width(ws::Workspace, ::Type{S}, C, i, k, kexact, window, hptr, htgt, rs, re, rv, resnode, cache) where {S <: Store}
    M, E, ids, b = localmasks(ws, S, C, i, k, hptr, htgt, rs, re, rv, resnode)
    k <= kexact && return width_dp!(ws, M, M, E, 1, k)
    t = widthtree(ws, M, E, k, b, window)
    cache[i] = t
    return maxwidth(ws, t)
end

# pass 2: the cheapest binarization of clique i's merge within width W
function clique_build(ws::Workspace, ::Type{S}, C, i, k, kexact, window, W, hptr, htgt, rs, re, rv, resnode, cache) where {S <: Store}
    M, E, ids, b = localmasks(ws, S, C, i, k, hptr, htgt, rs, re, rv, resnode)
    if k <= kexact
        c = flops_dp!(ws, M, M, E, 1, k, W)
        @assert c < Inf "no binarization within width $W"
        return emit_dp!(ws, (1 << k) - 1, ids)
    end
    # starts: the capped greedy, and (if pass 1 built it) the narrowest tree, which fits in W
    # by construction; each refined for flops within W; keep the cheaper
    best = nothing
    if haskey(cache, i)
        best = reconfigure!(ws, cache[i]::LocalTree{S}, M, E, 1, W, :flops, window)
        @assert maxwidth(ws, best) <= W + EPS
    end
    t = greedy_tree(ws, M, E, 1, k, b, W, true)
    if maxwidth(ws, t) <= W + EPS
        reconfigure!(ws, t, M, E, 1, W, :flops, window)
        (best === nothing || treecost(ws, t) < treecost(ws, best)) && (best = t)
    end
    @assert best !== nothing "no binarization within width $W"
    return emit_local!(ws, best::LocalTree{S}, ids)
end

# ----------------------------------------------------------------------------------------
# global refinement: subtree reconfiguration across cliques, within width and tc caps
# ----------------------------------------------------------------------------------------

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
function branchrefine(weights::AbstractVector, hypergraph::BipartiteGraph, point::Integer, parent::AbstractVector{<:Integer};
        window::Integer = 8, maxsweeps::Integer = 8, widthfirst::Bool = false, W = nothing, TC = nothing)
    H = simplehypergraph(hypergraph)
    m = CliqueTrees.nv(H); n = CliqueTrees.nov(H); hptr = pointers(H); htgt = targets(H)
    nn = length(parent); @assert nn == 2m - 2 && parent[point] == 0
    m <= 3 && return collect(Int, parent)
    window = clamp(window, 3, 16)
    deg = zeros(Int, n)
    @inbounds for t in htgt
        deg[t] += 1
    end
    par = collect(Int, parent); left = zeros(Int, nn); right = zeros(Int, nn); root = 0
    @inbounds for x in 1:nn
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
    legs = Vector{Vector{Int}}(undef, nn); cnts = Vector{Vector{Int}}(undef, nn)
    acc = zeros(Int, n); st = zeros(Int, n)
    @inbounds for e in 1:m
        legs[e] = [v for v in hedge(hptr, htgt, e) if deg[v] > 1]; cnts[e] = ones(Int, length(legs[e]))
    end
    @inbounds for x in m + 1:nn
        touched = Int[]
        for c in (left[x], right[x]), (v, k) in zip(legs[c], cnts[c])
            if st[v] != x
                st[v] = x; acc[v] = 0; push!(touched, v)
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
    end
    w = Float64.(weights)
    wleg(x) = sum((w[v] for v in legs[x]); init = 0.0)
    function tcof(x)
        s = 0.0; tick = -x
        for c in (left[x], right[x]), v in legs[c]
            if st[v] != tick
                st[v] = tick; s += w[v]
            end
        end
        return s
    end
    Wcap = W === nothing ? maximum(wleg(x) for x in m + 1:nn if x != point) : Float64(W)
    TCcap = TC === nothing ? maximum(tcof(x) for x in m + 1:nn if x != point) : Float64(TC)
    ws = Workspace(n, m, window)
    R = RefineState(par, left, right, legs, root, m, zeros(Int, nn), zeros(Int, nn), 0)
    if widthfirst
        sweeps!(ws, R, w, Wcap, TCcap, :width, window, maxsweeps)
        Wcap = maximum(wleg(x) for x in m + 1:nn if x != point)
    end
    sweeps!(ws, R, w, Wcap, TCcap, :flops, window, maxsweeps)
    # renumber the contractions children first
    newid = zeros(Int, nn); next = m
    stack = [(R.root, false)]
    while !isempty(stack)
        x, done = pop!(stack)
        if x <= m
            newid[x] = x
        elseif done
            next += 1; newid[x] = next
        else
            push!(stack, (x, true)); push!(stack, (R.right[x], false)); push!(stack, (R.left[x], false))
        end
    end
    out = zeros(Int, nn)
    @inbounds for x in 1:nn
        x == point && continue
        p = R.par[x]
        out[newid[x]] = p == point ? point : newid[p]
    end
    return out
end

mutable struct RefineState
    par::Vector{Int}
    left::Vector{Int}
    right::Vector{Int}
    legs::Vector{Vector{Int}}
    root::Int
    m::Int
    stamp::Vector{Int}       # when the node last changed
    solved::Vector{Int}      # when the window at the node was last found optimal
    clock::Int
end

function sweeps!(ws::Workspace, R::RefineState, w, Wcap, TCcap, mode::Symbol, window::Int, maxsweeps::Int)
    m = R.m; pieces = Int[]; win = Int[]; order = Int[]; stack = Int[]
    fill!(R.solved, 0)
    for _ in 1:maxsweeps
        empty!(order); empty!(stack); push!(stack, R.root)
        while !isempty(stack)                   # preorder: top-down
            x = pop!(stack)
            if x > m
                push!(order, x); push!(stack, R.right[x]); push!(stack, R.left[x])
            end
        end
        improved = false
        for x in order
            empty!(pieces); empty!(win); push!(pieces, R.left[x], R.right[x]); push!(win, x)
            while length(pieces) < window
                best = 0; bv = -Inf
                for (j, p) in enumerate(pieces)
                    if p > m
                        v = sum((w[u] for u in R.legs[p]); init = 0.0)
                        v > bv && (bv = v; best = j)
                    end
                end
                best == 0 && break
                p = pieces[best]; pieces[best] = R.left[p]; push!(pieces, R.right[p]); push!(win, p)
            end
            np = length(pieces)
            np <= 2 && continue
            if R.solved[x] > 0
                newest = 0
                for y in Iterators.flatten((win, pieces))
                    newest = max(newest, R.stamp[y])
                end
                newest < R.solved[x] && continue
            end
            # local bits: the vertices of the pieces' legs
            ws.tick += 1; tick = ws.tick; touched = ws.touched; empty!(touched)
            for p in pieces, v in R.legs[p]
                if ws.stamp[v] != tick
                    ws.stamp[v] = tick; push!(touched, v)
                end
            end
            S = length(touched) <= 64 ? Vector{UInt64} : Matrix{UInt64}
            if solve_window!(ws, S, R, x, pieces, win, w, Wcap, TCcap, mode)
                R.clock += 1
                for y in win
                    R.stamp[y] = R.clock
                end
                improved = true
            else
                R.clock += 1; R.solved[x] = R.clock
            end
        end
        improved || break
    end
end

# re-solve the window (pieces under x); returns true if it was replaced
function solve_window!(ws::Workspace, ::Type{S}, R::RefineState, x, pieces, win, w, Wcap, TCcap, mode) where {S <: Store}
    touched = ws.touched; b = length(touched); nw = max(cld(b, 64), 1); np = length(pieces)
    resize!(ws.wloc, b)
    @inbounds for (j, v) in enumerate(touched)
        ws.localid[v] = j - 1; ws.wloc[j] = w[v]
    end
    buildtables!(ws, b)
    # columns 1:np the pieces, np+1:np+|win| the window's nodes (their legs ⊆ the pieces')
    nwin = length(win)
    X = newstore(S, nw, np + nwin)
    col(y) = (j = findfirst(==(y), pieces); j === nothing ? np + findfirst(==(y), win) : j)
    for (j, y) in enumerate(Iterators.flatten((pieces, win)))
        for v in R.legs[y]
            msetbit!(X, j, ws.localid[v])
        end
    end
    tcf = mexp2(ws, TCcap) * (1 + 1e-12)
    if mode === :width
        oldw = 0.0
        for y in win
            oldw = max(oldw, mwt(ws, X, col(y)))
        end
        bwin = width_dp!(ws, X, X, X, np + 1, np; tcf)     # E = legs of x (column np + 1)
        bwin < oldw - EPS || return false
        c = flops_dp!(ws, X, X, X, np + 1, np, bwin; tcf)
        c < Inf || return false
    else
        oldc = 0.0
        for y in win
            oldc += mflopsor(ws, X, col(R.left[y]), X, col(R.right[y]))
        end
        c = flops_dp!(ws, X, X, X, np + 1, np, Wcap; tcf)
        c < oldc * (1 - 1e-12) || return false
    end
    # rebuild: x stays on top, the other window nodes are reused; legs from the DP's masks
    _, L = dpbufs(ws, S, nw)
    full = (1 << np) - 1; slot = Ref(1)
    function build(T)
        T & (T - 1) == 0 && return pieces[trailing_zeros(T) + 1]
        A = ws.split[T + 1]
        y = T == full ? x : (slot[] += 1; win[slot[]])
        l = build(A); r = build(T ⊻ A)
        R.left[y] = l; R.right[y] = r; R.par[l] = y; R.par[r] = y
        if y != x
            lg = Int[]
            for i in 1:nw
                word = getw(L, i, T + 1)
                while word != 0
                    push!(lg, touched[64 * (i - 1) + trailing_zeros(word) + 1]); word &= word - 1
                end
            end
            R.legs[y] = lg
        end
        return y
    end
    build(full)
    return true
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
function branchtree(weights::AbstractVector, hypergraph::BipartiteGraph, point::Integer;
        alg = DEFAULT_ELIMINATION_ALGORITHM, snd = DEFAULT_SUPERNODE_TYPE, kexact::Integer = 10, window::Integer = 8)
    H = simplehypergraph(hypergraph)
    n = CliqueTrees.nov(H)
    @assert length(weights) == n
    @assert all(>=(0), weights) "weights must be nonnegative"
    if any(iszero, weights)       # drop dimension-1 indices (they cost nothing)
        keep = findall(!iszero, weights); new = zeros(Int, n); new[keep] = 1:length(keep)
        H = restrict(H, new, length(keep)); weights = weights[keep]
        alg isa AbstractVector && (alg = [new[v] for v in alg if new[v] > 0])
    end
    if CliqueTrees.nov(H) == 0
        return scalartree(CliqueTrees.nv(H), point)
    end
    label, tree = cliquetree(weights, linegraph(reverse(H), H); alg, snd)
    return branchtree(weights, H, point, label, tree; kexact, window)
end

"""
    branchtree(weights, hypergraph, point, label, tree; kexact=10, window=8)

As above, from a clique tree `(label, tree)` of the line graph `linegraph(reverse(H), H)`,
e.g. from `cliquetree`.
"""
function branchtree(weights::AbstractVector, hypergraph::BipartiteGraph, point::Integer,
        label::AbstractVector, tree::CliqueTree; kexact::Integer = 10, window::Integer = 8)
    H = simplehypergraph(hypergraph)
    m = CliqueTrees.nv(H); n = CliqueTrees.nov(H)
    @assert 1 <= point <= m
    @assert m >= 2 "a contraction needs at least one tensor besides the point"
    @assert 2 <= kexact <= 20
    @assert length(weights) == n
    window = min(max(window, 3), kexact)
    length(tree) == 0 && return scalartree(m, point)
    hptr = pointers(H); htgt = targets(H)
    Hr = reverse(H); vptr = pointers(Hr)
    label = collect(Int, label)

    # re-root at the clique holding the point's vertices
    pv = collect(hedge(hptr, htgt, point))
    if !isempty(pv)
        invl = invperm(label)
        perm, tree = cliquetree(tree, invl[pv])
        label = label[perm]
    end
    C = cliques(tree, label, H, Hr, point, n, m)

    ws = Workspace(n, m, kexact)
    @inbounds for v in 1:n
        ws.weights[v] = weights[v]; ws.deg[v] = vptr[v + 1] - vptr[v]
    end
    nc = C.nc
    rs = zeros(Int, nc); re = zeros(Int, nc)   # result legs of clique i: rv[rs[i]:re[i]-1]
    rv = Int[]; rc = Int[]
    nitems = zeros(Int, nc); tw = zeros(Float64, nc)
    resnode = zeros(Int, nc)
    cache = Dict{Int, Any}()

    # pass 1a, bottom-up: the legs of every clique's merge; W starts at the widest result
    W = 0.0
    @inbounds for i in 1:nc
        k = gather!(ws, C, i, hptr, htgt, rs, re, rv, rc)
        nitems[i] = k
        k == 0 && continue
        rs[i] = length(rv) + 1; s = 0.0
        for v in ws.touched
            # a single item passes up unchanged
            if k == 1 || ws.acc[v] < ws.deg[v]
                push!(rv, v); push!(rc, ws.acc[v]); s += ws.weights[v]
            end
        end
        re[i] = length(rv) + 1
        k >= 2 && (W = max(W, s))
        tw[i] = touchedweight(ws)
    end
    # pass 1b: the smallest width of each merge of k ≥ 3 items, heaviest first; a merge
    # whose vertices weigh at most W in total cannot raise W
    big = [i for i in 1:nc if nitems[i] >= 3]
    sort!(big; by = i -> tw[i], rev = true)
    @inbounds for i in big
        tw[i] <= W + EPS && break
        gather!(ws, C, i, hptr, htgt, rs, re, rv, rc)
        W = max(W, clique_width(ws, storetype(ws), C, i, nitems[i], kexact, window, hptr, htgt, rs, re, rv, resnode, cache))
    end

    # pass 2, bottom-up: the cheapest binarization of each clique within W
    @inbounds for i in 1:nc
        k = nitems[i]
        k == 0 && continue
        gather!(ws, C, i, hptr, htgt, rs, re, rv, rc)
        if k == 1
            resnode[i] = firstitem(C, i, rs, resnode)
        elseif k == 2
            a, c = twoitems(C, i, rs, resnode)
            x = ws.next; ws.next += 1; ws.parent[a] = x; ws.parent[c] = x; resnode[i] = x
        else
            resnode[i] = clique_build(ws, storetype(ws), C, i, k, kexact, window, W, hptr, htgt, rs, re, rv, resnode, cache)
        end
    end
    # the scalars (other components, hyperedges without vertices) are multiplied together,
    # then into the root's result
    top = nitems[C.root] >= 1 ? resnode[C.root] : 0
    s = 0
    for t in Iterators.flatten(((resnode[c] for c in C.scalarroots if nitems[c] >= 1), C.free))
        if s == 0
            s = t
        else
            x = ws.next; ws.next += 1; ws.parent[s] = x; ws.parent[t] = x; s = x
        end
    end
    if top != 0 && s != 0
        x = ws.next; ws.next += 1; ws.parent[top] = x; ws.parent[s] = x; top = x
    elseif top == 0
        top = s
    end
    @assert top != 0 "nothing to contract"
    ws.parent[top] = point
    ws.parent[point] = 0
    @assert ws.next == 2m - 1 "expected $(m - 2) contractions, made $(ws.next - m - 1)"
    return ws.parent
end

# every tensor is a scalar: multiply them in a chain
function scalartree(m::Int, point::Int)
    @assert m >= 2 "a contraction needs at least one tensor besides the point"
    parent = zeros(Int, 2m - 2); next = m + 1; s = 0
    for e in 1:m
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

function firstitem(C::Cliques, i, rs, resnode)
    @inbounds for p in C.chptr[i]:C.chptr[i + 1] - 1
        c = C.ch[p]; rs[c] == 0 || return resnode[c]
    end
    return @inbounds C.as[C.asptr[i]]
end

function twoitems(C::Cliques, i, rs, resnode)
    a = 0
    @inbounds for p in C.chptr[i]:C.chptr[i + 1] - 1
        c = C.ch[p]
        rs[c] == 0 && continue
        a == 0 ? (a = resnode[c]) : return a, resnode[c]
    end
    @inbounds for p in C.asptr[i]:C.asptr[i + 1] - 1
        e = C.as[p]
        a == 0 ? (a = e) : return a, e
    end
    error("clique $i has fewer than two items")
end

# Children and assigned hyperedges of each clique. The roots of the other components of a
# forest (their results are scalars) and the hyperedges without vertices are multiplied in
# at the end, outside the root's merge. The point is assigned nowhere: the root's merge
# is the output.
function cliques(tree::CliqueTree, label::Vector{Int}, H, Hr, point::Int, n::Int, m::Int)
    nc = length(tree)
    hptr = pointers(H); htgt = targets(H); vptr = pointers(Hr); vtgt = targets(Hr)
    prnt = zeros(Int, nc)
    @inbounds for i in 1:nc
        p = AbstractTrees.parentindex(tree, i)
        prnt[i] = isnothing(p) ? 0 : p
    end
    cliqueof = zeros(Int, n)
    @inbounds for i in 1:nc, q in residual(tree, i)
        cliqueof[label[q]] = i
    end
    pv = hedge(hptr, htgt, point)
    root = isempty(pv) ? findlast(iszero, prnt) : cliqueof[first(pv)]
    @assert prnt[root] == 0 "the point's clique is not a root"
    @assert root == nc "the root clique must come last in postorder"
    scalarroots = [i for i in 1:nc if prnt[i] == 0 && i != root]
    @inbounds for i in 1:nc
        prnt[i] == 0 || @assert prnt[i] > i "the clique tree is not in postorder"
    end
    chptr = zeros(Int, nc + 1); chptr[1] = 1
    @inbounds for i in 1:nc
        prnt[i] != 0 && (chptr[prnt[i] + 1] += 1)
    end
    @inbounds for i in 1:nc
        chptr[i + 1] += chptr[i]
    end
    ch = zeros(Int, chptr[end] - 1); fillp = copy(chptr)
    @inbounds for i in 1:nc
        p = prnt[i]
        if p != 0
            ch[fillp[p]] = i; fillp[p] += 1
        end
    end
    # assignment: the first clique (in postorder) whose residual holds a vertex of e
    owner = zeros(Int, m)
    @inbounds for i in 1:nc, q in residual(tree, i)
        v = label[q]
        for t in vptr[v]:vptr[v + 1] - 1
            e = vtgt[t]
            (e != point && owner[e] == 0) && (owner[e] = i)
        end
    end
    free = [e for e in 1:m if e != point && owner[e] == 0]
    asptr = zeros(Int, nc + 1); asptr[1] = 1
    @inbounds for e in 1:m
        owner[e] != 0 && (asptr[owner[e] + 1] += 1)
    end
    @inbounds for i in 1:nc
        asptr[i + 1] += asptr[i]
    end
    as = zeros(Int, asptr[end] - 1); fillp = copy(asptr)
    @inbounds for e in 1:m
        i = owner[e]
        if i != 0
            as[fillp[i]] = e; fillp[i] += 1
        end
    end
    return Cliques(nc, root, chptr, ch, asptr, as, scalarroots, free)
end

# a copy of the hypergraph without repeated vertices in a hyperedge
function simplehypergraph(H::BipartiteGraph)
    m = CliqueTrees.nv(H); n = CliqueTrees.nov(H)
    ptr = pointers(H); tgt = targets(H)
    mark = zeros(Int, n); newptr = Vector{Int}(undef, m + 1); newtgt = Int[]
    sizehint!(newtgt, length(tgt)); newptr[1] = 1
    @inbounds for e in 1:m
        for p in ptr[e]:ptr[e + 1] - 1
            v = tgt[p]
            if mark[v] != e
                mark[v] = e; push!(newtgt, v)
            end
        end
        newptr[e + 1] = length(newtgt) + 1
    end
    return BipartiteGraph{Int, Int}(n, m, length(newtgt), newptr, newtgt)
end

# the hypergraph on the vertices v with new[v] > 0, renumbered to new[v]
function restrict(H::BipartiteGraph, new::Vector{Int}, n2::Int)
    m = CliqueTrees.nv(H); ptr = pointers(H); tgt = targets(H)
    newptr = Vector{Int}(undef, m + 1); newtgt = Int[]; newptr[1] = 1
    @inbounds for e in 1:m
        for p in ptr[e]:ptr[e + 1] - 1
            new[tgt[p]] > 0 && push!(newtgt, new[tgt[p]])
        end
        newptr[e + 1] = length(newtgt) + 1
    end
    return BipartiteGraph{Int, Int}(n2, m, length(newtgt), newptr, newtgt)
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
    m = CliqueTrees.nv(H); n = CliqueTrees.nov(H)
    hptr = pointers(H); htgt = targets(H)
    nn = length(parent); @assert nn == 2m - 2
    deg = zeros(Int, n)
    @inbounds for t in htgt
        deg[t] += 1
    end
    kids = [Int[] for _ in 1:nn]
    @inbounds for x in 1:nn
        parent[x] == 0 || push!(kids[parent[x]], x)
    end
    @assert parent[point] == 0 && length(kids[point]) == 1
    legs = Vector{Vector{Int}}(undef, nn); cnts = Vector{Vector{Int}}(undef, nn)
    @inbounds for e in 1:m
        legs[e] = [v for v in hedge(hptr, htgt, e) if deg[v] > 1]; cnts[e] = ones(Int, length(legs[e]))
    end
    acc = zeros(Int, n); stamp = zeros(Int, n)
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
    mx = isempty(costs) ? 0.0 : maximum(costs)
    log2cost = isempty(costs) ? -Inf : mx + log2(sum(exp2(c - mx) for c in costs))
    return (; width, peak = tc, cost = log2cost)
end

"""
    branchssa(parent, point) -> Vector{Tuple{Int, Int}}

The contraction as a 0-based SSA path over the tensors (the hyperedges other than the
point, in order), e.g. for cotengra's `ContractionTree.from_path(..., ssa_path=...)`.
"""
function branchssa(parent::AbstractVector{<:Integer}, point::Integer)
    nn = length(parent); m = (nn + 2) ÷ 2
    id = zeros(Int, nn); t = 0
    for e in 1:m
        e == point && continue
        id[e] = t; t += 1
    end
    for x in m + 1:nn
        id[x] = t; t += 1
    end
    kids = [Int[] for _ in 1:nn]
    for x in 1:nn
        (parent[x] == 0 || parent[x] == point) || push!(kids[parent[x]], x)
    end
    return [(id[kids[x][1]], id[kids[x][2]]) for x in m + 1:nn]
end


# thin accessors over branchstats (one bottom-up pass each)
branchwidth(weights, hypergraph, point, parent) = branchstats(weights, hypergraph, point, parent).width
branchpeak(weights, hypergraph, point, parent)  = branchstats(weights, hypergraph, point, parent).peak
branchcost(weights, hypergraph, point, parent)  = branchstats(weights, hypergraph, point, parent).cost
