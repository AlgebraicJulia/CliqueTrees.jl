# The safe triconnected components of a biconnected graph, for
# `ConnectedComponents{3}` (see components.jl): the Tutte decomposition, cut
# along the separation pairs that pass the test of `pairsafe`.
#
# `tuttetree` returns a clique tree in the format of `atomtree!`: `order` maps
# positions to vertices, the residual of each bag is a range of positions, and
# the separator of each bag lists positions in the residuals of its ancestors.


# The Tutte decomposition of a biconnected graph: the tree
# decomposition induced by its totally-nested 2-separations (those that cross
# no other 2-separation; Cunningham and Edmonds). Its bags, with the
# separators filled, are the triconnected components: 3-connected graphs,
# cycles, and (merged here into a neighbor) the pairs of bonds.
#
# The separations are found in linear time by the algorithm of Bourneuf and
# Planken. Section and lemma numbers below refer to their paper.
#
#   Bourneuf, Romain, and Tim Planken. "A Structural Linear-Time Algorithm
#   for Computing the Tutte Decomposition." arXiv:2508.06212 (2025).
#
#   Cunningham, William H., and Jack Edmonds. "A combinatorial decomposition
#   theory." Canadian Journal of Mathematics 32.3 (1980): 734-765.

@inline function insert2!(l1::AbstractVector{V}, l2::AbstractVector{V}, i::V, x::V) where {V}
    if x < l1[i]
        l2[i] = l1[i]; l1[i] = x
    elseif l1[i] < x < l2[i]
        l2[i] = x
    end

    return
end

# The first two lowpoints of every vertex in a numbering `lab` (vertex =>
# number) of a DFS tree with parents `par` (by number): lwpt_k(v) is the k-th
# smallest proper ancestor of v adjacent to a descendant of v, or v.
function lowpoints!(lw1::AbstractVector{V}, lw2::AbstractVector{V}, graph::AbstractGraph{V}, lab::AbstractVector{V}, inv::AbstractVector{V}, par::AbstractVector{V}) where {V}
    n = nv(graph); inf = n + one(V)

    for i in oneto(n)
        lw1[i] = lw2[i] = inf
    end

    for i in Iterators.reverse(oneto(n))
        for w in neighbors(graph, inv[i])
            j = lab[w]
            j < i && insert2!(lw1, lw2, i, j)
        end

        lw1[i] > n && (lw1[i] = i)
        lw2[i] > n && (lw2[i] = i)
        p = par[i]

        if ispositive(p)
            lw1[i] < p && insert2!(lw1, lw2, p, lw1[i])
            lw2[i] < p && insert2!(lw1, lw2, p, lw2[i])
        end
    end

    return
end

# The precomputation of Section 4.2. Vertices are renumbered by a numbering
# compatible with a DFS tree (Definition 2.5): preorder, the children of every
# vertex visited by decreasing lowpoint and then increasing second lowpoint.
# All arrays except `vert` and `num` are indexed by the new numbers.
struct TutteTree{V}
    n::V
    vert::FVector{V}    # number => vertex
    num::FVector{V}     # vertex => number
    par::FVector{V}     # parent (0: the root, 1)
    chp::FVector{V}     # the children of v are cht[chp[v]:chp[v + 1] - 1], increasing
    cht::FVector{V}
    rnk::FVector{V}     # the rank of v among its siblings
    lw1::FVector{V}     # lowpoint
    lw2::FVector{V}     # second lowpoint
    hgp::FVector{V}     # highpoint
    lds::FVector{V}     # the largest descendant
    lft::FVector{V}     # the left (largest) child, or 0
    wit::FVector{V}     # witn
    mxh::FVector{V}     # maxhgpt
    parr::FVector{V}    # the maximal leftmost paths, one after another
    pof::FVector{V}     # the path in which v is not the first vertex
    idx::FVector{V}     # the position of v in that path
    pst::FVector{V}     # the position of the first vertex of a path
    pen::FVector{V}     # the position of the last vertex of a path
    witq::RMQ{V, V}     # minimum of witn along parr
    nmnq::RMQ{V, V}     # minimum neighbor, by number
    nmxq::RMQ{V, V}     # maximum neighbor, by number
    stb::FVector{V}     # stab(witn(v), v)
end

@inline function children(T::TutteTree, v::Integer)
    return view(T.cht, T.chp[v]:T.chp[v + 1] - 1)
end

# a stable counting sort of src[1:k] by key[src[i]] in 1:n (descending if rev)
function countsort!(dst::AbstractVector{V}, src::AbstractVector{V}, key::AbstractVector{V}, cnt::AbstractVector{V}, k::Integer, n::Integer, rev::Bool) where {V}
    for j in 1:n + 1
        cnt[j] = zero(V)
    end

    for i in 1:k
        x = key[src[i]]; j = rev ? n + 1 - x : x
        cnt[j + 1] += one(V)
    end

    for j in 2:n + 1
        cnt[j] += cnt[j - 1]
    end

    for i in 1:k
        x = key[src[i]]; j = rev ? n + 1 - x : x
        cnt[j] += one(V); dst[cnt[j]] = src[i]
    end

    return dst
end

function TutteTree(graph::AbstractGraph{V}) where {V}
    n = nv(graph)

    # a DFS tree from vertex 1, and its preorder ranks
    par0 = FVector{V}(undef, n); rk = FVector{V}(undef, n); ord = FVector{V}(undef, n)
    nxt = FVector{V}(undef, n); stk = FVector{V}(undef, n)

    for v in vertices(graph)
        rk[v] = zero(V)
    end

    t = one(V); rk[1] = one(V); ord[1] = one(V); par0[1] = zero(V); sp = 1; stk[1] = one(V); nxt[1] = one(V)

    while ispositive(sp)
        v = stk[sp]; nbr = neighbors(graph, v); k = nxt[v]

        if k <= length(nbr)
            nxt[v] = k + one(V); w = nbr[k]

            if iszero(rk[w])
                t += one(V); rk[w] = t; ord[t] = w; par0[w] = v
                sp += 1; stk[sp] = w; nxt[w] = one(V)
            end
        else
            sp -= 1
        end
    end

    @assert t == n

    # lowpoints in the DFS ranks; sort the children by (-lwpt, lwpt2)
    parr0 = FVector{V}(undef, n)

    for i in oneto(n)
        p = par0[ord[i]]; parr0[i] = iszero(p) ? zero(V) : rk[p]
    end

    lw1 = FVector{V}(undef, n); lw2 = FVector{V}(undef, n)
    lowpoints!(lw1, lw2, graph, rk, ord, parr0)

    a1 = FVector{V}(undef, n); a2 = FVector{V}(undef, n); cnt = FVector{V}(undef, n + 2)

    for i in 1:n - 1
        a1[i] = V(i + 1)
    end

    countsort!(a2, a1, lw2, cnt, n - 1, n, false)
    countsort!(a1, a2, lw1, cnt, n - 1, n, true)
    countsort!(a2, a1, parr0, cnt, n - 1, n, false)

    # children of rank i in a2[cnt0[i]:cnt0[i + 1] - 1], in visiting order
    chp0 = FVector{V}(undef, n + 1)

    for i in 1:n + 1
        chp0[i] = zero(V)
    end

    for i in 2:n
        chp0[parr0[i]] += one(V)
    end

    s = one(V)

    for i in 1:n
        c = chp0[i]; chp0[i] = s; s += c
    end

    chp0[n + 1] = s

    # the compatible numbering: preorder, visiting children in that order
    num = FVector{V}(undef, n); vert = FVector{V}(undef, n)
    sp = 1; stk[1] = one(V); t = zero(V)

    while ispositive(sp)
        i = stk[sp]; sp -= 1; t += one(V)
        v = ord[i]; num[v] = t; vert[t] = v

        for j in Iterators.reverse(chp0[i]:chp0[i + 1] - one(V))
            sp += 1; stk[sp] = a2[j]
        end
    end

    # the tree in the new numbering
    par = FVector{V}(undef, n)

    for i in oneto(n)
        p = par0[vert[i]]; par[i] = iszero(p) ? zero(V) : num[p]
    end

    chp = FVector{V}(undef, n + 1); cht = FVector{V}(undef, max(n - 1, 1)); rnk = FVector{V}(undef, n)

    for i in 1:n + 1
        chp[i] = zero(V)
    end

    for i in 2:n
        chp[par[i]] += one(V)
    end

    s = one(V)

    for i in 1:n
        c = chp[i]; chp[i] = s; s += c
    end

    chp[n + 1] = s

    for i in 1:n
        nxt[i] = chp[i]
    end

    for i in 2:n
        p = par[i]; j = nxt[p]; nxt[p] = j + one(V)
        cht[j] = V(i); rnk[i] = j - chp[p] + one(V)
    end

    rnk[1] = one(V)
    lowpoints!(lw1, lw2, graph, num, vert, par)

    lds = FVector{V}(undef, n); dep = FVector{V}(undef, n)

    for i in oneto(n)
        lds[i] = i
    end

    for i in Iterators.reverse(2:n)
        lds[par[i]] = max(lds[par[i]], lds[i])
    end

    dep[1] = zero(V)

    for i in 2:n
        dep[i] = dep[par[i]] + one(V)
    end

    # highpoints: paint the vertices v on the tree path above x with
    # p(v) > y, for the back-edges (x, y) by decreasing y (union-find)
    bkp = FVector{V}(undef, n + 1); bkt = FVector{V}(undef, max(ne(graph), 1))

    for i in 1:n + 1
        bkp[i] = zero(V)
    end

    for x in oneto(n), w in neighbors(graph, vert[x])
        y = num[w]

        if y < x && y != par[x]
            bkp[y] += one(V)
        end
    end

    s = one(V)

    for i in 1:n
        c = bkp[i]; bkp[i] = s; s += c
    end

    bkp[n + 1] = s

    for i in 1:n
        nxt[i] = bkp[i]
    end

    for x in oneto(n), w in neighbors(graph, vert[x])
        y = num[w]

        if y < x && y != par[x]
            j = nxt[y]; nxt[y] = j + one(V); bkt[j] = x
        end
    end

    hgp = FVector{V}(undef, n); up = nxt

    for i in oneto(n)
        hgp[i] = zero(V); up[i] = i
    end

    for y in Iterators.reverse(oneto(n)), j in bkp[y]:bkp[y + 1] - one(V)
        v = ufroot!(up, bkt[j])

        while dep[v] >= dep[y] + two(V)
            hgp[v] = y; up[v] = par[v]
            v = ufroot!(up, v)
        end
    end

    for i in oneto(n)
        iszero(hgp[i]) && (hgp[i] = i)
    end

    nmn = FVector{V}(undef, n); nmx = FVector{V}(undef, n)

    for i in oneto(n)
        lo = n; hi = zero(V)

        for w in neighbors(graph, vert[i])
            j = num[w]

            if j != i
                lo = min(lo, j); hi = max(hi, j)
            end
        end

        nmn[i] = lo; nmx[i] = hi
    end

    lft = FVector{V}(undef, n); wit = FVector{V}(undef, n); mxh = FVector{V}(undef, n)

    for i in oneto(n)
        a = chp[i]; b = chp[i + 1] - one(V)
        lft[i] = a <= b ? cht[b] : zero(V)
        wit[i] = b - a >= one(V) ? min(nmn[i], lw1[cht[b - one(V)]]) : nmn[i]
        m = zero(V)

        for j in Iterators.reverse(a:b)
            c = cht[j]; m = max(m, hgp[c]); mxh[c] = m
        end
    end

    # the maximal leftmost paths: a vertex c starts a path [p(c), c, ℓ(c), ...]
    # unless it is the left child of a vertex other than the root
    np = 0

    for c in 2:n
        (par[c] == 1 || c != lft[par[c]]) && (np += 1)
    end

    parr = FVector{V}(undef, n - 1 + np); pof = FVector{V}(undef, n); idx = FVector{V}(undef, n)
    pst = FVector{V}(undef, max(np, 1)); pen = FVector{V}(undef, max(np, 1))
    pof[1] = idx[1] = zero(V); k = 0; pos = 0

    for c in 2:n
        (par[c] == 1 || c != lft[par[c]]) || continue
        k += 1; pos += 1; parr[pos] = par[c]; pst[k] = pos; u = V(c)

        while true
            pos += 1; parr[pos] = u; pof[u] = k; idx[u] = pos
            iszero(lft[u]) && break
            u = lft[u]
        end

        pen[k] = pos
    end

    wv = FVector{V}(undef, length(parr))

    for i in eachindex(parr)
        wv[i] = wit[parr[i]]
    end

    T = TutteTree{V}(V(n), vert, num, par, chp, cht, rnk, lw1, lw2, hgp, lds, lft, wit, mxh,
        parr, pof, idx, pst, pen, RMQ(wv, false), RMQ(nmn, false), RMQ(nmx, true), FVector{V}(undef, n))

    # stab(witn(v), v) along every path (Lemma 4.13)
    stb = T.stb; stb[1] = zero(V)

    for k in 1:np
        v0 = parr[pst[k]]

        for q in pst[k] + 1:pen[k]
            v = parr[q]; w = wit[v]

            if q == pst[k] + 1
                stb[v] = w == v0 ? v0 : zero(V)
            elseif w < v0
                stb[v] = zero(V)
            else
                ia, ib = onpath(T, w, v)

                if ib - ia <= 1
                    stb[v] = w
                else
                    j = rmq(T.witq, ia + 1, ib - 1)
                    stb[v] = wv[j] >= w ? w : stb[parr[j]]
                end
            end
        end
    end

    return T
end

@inline function ufroot!(up::AbstractVector{V}, v::V) where {V}
    r = v

    while up[r] != r
        r = up[r]
    end

    while up[v] != r
        w = up[v]; up[v] = r; v = w
    end

    return r
end

# The positions of a and b on a common maximal leftmost path, a before b;
# (0, 0) if there is none.
@inline function onpath(T::TutteTree{V}, a::Integer, b::Integer) where {V}
    (a <= 0 || b <= 1 || a >= b) && return zero(V), zero(V)
    p = T.pof[b]; ib = T.idx[b]
    T.parr[T.pst[p]] == a && return T.pst[p], ib
    a != 1 && T.pof[a] == p && return T.idx[a], ib
    return zero(V), zero(V)
end

# The largest ancestor α of a such that T[α, b] is stable, or 0
# (Definition 3.4, Proposition 4.11).
function stab(T::TutteTree{V}, a::Integer, b::Integer) where {V}
    ia, ib = onpath(T, a, b)
    iszero(ia) && return zero(V)
    ib - ia <= 1 && return V(a)
    j = rmq(T.witq, ia + 1, ib - 1)
    T.witq.val[j] >= a && return V(a)
    return T.stb[T.parr[j]]
end

# Half-connected 2-separations with separator {a < b} and the side
# B - {a, b} = [f, l] - {a, b}; m and M are the smallest and largest child
# of b in B (0 if none).
struct TutteSeps{V}
    a::Vector{V}
    b::Vector{V}
    f::Vector{V}
    l::Vector{V}
    m::Vector{V}
    M::Vector{V}
    t::Vector{Int8}
end

function addsep!(S::TutteSeps{V}, a, b, f, l, m, M, t) where {V}
    push!(S.a, a); push!(S.b, b); push!(S.f, f); push!(S.l, l)
    push!(S.m, m); push!(S.M, M); push!(S.t, t)
    return
end

# all half-connected type-1 separations (Lemmas 4.14 and 4.15)
function tuttetype1!(S::TutteSeps{V}, T::TutteTree{V}) where {V}
    n = T.n

    for b in 2:n
        cs = children(T, b); k = 1

        while k <= length(cs)
            d = cs[k]

            if T.lw2[d] != b
                k += 1; continue
            end

            a = T.lw1[d]; k2 = k

            while k2 < length(cs) && T.lw2[cs[k2 + 1]] == b && T.lw1[cs[k2 + 1]] == a
                k2 += 1
            end

            for j in k:k2
                d = cs[j]

                if !(a == 1 && b == 2 && d == 3 && length(children(T, 2)) == 1)
                    addsep!(S, a, b, d, T.lds[d], d, d, 1)
                end
            end

            if k2 > k && !(a == 1 && b == 2)
                m = cs[k]; M = cs[k2]
                ac = children(T, a); ap = ac[searchsortedlast(ac, b)]
                ok = ap == b || a == 1

                for (lo, hi) in ((ap, b - 1), (b + 1, m - 1))
                    (ok || lo > hi) && continue
                    T.nmnq.val[rmq(T.nmnq, lo, hi)] < a && (ok = true)
                    T.nmxq.val[rmq(T.nmxq, lo, hi)] > T.lds[M] && (ok = true)
                end

                ok && addsep!(S, a, b, m, T.lds[M], m, M, 1)
            end

            k = k2 + 1
        end
    end

    return
end

# a maximal nested set of P-good tuples for every maximal leftmost path P
# (Algorithm 1), and the half-connected type-2 separations they give
# (Lemma 4.22)
function tuttetype2!(S::TutteSeps{V}, T::TutteTree{V}) where {V}
    n = T.n; par = T.par
    stack = V[]; pend = V[]; good = NTuple{4, V}[]

    for k in eachindex(T.pst)
        ps = T.pst[k]; pe = T.pen[k]; v1 = T.parr[ps]
        empty!(stack); empty!(good)

        for q in Iterators.reverse(ps + 1:pe)
            b = T.parr[q]

            while !isempty(stack) && stack[end] == b
                pop!(stack)
            end

            cs = children(T, b); nc = length(cs)
            i1 = 0; i2 = 1
            pp = iszero(par[b]) ? zero(V) : par[par[b]]
            a = stab(T, pp, b)
            empty!(pend)

            while a > 1 && (isempty(stack) || a >= stack[end])
                while i2 <= nc && T.lw1[cs[i2]] >= a
                    i2 += 1
                end

                while i1 < nc && (T.lw1[cs[i1 + 1]] > a || (T.lw1[cs[i1 + 1]] == a && T.hgp[cs[i1 + 1]] > a))
                    i1 += 1
                end

                if i2 == nc + 1 || T.mxh[cs[i2]] <= a
                    push!(pend, a)
                    push!(good, (a, b, iszero(i1) ? zero(V) : cs[i1], i2 > nc ? n + one(V) : cs[i2]))
                    a = a != v1 ? stab(T, par[a], b) : zero(V)
                elseif i2 <= nc
                    a = stab(T, T.lw1[cs[i2]], b)
                else
                    a = zero(V)
                end
            end

            while !isempty(pend)
                push!(stack, pop!(pend))
            end
        end

        for (a, b, d1, d2) in good
            ia, _ = onpath(T, a, b); ap = T.parr[ia + 1]
            cs = children(T, b); leaf = isempty(cs)
            l1 = ispositive(d1) ? T.lds[d1] : b - one(V)
            m1 = ispositive(d1) ? b + one(V) : zero(V)
            M1 = d1
            addsep!(S, a, b, ap, l1, m1, M1, 2)
            l2 = d2 <= n ? d2 - one(V) : (leaf ? b - one(V) : T.lds[b])
            l2 == b && (l2 = b - one(V))
            m2 = (d2 > b + 1 && !leaf) ? b + one(V) : zero(V)
            M2 = (b + 1 < d2 <= n) ? cs[T.rnk[d2] - 1] : ((d2 > n && !leaf) ? cs[end] : zero(V))
            l2 != l1 && addsep!(S, a, b, ap, l2, m2, M2, 2)
        end
    end

    return
end

# every child c of β satisfies lwpt(c) >= α or hgpt(c) <= α
function tuttechildren(T::TutteTree{V}, β::V, α::V) where {V}
    cs = children(T, β); lo = 1; hi = length(cs) + 1

    # lwpt is non-increasing along the children
    while lo < hi
        mid = (lo + hi) >>> 1
        T.lw1[cs[mid]] < α ? (hi = mid) : (lo = mid + 1)
    end

    return lo > length(cs) || T.mxh[cs[lo]] <= α
end

# The totally-nested 2-separations (Section 6.1): the half-connected ones
# that pass the tests of Propositions 6.4 and 6.5. Duplicates are removed.
function tutteseps(T::TutteTree{V}) where {V}
    n = T.n
    S = TutteSeps{V}(V[], V[], V[], V[], V[], V[], Int8[])
    tuttetype1!(S, T); tuttetype2!(S, T)

    # α(v), β(v) (Lemma 6.2) and Z(v) (Lemma 6.3)
    α = FVector{V}(undef, n); β = FVector{V}(undef, n); Z = FVector{Bool}(undef, n)

    for v in oneto(n)
        α[v] = β[v] = zero(V)
        (v == 1 || iszero(T.lft[v])) && continue
        α[v] = stab(T, T.par[v], T.lft[v])
    end

    q1 = V[]; q2 = V[]

    for k in eachindex(T.pst)
        ps = T.pst[k]; pe = T.pen[k]; vk = T.parr[pe]
        empty!(q1); empty!(q2); qh = 1

        # β1 into β, then β2 as the minimum
        for q in ps + 1:pe - 1
            vi = T.parr[q]; vn = T.parr[q + 1]

            while qh <= length(q1) && T.lw1[vn] >= q1[qh]
                β[q1[qh]] = vi; qh += 1
            end

            push!(q1, vi)
        end

        for j in qh:length(q1)
            β[q1[j]] = vk
        end

        for q in ps + 1:pe - 1
            vi = T.parr[q]; vn = T.parr[q + 1]

            while !isempty(q2) && T.hgp[vn] < q2[end]
                β[q2[end]] = min(β[q2[end]], vi); pop!(q2)
            end

            push!(q2, vi)
        end

        β[vk] = zero(V)
    end

    for v in oneto(n)
        Z[v] = false
        (iszero(α[v]) || iszero(β[v])) && continue
        stab(T, α[v], β[v]) == α[v] || continue
        Z[v] = tuttechildren(T, β[v], α[v])
    end

    np = length(T.parr)
    ycnt = FVector{V}(undef, np + 1); yp = FVector{V}(undef, np); ycnt[1] = zero(V)

    for i in 1:np
        v = T.parr[i]; l = T.lft[v]
        ycnt[i + 1] = ycnt[i] + (!iszero(l) && T.lw2[l] == v ? one(V) : zero(V))
        yp[i] = iszero(l) ? zero(V) : T.hgp[l]
    end

    ypq = RMQ(yp, false)
    out = TutteSeps{V}(V[], V[], V[], V[], V[], V[], Int8[]); seen = Set{NTuple{4, V}}()

    for i in eachindex(S.a)
        a = S.a[i]; b = S.b[i]; ℓ = T.lft[b]

        if S.t[i] == 1
            crossed = (S.m[i] == S.M[i] ? S.m[i] == ℓ : S.M[i] == ℓ) && α[b] > a && Z[b]
        else
            ia, ib = onpath(T, a, b); crossed = false

            if ib - ia >= 2
                crossed = ycnt[ib] > ycnt[ia + 1]
                crossed = crossed || yp[rmq(ypq, ia + 1, ib - 1)] <= stab(T, T.par[a], S.f[i])
            end

            crossed = crossed || (!iszero(ℓ) && S.M[i] < ℓ && α[b] > a && Z[b])
        end

        crossed && continue
        key = (a, b, S.f[i], S.l[i])
        key in seen && continue
        push!(seen, key); addsep!(out, a, b, S.f[i], S.l[i], S.m[i], S.M[i], S.t[i])
    end

    return out
end

# The tree decomposition induced by the nested separations S (Theorem 6.6).
# Orient every separation away from a vertex x in no separator: the far sides
# form a laminar family whose Hasse diagram is the tree. In the cyclic order
# of the numbering starting at x, each far side is an interval minus the two
# separator vertices. Nodes whose residual would be empty (the pairs of
# bonds) are merged into their parents.
#
# Returns the number of nodes q, the node of each number, and for each node
# its parent and separator (numbers); nodes are in postorder, the root last.
function tuttebuild(T::TutteTree{V}, S::TutteSeps{V}) where {V}
    n = T.n; p = length(S.a)
    cnt = FVector{V}(undef, n + 2)

    for v in oneto(n)
        cnt[v] = zero(V)
    end

    for i in 1:p
        cnt[S.a[i]] += one(V); cnt[S.b[i]] += one(V)
    end

    x = one(V)

    while ispositive(cnt[x])
        x += one(V)
    end

    lo = FVector{V}(undef, p); hi = FVector{V}(undef, p); sz = FVector{V}(undef, p)

    # positions in the cyclic order starting at x: tuttepos(v, x, n)
    for i in 1:p
        f = S.f[i]; l = S.l[i]; a = S.a[i]; b = S.b[i]

        if f <= x <= l
            lo[i] = tuttepos(l == n ? one(V) : l + one(V), x, n); hi[i] = tuttepos(isone(f) ? n : f - one(V), x, n)
        else
            lo[i] = tuttepos(f, x, n); hi[i] = tuttepos(l, x, n)
        end

        pa = tuttepos(a, x, n); pb = tuttepos(b, x, n)
        sz[i] = hi[i] - lo[i] + one(V) - (lo[i] <= pa <= hi[i]) - (lo[i] <= pb <= hi[i])
    end

    # sort by (lo ascending, hi descending, size descending)
    o1 = FVector{V}(undef, p); o2 = FVector{V}(undef, p)

    for i in 1:p
        o1[i] = V(i)
    end

    countsort!(o2, o1, sz, cnt, p, n, true)
    countsort!(o1, o2, hi, cnt, p, n, true)
    countsort!(o2, o1, lo, cnt, p, n, false)

    sparent = FVector{V}(undef, p); vnode = FVector{V}(undef, n); stack = V[]; j = 1

    for q in 1:n - 1
        while !isempty(stack) && hi[stack[end]] < q
            pop!(stack)
        end

        while j <= p && lo[o2[j]] == q
            i = o2[j]; j += 1
            sparent[i] = isempty(stack) ? zero(V) : stack[end]
            push!(stack, i)
        end

        t = isempty(stack) ? zero(V) : stack[end]; v = V(mod(q + x - 1, n) + 1)

        while ispositive(t) && (v == S.a[t] || v == S.b[t])
            t = sparent[t]
        end

        vnode[v] = t
    end

    vnode[x] = zero(V)

    # alive sets: those with a nonempty residual
    alive = FVector{Bool}(undef, p)

    for i in 1:p
        alive[i] = false
    end

    for v in oneto(n)
        ispositive(vnode[v]) && (alive[vnode[v]] = true)
    end

    for i in 1:p
        t = sparent[i]

        while ispositive(t) && !alive[t]
            t = sparent[t]
        end

        sparent[i] = t
    end

    # postorder of the alive sets, with the root (0) last
    chp = FVector{V}(undef, p + 2); cht = FVector{V}(undef, max(p, 1))

    for i in 1:p + 2
        chp[i] = zero(V)
    end

    for i in 1:p
        alive[i] && (chp[sparent[i] + 1] += one(V))
    end

    s = one(V)

    for i in 1:p + 1
        c = chp[i]; chp[i] = s; s += c
    end

    chp[p + 2] = s
    nxt = FVector{V}(undef, p + 1)

    for i in 1:p + 1
        nxt[i] = chp[i]
    end

    for i in 1:p
        if alive[i]
            k = sparent[i] + 1; cht[nxt[k]] = V(i); nxt[k] += one(V)
        end
    end

    post = FVector{V}(undef, p + 1); q = zero(V)   # post[i + 1]: the postorder number of set i
    st = V[zero(V)]; it = V[chp[1]]

    while !isempty(st)
        u = st[end]; k = it[end]

        if k < chp[u + 2]
            it[end] = k + one(V); push!(st, cht[k]); push!(it, chp[cht[k] + 1])
        else
            pop!(st); pop!(it); q += one(V); post[u + 1] = q
        end
    end

    node = FVector{V}(undef, n); prnt = FVector{V}(undef, q); sep1 = FVector{V}(undef, q); sep2 = FVector{V}(undef, q)

    for v in oneto(n)
        node[v] = post[vnode[v] + 1]
    end

    prnt[q] = sep1[q] = sep2[q] = zero(V)

    for i in 1:p
        alive[i] || continue
        k = post[i + 1]; prnt[k] = post[sparent[i] + 1]; sep1[k] = S.a[i]; sep2[k] = S.b[i]
    end

    return q, node, prnt, sep1, sep2
end

@inline function tuttepos(v::V, x::Integer, n::V) where {V}
    return V(mod(v - x, n))
end

# The decomposition of a biconnected graph along the safe separators
# of its Tutte decomposition. A separator {a, b} is used if it is safe
# (`pairsafe`); the bags on the two sides of a separator that is not used
# are merged, and a separator that is used is filled. Returns `nothing`, or
# the filled graph, `order`, and the tree. The complexity is linear with
# unit weights (or when every separator passes the first two tests of
# `pairsafe`).
function tuttetree(weights::AbstractVector{W}, graph::AbstractGraph{V}) where {W, V}
    E = etype(graph); n = nv(graph)

    # a biconnected graph with at most three vertices has no separation pair
    n < 4 && return nothing

    T = TutteTree(graph); S = tutteseps(T)
    isempty(S.a) && return nothing
    q, node, prnt, sep1, sep2 = tuttebuild(T, S)
    isone(q) && return nothing

    # relabel by vertex
    nodev = FVector{V}(undef, n)

    for i in oneto(n)
        nodev[T.vert[i]] = node[i]
    end

    for k in oneto(q - one(V))
        sep1[k] = T.vert[sep1[k]]; sep2[k] = T.vert[sep2[k]]
    end

    # the first descendant of each node; the nodes with the same separator
    # are siblings (bonds were merged into their parents): link them
    fdes = FVector{V}(undef, q); gnxt = FVector{V}(undef, q); gfst = FVector{V}(undef, q)
    safe = FVector{Bool}(undef, q)

    for k in oneto(q)
        fdes[k] = k; gfst[k] = gnxt[k] = zero(V); safe[k] = true
    end

    for k in oneto(q - one(V))
        p = prnt[k]; fdes[p] = min(fdes[p], fdes[k])
    end

    # group the children of each node by separator (sort by (sep1, sep2, parent))
    o1 = FVector{V}(undef, q); o2 = FVector{V}(undef, q); cnt = FVector{V}(undef, max(n, q) + 2)
    nq = q - one(V)

    for k in oneto(nq)
        o1[k] = k
    end

    countsort!(o2, o1, sep2, cnt, nq, n, false)
    countsort!(o1, o2, sep1, cnt, nq, n, false)
    countsort!(o2, o1, prnt, cnt, nq, q, false)

    cap = FVector{W}(undef, n); ctag = FVector{Int}(undef, n); vtag = FVector{Int}(undef, n)
    pred = FVector{V}(undef, n); queue = FVector{V}(undef, n)
    minw = weights[one(V)]

    for v in vertices(graph)
        ctag[v] = vtag[v] = 0
        minw = min(minw, weights[v])
    end

    tags = (0, 0); j = one(V)

    while j <= nq
        k = o2[j]; head = zero(V); jj = j

        while jj <= nq && prnt[o2[jj]] == prnt[k] && sep1[o2[jj]] == sep1[k] && sep2[o2[jj]] == sep2[k]
            t = o2[jj]; gnxt[t] = head; head = t; jj += one(V)
        end

        a = sep1[k]; b = sep2[k]
        ok, tags = pairsafe(cap, ctag, vtag, pred, queue, weights, graph, nodev, fdes, gnxt, head, isadjacent(graph, a, b), a, b, minw, tags)

        if !ok
            t = head

            while ispositive(t)
                safe[t] = false; t = gnxt[t]
            end
        end

        j = jj
    end

    # merge the bags across the separators that are not used
    rep = gfst

    for k in Iterators.reverse(oneto(q))
        rep[k] = (k == q || safe[k]) ? k : rep[prnt[k]]
    end

    idx = fdes; qq = zero(V)

    for k in oneto(q)
        rep[k] == k && (idx[k] = qq += one(V))
    end

    isone(qq) && return nothing
    nprnt = FVector{V}(undef, qq); s1 = FVector{V}(undef, qq); s2 = FVector{V}(undef, qq)

    for k in oneto(q)
        rep[k] == k || continue
        i = idx[k]

        if k == q
            nprnt[i] = s1[i] = s2[i] = zero(V)
        else
            nprnt[i] = idx[rep[prnt[k]]]; s1[i] = sep1[k]; s2[i] = sep2[k]
        end
    end

    for v in vertices(graph)
        nodev[v] = idx[rep[nodev[v]]]
    end

    # fill the separators that are used (once each)
    pairs = Set{Tuple{V, V}}()

    for i in oneto(qq - one(V))
        a = s1[i]; b = s2[i]
        isadjacent(graph, a, b) || push!(pairs, (min(a, b), max(a, b)))
    end

    for v in vertices(graph)
        cnt[v] = zero(V)
    end

    for (a, b) in pairs
        cnt[a] += one(V); cnt[b] += one(V)
    end

    m = de(graph) + twice(E(length(pairs)))
    ptr = FVector{E}(undef, n + one(V)); tgt = FVector{V}(undef, m); fill = FVector{E}(undef, n)
    pp = one(E)

    for v in vertices(graph)
        ptr[v] = pp

        for w in neighbors(graph, v)
            tgt[pp] = w; pp += one(E)
        end

        fill[v] = pp; pp += cnt[v]
    end

    ptr[n + one(V)] = pp

    for (a, b) in pairs
        tgt[fill[a]] = b; fill[a] += one(E)
        tgt[fill[b]] = a; fill[b] += one(E)
    end

    cmpgraph = BipartiteGraph(n, n, m, ptr, tgt)
    order, tree = partitiontree(nodev, nprnt, s1, s2, n, qq, E)
    return cmpgraph, order, tree
end

@inline function isadjacent(graph::AbstractGraph, a::Integer, b::Integer)
    if eltypedegree(graph, a) > eltypedegree(graph, b)
        a, b = b, a
    end

    for w in neighbors(graph, a)
        w == b && return true
    end

    return false
end

# Is the minimal separator S = {c, x} safe?
#
# The weighted form of Theorem 16 of Medema and Lazovik with the clique
# K = {heavier of c and x}: S is safe if, for every component C of G - S,
# the vertex-capacitated flow from c to x in G - C (capacities w; c and x
# uncapacitated) is at least min(w(c), w(x)). (A weight-w vertex stands for w
# true twins.) A flow from c to x avoiding C passes through the other
# components, so with f(D) the flow through a component D, capped at the
# requirement r, the condition is
#
#     sum_D f(D) - max_D f(D) >= r.
#
# Every component is full, so f(D) >= min(r, min_D w). In particular, with
# unit weights every minimal separator of size two is safe.
#
# The components are the subtrees of the nodes in the list that starts at
# `head`, and the rest of the graph. The flows are found greedily (augmenting
# paths without reverse arcs), which gives a lower bound.
#
#   Medema, Michel, and Alexander Lazovik. "A safeness condition for minimal
#   separators based on vertex connectivity." Discrete Mathematics 348
#   (2025): 114524.
function pairsafe(
        cap::AbstractVector{W},
        ctag::AbstractVector{Int},
        vtag::AbstractVector{Int},
        pred::AbstractVector{V},
        queue::AbstractVector{V},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        node::AbstractVector{V},
        fdes::AbstractVector{V},
        gnxt::AbstractVector{V},
        head::V,
        adjacent::Bool,
        c::V,
        x::V,
        minw::W,
        tags::Tuple{Int, Int},
    ) where {W, V}
    # S is a clique
    adjacent && return true, tags

    r = min(weights[c], weights[x])
    r <= minw && return true, tags

    fsum = fmax = zero(W); t = head

    while ispositive(t)
        f, tags = pairflow!(cap, ctag, vtag, pred, queue, weights, graph,
            node, fdes, gnxt, c, x, fdes[t], t, zero(V), r, tags)

        fsum += f; fmax = max(fmax, f)
        fsum - fmax >= r && return true, tags
        t = gnxt[t]
    end

    # the rest of the graph
    f, tags = pairflow!(cap, ctag, vtag, pred, queue, weights, graph,
        node, fdes, gnxt, c, x, zero(V), zero(V), head, r, tags)

    fsum += f; fmax = max(fmax, f)
    return fsum - fmax >= r, tags
end

# Is the node b in the component of G - {c, x} given by (lo, hi, head)?
# lo > 0: the nodes lo:hi (a subtree); lo = 0: the nodes outside the
# subtrees of the nodes in the list that starts at `head`.
@inline function pairmember(fdes::AbstractVector{V}, hnxt::AbstractVector{V}, b::V, lo::V, hi::V, head::V) where {V}
    ispositive(lo) && return lo <= b <= hi
    t = head

    while ispositive(t)
        fdes[t] <= b <= t && return false
        t = hnxt[t]
    end

    return true
end

# A flow from c to x through one component of G - {c, x}, capped at r. The
# flow is a sum of augmenting paths found by breadth-first search, without
# reverse arcs, so it is a lower bound on the maximum flow.
function pairflow!(
        cap::AbstractVector{W},
        ctag::AbstractVector{Int},
        vtag::AbstractVector{Int},
        pred::AbstractVector{V},
        queue::AbstractVector{V},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        blk::AbstractVector{V},
        fdes::AbstractVector{V},
        hnxt::AbstractVector{V},
        c::V,
        x::V,
        lo::V,
        hi::V,
        head::V,
        r::W,
        tags::Tuple{Int, Int},
    ) where {W, V}
    cs, vs = tags; cs += 1; total = zero(W)

    while total < r
        vs += 1; qh = qt = one(V); queue[qt] = c; vtag[c] = vs; last = zero(V)

        while qh <= qt && iszero(last)
            u = queue[qh]; qh += one(V)

            for v in neighbors(graph, u)
                if v == x
                    if u != c
                        last = u
                        break
                    end
                elseif v != c && vtag[v] != vs && pairmember(fdes, hnxt, blk[v], lo, hi, head)
                    if ctag[v] != cs
                        ctag[v] = cs; cap[v] = weights[v]
                    end

                    if ispositive(cap[v])
                        vtag[v] = vs; pred[v] = u
                        qt += one(V); queue[qt] = v
                    end
                end
            end
        end

        iszero(last) && break

        d = r - total; v = last

        while v != c
            d = min(d, cap[v]); v = pred[v]
        end

        v = last

        while v != c
            cap[v] -= d; v = pred[v]
        end

        total += d
    end

    return total, (cs, vs)
end
