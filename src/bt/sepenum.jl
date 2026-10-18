#
# List the minimal separators S of a graph with w(S) < maxk.
# Every minimal separator S has a small full component A, that
# is, 2|A| <= |V| - |S|, and it is found from A by a branching
# search (Tamaki's MinSepsGenerator):
#
#   - start from A = {a} and S = N(a), for each vertex a;
#   - choose a vertex v of S that is not fixed: either move v
#     into A, so that S becomes N(A + v), or fix v in S;
#   - a full component B of S other than A gives the pair
#     (S, A, B); a component of G - A - S that is not full
#     gives a smaller separator N(B) with the same A.
#
# A branch is cut when the fixed vertices weigh maxk or more,
# or when A cannot stay small, which for a state whose S is too
# heavy to be listed is decided by Tamaki's hanging trees (see
# sepenum_ste). A vertex that has been a start
# vertex is never moved into A again, and is fixed in S as soon
# as it joins it. With maxk = typemax(W), every minimal
# separator is listed. The graph need not be connected: if it is
# not, ∅ is a minimal separator, and so is every minimal
# separator of a component.
#
# Each separator is listed once. The search finds S from each of
# its small full components A, starting from the first vertex of
# A, and within that search once for each other full component B.
# So S is listed only with the A that is searched first, when no
# other small full component contains an earlier start vertex,
# and with the first B. Both tests read the components of the rest
# R of the current state. R is not all of G - A - S, since moving
# a vertex into A keeps only the side of the chosen B, but it holds
# every full component, other than the one containing a, of the
# separator being listed. Suppose a full component P lies outside
# R, in a part dropped at an earlier state (S', A', B'). P touches
# only that part and S', and the full component B of the listing
# touches only B' ⊇ B and S', so the separator N(P) = N(B) lies in
# S'. The vertex moved into A at that state is in S' but not in
# the separator, and is next to B', so B and a are joined outside
# the separator, which is impossible. ∅ is listed apart, since
# every state with no fixed vertex finds it. The vertex sets are
# packed sets.
#
# The search from each start reads only the starts before it, so
# the starts can be searched in parallel: with nt > 1, nt tasks
# share them (sepenum_mt), and the list is the same as with one
# (sepenum_st). The result is (sep, status): the list and
# BT_FOUND, or, if the search runs for more than mt seconds, an
# empty list and BT_TIMEOUT. The positional method takes the
# deadline dl, in the sense of time(), instead.
#
#   - Tamaki, Hisao. "Computing treewidth via exact and heuristic
#     lists of minimal separators." SEA 2019.
#
function sepenum(weights::AbstractVector, graph::AbstractGraph, maxk; mt::Real = Inf, nt::Integer = Threads.nthreads())
    return sepenum(weights, graph, maxk, time() + mt, nt)
end

function sepenum(weights::AbstractVector{W}, graph::AbstractGraph{V}, maxk, dl::Real, nt::Integer) where {W, V}
    nvtx = nv(graph)

    if nt <= 1 || nvtx < SEPENUM_MT
        sep = sepenum_st(weights, graph, maxk, dl)
    else
        sep = sepenum_mt(weights, graph, maxk, nt, dl)
    end

    if isnothing(sep)
        return bt_out(UInt64[], nvtx, convert(V, cld(nvtx, 64))), BT_TIMEOUT
    else
        return sep, BT_FOUND
    end
end

# With fewer vertices than this, the search uses one task.
const SEPENUM_MT = 20

# One thread: the starts in order.
function sepenum_st(weights::AbstractVector{W}, graph::AbstractGraph{V}, maxk, deadline::Real = Inf) where {W, V}
    nvtx = nv(graph); nwrd = convert(V, cld(nvtx, 64)); maxk = convert(W, maxk); clk = bt_clk(deadline)

    # the work arrays are slots 161 to 166 of the workspace
    ws   = workspace()
    adj  = bt_mat(nwrd, nvtx)
    excl = workvector(ws, UInt64, 161, nwrd)
    pool = worklist(ws, UInt64, 162)
    out  = worklist(ws, UInt64, 163)

    order = workvector(ws, V, 164, nvtx)
    key   = workvector(ws, V, 165, nvtx)
    cnt   = workvector(ws, V, 166, nvtx)

    smax = sepenum_adj!(adj, order, key, cnt, weights, graph, maxk)
    sepenum_emp!(out, pool, adj, maxk, nvtx, nwrd)
    fill!(excl, zero(UInt64))

    @inbounds for a in order
        sepenum_one!(pool, out, adj, excl, weights, maxk, smax, nvtx, nwrd, a, clk)
        clk.out[] && return nothing
        excl[bt_wrd(a)] |= bt_bit(a)
    end

    return bt_out(out, nvtx, nwrd)
end

# nt tasks: each takes the next start, with its own pool, list, and
# set of the starts before it. The separators of the start i are
# lsts[task[i]][lo[i] + 1:hi[i]], and they are put back in the order
# of the starts, so the result is that of sepenum_st.
function sepenum_mt(weights::AbstractVector{W}, graph::AbstractGraph{V}, maxk, nt::Integer, deadline::Real = Inf) where {W, V}
    nvtx = nv(graph); nwrd = convert(V, cld(nvtx, 64)); maxk = convert(W, maxk); clk = bt_clk(deadline)

    adj = bt_mat(nwrd, nvtx)
    out = UInt64[]

    order = FVector{V}(undef, nvtx)
    key   = FVector{V}(undef, nvtx)
    cnt   = FVector{V}(undef, nvtx)

    smax = sepenum_adj!(adj, order, key, cnt, weights, graph, maxk)
    sepenum_emp!(out, UInt64[], adj, maxk, nvtx, nwrd)

    task = FVector{Int}(undef, nvtx)
    lo   = FVector{Int}(undef, nvtx)
    hi   = FVector{Int}(undef, nvtx)
    next = Threads.Atomic{Int}(1)

    # each task makes its own list, apart from the others in memory
    tasks = [Threads.@spawn sepenum_run!(task, lo, hi, next, t, adj, order, weights, maxk, smax, nvtx, nwrd, bt_clk(clk)) for t in oneto(nt)]
    lsts = Vector{UInt64}[fetch(t) for t in tasks]
    clk.out[] && return nothing

    @inbounds for i in oneto(nvtx)
        lst = lsts[task[i]]

        for j in lo[i] + 1:hi[i]
            push!(out, lst[j])
        end
    end

    return bt_out(out, nvtx, nwrd)
end

# The task t: take starts until none are left, and return its list.
function sepenum_run!(
        task::AbstractVector{Int},
        lo::AbstractVector{Int},
        hi::AbstractVector{Int},
        next::Threads.Atomic{Int},
        t::Int,
        adj::AbstractVecOrMat{UInt64},
        order::AbstractVector{V},
        weights::AbstractVector{W},
        maxk::W,
        smax::Int,
        nvtx::Integer,
        nwrd::V,
        clk::NamedTuple,
    ) where {W, V}
    excl = FVector{UInt64}(undef, nwrd); pool = UInt64[]; lst = UInt64[]; j = 1
    fill!(excl, zero(UInt64))

    @inbounds while true
        i = Threads.atomic_add!(next, 1)
        (i > nvtx || clk.out[]) && break

        # the starts before i
        while j < i
            v = order[j]; excl[bt_wrd(v)] |= bt_bit(v); j += 1
        end

        task[i] = t; lo[i] = length(lst)
        sepenum_one!(pool, lst, adj, excl, weights, maxk, smax, nvtx, nwrd, order[i], clk)
        hi[i] = length(lst)
    end

    return lst
end

# ∅, if the graph is disconnected.
function sepenum_emp!(out::Vector{UInt64}, pool::Vector{UInt64}, adj::AbstractVecOrMat{UInt64}, maxk::W, nvtx::Integer, nwrd::V) where {W, V}
    if !iszero(nvtx) && zero(W) < maxk
        pv = bt_new!(pool, nwrd)
        pc = bt_new!(pool, nwrd)

        @inbounds for i in oneto(nwrd)
            pool[pv + i] = ~zero(UInt64)
        end

        bt_cap!(pool, nvtx, pv, nwrd)
        bt_cmp!(pool, adj, pc, pv, 1, nwrd)

        if !bt_eq(pool, pc, pv, nwrd)
            for _ in oneto(nwrd)
                push!(out, zero(UInt64))
            end
        end

        empty!(pool)
    end

    return
end

# The search from the start a: A = {a}, S = N(a), and the rest
# R = V - N[a], with the starts before a in excl.
function sepenum_one!(
        pool::Vector{UInt64},
        out::Vector{UInt64},
        adj::AbstractVecOrMat{UInt64},
        excl::AbstractVector{UInt64},
        weights::AbstractVector{W},
        maxk::W,
        smax::Int,
        nvtx::Integer,
        nwrd::V,
        a::Integer,
        clk::NamedTuple,
    ) where {W, V}
    pa = bt_new!(pool, nwrd)
    ps = bt_new!(pool, nwrd)
    pr = bt_new!(pool, nwrd)
    pf = bt_new!(pool, nwrd)

    @inbounds pool[pa + bt_wrd(a)] = bt_bit(a)

    @inbounds for i in oneto(nwrd)
        na = bt_get(adj, i, a)
        pool[ps + i] = na
        pool[pf + i] = na & excl[i]
        pool[pr + i] = ~(na | pool[pa + i])
    end

    bt_cap!(pool, nvtx, pr, nwrd)

    if bt_wgt(pool, pf, weights, nwrd) < maxk && sepenum_sml(1, bt_len(pool, ps, nwrd), smax, nvtx)
        sepenum_from!(pool, out, adj, excl, weights, maxk, smax, nvtx, nwrd, a, pa, pr, ps, pf, bt_wgt(pool, ps, weights, nwrd), clk)
    end

    empty!(pool)
    return
end

# The neighborhoods as packed sets, the start vertices in order
# of decreasing degree, and the largest number of vertices that
# weigh less than maxk.
function sepenum_adj!(
        adj::AbstractVecOrMat{UInt64},
        order::AbstractVector{V},
        key::AbstractVector{V},
        cnt::AbstractVector{V},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        maxk::W,
    ) where {W, V}
    nvtx = nv(graph)

    @assert nvtx <= bt_nst(adj)
    @assert nvtx <= length(order)
    @assert nvtx <= length(key)
    @assert nvtx <= length(cnt)
    @assert nvtx <= length(weights)

    bt_adj!(adj, graph)

    @inbounds for v in vertices(graph)
        d = zero(V)

        for i in oneto(bt_nwd(adj))
            d += convert(V, count_ones(bt_get(adj, i, v)))
        end

        key[v] = nvtx - d
    end

    pmcdp_sort!(order, vertices(graph), key, cnt, nvtx, nvtx)

    # the lightest vertices first
    @inbounds for v in vertices(graph)
        key[v] = v
    end

    sort!(view(key, oneto(nvtx)); by = v -> weights[v])
    smax = 0; w = zero(W)

    @inbounds for v in view(key, oneto(nvtx))
        w += weights[v]
        w < maxk || break
        smax += 1
    end

    return smax
end

# Grow the bag A = pa into the rest R = pr, where S = ps = N(A)
# and F = pf ⊆ S is fixed. Each component B of R is either
# full, so that (S, A, B) is a branch, or not full, so that
# N(B) is a smaller separator with a full component containing A.
# Every full component of S, or of N(B), other than the one
# containing a is a component of R (see above).
function sepenum_from!(
        pool::Vector{UInt64},
        out::Vector{UInt64},
        adj::AbstractVecOrMat{UInt64},
        excl::AbstractVector{UInt64},
        weights::AbstractVector{W},
        maxk::W,
        smax::Int,
        nvtx::Integer,
        nwrd::V,
        a::Integer,
        pa::V,
        pr::V,
        ps::V,
        pf::V,
        ws::W,
        clk::NamedTuple,
    ) where {W, V}
    top = convert(V, length(pool))
    pR = bt_new!(pool, nwrd)
    pC = bt_new!(pool, nwrd)
    pT = bt_new!(pool, nwrd)    # the vertices outside B ∪ N

    # a component B is wanted only if F ⊆ N(B), and then it meets
    # N(f) for every f ∈ F: so the components are searched only from
    # the vertices of R next to the first f
    f = bt_min(pool, pf, nwrd)

    @inbounds for i in oneto(nwrd)
        pool[pR + i] = iszero(f) ? pool[pr + i] : bt_get(adj, i, f) & pool[pr + i]
    end

    # the components B of R next to f, each followed by N = N(B) ⊆ S
    lo = convert(V, length(pool))

    @inbounds while true
        v = bt_min(pool, pR, nwrd)
        iszero(v) && break
        pB = bt_new!(pool, nwrd)
        pN = bt_new!(pool, nwrd)
        bt_cmp!(pool, adj, pB, pr, v, nwrd)

        for i in oneto(nwrd)
            pool[pR + i] &= ~pool[pB + i]
        end

        bt_nbr!(pool, adj, pN, pB, nwrd)
    end

    hi = convert(V, length(pool)) - one(V)

    # can S be listed with A?
    nS = bt_len(pool, ps, nwrd)
    lst = !iszero(nS) && twice(bt_len(pool, pa, nwrd)) <= nvtx - nS && ws < maxk

    @inbounds for pB in lo:twice(nwrd):hi
        clk.out[] && break
        pN = pB + nwrd

        if bt_eq(pool, pN, ps, nwrd)
            put = lst && sepenum_put(pool, excl, lo, hi, pB, nS, nvtx, nwrd)
            sepenum_branch!(pool, out, adj, excl, weights, maxk, smax, nvtx, nwrd, a, pa, pB, ps, pf, ws, put, clk)
        elseif bt_sub(pool, pf, pN, nwrd)
            # C: the component of a in G - B - N
            for i in oneto(nwrd)
                pool[pT + i] = ~(pool[pB + i] | pool[pN + i])
            end

            bt_cap!(pool, nvtx, pT, nwrd)
            bt_cmp!(pool, adj, pC, pT, a, nwrd)

            if bt_dsj(pool, pC, excl, nwrd)
                # can N be listed with C?
                nN = bt_len(pool, pN, nwrd); wN = bt_wgt(pool, pN, weights, nwrd)
                put = !iszero(nN) && twice(bt_len(pool, pC, nwrd)) <= nvtx - nN && wN < maxk &&
                    sepenum_put(pool, excl, lo, hi, pB, nN, nvtx, nwrd)
                sepenum_branch!(pool, out, adj, excl, weights, maxk, smax, nvtx, nwrd, a, pC, pB, pN, pf, wN, put, clk)
            end
        end
    end

    resize!(pool, top)
    return
end

# The components of R next to f, among them every one whose
# neighborhood holds F, are pool[q + 1:q + nwrd], each followed by
# its neighborhood, for q in lo:2nwrd:hi. With N = N(B) ⊇ F, B = pb:
# is B the first component with neighborhood N, and does every
# small one of them avoid the start vertices that came before a?
# These are the full components of N other than the one that
# contains a.
function sepenum_put(pool::Vector{UInt64}, excl::AbstractVector{UInt64}, lo::V, hi::V, pb::V, nn::Int, nvtx::Integer, nwrd::V) where {V}
    pn = pb + nwrd

    for q in lo:twice(nwrd):hi
        if bt_eq(pool, q + nwrd, pn, nwrd)
            q < pb && return false

            if twice(bt_len(pool, q, nwrd)) <= nvtx - nn && !bt_dsj(pool, q, excl, nwrd)
                return false
            end
        end
    end

    return true
end

# The separator S = ps has full components A = pa (containing
# a) and B = pb; F = pf ⊆ S is fixed. S is listed if put.
function sepenum_branch!(
        pool::Vector{UInt64},
        out::Vector{UInt64},
        adj::AbstractMatrix{UInt64},
        excl::AbstractVector{UInt64},
        weights::AbstractVector{W},
        maxk::W,
        smax::Int,
        nvtx::Integer,
        nwrd::V,
        a::Integer,
        pa::V,
        pb::V,
        ps::V,
        pf::V,
        ws::W,
        put::Bool,
        clk::NamedTuple,
    ) where {W, V}
    bt_tck!(clk) && return
    nA = bt_len(pool, pa, nwrd); nS = bt_len(pool, ps, nwrd)

    sepenum_sml(nA, nS, smax, nvtx) || return
    iszero(nS) && return

    # S is too heavy to be listed: can it lighten below?
    if ws >= maxk
        sepenum_ste(pool, adj, weights, pb, ps, pf, nA, maxk, nvtx, nwrd) && return
    end

    if put
        @inbounds for i in oneto(nwrd)
            push!(out, pool[ps + i])
        end
    end

    # the fixed vertices, which grow as each branch is closed, and
    # their weight
    top = convert(V, length(pool))
    pg = bt_new!(pool, nwrd)

    @inbounds for i in oneto(nwrd)
        pool[pg + i] = pool[pf + i]
    end

    wg = bt_wgt(pool, pf, weights, nwrd)

    # the unfixed vertices of S, in the order in which they are
    # taken: the most neighbors in B first, then the smallest
    lo = convert(V, length(pool))

    @inbounds for i in oneto(nwrd)
        word = pool[ps + i] & ~pool[pf + i]

        while !iszero(word)
            u = bt_vtx(Int, i, word); word &= word - one(UInt64); c = 0

            for j in oneto(nwrd)
                c += count_ones(adj[j, u] & pool[pb + j])
            end

            push!(pool, sepenum_key(c, u))
        end
    end

    hi = convert(V, length(pool))
    sepenum_srt!(pool, lo, hi)

    @inbounds for t in lo + 1:hi
        v = sepenum_vtx(pool[t])
        mid = convert(V, length(pool))

        # v into A: S becomes N(A + v), the rest becomes B - N(v)
        pA = bt_new!(pool, nwrd)
        pS = bt_new!(pool, nwrd)
        pF = bt_new!(pool, nwrd)
        pR = bt_new!(pool, nwrd)
        wF = wg; wS = ws - weights[v]

        for i in oneto(nwrd)
            nb = adj[i, v] & ~pool[ps + i] & ~pool[pa + i]
            fx = nb & excl[i]; nx = nb

            while !iszero(nx)
                wS += weights[bt_vtx(Int, i, nx)]; nx &= nx - one(UInt64)
            end

            pool[pA + i] = pool[pa + i]
            pool[pS + i] = pool[ps + i] | nb
            pool[pF + i] = pool[pg + i] | fx
            pool[pR + i] = pool[pb + i] & ~adj[i, v]

            while !iszero(fx)
                wF += weights[bt_vtx(Int, i, fx)]; fx &= fx - one(UInt64)
            end
        end

        pool[pA + bt_wrd(v)] |= bt_bit(v)
        pool[pS + bt_wrd(v)] &= ~bt_bit(v)

        # the test of sepenum_branch!, for every state below
        nS1 = 0

        for i in oneto(nwrd)
            nS1 += count_ones(pool[pS + i])
        end

        if wF < maxk && sepenum_sml(nA + 1, nS1, smax, nvtx)
            sepenum_from!(pool, out, adj, excl, weights, maxk, smax, nvtx, nwrd, a, pA, pR, pS, pF, wS, clk)
        end

        resize!(pool, mid)
        clk.out[] && break

        # v fixed in S
        wg + weights[v] < maxk || break
        wg += weights[v]; pool[pg + bt_wrd(v)] |= bt_bit(v)
    end

    resize!(pool, top)
    return
end

# sepenum_branch! with one word: the sets are words, and those that
# sepenum_from! reads are pushed onto the pool.
function sepenum_branch!(
        pool::Vector{UInt64},
        out::Vector{UInt64},
        adj::AbstractVector{UInt64},
        excl::AbstractVector{UInt64},
        weights::AbstractVector{W},
        maxk::W,
        smax::Int,
        nvtx::Integer,
        nwrd::V,
        a::Integer,
        pa::V,
        pb::V,
        ps::V,
        pf::V,
        ws::W,
        put::Bool,
        clk::NamedTuple,
    ) where {W, V}
    bt_tck!(clk) && return
    @inbounds A = pool[pa + one(V)]; B = pool[pb + one(V)]; S = pool[ps + one(V)]; F = pool[pf + one(V)]; X = excl[1]
    nA = count_ones(A); nS = count_ones(S)

    sepenum_sml(nA, nS, smax, nvtx) || return
    iszero(nS) && return

    # S is too heavy to be listed: can it lighten below?
    if ws >= maxk
        sepenum_ste(pool, adj, weights, pb, ps, pf, nA, maxk, nvtx, nwrd) && return
    end

    put && push!(out, S)

    # the fixed vertices, which grow as each branch is closed, and
    # their weight
    G = F; wg = sepenum_wgt(zero(W), F, weights)

    # the unfixed vertices of S, in the order in which they are
    # taken: the most neighbors in B first, then the smallest
    top = lo = convert(V, length(pool)); word = S & ~F

    @inbounds while !iszero(word)
        u = trailing_zeros(word) + 1; word &= word - one(UInt64)
        push!(pool, sepenum_key(count_ones(adj[u] & B), u))
    end

    hi = convert(V, length(pool))
    sepenum_srt!(pool, lo, hi)

    @inbounds for t in lo + 1:hi
        v = sepenum_vtx(pool[t]); bv = bt_bit(v); nbv = adj[v]
        mid = convert(V, length(pool))

        # v into A: S becomes N(A + v), the rest becomes B - N(v)
        nb = nbv & ~S & ~A; fx = nb & X; S1 = (S | nb) & ~bv
        wS = sepenum_wgt(ws - weights[v], nb, weights); wF = sepenum_wgt(wg, fx, weights)
        push!(pool, A | bv, S1, G | fx, B & ~nbv)

        if wF < maxk && sepenum_sml(nA + 1, count_ones(S1), smax, nvtx)
            pA = mid; pS = mid + one(V); pF = pS + one(V); pR = pF + one(V)
            sepenum_from!(pool, out, adj, excl, weights, maxk, smax, nvtx, nwrd, a, pA, pR, pS, pF, wS, clk)
        end

        resize!(pool, mid)
        clk.out[] && break

        # v fixed in S
        wg + weights[v] < maxk || break
        wg += weights[v]; G |= bv
    end

    resize!(pool, top)
    return
end

# w + the weight of the word, added from the smallest vertex.
function sepenum_wgt(w::W, word::UInt64, weights::AbstractVector{W}) where {W}
    @inbounds while !iszero(word)
        w += weights[trailing_zeros(word) + 1]; word &= word - one(UInt64)
    end

    return w
end

# Can A stay small? If S ends as S' with s' vertices, then the
# vertices of S - S' join A, and 2|A'| <= |V| - s'. This holds for
# every state below (A, S) as well.
function sepenum_sml(nA::Int, nS::Int, smax::Int, nvtx::Integer)
    return (nS <= smax && 2nA <= nvtx - nS) || 2nA + 2nS - nvtx <= min(smax, nS - 1)
end

# The key of a vertex u with c neighbors in B: larger keys are
# taken first.
function sepenum_key(c::Int, u::Int)
    return (convert(UInt64, c) << 32) | (typemax(UInt32) - convert(UInt64, u))
end

function sepenum_vtx(key::UInt64)
    return convert(Int, typemax(UInt32) - (key & typemax(UInt32)))
end

# Sort the keys pool[lo + 1:hi] in decreasing order (by insertion:
# there are at most |S| of them).
function sepenum_srt!(pool::Vector{UInt64}, lo::V, hi::V) where {V}
    @inbounds for t in lo + 2:hi
        key = pool[t]; s = t

        while s > lo + 1 && pool[s - 1] < key
            pool[s] = pool[s - 1]; s -= 1
        end

        pool[s] = key
    end

    return
end

# Is the state with separator S = ps, fixed vertices F = pf, and
# full components A (with nA vertices) and B = pb sterile, given
# that S is too heavy to be listed? Grow vertex-disjoint trees from
# the vertices of S - F into B, one vertex per tree per round
# (Tamaki's hanging trees). A separator S' listed below has F ⊆ S'
# and w(S') < maxk, so it meets at most jmax of the trees, and each
# tree that it misses lies in the side A' of a, with A. If S' meets
# j trees, then |S'| >= |F| + j, and A' is small only if the trees
# that it misses, with A, have at most (|V| - |F| - j) / 2 vertices.
# Each tree it meets lowers the left side by at least 2 and the
# right side by 1, so j = jmax is the case to check: if even the
# smallest trees that it can miss are too large, nothing below is
# listed.
function sepenum_ste(
        pool::Vector{UInt64},
        adj::AbstractMatrix{UInt64},
        weights::AbstractVector{W},
        pb::V,
        ps::V,
        pf::V,
        nA::Int,
        maxk::W,
        nvtx::Integer,
        nwrd::V,
    ) where {W, V}
    nF = bt_len(pool, pf, nwrd); nU = bt_len(pool, ps, nwrd) - nF; nB = bt_len(pool, pb, nwrd)

    # each tree that S' meets costs at least wmin, the weight of
    # the lightest vertex of (S - F) ∪ B
    wmin = typemax(W)

    @inbounds for i in oneto(nwrd)
        word = (pool[ps + i] & ~pool[pf + i]) | pool[pb + i]

        while !iszero(word)
            wmin = min(wmin, weights[bt_vtx(Int, i, word)]); word &= word - one(UInt64)
        end
    end

    jmax = 0; w = bt_wgt(pool, pf, weights, nwrd)

    while jmax < nU && w + wmin < maxk
        jmax += 1; w += wmin
    end

    # S' misses m trees; not even all of B in them is enough?
    m = nU - jmax
    (m <= 0 || twice(nA + m + nB) <= nvtx - nF - jmax) && return false

    top = convert(V, length(pool))
    pr = bt_new!(pool, nwrd)    # the vertices of B in no tree

    @inbounds for i in oneto(nwrd)
        pool[pr + i] = pool[pb + i]
    end

    # the frontiers of the trees
    lo = convert(V, length(pool))

    @inbounds for i in oneto(nwrd)
        word = pool[ps + i] & ~pool[pf + i]

        while !iszero(word)
            u = bt_vtx(Int, i, word); word &= word - one(UInt64)
            p = bt_new!(pool, nwrd)

            for j in oneto(nwrd)
                pool[p + j] = adj[j, u] & pool[pr + j]
            end
        end
    end

    # after r rounds, the nlive trees that still grow come first
    # and have r + 1 vertices; the nfrz others have stopped and are
    # smaller, and the first min(m, nfrz) of them to stop have sfrz
    # vertices
    r = 0; nlive = nU; nfrz = 0; sfrz = 0; nrst = nB; ste = false

    @inbounds while true
        sum = sfrz + max(0, m - nfrz) * (r + 1)

        if twice(nA + sum) > nvtx - nF - jmax
            ste = true; break
        end

        # not even if every vertex left joins them
        (iszero(nlive) || twice(nA + sum + nrst) <= nvtx - nF - jmax) && break
        t = zero(V); stop = 0

        while t < nlive
            p = lo + t * nwrd

            for j in oneto(nwrd)
                pool[p + j] &= pool[pr + j]
            end

            x = bt_min(pool, p, nwrd)

            if iszero(x)
                # the last tree that grows takes its place
                stop += 1; nlive -= 1; q = lo + nlive * nwrd

                for j in oneto(nwrd)
                    pool[p + j] = pool[q + j]
                end
            else
                nrst -= 1; pool[pr + bt_wrd(x)] &= ~bt_bit(x)

                for j in oneto(nwrd)
                    pool[p + j] = (pool[p + j] | adj[j, x]) & pool[pr + j]
                end

                t += one(V)
            end
        end

        # the trees that stopped in this round have r + 1 vertices
        sfrz += (r + 1) * min(stop, max(0, m - nfrz))
        nfrz += stop; r += 1
    end

    resize!(pool, top)
    return ste
end

# sepenum_ste with one word: the sets are words, and the frontiers of
# the trees are words on the pool.
function sepenum_ste(
        pool::Vector{UInt64},
        adj::AbstractVector{UInt64},
        weights::AbstractVector{W},
        pb::V,
        ps::V,
        pf::V,
        nA::Int,
        maxk::W,
        nvtx::Integer,
        nwrd::V,
    ) where {W, V}
    @inbounds B = pool[pb + one(V)]; S = pool[ps + one(V)]; F = pool[pf + one(V)]
    nF = count_ones(F); nU = count_ones(S) - nF; nB = count_ones(B)

    # each tree that S' meets costs at least wmin, the weight of
    # the lightest vertex of (S - F) ∪ B
    wmin = typemax(W); word = (S & ~F) | B

    @inbounds while !iszero(word)
        wmin = min(wmin, weights[trailing_zeros(word) + 1]); word &= word - one(UInt64)
    end

    jmax = 0; w = sepenum_wgt(zero(W), F, weights)

    while jmax < nU && w + wmin < maxk
        jmax += 1; w += wmin
    end

    # S' misses m trees; not even all of B in them is enough?
    m = nU - jmax
    (m <= 0 || twice(nA + m + nB) <= nvtx - nF - jmax) && return false

    # the vertices of B in no tree, and the frontiers of the trees
    R = B; top = lo = length(pool); word = S & ~F

    @inbounds while !iszero(word)
        push!(pool, adj[trailing_zeros(word) + 1] & R); word &= word - one(UInt64)
    end

    # after r rounds, the nlive trees that still grow come first
    # and have r + 1 vertices; the nfrz others have stopped and are
    # smaller, and the first min(m, nfrz) of them to stop have sfrz
    # vertices
    r = 0; nlive = nU; nfrz = 0; sfrz = 0; nrst = nB; ste = false

    @inbounds while true
        sum = sfrz + max(0, m - nfrz) * (r + 1)

        if twice(nA + sum) > nvtx - nF - jmax
            ste = true; break
        end

        # not even if every vertex left joins them
        (iszero(nlive) || twice(nA + sum + nrst) <= nvtx - nF - jmax) && break
        t = 0; stop = 0

        while t < nlive
            f = pool[lo + t + 1] & R

            if iszero(f)
                # the last tree that grows takes its place
                stop += 1; nlive -= 1; pool[lo + t + 1] = pool[lo + nlive + 1]
            else
                x = trailing_zeros(f) + 1; nrst -= 1; R &= ~bt_bit(x)
                pool[lo + t + 1] = (f | adj[x]) & R; t += 1
            end
        end

        # the trees that stopped in this round have r + 1 vertices
        sfrz += (r + 1) * min(stop, max(0, m - nfrz))
        nfrz += stop; r += 1
    end

    resize!(pool, top)
    return ste
end
