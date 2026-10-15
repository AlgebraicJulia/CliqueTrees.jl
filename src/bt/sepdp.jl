#
# Decide whether a graph has a tree decomposition of width
# at most maxk, given a collection S of minimal separators.
# A block is a pair (S, D), where S ∈ S and D is a small full
# component of G - S, that is, 2|D| <= |V| - |S|. A cap of a
# block (S, D) is a set Ω with S ⊂ Ω ⊆ S ∪ D such that every
# component of D - Ω is a block. The value of a block is
#
#   min over caps Ω of max(w(Ω), the values of the components of D - Ω)
#
# The components of D - Ω are small, so the recurrence never
# needs a large component. The root (∅, V) is searched in the
# same way, except that its bag may grow into one large full
# component even when it has other full components, which must
# be blocks.
#
# The caps are not listed. They are found by Tamaki's search,
# which stops at the first cap of value at most maxk. Blocks
# are evaluated in order of increasing |D|, and each is stored
# as the smallest vertex of D, its separator S, and D and S as
# packed sets. Each component D is found once, by the search
# for the blocks. The searches split sets into components by
# breadth-first search on packed sets, and a component of the
# search is matched to its block by its smallest vertex and its
# neighborhood, which determine it. The children that a bag can
# take are found through a trie of the separators (sepdp_tri). After
# the blocks are found, the vertices are renumbered so that those in
# fewest blocks come first (sepdp_lab!): the root, which is searched
# by its smallest vertex as any block is, then branches first on
# vertices with few blocks, as in Tamaki's search for the root.
#
# With nt > 1, the blocks are found (sepdp_blk_mt!) and evaluated
# (sepdp_val_mt!) by nt tasks, and the result is the same as with
# one.
#
# If mink <= maxk, then (perm, invp) is an elimination ordering
# of width at most mink. Otherwise, mink is a lower bound on the
# width of every decomposition that the search can build, and
# (perm, invp) is the identity. If the weights are positive and
# S contains every minimal separator of weight less than maxk,
# then mink <= maxk if and only if the graph has a tree
# decomposition of width at most maxk. In that case, a failed
# call gives the lower bound min(mink, maxk + w), where w is the
# smallest weight: a decomposition that needs a separator of
# weight at least maxk has a bag heavier than it.
#
#   - Tamaki, Hisao. "Computing treewidth via exact and heuristic
#     lists of minimal separators." SEA 2019.
#
# With fewer separators or blocks than this, a phase uses one task.
const SEPDP_MT = 256

function sepdp(weights::AbstractVector{W}, graph::AbstractGraph{V}, sep::BipartiteGraph{V}, maxk; nt::Integer = 1) where {W, V}
    nvtx = nv(graph); nsep = nv(sep); maxk = convert(W, maxk)

    perm = FVector{V}(undef, nvtx)
    invp = FVector{V}(undef, nvtx)

    if iszero(nvtx)
        mink = zero(W)
    else
        nwrd   = cld(nvtx, 64)
        adj    = bt_adj!(FMatrix{UInt64}(undef, nwrd, nvtx), graph)
        work1  = FVector{V}(undef, nvtx)
        sepwgt = FVector{W}(undef, nsep)

        if nt <= 1 || nsep < SEPDP_MT
            lstmin, lstlen, lstwgt, lstsep, lstset = sepdp_blk!(
                sepwgt, adj, weights, graph, sep)
        else
            lstmin, lstlen, lstwgt, lstsep, lstset = sepdp_blk_mt!(
                sepwgt, adj, weights, graph, sep, nt)
        end

        nblk = convert(V, length(lstmin))

        # the rest is done in the work numbering, in which the old
        # vertex of the vertex v is new2old[v]
        new2old, old2new, moved = sepdp_lab!(lstmin, lstset, nvtx, nwrd)

        if moved
            weights, graph, sep, adj = sepdp_rel(weights, graph, sep, adj, new2old, old2new)
        end

        blkmin = FVector{V}(undef, nblk)
        blklen = FVector{V}(undef, nblk)
        blkwgt = FVector{W}(undef, nblk)
        blksep = FVector{V}(undef, nblk)
        order  = FVector{V}(undef, nblk)

        fst = sepdp_srt!(blkmin, blklen, blkwgt, blksep, order, work1,
            lstmin, lstlen, lstwgt, lstsep, nvtx)

        blkset, sepset = sepdp_set!(sep, blksep, order, lstset, nvtx)
        trie = sepdp_tri(fst, blklen, sepset, weights)

        valk   = FVector{W}(undef, nblk + 1)
        capptr = FVector{V}(undef, nblk + 2)

        if nt <= 1 || nblk < SEPDP_MT
            cap = sepdp_val!(valk, capptr, adj, weights, sep, sepwgt,
                blkset, sepset, fst, trie, blkmin, blklen, blkwgt, blksep, maxk)
        else
            cap = sepdp_val_mt!(valk, capptr, adj, weights, sep, sepwgt,
                blkset, sepset, fst, trie, blkmin, blklen, blkwgt, blksep, maxk, nt)
        end

        mink = valk[nblk + one(V)]

        if mink <= maxk
            tag   = FScalar{V}(undef); tag[] = zero(V)
            stack = FVector{V}(undef, nvtx + 1)
            queue = work1
            insep = FVector{V}(undef, nvtx)

            sepdp_ord!(perm, invp, stack, queue, insep, tag,
                graph, sep, fst, blklen, blksep, cap)

            # back to the numbering of the graph
            if moved
                for i in oneto(nvtx)
                    perm[i] = new2old[perm[i]]
                end

                for i in oneto(nvtx)
                    invp[perm[i]] = i
                end
            end
        else
            for v in oneto(nvtx)
                perm[v] = invp[v] = v
            end
        end
    end

    return perm, invp, mink
end

# The blocks: the small full components of G - S, for each S.
function sepdp_blk!(
        sepwgt::AbstractVector{W},
        adj::AbstractMatrix{UInt64},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        sep::BipartiteGraph{V},
    ) where {W, V}
    nvtx = nv(graph); nsep = nv(sep); nwrd = size(adj, 1)

    @assert nsep <= length(sepwgt)
    @assert nvtx <= size(adj, 2)
    @assert nvtx <= length(weights)

    lsts = (V[], V[], W[], V[], UInt64[]); pool = UInt64[]

    @inbounds for s in vertices(sep)
        sepdp_bk1!(lsts, pool, sepwgt, adj, weights, sep, s, nvtx, nwrd)
    end

    return lsts
end

# The same with nt tasks, each taking the next separator, with its
# own pool and lists. The blocks of the separator s are blocks
# lo[s] + 1:hi[s] of the lists of the task task[s], and they are put
# back in the order of the separators.
function sepdp_blk_mt!(
        sepwgt::AbstractVector{W},
        adj::AbstractMatrix{UInt64},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        sep::BipartiteGraph{V},
        nt::Integer,
    ) where {W, V}
    nvtx = nv(graph); nsep = nv(sep); nwrd = size(adj, 1)

    @assert nsep <= length(sepwgt)
    @assert nvtx <= size(adj, 2)
    @assert nvtx <= length(weights)

    task = FVector{Int}(undef, nsep)
    lo   = FVector{Int}(undef, nsep)
    hi   = FVector{Int}(undef, nsep)
    next = Threads.Atomic{Int}(1)

    # each task makes its own lists, apart from the others in memory
    tasks = [Threads.@spawn sepdp_bkt!(task, lo, hi, next, t, sepwgt, adj, weights, sep, nvtx, nwrd) for t in oneto(nt)]
    lstt = Tuple{Vector{V}, Vector{V}, Vector{W}, Vector{V}, Vector{UInt64}}[fetch(t) for t in tasks]

    lsts = (V[], V[], W[], V[], UInt64[])

    @inbounds for s in oneto(nsep)
        src = lstt[task[s]]

        for j in lo[s] + 1:hi[s]
            for l in 1:4
                push!(lsts[l], src[l][j])
            end

            for i in oneto(nwrd)
                push!(lsts[5], src[5][(j - 1) * nwrd + i])
            end
        end
    end

    return lsts
end

# The task t of sepdp_blk_mt!: take separators until none are left,
# and return its lists.
function sepdp_bkt!(
        task::AbstractVector{Int},
        lo::AbstractVector{Int},
        hi::AbstractVector{Int},
        next::Threads.Atomic{Int},
        t::Int,
        sepwgt::AbstractVector{W},
        adj::AbstractMatrix{UInt64},
        weights::AbstractVector{W},
        sep::BipartiteGraph{V},
        nvtx::Integer,
        nwrd::Int,
    ) where {W, V}
    nsep = nv(sep); pool = UInt64[]; lsts = (V[], V[], W[], V[], UInt64[])

    @inbounds while true
        s = Threads.atomic_add!(next, 1)
        s > nsep && break
        task[s] = t; lo[s] = length(lsts[1])
        sepdp_bk1!(lsts, pool, sepwgt, adj, weights, sep, convert(V, s), nvtx, nwrd)
        hi[s] = length(lsts[1])
    end

    return lsts
end

# Append the blocks of the separator s to the lists (lstmin,
# lstlen, lstwgt, lstsep, lstset).
function sepdp_bk1!(
        lsts::Tuple,
        pool::Vector{UInt64},
        sepwgt::AbstractVector{W},
        adj::AbstractMatrix{UInt64},
        weights::AbstractVector{W},
        sep::BipartiteGraph{V},
        s::V,
        nvtx::Integer,
        nwrd::Int,
    ) where {W, V}
    lstmin, lstlen, lstwgt, lstsep, lstset = lsts; top = length(pool)
    ps = bt_new!(pool, nwrd)    # S
    pr = bt_new!(pool, nwrd)    # the rest of G - S
    pd = bt_new!(pool, nwrd)    # a component D
    pn = bt_new!(pool, nwrd)    # N(D)
    n = 0; w = zero(W)

    @inbounds for x in neighbors(sep, s)
        pool[ps + bt_wrd(x)] |= bt_bit(x); n += 1; w += weights[x]
    end

    @inbounds sepwgt[s] = w

    @inbounds for i in oneto(nwrd)
        pool[pr + i] = ~pool[ps + i]
    end

    bt_cap!(pool, nvtx, pr, nwrd)

    @inbounds while true
        v = bt_min(pool, pr, nwrd)
        iszero(v) && break
        bt_cmp!(pool, adj, pd, pr, v, nwrd)

        for i in oneto(nwrd)
            pool[pr + i] &= ~pool[pd + i]
            pool[pn + i] = zero(UInt64)
        end

        bt_nbr!(pool, adj, pn, pd, nwrd)
        t = bt_len(pool, pd, nwrd)

        if twice(t) <= nvtx - n && bt_eq(pool, pn, ps, nwrd)
            for i in oneto(nwrd)
                push!(lstset, pool[pd + i])
            end

            push!(lstmin, convert(V, v))
            push!(lstlen, convert(V, t))
            push!(lstwgt, bt_wgt(pool, pd, weights, nwrd))
            push!(lstsep, s)
        end
    end

    resize!(pool, top)
    return
end

# The work numbering: the vertices in order of the number of blocks
# whose component contains them, fewest first. This is Tamaki's order
# for the root, and it also groups the blocks well, by their smallest
# vertex. The blocks (lstmin and lstset) are renumbered in place;
# returns the old vertex of each new vertex, its inverse, and whether
# any vertex moved.
function sepdp_lab!(lstmin::AbstractVector{V}, lstset::AbstractVector{UInt64}, nvtx::V, nwrd::Int) where {V}
    nblk = length(lstmin); cnt = zeros(Int, nvtx); new = zeros(UInt64, nwrd)

    @inbounds for p in 0:nwrd:nblk * nwrd - 1, i in oneto(nwrd)
        word = lstset[p + i]

        while !iszero(word)
            cnt[bt_vtx(Int, i, word)] += 1; word &= word - one(UInt64)
        end
    end

    new2old = convert(Vector{V}, sortperm(cnt))
    old2new = invperm(new2old)
    issorted(cnt) && return new2old, old2new, false

    @inbounds for b in oneto(nblk)
        p = (b - 1) * nwrd; fill!(new, zero(UInt64)); m = nvtx

        for i in oneto(nwrd)
            word = lstset[p + i]

            while !iszero(word)
                v = old2new[bt_vtx(Int, i, word)]; word &= word - one(UInt64)
                new[bt_wrd(v)] |= bt_bit(v); m = min(m, v)
            end
        end

        for i in oneto(nwrd)
            lstset[p + i] = new[i]
        end

        lstmin[b] = m
    end

    return new2old, old2new, true
end

# The weights, graph, separators, and packed neighborhoods in the work
# numbering.
function sepdp_rel(
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        sep::BipartiteGraph{V},
        adj::AbstractMatrix{UInt64},
        new2old::AbstractVector{V},
        old2new::AbstractVector{V},
    ) where {W, V}
    nvtx = nv(graph); nwrd = size(adj, 1)
    wgt = FVector{W}(undef, nvtx)
    nbr = FMatrix{UInt64}(undef, nwrd, nvtx); fill!(nbr, zero(UInt64))
    ptr = V[one(V)]; tgt = V[]

    @inbounds for v in oneto(nvtx)
        u = new2old[v]; wgt[v] = weights[u]

        for x in neighbors(graph, u)
            push!(tgt, old2new[x])
        end

        push!(ptr, convert(V, length(tgt)) + one(V))

        for i in oneto(nwrd)
            word = adj[i, u]

            while !iszero(word)
                y = old2new[bt_vtx(Int, i, word)]; word &= word - one(UInt64)
                nbr[bt_wrd(y), v] |= bt_bit(y)
            end
        end
    end

    gph = BipartiteGraph{V, V}(convert(V, nvtx), convert(V, nvtx), length(tgt), ptr, tgt)
    sptr = V[one(V)]; stgt = V[]

    @inbounds for s in vertices(sep)
        for x in neighbors(sep, s)
            push!(stgt, old2new[x])
        end

        push!(sptr, convert(V, length(stgt)) + one(V))
    end

    spr = BipartiteGraph{V, V}(convert(V, nvtx), nv(sep), length(stgt), sptr, stgt)
    return wgt, gph, spr, nbr
end

function sepdp_srt!(
        blkmin::AbstractVector{V},
        blklen::AbstractVector{V},
        blkwgt::AbstractVector{W},
        blksep::AbstractVector{V},
        order::AbstractVector{V},
        cnt::AbstractVector{V},
        lstmin::AbstractVector{V},
        lstlen::AbstractVector{V},
        lstwgt::AbstractVector{W},
        lstsep::AbstractVector{V},
        nvtx::V,
    ) where {W, V}
    nblk = convert(V, length(lstmin))

    @assert nblk <= length(blkmin)
    @assert nblk <= length(blklen)
    @assert nblk <= length(blkwgt)
    @assert nblk <= length(blksep)
    @assert nblk <= length(order)
    @assert nvtx <= length(cnt)
    @assert nblk <= length(lstlen)
    @assert nblk <= length(lstwgt)
    @assert nblk <= length(lstsep)

    pmcdp_sort!(order, oneto(nblk), lstlen, cnt, nblk, nvtx)

    @inbounds for b in oneto(nblk)
        c = order[b]

        blkmin[b] = lstmin[c]
        blklen[b] = lstlen[c]
        blkwgt[b] = lstwgt[c]
        blksep[b] = lstsep[c]
    end

    fst = reverse(BipartiteGraph{V, V}(nvtx, nblk, nblk, oneto(nblk + one(V)), blkmin))
    return fst
end

function sepdp_set!(
        sep::BipartiteGraph{V},
        blksep::AbstractVector{V},
        order::AbstractVector{V},
        lstset::AbstractVector{UInt64},
        nvtx::V,
    ) where {V}
    nblk = convert(V, length(blksep)); nwrd = cld(nvtx, 64)

    @assert nblk <= length(order)
    @assert nblk * nwrd <= length(lstset)

    blkset = FMatrix{UInt64}(undef, nwrd, nblk)
    sepset = FMatrix{UInt64}(undef, nwrd, nblk)
    fill!(sepset, zero(UInt64))

    @inbounds for b in oneto(nblk)
        p = (order[b] - one(V)) * nwrd

        for i in oneto(nwrd)
            blkset[i, b] = lstset[p + i]
        end

        for x in neighbors(sep, blksep[b])
            sepset[bt_wrd(x), b] |= bt_bit(x)
        end
    end

    return blkset, sepset
end

function sepdp_val!(
        valk::AbstractVector{W},
        capptr::AbstractVector{V},
        adj::AbstractMatrix{UInt64},
        weights::AbstractVector{W},
        sep::BipartiteGraph{V},
        sepwgt::AbstractVector{W},
        blkset::AbstractMatrix{UInt64},
        sepset::AbstractMatrix{UInt64},
        fst::BipartiteGraph{V},
        trie::NamedTuple,
        blkmin::AbstractVector{V},
        blklen::AbstractVector{V},
        blkwgt::AbstractVector{W},
        blksep::AbstractVector{V},
        maxk::W,
    ) where {W, V}
    nwrd = size(adj, 1); nvtx = size(adj, 2); nblk = convert(V, length(blkmin)); nsep = nv(sep)

    @assert nblk < length(valk)
    @assert nblk + one(V) < length(capptr)
    @assert nvtx <= length(weights)
    @assert nsep <= length(sepwgt)
    @assert nblk <= length(blklen)
    @assert nblk <= length(blkwgt)
    @assert nblk <= length(blksep)

    pool   = UInt64[]
    captgt = V[]
    bagset = FVector{UInt64}(undef, nwrd)

    @inbounds for b in oneto(nblk)
        capptr[b] = convert(V, length(captgt)) + one(V)
        sepdp_bv1!(valk, captgt, pool, bagset, adj, weights, sep, sepwgt,
            blkset, sepset, fst, trie, blkmin, blklen, blkwgt, blksep, maxk, b)
    end

    return sepdp_rot!(valk, capptr, captgt, pool, bagset, adj, weights,
        blkset, sepset, fst, trie, blklen, maxk)
end

# The same with nt tasks. The blocks are in order of size, and the
# children of a block are smaller, so the blocks of each size are
# evaluated in parallel: the tasks take blocks in order, each with
# its own workspace and list of caps, and a block waits only for the
# smaller ones. The cap of the block b is caps[task[b]][lo[b] +
# 1:hi[b]]; the caps are put back in the order of the blocks, and
# the root is searched by one task.
function sepdp_val_mt!(
        valk::AbstractVector{W},
        capptr::AbstractVector{V},
        adj::AbstractMatrix{UInt64},
        weights::AbstractVector{W},
        sep::BipartiteGraph{V},
        sepwgt::AbstractVector{W},
        blkset::AbstractMatrix{UInt64},
        sepset::AbstractMatrix{UInt64},
        fst::BipartiteGraph{V},
        trie::NamedTuple,
        blkmin::AbstractVector{V},
        blklen::AbstractVector{V},
        blkwgt::AbstractVector{W},
        blksep::AbstractVector{V},
        maxk::W,
        nt::Integer,
    ) where {W, V}
    nwrd = size(adj, 1); nvtx = size(adj, 2); nblk = convert(V, length(blkmin)); nsep = nv(sep)

    @assert nblk < length(valk)
    @assert nblk + one(V) < length(capptr)
    @assert nvtx <= length(weights)
    @assert nsep <= length(sepwgt)
    @assert nblk <= length(blklen)
    @assert nblk <= length(blkwgt)
    @assert nblk <= length(blksep)

    task  = FVector{Int}(undef, nblk)
    lo    = FVector{Int}(undef, nblk)
    hi    = FVector{Int}(undef, nblk)

    # the blocks of each size form a class, and left[c] counts the
    # blocks of the class c not yet evaluated
    cls  = FVector{Int}(undef, nblk)
    left = Threads.Atomic{Int}[]

    @inbounds for b in oneto(nblk)
        if isone(b) || blklen[b] != blklen[b - 1]
            push!(left, Threads.Atomic{Int}(0))
        end

        cls[b] = length(left); left[end][] += 1
    end

    next = Threads.Atomic{Int}(1)

    # each task makes its own workspace and list of caps, apart from
    # the others in memory
    tasks = [Threads.@spawn sepdp_bvt!(valk, task, lo, hi, cls, left, next, t,
        adj, weights, sep, sepwgt, blkset, sepset, fst, trie, blkmin, blklen, blkwgt, blksep, maxk) for t in oneto(nt)]
    caps = Vector{V}[fetch(t) for t in tasks]

    captgt = V[]

    @inbounds for b in oneto(nblk)
        capptr[b] = convert(V, length(captgt)) + one(V)
        src = caps[task[b]]

        for j in lo[b] + 1:hi[b]
            push!(captgt, src[j])
        end
    end

    return sepdp_rot!(valk, capptr, captgt, UInt64[], FVector{UInt64}(undef, nwrd), adj, weights,
        blkset, sepset, fst, trie, blklen, maxk)
end

# The task t of sepdp_val_mt!: take the next block until none are
# left, and return its list of caps. A block waits until the blocks of the class before its own,
# which are smaller, have been evaluated; the classes before that
# finished first.
function sepdp_bvt!(
        valk::AbstractVector{W},
        task::AbstractVector{Int},
        lo::AbstractVector{Int},
        hi::AbstractVector{Int},
        cls::AbstractVector{Int},
        left::Vector{Threads.Atomic{Int}},
        next::Threads.Atomic{Int},
        t::Int,
        adj::AbstractMatrix{UInt64},
        weights::AbstractVector{W},
        sep::BipartiteGraph{V},
        sepwgt::AbstractVector{W},
        blkset::AbstractMatrix{UInt64},
        sepset::AbstractMatrix{UInt64},
        fst::BipartiteGraph{V},
        trie::NamedTuple,
        blkmin::AbstractVector{V},
        blklen::AbstractVector{V},
        blkwgt::AbstractVector{W},
        blksep::AbstractVector{V},
        maxk::W,
    ) where {W, V}
    nblk = length(blkmin); captgt = V[]; pool = UInt64[]
    bagset = FVector{UInt64}(undef, size(adj, 1)); trie = merge(trie, (; stack = Int[]))

    @inbounds while true
        b = Threads.atomic_add!(next, 1)
        b > nblk && break
        c = cls[b]

        if c > 1
            while ispositive(left[c - 1][])
                ccall(:jl_cpu_pause, Cvoid, ())
                GC.safepoint()
            end
        end

        task[b] = t; lo[b] = length(captgt)
        sepdp_bv1!(valk, captgt, pool, bagset, adj, weights, sep, sepwgt,
            blkset, sepset, fst, trie, blkmin, blklen, blkwgt, blksep, maxk, convert(V, b))
        hi[b] = length(captgt)
        Threads.atomic_sub!(left[c], 1)
    end

    return captgt
end

# Evaluate the block b, and append its cap to captgt.
function sepdp_bv1!(
        valk::AbstractVector{W},
        captgt::AbstractVector{V},
        pool::Vector{UInt64},
        bagset::AbstractVector{UInt64},
        adj::AbstractMatrix{UInt64},
        weights::AbstractVector{W},
        sep::BipartiteGraph{V},
        sepwgt::AbstractVector{W},
        blkset::AbstractMatrix{UInt64},
        sepset::AbstractMatrix{UInt64},
        fst::BipartiteGraph{V},
        trie::NamedTuple,
        blkmin::AbstractVector{V},
        blklen::AbstractVector{V},
        blkwgt::AbstractVector{W},
        blksep::AbstractVector{V},
        maxk::W,
        b::V,
    ) where {W, V}
    nwrd = length(bagset); top = length(pool); s = blksep[b]
    pc = bt_new!(pool, nwrd)    # the scope

    @inbounds for i in oneto(nwrd)
        bagset[i] = sepset[i, b]; pool[pc + i] = blkset[i, b]
    end

    @inbounds valk[b] = sepdp_cap!(captgt, pool, bagset, adj, weights, blkset, sepset,
        fst, trie, blklen, valk, maxk, convert(Int, eltypedegree(sep, s)), sepwgt[s],
        pc, convert(Int, blklen[b]), convert(Int, blkmin[b]), blkwgt[b], zero(W), false)

    resize!(pool, top)
    return
end

# Search the root, append its cap to captgt, and return the caps.
function sepdp_rot!(
        valk::AbstractVector{W},
        capptr::AbstractVector{V},
        captgt::Vector{V},
        pool::Vector{UInt64},
        bagset::AbstractVector{UInt64},
        adj::AbstractMatrix{UInt64},
        weights::AbstractVector{W},
        blkset::AbstractMatrix{UInt64},
        sepset::AbstractMatrix{UInt64},
        fst::BipartiteGraph{V},
        trie::NamedTuple,
        blklen::AbstractVector{V},
        maxk::W,
    ) where {W, V}
    nwrd = size(adj, 1); nvtx = size(adj, 2); r = convert(V, size(blkset, 2)) + one(V)
    top = length(pool); wall = zero(W)
    pc = bt_new!(pool, nwrd)    # the scope

    @inbounds for v in oneto(nvtx)
        wall += weights[v]
    end

    @inbounds for i in oneto(nwrd)
        bagset[i] = zero(UInt64); pool[pc + i] = ~zero(UInt64)
    end

    bt_cap!(pool, nvtx, pc, nwrd)
    @inbounds capptr[r] = convert(V, length(captgt)) + one(V)

    @inbounds valk[r] = sepdp_cap!(captgt, pool, bagset, adj, weights, blkset, sepset,
        fst, trie, blklen, valk, maxk, 0, zero(W), pc, nvtx, 1, wall, zero(W), true)

    resize!(pool, top)
    ncap = convert(V, length(captgt))
    @inbounds capptr[r + one(V)] = ncap + one(V)
    return BipartiteGraph{V, V}(convert(V, nvtx), r, ncap, capptr, captgt)
end

# Search the caps that grow the bag (bagset, with nbag vertices
# and weight wbag) into the scope (the set pc, with clen vertices,
# smallest vertex cmin, and weight cwgt), a full component of the
# bag; m is the largest value of the children chosen so far.
# Returns the value of the first cap of value at most maxk, which
# is written to captgt, or else a lower bound greater than maxk.
function sepdp_cap!(
        captgt::AbstractVector{V},
        pool::Vector{UInt64},
        bagset::AbstractVector{UInt64},
        adj::AbstractMatrix{UInt64},
        weights::AbstractVector{W},
        blkset::AbstractMatrix{UInt64},
        sepset::AbstractMatrix{UInt64},
        fst::BipartiteGraph{V},
        trie::NamedTuple,
        blklen::AbstractVector{V},
        valk::AbstractVector{W},
        maxk::W,
        nbag::Int,
        wbag::W,
        pc::Int,
        clen::Int,
        cmin::Int,
        cwgt::W,
        m::W,
        root::Bool,
    ) where {W, V}
    nwrd = length(bagset)

    # the bag ∪ the scope
    k = max(wbag + cwgt, m)

    if k <= maxk
        @inbounds for i in oneto(nwrd)
            word = bagset[i] | pool[pc + i]

            while !iszero(word)
                push!(captgt, bt_vtx(V, i, word)); word &= word - one(UInt64)
            end
        end

        return k
    end

    mink = k; top = length(pool)
    po = bt_new!(pool, nwrd)    # the bag

    @inbounds for i in oneto(nwrd)
        pool[po + i] = bagset[i]
    end

    # cmin in a child x: its component lies in the scope, so it is
    # disjoint from the bag. The blocks of cmin are in fst by size;
    # if many are smaller than the scope, they are searched through
    # the trie of their separators, where a node is cut if no
    # separator below can fit the bag below mink
    tgt = targets(fst); rng = incident(fst, cmin); lo = first(rng); hi = last(rng) + 1

    while lo < hi
        p = lo + half(hi - lo)

        if blklen[tgt[p]] < clen
            lo = p + 1
        else
            hi = p
        end
    end

    q = first(rng); qend = lo - 1; lin = qend - q < SEPDP_TRIE
    tord = trie.tord; nodl = trie.nodl; nodr = trie.nodr; wmin = trie.wmin
    stack = trie.stack; sp = length(stack)

    if !lin
        push!(stack, trie.head[cmin], first(rng), last(rng))
    end

    @inbounds while true
        if lin
            q > qend && break
            x = tgt[q]; q += 1
        else
            x = 0

            while length(stack) > sp
                hi = pop!(stack); lo = pop!(stack); p = pop!(stack)

                if iszero(p)
                    x = tord[lo]
                    break
                end

                trie.nodn[p] < clen || continue

                # (2) the bag and the vertices of every separator
                # below, and (2') the lightest separator below and
                # the vertices of the bag in none of them
                c = 0; d = 0

                for i in oneto(nwrd)
                    c += count_ones(trie.nodall[i, p] & ~bagset[i])
                    d += count_ones(bagset[i] & ~trie.nodany[i, p])
                end

                max(m, wbag + c * wmin, trie.nodw[p] + d * wmin) < mink || continue

                # the side without the split vertex first
                push!(stack, nodr[p], p + 1, hi)
                push!(stack, nodl[p], lo, p)
            end

            iszero(x) && break
            blklen[x] < clen || continue
        end

        kx = max(m, valk[x])
        kx <= maxk || kx < mink || continue
        out = true

        for i in oneto(nwrd)
            if !iszero(blkset[i, x] & bagset[i])
                out = false
                break
            end
        end

        out || continue

        if kx > maxk
            mink = kx
            continue
        end

        # the weight of the bag ∪ the separator of x
        w = wbag

        for i in oneto(nwrd)
            word = sepset[i, x] & ~bagset[i]

            while !iszero(word)
                w += weights[bt_vtx(V, i, word)]; word &= word - one(UInt64)
            end
        end

        k = max(w, kx)

        if k > maxk
            mink = min(mink, k)
        else
            n = nbag

            for i in oneto(nwrd)
                n += count_ones(sepset[i, x] & ~bagset[i]); bagset[i] |= sepset[i, x]
            end

            k = sepdp_uni!(captgt, pool, bagset, adj, weights, blkset, sepset,
                fst, trie, blklen, valk, maxk, n, w, pc, convert(Int, x), kx, root)

            for i in oneto(nwrd)
                bagset[i] = pool[po + i]
            end

            if k <= maxk
                resize!(stack, sp)
                resize!(pool, top)
                return k
            end

            mink = min(mink, k)
        end
    end

    # cmin in the bag
    w = wbag + weights[cmin]; k = max(w, m)

    if k <= maxk
        @inbounds bagset[bt_wrd(cmin)] |= bt_bit(cmin)

        k = sepdp_uni!(captgt, pool, bagset, adj, weights, blkset, sepset,
            fst, trie, blklen, valk, maxk, nbag + 1, w, pc, 0, m, root)

        @inbounds bagset[bt_wrd(cmin)] &= ~bt_bit(cmin)
    end

    resize!(pool, top)
    return min(mink, k)
end

# The bag is bagset, with nbag vertices and weight wbag; the
# rest is the scope pc minus the bag and minus the component of
# the child x (if x > 0). Every component Y of the rest has
# N(Y) ⊆ bag, since N(D_x) ⊆ bag. Each Y that is not full must
# be a block, and is a child. If every full component is a
# block, then the bag is a cap. Otherwise, if one full component
# is not a block, the search continues into it, provided that it
# is the only full component or, at the root, that it is large.
function sepdp_uni!(
        captgt::AbstractVector{V},
        pool::Vector{UInt64},
        bagset::AbstractVector{UInt64},
        adj::AbstractMatrix{UInt64},
        weights::AbstractVector{W},
        blkset::AbstractMatrix{UInt64},
        sepset::AbstractMatrix{UInt64},
        fst::BipartiteGraph{V},
        trie::NamedTuple,
        blklen::AbstractVector{V},
        valk::AbstractVector{W},
        maxk::W,
        nbag::Int,
        wbag::W,
        pc::Int,
        x::Int,
        m::W,
        root::Bool,
    ) where {W, V}
    nwrd = length(bagset); nvtx = size(adj, 2); top = length(pool)
    pr = bt_new!(pool, nwrd)    # the rest
    py = bt_new!(pool, nwrd)    # a component Y of the rest
    pn = bt_new!(pool, nwrd)    # N(Y)
    pf = bt_new!(pool, nwrd)    # the full component that is not a block

    @inbounds for i in oneto(nwrd)
        pool[pr + i] = pool[pc + i] & ~bagset[i]
    end

    if ispositive(x)
        @inbounds for i in oneto(nwrd)
            pool[pr + i] &= ~blkset[i, x]
        end
    end

    # nnon full components are not blocks; the last one is pf,
    # with flen vertices and smallest vertex fmin
    nful = nnon = flen = fmin = 0; mful = m

    @inbounds while true
        v = bt_min(pool, pr, nwrd)
        iszero(v) && break
        bt_cmp!(pool, adj, py, pr, v, nwrd)

        for i in oneto(nwrd)
            pool[pr + i] &= ~pool[py + i]
            pool[pn + i] = zero(UInt64)
        end

        bt_nbr!(pool, adj, pn, py, nwrd)
        ylen = bt_len(pool, py, nwrd)
        y = sepdp_get(fst, blklen, sepset, pool, pn, v, ylen, nwrd)

        if bt_len(pool, pn, nwrd) == nbag
            nful += 1

            if iszero(y)
                nnon += 1; flen = ylen; fmin = v

                for i in oneto(nwrd)
                    pool[pf + i] = pool[py + i]
                end
            else
                mful = max(mful, valk[y])
            end
        else
            k = iszero(y) ? typemax(W) : max(wbag, m, valk[y])

            if k > maxk
                resize!(pool, top)
                return k
            end

            m = max(m, valk[y]); mful = max(mful, valk[y])
        end
    end

    if iszero(nnon)
        k = max(wbag, mful)

        if k <= maxk
            @inbounds for i in oneto(nwrd)
                word = bagset[i]

                while !iszero(word)
                    push!(captgt, bt_vtx(V, i, word)); word &= word - one(UInt64)
                end
            end
        end
    elseif isone(nnon) && (isone(nful) || root && twice(flen) > nvtx - nbag)
        # the weight of pf, and of its lightest vertex
        fwgt = zero(W); fone = typemax(W)

        @inbounds for i in oneto(nwrd)
            word = pool[pf + i]

            while !iszero(word)
                u = bt_vtx(V, i, word); word &= word - one(UInt64)
                fwgt += weights[u]; fone = min(fone, weights[u])
            end
        end

        k = max(wbag + fone, mful)

        if k <= maxk
            k = sepdp_cap!(captgt, pool, bagset, adj, weights, blkset, sepset,
                fst, trie, blklen, valk, maxk, nbag, wbag, pf, flen, fmin, fwgt, mful, root)
        end
    else
        k = typemax(W)
    end

    resize!(pool, top)
    return k
end

# With fewer blocks smaller than the scope, the scan is linear.
const SEPDP_TRIE = 256

# The trie of the separators of the blocks of each vertex v, the
# blocks of fst whose component has smallest vertex v. Within the
# positions of v in fst, tord lists them by separator: at the first
# vertex where two separators differ, the one without it comes
# first. With crit[p] the first vertex where the separators at p
# and p + 1 differ, the trie is the Cartesian tree on crit, whose
# minimum over a range is unique: the internal node p, between p
# and p + 1, has children nodl[p] (up to p) and nodr[p] (after p),
# zero for a leaf, and head[v] is the root, zero if v has one block.
# Each node p stores, over the blocks below it, the size of the
# smallest component (nodn[p]), the weight of the lightest
# separator (nodw[p]), and the intersection and union of the
# separators (nodall[:, p] and nodany[:, p]).
function sepdp_tri(fst::BipartiteGraph{V}, blklen::AbstractVector{V}, sepset::AbstractMatrix{UInt64}, weights::AbstractVector{W}) where {W, V}
    nvtx = nv(fst); nwrd = size(sepset, 1); tgt = targets(fst); nblk = length(tgt)

    tord   = Vector{V}(undef, nblk)
    crit   = zeros(Int, nblk)
    nodl   = zeros(Int, nblk)
    nodr   = zeros(Int, nblk)
    nodn   = zeros(Int, nblk)
    nodw   = zeros(W, nblk)
    nodall = zeros(UInt64, nwrd, nblk)
    nodany = zeros(UInt64, nwrd, nblk)
    head   = zeros(Int, nvtx)
    stack  = Int[]

    @inbounds for p in oneto(nblk)
        tord[p] = tgt[p]
    end

    function less(x, y)
        @inbounds for i in oneto(nwrd)
            d = sepset[i, x] ⊻ sepset[i, y]
            iszero(d) || return iszero(sepset[i, x] & d & -d)
        end

        return false
    end

    @inbounds for v in oneto(nvtx)
        rng = incident(fst, v)
        isempty(rng) && continue
        sort!(view(tord, rng); lt = less)
        empty!(stack)

        for p in first(rng):last(rng) - 1
            x = tord[p]; y = tord[p + 1]

            for i in oneto(nwrd)
                d = sepset[i, x] ⊻ sepset[i, y]

                if !iszero(d)
                    crit[p] = 64(i - 1) + trailing_zeros(d) + 1
                    break
                end
            end

            # the Cartesian tree, with the smallest crit at the root
            q = 0

            while !isempty(stack) && crit[last(stack)] > crit[p]
                q = pop!(stack)
            end

            nodl[p] = q
            isempty(stack) || (nodr[last(stack)] = p)
            push!(stack, p)
        end

        head[v] = isempty(stack) ? 0 : first(stack)

        # the summaries, children first
        if ispositive(head[v])
            empty!(stack); push!(stack, head[v], first(rng), last(rng), 0)

            while !isempty(stack)
                done = pop!(stack); hi = pop!(stack); lo = pop!(stack); p = pop!(stack)

                if iszero(done)
                    push!(stack, p, lo, hi, 1)
                    ispositive(nodr[p]) && push!(stack, nodr[p], p + 1, hi, 0)
                    ispositive(nodl[p]) && push!(stack, nodl[p], lo, p, 0)
                else
                    l = ispositive(nodl[p]) ? nodn[nodl[p]] : convert(Int, blklen[tord[lo]])
                    r = ispositive(nodr[p]) ? nodn[nodr[p]] : convert(Int, blklen[tord[hi]])
                    nodn[p] = min(l, r)
                    wl = ispositive(nodl[p]) ? nodw[nodl[p]] : sepdp_wgt(sepset, tord[lo], weights)
                    wr = ispositive(nodr[p]) ? nodw[nodr[p]] : sepdp_wgt(sepset, tord[hi], weights)
                    nodw[p] = min(wl, wr)

                    for i in oneto(nwrd)
                        al = ispositive(nodl[p]) ? nodall[i, nodl[p]] : sepset[i, tord[lo]]
                        ar = ispositive(nodr[p]) ? nodall[i, nodr[p]] : sepset[i, tord[hi]]
                        bl = ispositive(nodl[p]) ? nodany[i, nodl[p]] : sepset[i, tord[lo]]
                        br = ispositive(nodr[p]) ? nodany[i, nodr[p]] : sepset[i, tord[hi]]
                        nodall[i, p] = al & ar; nodany[i, p] = bl | br
                    end
                end
            end
        end
    end

    empty!(stack)
    wmin = iszero(nvtx) ? zero(W) : minimum(view(weights, oneto(nvtx)))
    return (; tord, nodl, nodr, nodn, nodw, nodall, nodany, head, stack, wmin)
end

function sepdp_wgt(sepset::AbstractMatrix{UInt64}, x::Integer, weights::AbstractVector{W}) where {W}
    w = zero(W)

    @inbounds for i in axes(sepset, 1)
        word = sepset[i, x]

        while !iszero(word)
            w += weights[bt_vtx(Int, i, word)]; word &= word - one(UInt64)
        end
    end

    return w
end

# The block whose component has smallest vertex ymin and ylen
# vertices and whose separator is the set pn, or zero if there
# is none.
function sepdp_get(
        fst::BipartiteGraph{V},
        blklen::AbstractVector{V},
        sepset::AbstractMatrix{UInt64},
        pool::Vector{UInt64},
        pn::Int,
        ymin::Integer,
        ylen::Integer,
        nwrd::Int,
    ) where {V}
    tgt = targets(fst); rng = incident(fst, ymin)
    lo = first(rng); hi = last(rng) + one(eltype(rng))

    @inbounds while lo < hi
        p = lo + half(hi - lo)

        if blklen[tgt[p]] < ylen
            lo = p + one(p)
        else
            hi = p
        end
    end

    @inbounds for p in lo:last(rng)
        b = tgt[p]
        blklen[b] == ylen || break
        found = true

        for i in oneto(nwrd)
            if sepset[i, b] != pool[pn + i]
                found = false
                break
            end
        end

        found && return b
    end

    return zero(V)
end

# The block whose separator is marked insep[x] == c (with n
# vertices) and whose component has smallest vertex ymin and
# ylen vertices, or zero if there is none.
function sepdp_fnd(
        fst::BipartiteGraph{V},
        blklen::AbstractVector{V},
        sep::BipartiteGraph{V},
        blksep::AbstractVector{V},
        insep::AbstractVector{V},
        c::V,
        ymin::V,
        ylen::V,
        n::V,
    ) where {V}
    tgt = targets(fst); rng = incident(fst, ymin)
    lo = first(rng); hi = last(rng) + one(eltype(rng))

    @inbounds while lo < hi
        p = lo + half(hi - lo)

        if blklen[tgt[p]] < ylen
            lo = p + one(p)
        else
            hi = p
        end
    end

    @inbounds for p in lo:last(rng)
        b = tgt[p]
        blklen[b] == ylen || break
        s = blksep[b]

        if eltypedegree(sep, s) == n
            found = true

            for x in neighbors(sep, s)
                if insep[x] != c
                    found = false
                    break
                end
            end

            found && return b
        end
    end

    return zero(V)
end

function sepdp_ord!(
        perm::AbstractVector{V},
        invp::AbstractVector{V},
        stack::AbstractVector{V},
        queue::AbstractVector{V},
        insep::AbstractVector{V},
        tag::AbstractScalar{V},
        graph::AbstractGraph{V},
        sep::BipartiteGraph{V},
        fst::BipartiteGraph{V},
        blklen::AbstractVector{V},
        blksep::AbstractVector{V},
        cap::BipartiteGraph{V},
    ) where {V}
    nvtx = nv(graph); r = nv(cap)

    @assert nvtx <= length(perm)
    @assert nvtx <= length(invp)
    @assert nvtx < length(stack)
    @assert nvtx <= length(queue)
    @assert nvtx <= length(insep)

    @inbounds for v in vertices(graph)
        invp[v] = insep[v] = zero(V)
    end

    i = nvtx; t = one(V); stack[t] = r

    @inbounds while ispositive(t)
        b = stack[t]; t -= one(V); j = i; c = tag[] += one(V)

        for x in neighbors(cap, b)
            if !ispositive(invp[x])
                perm[i] = x; invp[x] = i; i -= one(V)
            end
        end

        # the components of D - Ω: for the root, the
        # components of G - Ω; otherwise, the components
        # next to the vertices just placed
        if b == r
            for v in vertices(graph)
                t = sepdp_chd!(stack, t, queue, insep, tag, invp,
                    graph, sep, fst, blklen, blksep, c, v)
            end
        else
            for k in i + one(V):j, v in neighbors(graph, perm[k])
                t = sepdp_chd!(stack, t, queue, insep, tag, invp,
                    graph, sep, fst, blklen, blksep, c, v)
            end
        end
    end

    return
end

# If the vertex v has neither been placed (invp[v] > 0) nor
# visited in round c (invp[v] = -c), then push the block of its
# component onto the stack.
function sepdp_chd!(
        stack::AbstractVector{V},
        t::V,
        queue::AbstractVector{V},
        insep::AbstractVector{V},
        tag::AbstractScalar{V},
        invp::AbstractVector{V},
        graph::AbstractGraph{V},
        sep::BipartiteGraph{V},
        fst::BipartiteGraph{V},
        blklen::AbstractVector{V},
        blksep::AbstractVector{V},
        c::V,
        v::V,
    ) where {V}
    @inbounds if !ispositive(invp[v]) && invp[v] != -c
        d = tag[] += one(V)
        h = one(V)
        q = one(V)
        n = zero(V)
        ymin = v

        invp[v] = -c
        queue[q] = v

        while h <= q
            u = queue[h]; h += one(V); ymin = min(ymin, u)

            for y in neighbors(graph, u)
                if ispositive(invp[y])
                    if insep[y] != d
                        insep[y] = d; n += one(V)
                    end
                elseif invp[y] != -c
                    invp[y] = -c; q += one(V); queue[q] = y
                end
            end
        end

        t += one(V); stack[t] = sepdp_fnd(fst, blklen, sep, blksep, insep, d, ymin, q, n)
    end

    return t
end
