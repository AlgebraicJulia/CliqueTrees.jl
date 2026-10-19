#
# Heuristic upper bounds on treewidth by BT dynamic programming over sets
# of potential maximal cliques (Tamaki 2019, 2022).
#
# A *solution* is a set Π of potential maximal cliques (PMCs). Its value is
# the smallest refined width of a tree decomposition all of whose bags are
# PMCs in Π or sets N[C] of blocks C, computed by the Bouchitté-Todinca
# recurrence. The refined width of a decomposition is the pair (k, f), where
# k is the largest bag weight and f the number of bags of weight k, ordered
# lexicographically (Tamaki 2019, Section 2): lowering f is progress even
# when k stays put. (Bag weights are sums in the order of the vertices, so
# with floats two bags of the same weight count as one only if they are
# summed alike; this changes only which progress is seen.)
#
# The sets of the graph (PMCs, separators, components) are sorted lists of
# vertices, not packed sets: the graph can be large. A PMC X is stored with
# the components C₁, …, Cₜ of G - X. A *block* is a component C whose
# neighborhood N(C) is a minimal separator; it is identified by its smallest
# vertex and its separator, and stored once per graph (hbt_blk!). For each
# component Cⱼ, a PMC stores its block and the block of its superblock: the
# full component of N(Cⱼ) that contains X - N(Cⱼ). X is a cap of each
# superblock. The superblock of Cⱼ consists of X - N(Cⱼ) and the components
# Cᵢ with N(Cᵢ) ⊈ N(Cⱼ), the *inner* components of j.
#
#   - Tamaki, Hisao. "A heuristic use of dynamic programming to upperbound
#     treewidth." arXiv:1909.07647 (2019).
#   - Tamaki, Hisao. "Heuristic computation of exact treewidth."
#     arXiv:2202.07793 (2022).
#

# ---------------------------------------------------------------------------
# Tables of sequences of integers
# ---------------------------------------------------------------------------

# The sequence i is tgt[ptr[i]:ptr[i + 1] - 1]; slot is an open-addressing
# table of the indices, by hash.
function bt_tab()
    return (; ptr = Int[1], tgt = Int[], slot = zeros(Int, 16))
end

bt_tab_len(tab::NamedTuple) = length(tab.ptr) - 1
bt_tab_seq(tab::NamedTuple, i::Integer) = view(tab.tgt, tab.ptr[i]:tab.ptr[i + 1] - 1)

# FNV over the integers, then the finalizer of MurmurHash3 (fmix64), so
# that every bit of the integers reaches the low bits that pick a slot:
# multiplication carries bits only upward, and the integers can be words
# of packed sets, whose high bits are vertices too.
function bt_tab_hsh(x::AbstractVector{Int})
    h = 0xcbf29ce484222325

    @inbounds for v in x
        h = (h ⊻ (v % UInt64)) * 0x00000100000001b3
    end

    h ⊻= h >> 33; h *= 0xff51afd7ed558ccd
    h ⊻= h >> 33; h *= 0xc4ceb9fe1a85ec53
    return h ⊻ (h >> 33)
end

function bt_tab_eq(tab::NamedTuple, i::Int, x::AbstractVector{Int})
    lo = tab.ptr[i]; hi = tab.ptr[i + 1]
    hi - lo == length(x) || return false

    @inbounds for k in eachindex(x)
        tab.tgt[lo + k - 1] == x[k] || return false
    end

    return true
end

# The index of the sequence x (with hash h), or zero.
function bt_tab_find(tab::NamedTuple, x::AbstractVector{Int}, h::UInt64 = bt_tab_hsh(x))
    slot = tab.slot; m = length(slot) - 1; s = (h % Int) & m

    @inbounds while true
        i = slot[s + 1]
        iszero(i) && return 0
        bt_tab_eq(tab, i, x) && return i
        s = (s + 1) & m
    end
end

# The index of the sequence x (with hash h), added if it is new, and whether it is.
function bt_tab_add!(tab::NamedTuple, x::AbstractVector{Int}, h::UInt64 = bt_tab_hsh(x))
    i = bt_tab_find(tab, x, h)
    ispositive(i) && return i, false
    append!(tab.tgt, x); push!(tab.ptr, length(tab.tgt) + 1)
    i = bt_tab_len(tab)

    if 2i > length(tab.slot)
        bt_tab_rhs!(tab, 2length(tab.slot))
    else
        bt_tab_slt!(tab, i, h)
    end

    return i, true
end

function bt_tab_slt!(tab::NamedTuple, i::Int, h::UInt64 = bt_tab_hsh(bt_tab_seq(tab, i)))
    slot = tab.slot; m = length(slot) - 1; s = (h % Int) & m

    @inbounds while ispositive(slot[s + 1])
        s = (s + 1) & m
    end

    @inbounds slot[s + 1] = i
    return
end

# Rebuild the slots, with nslot of them (a power of two).
function bt_tab_rhs!(tab::NamedTuple, nslot::Int)
    resize!(tab.slot, nslot); fill!(tab.slot, 0)

    for i in oneto(bt_tab_len(tab))
        bt_tab_slt!(tab, i)
    end

    return
end

# ---------------------------------------------------------------------------
# Refined widths
# ---------------------------------------------------------------------------

hbt_bag(k::W) where {W} = (k, 1)
hbt_inf(::Type{W}) where {W} = (typemax(W), typemax(Int))

function hbt_plus(a::Tuple{W, Int}, b::Tuple{W, Int}) where {W}
    return a[1] == b[1] ? (a[1], a[2] + b[2]) : max(a, b)
end

# ---------------------------------------------------------------------------
# The graph: its separators and blocks, and scratch space
# ---------------------------------------------------------------------------

# A simple graph with sorted neighborhoods, from any graph, with the weights
# of its vertices.
function hbt_graph(weights::AbstractVector, graph::AbstractGraph)
    n = nv(graph); W = eltype(weights)
    ptr = Vector{Int}(undef, n + 1); tgt = Int[]; wgt = Vector{W}(undef, n)
    ptr[1] = 1

    @inbounds for v in oneto(n)
        wgt[v] = weights[v]; lo = length(tgt) + 1

        for x in neighbors(graph, v)
            x != v && push!(tgt, x)
        end

        sort!(view(tgt, lo:length(tgt)))
        q = lo

        for i in lo:length(tgt)
            if q == lo || tgt[i] != tgt[q - 1]
                tgt[q] = tgt[i]; q += 1
            end
        end

        resize!(tgt, q - 1); ptr[v + 1] = q
    end

    return BipartiteGraph{Int, Int}(n, n, length(tgt), ptr, tgt), wgt
end

function hbt_ctx(graph::BipartiteGraph{Int, Int}, wgt::Vector{W}) where {W}
    n = nv(graph)

    return (;
        graph, wgt,
        seps = bt_tab(), swgt = W[],            # separators and their weights
        blks = bt_tab(),                        # blocks, by (smallest vertex, separator)
        bmin = Int[], bsep = Int[], bsize = Int[], bwgt = W[],
        stamp = Ref(0),                         # mark*[v] == stamp: marked
        mark1 = zeros(Int, n), mark2 = zeros(Int, n), mark3 = zeros(Int, n),
        loc = zeros(Int, n), queue = Int[], buf = Int[],

        # the dynamic program, by block
        bstamp = Int[], bloc = Int[], bwid = Tuple{W, Int}[], bcapp = Int[], bcapj = Int[],
        active = Int[], cnt = Int[], sorted = Int[], cptr = Int[],
        capb = Int[], capp = Int[], capj = Int[], capp2 = Int[], capj2 = Int[],

        # the groups of the components of a PMC (hbt_add!)
        gdst = Int[], gid = Int[], gsmk = Int[], gsof = Int[], gin = Bool[], gptr = Int[], gtgt = Int[],
        gm = Int[], gsz = Int[], gw = W[],

        # the graphs H + S of a diversification (hbt_div), and the relabeled
        # graphs of the greedy triangulations (hbt_grd)
        hsp = Int[], hst = Int[], rprm = Int[], rinv = Int[], rptr = Int[], rtgt = Int[], rwgt = W[],
    )
end

hbt_stp!(ctx::NamedTuple) = ctx.stamp[] += 1

function hbt_sep!(ctx::NamedTuple, sep::AbstractVector{Int})
    s, new = bt_tab_add!(ctx.seps, sep)

    if new
        w = zero(eltype(ctx.wgt))

        @inbounds for v in sep
            w += ctx.wgt[v]
        end

        push!(ctx.swgt, w)
    end

    return s
end

function hbt_blk!(ctx::NamedTuple, m::Int, s::Int, size::Int, wgt)
    key = ctx.buf; empty!(key); push!(key, m, s)
    b, new = bt_tab_add!(ctx.blks, key)

    if new
        push!(ctx.bmin, m); push!(ctx.bsep, s); push!(ctx.bsize, size); push!(ctx.bwgt, wgt)
    end

    return b
end

# ---------------------------------------------------------------------------
# Solutions: sets of PMCs
# ---------------------------------------------------------------------------

# The PMC p is bt_tab_seq(sol.pmcs, p), with weight pwgt[p] and components
# cfst[p]:cfst[p + 1] - 1; the component c has the block csub[c], the
# superblock csup[c], and the inner components iptr[c]:iptr[c + 1] - 1 of
# itgt (as offsets: component cfst[p] + itgt[i] - 1).
function hbt_sol(::Type{W}) where {W}
    return (;
        pmcs = bt_tab(), pwgt = W[],
        cfst = Int[1], csub = Int[], csup = Int[], iptr = Int[1], itgt = Int[],
        root = Tuple{W, Int}[],                 # the root value of each PMC (hbt_dp!)
    )
end

hbt_len(sol::NamedTuple) = bt_tab_len(sol.pmcs)
hbt_pmc(sol::NamedTuple, p::Integer) = bt_tab_seq(sol.pmcs, p)

# Add the PMC X (sorted) to the solution, unless it is there: its
# components have smallest vertices cmin, sizes csize, weights cwgt, and
# separators csep. Returns its index.
function hbt_add!(ctx::NamedTuple, sol::NamedTuple, X::AbstractVector{Int}, cmin::AbstractVector{Int}, csize::AbstractVector{Int}, cwgt::AbstractVector, csep::AbstractVector{Int})
    p, new = bt_tab_add!(sol.pmcs, X)
    new || return p
    mark = ctx.mark3; wgt = ctx.wgt; seps = ctx.seps; t = length(cmin)
    W = eltype(wgt); xw = zero(W)

    @inbounds for x in X
        xw += wgt[x]
    end

    push!(sol.pwgt, xw)

    # Components with the same separator have the same inner components (those
    # whose separator is not inside it, the component's own separator aside)
    # and the same superblock, so these are found once per separator: in time
    # about d^2 |S| + d t for d distinct separators, not t^2 |S|. (A PMC can
    # have thousands of components, most with one of a few separators.) The
    # lists and sums are made in the same order as one component at a time.
    # (The arrays are kept in the context: the group of the separator s is
    # gsof[s] if gsmk[s] is the stamp sg.)
    dist = ctx.gdst; gid = ctx.gid; gsmk = ctx.gsmk; gsof = ctx.gsof
    empty!(dist); resize!(gid, t); sg = hbt_stp!(ctx)

    if length(gsmk) < bt_tab_len(seps)
        m = length(gsmk); resize!(gsmk, bt_tab_len(seps)); resize!(gsof, bt_tab_len(seps))
        fill!(view(gsmk, m + 1:length(gsmk)), 0)
    end

    @inbounds for j in oneto(t)
        c = csep[j]

        if gsmk[c] != sg
            gsmk[c] = sg; push!(dist, c); gsof[c] = length(dist)
        end

        gid[j] = gsof[c]
    end

    d = length(dist); isin = resize!(ctx.gin, d)
    gptr = resize!(ctx.gptr, d + 1); gtgt = empty!(ctx.gtgt)
    gm = resize!(ctx.gm, d); gsz = resize!(ctx.gsz, d); gw = resize!(ctx.gw, d)

    @inbounds for g in oneto(d)
        s = hbt_stp!(ctx)

        for v in bt_tab_seq(seps, dist[g])
            mark[v] = s
        end

        m = typemax(Int); sz = 0; w = zero(W)

        for x in X
            if mark[x] != s
                m = min(m, x); sz += 1; w += wgt[x]
            end
        end

        for h in oneto(d)
            isin[h] = false
            h == g && continue

            for v in bt_tab_seq(seps, dist[h])
                if mark[v] != s
                    isin[h] = true
                    break
                end
            end
        end

        gptr[g] = length(gtgt) + 1

        for i in oneto(t)
            if isin[gid[i]]
                push!(gtgt, i)
                m = min(m, cmin[i]); sz += csize[i]; w += cwgt[i]
            end
        end

        gm[g] = m; gsz[g] = sz; gw[g] = w
    end

    gptr[d + 1] = length(gtgt) + 1

    @inbounds for j in oneto(t)
        g = gid[j]

        for i in gptr[g]:gptr[g + 1] - 1
            push!(sol.itgt, gtgt[i])
        end

        push!(sol.iptr, length(sol.itgt) + 1)
        push!(sol.csub, hbt_blk!(ctx, cmin[j], csep[j], csize[j], cwgt[j]))
        push!(sol.csup, hbt_blk!(ctx, gm[g], csep[j], gsz[g], gw[g]))
    end

    push!(sol.cfst, length(sol.csub) + 1)
    return p
end

function hbt_add_ref!(ctx::NamedTuple, sol::NamedTuple, X::AbstractVector{Int}, cmin::AbstractVector{Int}, csize::AbstractVector{Int}, cwgt::AbstractVector, csep::AbstractVector{Int})
    p, new = bt_tab_add!(sol.pmcs, X)
    new || return p
    mark = ctx.mark3; wgt = ctx.wgt; seps = ctx.seps; t = length(cmin)
    W = eltype(wgt); xw = zero(W)

    @inbounds for x in X
        xw += wgt[x]
    end

    push!(sol.pwgt, xw)

    @inbounds for j in oneto(t)
        s = hbt_stp!(ctx)

        for v in bt_tab_seq(seps, csep[j])
            mark[v] = s
        end

        m = typemax(Int); sz = 0; w = zero(W)

        for x in X
            if mark[x] != s
                m = min(m, x); sz += 1; w += wgt[x]
            end
        end

        for i in oneto(t)
            i == j && continue
            csep[i] == csep[j] && continue
            inner = false

            for v in bt_tab_seq(seps, csep[i])
                if mark[v] != s
                    inner = true
                    break
                end
            end

            if inner
                push!(sol.itgt, i)
                m = min(m, cmin[i]); sz += csize[i]; w += cwgt[i]
            end
        end

        push!(sol.iptr, length(sol.itgt) + 1)
        push!(sol.csub, hbt_blk!(ctx, cmin[j], csep[j], csize[j], cwgt[j]))
        push!(sol.csup, hbt_blk!(ctx, m, csep[j], sz, w))
    end

    push!(sol.cfst, length(sol.csub) + 1)
    return p
end

# Add the PMC X (sorted) of the graph, finding its components by search.
function hbt_add!(ctx::NamedTuple, sol::NamedTuple, X::AbstractVector{Int})
    p = bt_tab_find(sol.pmcs, X)
    ispositive(p) && return p
    g = ctx.graph; n = nv(g); wgt = ctx.wgt; W = eltype(wgt)
    mark1 = ctx.mark1; mark2 = ctx.mark2; mark3 = ctx.mark3; queue = ctx.queue
    cmin = Int[]; csize = Int[]; cwgt = W[]; csep = Int[]; sep = Int[]
    sx = hbt_stp!(ctx)

    @inbounds for x in X
        mark1[x] = sx
    end

    sc = hbt_stp!(ctx)

    @inbounds for v0 in oneto(n)
        (mark1[v0] == sx || mark2[v0] == sc) && continue
        sn = hbt_stp!(ctx); empty!(sep); empty!(queue)
        push!(queue, v0); mark2[v0] = sc; sz = 0; w = zero(W); head = 1

        while head <= length(queue)
            u = queue[head]; head += 1; sz += 1; w += wgt[u]

            for x in neighbors(g, u)
                if mark1[x] == sx
                    if mark3[x] != sn
                        mark3[x] = sn; push!(sep, x)
                    end
                elseif mark2[x] != sc
                    mark2[x] = sc; push!(queue, x)
                end
            end
        end

        sort!(sep; alg = QuickSort)
        push!(cmin, v0); push!(csize, sz); push!(cwgt, w); push!(csep, hbt_sep!(ctx, sep))
    end

    return hbt_add!(ctx, sol, X, cmin, csize, cwgt, csep)
end

# Copy the PMC p of the solution src into sol, unless it is there. (The
# blocks belong to the graph, so the structure is copied as it is.)
function hbt_cpy!(sol::NamedTuple, src::NamedTuple, p::Int)
    q, new = bt_tab_add!(sol.pmcs, hbt_pmc(src, p))
    new || return q
    push!(sol.pwgt, src.pwgt[p])

    @inbounds for c in src.cfst[p]:src.cfst[p + 1] - 1
        push!(sol.csub, src.csub[c]); push!(sol.csup, src.csup[c])

        for i in src.iptr[c]:src.iptr[c + 1] - 1
            push!(sol.itgt, src.itgt[i])
        end

        push!(sol.iptr, length(sol.itgt) + 1)
    end

    push!(sol.cfst, length(sol.csub) + 1)
    return q
end

# Keep the PMCs p with keep(p), in order, and rebuild the table.
function hbt_flt!(sol::NamedTuple, keep::F) where {F}
    tab = sol.pmcs; np = hbt_len(sol); root = length(sol.root) == np
    nt = 0; nc = 0; ni = 0; k = 0

    @inbounds for p in oneto(np)
        keep(p) || continue
        k += 1; lo = tab.ptr[p]; hi = tab.ptr[p + 1] - 1

        for i in lo:hi
            nt += 1; tab.tgt[nt] = tab.tgt[i]
        end

        tab.ptr[k + 1] = nt + 1
        sol.pwgt[k] = sol.pwgt[p]; root && (sol.root[k] = sol.root[p])

        for c in sol.cfst[p]:sol.cfst[p + 1] - 1
            nc += 1; sol.csub[nc] = sol.csub[c]; sol.csup[nc] = sol.csup[c]

            for i in sol.iptr[c]:sol.iptr[c + 1] - 1
                ni += 1; sol.itgt[ni] = sol.itgt[i]
            end

            sol.iptr[nc + 1] = ni + 1
        end

        sol.cfst[k + 1] = nc + 1
    end

    resize!(tab.tgt, nt); resize!(tab.ptr, k + 1); resize!(sol.pwgt, k)
    resize!(sol.root, root ? k : 0)
    resize!(sol.csub, nc); resize!(sol.csup, nc); resize!(sol.iptr, nc + 1); resize!(sol.itgt, ni)
    resize!(sol.cfst, k + 1)
    bt_tab_rhs!(tab, max(16, nextpow(2, 2k + 1)))
    return sol
end

# ---------------------------------------------------------------------------
# The dynamic program
# ---------------------------------------------------------------------------

# Evaluate the solution: the width of every block and the root value of
# every PMC (in sol.root). Returns (width, root). The block data stay valid
# until the next call.
function hbt_dp!(ctx::NamedTuple, sol::NamedTuple)
    W = eltype(ctx.wgt); np = hbt_len(sol); nb = length(ctx.bmin)
    bstamp = ctx.bstamp; bloc = ctx.bloc; active = ctx.active; bwid = ctx.bwid

    if length(bstamp) < nb
        m = max(nb, 2length(bstamp))
        resize!(bstamp, m); fill!(bstamp, 0)
        resize!(bloc, m); resize!(bwid, m); resize!(ctx.bcapp, m); resize!(ctx.bcapj, m)
    end

    s = hbt_stp!(ctx)
    capb = ctx.capb; capp = ctx.capp; capj = ctx.capj
    empty!(active); empty!(capb); empty!(capp); empty!(capj)

    # the blocks of the solution, and the caps of each superblock
    @inbounds for p in oneto(np), c in sol.cfst[p]:sol.cfst[p + 1] - 1
        b = sol.csub[c]

        if bstamp[b] != s
            bstamp[b] = s; push!(active, b)
        end

        b = sol.csup[c]

        if bstamp[b] != s
            bstamp[b] = s; push!(active, b)
        end

        push!(capb, b); push!(capp, p); push!(capj, c)
    end

    # the blocks in order of size (a stable counting sort: sizes are at most n)
    bsize = ctx.bsize; na = length(active); n = nv(ctx.graph)
    cnt = ctx.cnt; resize!(cnt, n + 1); fill!(cnt, 0)
    sorted = ctx.sorted; resize!(sorted, na)

    @inbounds for b in active
        cnt[bsize[b] + 1] += 1
    end

    @inbounds for i in 2:n + 1
        cnt[i] += cnt[i - 1]
    end

    @inbounds for b in active
        sz = bsize[b]; cnt[sz] += 1; sorted[cnt[sz]] = b
    end

    copyto!(active, sorted)

    @inbounds for (i, b) in enumerate(active)
        bloc[b] = i
    end

    # the caps, by superblock
    cptr = ctx.cptr; resize!(cptr, na + 2); fill!(cptr, 0); nc = length(capb)
    capp2 = ctx.capp2; capj2 = ctx.capj2; resize!(capp2, nc); resize!(capj2, nc)

    @inbounds for b in capb
        cptr[bloc[b] + 2] += 1
    end

    cptr[1] = 1; cptr[2] = 1

    @inbounds for i in 2:na + 1
        cptr[i + 1] += cptr[i]
    end

    @inbounds for c in oneto(nc)
        i = bloc[capb[c]] + 1; k = cptr[i]; cptr[i] += 1
        capp2[k] = capp[c]; capj2[k] = capj[c]
    end

    # cptr[i]:cptr[i + 1] - 1 are the caps of active[i]; the bag N[C] is
    # always one
    @inbounds for (i, b) in enumerate(active)
        w = hbt_bag(ctx.bwgt[b] + ctx.swgt[ctx.bsep[b]]); bp = 0; bj = 0

        for k in cptr[i]:cptr[i + 1] - 1
            p = capp2[k]; c = capj2[k]; f = sol.cfst[p] - 1
            v = hbt_bag(sol.pwgt[p])
            v < w || continue

            for q in sol.iptr[c]:sol.iptr[c + 1] - 1
                v = hbt_plus(v, bwid[sol.csub[f + sol.itgt[q]]])
                v < w || break
            end

            if v < w
                w = v; bp = p; bj = c
            end
        end

        bwid[b] = w; ctx.bcapp[b] = bp; ctx.bcapj[b] = bj
    end

    # the root values
    root = sol.root; resize!(root, np)
    width = hbt_inf(W); best = 0

    @inbounds for p in oneto(np)
        v = hbt_bag(sol.pwgt[p])

        for c in sol.cfst[p]:sol.cfst[p + 1] - 1
            v = hbt_plus(v, bwid[sol.csub[c]])
        end

        root[p] = v

        if v < width
            width = v; best = p
        end
    end

    return width, best
end

# The tree decomposition of the dynamic program from the PMC root: its bags
# (bags[i] is bt_tab_seq-like: btgt[bptr[i]:bptr[i + 1] - 1], sorted), parents
# first, and the parent of each bag (zero for the root).
function hbt_td(ctx::NamedTuple, sol::NamedTuple, root::Int)
    g = ctx.graph; mark1 = ctx.mark1; mark2 = ctx.mark2; queue = ctx.queue
    bptr = Int[1]; btgt = Int[]; parent = Int[]; stack = Tuple{Int, Int}[]

    append!(btgt, hbt_pmc(sol, root)); push!(bptr, length(btgt) + 1); push!(parent, 0)

    @inbounds for c in sol.cfst[root]:sol.cfst[root + 1] - 1
        push!(stack, (sol.csub[c], 1))
    end

    @inbounds while !isempty(stack)
        b, pn = pop!(stack); p = ctx.bcapp[b]

        if iszero(p)
            # no cap: the bag N[C]
            s = hbt_stp!(ctx); sep = bt_tab_seq(ctx.seps, ctx.bsep[b])

            for v in sep
                mark1[v] = s
            end

            v0 = ctx.bmin[b]; empty!(queue); push!(queue, v0); mark2[v0] = s; head = 1

            while head <= length(queue)
                u = queue[head]; head += 1

                for x in neighbors(g, u)
                    if mark1[x] != s && mark2[x] != s
                        mark2[x] = s; push!(queue, x)
                    end
                end
            end

            lo = length(btgt) + 1; append!(btgt, queue); append!(btgt, sep)
            sort!(view(btgt, lo:length(btgt)); alg = QuickSort); push!(bptr, length(btgt) + 1); push!(parent, pn)
        else
            append!(btgt, hbt_pmc(sol, p)); push!(bptr, length(btgt) + 1); push!(parent, pn)
            node = length(parent); c = ctx.bcapj[b]; f = sol.cfst[p] - 1

            for q in sol.iptr[c]:sol.iptr[c + 1] - 1
                push!(stack, (sol.csub[f + sol.itgt[q]], node))
            end
        end
    end

    return bptr, btgt, parent
end

# An elimination order of the width of a tree decomposition: the bags
# children first, each eliminating its vertices that are not in its parent.
function hbt_ord(ctx::NamedTuple, bptr::Vector{Int}, btgt::Vector{Int}, parent::Vector{Int})
    n = nv(ctx.graph); mark = ctx.mark1; order = Int[]; sizehint!(order, n)

    @inbounds for i in length(parent):-1:1
        p = parent[i]; bag = view(btgt, bptr[i]:bptr[i + 1] - 1)

        if iszero(p)
            append!(order, bag)
        else
            s = hbt_stp!(ctx)

            for v in view(btgt, bptr[p]:bptr[p + 1] - 1)
                mark[v] = s
            end

            for v in bag
                mark[v] == s || push!(order, v)
            end
        end
    end

    @assert length(order) == n
    return order
end

# ---------------------------------------------------------------------------
# Local graphs
# ---------------------------------------------------------------------------

# The local graph on U (sorted): G[U], with N(B) made a clique for every
# component B of G - U, its vertices numbered by position in U. The
# components B have smallest vertices bmin (in G), sizes bsize, weights
# bwgt, and neighborhoods nbrs[nidx[b]] (local and sorted, in a table:
# components with the same neighborhood share it). Local graphs are usually
# dense, as the neighborhoods are separators of the size of a bag, so a
# small and dense one is also kept as packed sets (adj, with nwrd words a
# set), whose components are found with word operations; otherwise nwrd
# is zero.
const HBT_BITS = 1024

function hbt_loc(ctx::NamedTuple, U::AbstractVector{Int})
    g = ctx.graph; n = nv(g); nu = length(U); wgt = ctx.wgt; W = eltype(wgt)
    mark1 = ctx.mark1; mark2 = ctx.mark2; mark3 = ctx.mark3; loc = ctx.loc; queue = ctx.queue
    su = hbt_stp!(ctx)

    @inbounds for i in oneto(nu)
        mark1[U[i]] = su; loc[U[i]] = i
    end

    # the components of G - U and their neighborhoods
    bmin = Int[]; bsize = Int[]; bwgt = W[]; nidx = Int[]; nbrs = bt_tab(); nb = Int[]
    sc = hbt_stp!(ctx)

    @inbounds for v0 in oneto(n)
        (mark1[v0] == su || mark2[v0] == sc) && continue
        sn = hbt_stp!(ctx); empty!(nb); empty!(queue)
        push!(queue, v0); mark2[v0] = sc; sz = 0; w = zero(W); head = 1

        while head <= length(queue)
            u = queue[head]; head += 1; sz += 1; w += wgt[u]

            for x in neighbors(g, u)
                if mark1[x] == su
                    if mark3[x] != sn
                        mark3[x] = sn; push!(nb, loc[x])
                    end
                elseif mark2[x] != sc
                    mark2[x] = sc; push!(queue, x)
                end
            end
        end

        sort!(nb; alg = QuickSort)
        push!(bmin, v0); push!(bsize, sz); push!(bwgt, w); push!(nidx, first(bt_tab_add!(nbrs, nb)))
    end

    # the edges: G[U] and the cliques
    if nu <= HBT_BITS
        nwrd = cld(nu, 64); adj = bt_mat(nwrd, nu); fill!(adj, zero(UInt64))

        @inbounds for i in oneto(nu), x in neighbors(g, U[i])
            if mark1[x] == su
                j = loc[x]; bt_put!(adj, bt_get(adj, bt_wrd(j), i) | bt_bit(j), bt_wrd(j), i)
            end
        end

        # each clique as a row, ORed into the sets of its vertices (then
        # without the diagonal)
        row = zeros(UInt64, nwrd)

        @inbounds for c in oneto(bt_tab_len(nbrs))
            clq = bt_tab_seq(nbrs, c); fill!(row, zero(UInt64))

            for a in clq
                row[bt_wrd(a)] |= bt_bit(a)
            end

            for a in clq, k in oneto(nwrd)
                bt_put!(adj, bt_get(adj, k, a) | row[k], k, a)
            end
        end

        @inbounds for i in oneto(nu)
            bt_put!(adj, bt_get(adj, bt_wrd(i), i) & ~bt_bit(i), bt_wrd(i), i)
        end

        ptr = Vector{Int}(undef, nu + 1); ptr[1] = 1; m = 0

        @inbounds for i in oneto(nu), k in oneto(nwrd)
            m += count_ones(bt_get(adj, k, i))
        end

        tgt = Vector{Int}(undef, m); m = 0

        @inbounds for i in oneto(nu)
            for k in oneto(nwrd)
                word = bt_get(adj, k, i)

                while !iszero(word)
                    m += 1; tgt[m] = bt_vtx(Int, k, word); word &= word - one(UInt64)
                end
            end

            ptr[i + 1] = m + 1
        end

        # keep the packed sets only if a search over them is no slower than
        # over the lists: nwrd words a vertex against its degree
        if nwrd * nu > length(tgt)
            nwrd = 0; adj = bt_mat(1, 0)
        end
    else
        # the arcs, then sorted and without repeats, a vertex at a time
        src = Int[]; dst = Int[]

        @inbounds for i in oneto(nu), x in neighbors(g, U[i])
            mark1[x] == su && (push!(src, i); push!(dst, loc[x]))
        end

        @inbounds for c in oneto(bt_tab_len(nbrs))
            clq = bt_tab_seq(nbrs, c)

            for a in clq, b in clq
                a != b && (push!(src, a); push!(dst, b))
            end
        end

        ptr = zeros(Int, nu + 1); tgt = Vector{Int}(undef, length(src))

        @inbounds for a in src
            ptr[a + 1] += 1
        end

        ptr[1] = 1

        @inbounds for i in oneto(nu)
            ptr[i + 1] += ptr[i]
        end

        pos = ptr[1:nu]

        @inbounds for k in eachindex(src)
            a = src[k]; tgt[pos[a]] = dst[k]; pos[a] += 1
        end

        q = 1

        @inbounds for i in oneto(nu)
            lo = ptr[i]; hi = ptr[i + 1] - 1; sort!(view(tgt, lo:hi); alg = QuickSort); ptr[i] = q

            for k in lo:hi
                if k == lo || tgt[k] != tgt[k - 1]
                    tgt[q] = tgt[k]; q += 1
                end
            end
        end

        ptr[nu + 1] = q; resize!(tgt, q - 1)
        nwrd = 0; adj = bt_mat(1, 0)
    end

    graph = BipartiteGraph{Int, Int}(nu, nu, length(tgt), ptr, tgt)

    return (;
        U, graph, wgt = wgt[U], bmin, bsize, bwgt, nbrs, nidx, nwrd, adj,
        pool = UInt64[], lmark = zeros(Int, nu), lcomp = zeros(Int, nu),
        cmin = Int[], csize = Int[], cwgt = W[], qptr = Int[1], qtgt = Int[], csep = Int[],
        X = Int[], sep = Int[], queue = Int[],

        # the components of the cliques of its triangulations, if they are
        # found for all the cliques at once (see hbt_lfc!)
        fast = hbt_isfast(nwrd, wgt, U, bwgt), fcl = Int[], fptr = Int[1], fmin = Int[], fsize = Int[], fwgt = W[], fsptr = Int[1], fstgt = Int[],
    )
end

# Is the set K (local, sorted) the neighborhood of a component of G - U?
hbt_isnbr(L::NamedTuple, K::AbstractVector{Int}) = ispositive(bt_tab_find(L.nbrs, K))

# Add the PMC U[K] of G, for a PMC K (sorted) of the local graph, to the
# solution, unless it is there; returns its index. The components of
# G - U[K] are the components Q of H - K, each joined with the components
# B of G - U with ∅ ≠ N(B) - K ⊆ Q, and the components B of G - U with
# N(B) ⊆ K. The separator of the former is N_H(Q) ∩ K = N_H(Q). Linear in
# the size of the local graph, unless the components were found for all the
# cliques of its triangulation at once (K is the c-th, see hbt_cls!).
function hbt_lft!(ctx::NamedTuple, sol::NamedTuple, L::NamedTuple, c::Int, K::AbstractVector{Int})
    U = L.U; X = L.X; empty!(X)

    @inbounds for k in K
        push!(X, U[k])
    end

    p = bt_tab_find(sol.pmcs, X)
    ispositive(p) && return p

    lmark = L.lmark; lcomp = L.lcomp; cmin = L.cmin; csize = L.csize; cwgt = L.cwgt
    qptr = L.qptr; qtgt = L.qtgt; csep = L.csep
    empty!(cmin); empty!(csize); empty!(cwgt); resize!(qptr, 1); empty!(qtgt); empty!(csep)

    if L.fast && ispositive(L.fcl[c])
        # found for all the cliques at once (hbt_lfc!)
        f = L.fcl[c]

        @inbounds for i in L.fptr[f]:L.fptr[f + 1] - 1
            push!(cmin, L.fmin[i]); push!(csize, L.fsize[i]); push!(cwgt, L.fwgt[i])
            append!(qtgt, view(L.fstgt, L.fsptr[i]:L.fsptr[i + 1] - 1)); push!(qptr, length(qtgt) + 1)
        end
    else
        sk = hbt_stp!(ctx)

        @inbounds for k in K
            lmark[k] = sk
        end

        if ispositive(L.nwrd)
            hbt_cpb!(ctx, L, K)
        else
            hbt_cpl!(ctx, L, sk)
        end

        # the components of G - U
        @inbounds for b in eachindex(L.bmin)
            nb = bt_tab_seq(L.nbrs, L.nidx[b]); q = 0

            for v in nb
                if lmark[v] != sk
                    q = lcomp[v]
                    break
                end
            end

            if iszero(q)
                push!(cmin, L.bmin[b]); push!(csize, L.bsize[b]); push!(cwgt, L.bwgt[b])
                append!(qtgt, nb); push!(qptr, length(qtgt) + 1)
            else
                cmin[q] = min(cmin[q], L.bmin[b]); csize[q] += L.bsize[b]; cwgt[q] += L.bwgt[b]
            end
        end
    end

    # the separators, in G (U is sorted, so they stay sorted)
    sep = L.sep

    @inbounds for q in eachindex(cmin)
        empty!(sep)

        for i in qptr[q]:qptr[q + 1] - 1
            push!(sep, U[qtgt[i]])
        end

        push!(csep, hbt_sep!(ctx, sep))
    end

    return hbt_add!(ctx, sol, X, cmin, csize, cwgt, csep)
end

# The components Q of H - K, by search over the lists: their smallest
# vertices (in G), sizes, weights, and separators N_H(Q) (local). Sets
# lcomp[v] to the component of every v outside K, which is marked sk.
function hbt_cpl!(ctx::NamedTuple, L::NamedTuple, sk::Int)
    H = L.graph; nu = nv(H); U = L.U; wgt = L.wgt; W = eltype(wgt)
    lmark = L.lmark; lcomp = L.lcomp; queue = L.queue; sep = L.sep; mark3 = ctx.mark3
    sc = hbt_stp!(ctx)

    @inbounds for v0 in oneto(nu)
        (lmark[v0] == sk || lmark[v0] == sc) && continue
        q = length(L.cmin) + 1; ss = hbt_stp!(ctx)
        empty!(queue); push!(queue, v0); lmark[v0] = sc; lcomp[v0] = q; head = 1
        empty!(sep); sz = 0; w = zero(W)

        while head <= length(queue)
            u = queue[head]; head += 1; sz += 1; w += wgt[u]

            for x in neighbors(H, u)
                if lmark[x] == sk
                    # (mark3 is indexed in G)
                    if mark3[U[x]] != ss
                        mark3[U[x]] = ss; push!(sep, x)
                    end
                elseif lmark[x] != sc
                    lmark[x] = sc; lcomp[x] = q; push!(queue, x)
                end
            end
        end

        sort!(sep; alg = QuickSort)
        push!(L.cmin, U[v0]); push!(L.csize, sz); push!(L.cwgt, w)
        append!(L.qtgt, sep); push!(L.qptr, length(L.qtgt) + 1)
    end

    return
end

# The same, by search over the packed sets.
function hbt_cpb!(ctx::NamedTuple, L::NamedTuple, K::AbstractVector{Int})
    U = L.U; nu = length(U); wgt = L.wgt; W = eltype(wgt); adj = L.adj; nwrd = L.nwrd
    pool = L.pool; empty!(pool); lcomp = L.lcomp
    pr = bt_new!(pool, nwrd)    # the vertices outside K in no component yet
    pc = bt_new!(pool, nwrd)    # a component
    pn = bt_new!(pool, nwrd)    # its neighborhood

    @inbounds for i in oneto(nwrd)
        pool[pr + i] = ~zero(UInt64)
    end

    bt_cap!(pool, nu, pr, nwrd)

    @inbounds for k in K
        pool[pr + bt_wrd(k)] &= ~bt_bit(k)
    end

    @inbounds while true
        v0 = bt_min(pool, pr, nwrd)
        iszero(v0) && break
        q = length(L.cmin) + 1
        bt_cmp!(pool, adj, pc, pr, v0, nwrd); sz = 0; w = zero(W)

        for i in oneto(nwrd)
            pool[pr + i] &= ~pool[pc + i]; pool[pn + i] = zero(UInt64)
            word = pool[pc + i]

            while !iszero(word)
                v = bt_vtx(Int, i, word); word &= word - one(UInt64)
                lcomp[v] = q; sz += 1; w += wgt[v]
            end
        end

        bt_nbr!(pool, adj, pn, pc, nwrd)

        for i in oneto(nwrd)
            word = pool[pn + i]

            while !iszero(word)
                push!(L.qtgt, bt_vtx(Int, i, word)); word &= word - one(UInt64)
            end
        end

        # U is sorted, so the smallest local vertex is the smallest in G
        push!(L.cmin, U[v0]); push!(L.csize, sz); push!(L.cwgt, w); push!(L.qptr, length(L.qtgt) + 1)
    end

    return
end

# ---------------------------------------------------------------------------
# The search: solutions and their best decompositions
# ---------------------------------------------------------------------------

# A solution with its best decomposition (bags bptr/btgt, parents first, with
# parents parent), an elimination order of it, the size of its largest bag,
# and the steps since its last improvement (stall) and since the last
# improvement of its largest bag weight (kstall).
function hbt_st(::Type{W}, depth::Int) where {W}
    return (;
        sol = hbt_sol(W), width = Ref(hbt_inf(W)),
        bptr = Int[1], btgt = Int[], parent = Int[], order = Int[], maxbag = Ref(0),
        depth, stall = Ref(0), kstall = Ref(0),
    )
end

# What a diversification reads from a solution.
hbt_snp(st::NamedTuple) = (; st.bptr, st.btgt, st.parent, width = st.width[], kstall = st.kstall[])

# The bag weight just below k: bags lighter than k weigh at most this.
hbt_below(k::Integer) = k - one(k)
hbt_below(k::AbstractFloat) = prevfloat(k)
hbt_below(k) = k - one(k)

# The search: a context, its parameters, and the solutions (sols[d + 1] is
# the solution merged with sols[d], made when it is first needed).
function hbt_srh(ctx::NamedTuple, rng::AbstractRNG, par::NamedTuple)
    W = eltype(ctx.wgt)

    return (;
        ctx, rng, par,
        sols = typeof(hbt_st(W, 0))[],
        t0 = Ref(time()), deadline = Ref(Inf), cutoff = Ref(par.xtime),
        nstep = Ref(0), nlocal = Ref(0), nexact = Ref(0), ntimeout = Ref(0),
    )
end

hbt_expired(srh::NamedTuple) = time() > srh.deadline[]

# The time limit of the exact triangulations made while improving a
# solution whose largest bag has not shrunk for kstall steps: xmin, doubled
# every xstep steps, up to xtime. Most exact triangulations that succeed take
# a few milliseconds, but some take much longer; these are worth the wait only
# once the cheap improvements have run out.
function hbt_cut!(srh::NamedTuple, kstall::Int)
    par = srh.par; e = kstall ÷ par.xstep
    srh.cutoff[] = e >= 30 ? par.xtime : min(par.xtime, par.xmin * 2.0^e)
    return
end

# The size limit of exact triangulations of a region around bags with up to
# size vertices. A local graph only slightly larger than one bag is nearly
# complete, and septw solves it quickly even when the bag is wide.
hbt_limit(srh::NamedTuple, size::Int) = max(srh.par.base, size + srh.par.margin)

function hbt_rec!(srh::NamedTuple, st::NamedTuple, width::Tuple, root::Int)
    ctx = srh.ctx
    st.width[] = width
    bptr, btgt, parent = hbt_td(ctx, st.sol, root)
    copy!(st.bptr, bptr); copy!(st.btgt, btgt); copy!(st.parent, parent)
    copy!(st.order, hbt_ord(ctx, bptr, btgt, parent))
    st.maxbag[] = maximum(i -> bptr[i + 1] - bptr[i], eachindex(parent))
    return
end

# Evaluate a solution and, if it improved, record the decomposition and keep
# only the PMCs that root a decomposition of the new width. With refined
# widths, progress is any decrease of (k, f); otherwise only of k.
function hbt_upd!(srh::NamedTuple, st::NamedTuple)
    par = srh.par; sol = st.sol
    width, root = hbt_dp!(srh.ctx, sol); old = st.width[]
    better = par.refined ? width < old : width[1] < old[1]
    better || return false
    report = par.verbose && iszero(st.depth) && width[1] < old[1]
    hbt_rec!(srh, st, width, root)
    bound = par.refined ? width : (width[1], typemax(Int))
    hbt_flt!(sol, p -> sol.root[p] <= bound)

    if report
        println("hbt: width $(hbt_str(width[1])) at $(round(time() - srh.t0[]; digits = 2)) s (steps = $(hbt_stats(srh)[1]), pmcs = $(hbt_len(sol)))")
    end

    return true
end

hbt_str(k::Integer) = string(k - one(k))
hbt_str(k) = string(k)

# ---------------------------------------------------------------------------
# Triangulations of local graphs
# ---------------------------------------------------------------------------

# A greedy minimal triangulation: alg on a random relabeling of the graph
# (greedy algorithms break ties by label), then made minimal.
function hbt_grd(srh::NamedTuple, weights::AbstractVector, graph::AbstractGraph)
    n = nv(graph); buf = hbt_rbuf(srh, weights)
    perm = randperm!(srh.rng, resize!(buf.rprm, n)); invp = resize!(buf.rinv, n); pweights = resize!(buf.rwgt, n)
    ptr = resize!(buf.rptr, n + 1); tgt = resize!(buf.rtgt, de(graph)); ptr[1] = 1

    @inbounds for i in oneto(n)
        invp[perm[i]] = i; pweights[i] = weights[perm[i]]
    end

    @inbounds for i in oneto(n)
        k = ptr[i]

        for x in neighbors(graph, perm[i])
            tgt[k] = invp[x]; k += 1
        end

        sort!(view(tgt, ptr[i]:k - 1); alg = QuickSort); ptr[i + 1] = k
    end

    pgraph = BipartiteGraph{Int, Int}(n, n, ptr[n + 1] - 1, ptr, tgt)
    order, _ = permutation(pweights, pgraph, srh.par.alg)
    order, _ = permutation(weights, graph, MinimalChordal(perm[order]))
    return order
end

# The arrays of the relabeled graph: those of the context, if there is one.
hbt_rbuf(srh::NamedTuple, weights::AbstractVector) = haskey(srh, :ctx) && eltype(srh.ctx.rwgt) === eltype(weights) ? srh.ctx : (; rprm = Int[], rinv = Int[], rptr = Int[], rtgt = Int[], rwgt = Vector{eltype(weights)}())

# the reduction rules applied to local graphs (see hbt_exa)
const HBT_RULES = 5

# the largest number of missing edges of a safe separator found by the
# minor test (see SafeSeparators)
const HBT_MISSING = 50

# The exact search, as an elimination algorithm, so that SafeRules and
# SafeSeparators can apply it to the pieces of a kernel: septw solves each
# piece from the width lo, with the cap hi and the deadline dl, and status
# holds the worst outcome. A piece above the cap puts the whole graph above
# it; otherwise a piece that runs out of time spoils the result.
struct HBTExact{W} <: EliminationAlgorithm
    lo::W
    hi::W
    dl::Float64
    status::Base.RefValue{BTStatus}
end

# The pieces run one after another, on the task of the search: they share the
# deadline and the status, and a parallel search has a task on every thread
# (see isconcurrent).
isconcurrent(alg::HBTExact) = false

function permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::HBTExact; verbose::Bool = false) where {V}
    perm, invp, _, status = septw(weights, graph, alg.lo, alg.hi, alg.dl, 1)

    if status == BT_ABOVE || status == BT_TIMEOUT && alg.status[] == BT_FOUND
        alg.status[] = status
    end

    return convert(Vector{V}, perm), convert(Vector{V}, invp)
end

# HBTExact behind the safe reduction rules and safe separators. The rules
# get the lower bound lb only, so they preserve the treewidth.
function hbt_safe(weights::AbstractVector, graph::AbstractGraph, lb::W, lo::W, hi::W, dl::Float64) where {W}
    status = Ref(BT_FOUND)
    alg = SafeRules(SafeSeparators(HBTExact(lo, hi, dl, status); maxmissing = HBT_MISSING), lb)
    order, _ = permutation(weights, graph, alg)
    return order, status[]
end

# Triangulate a local graph exactly, giving up after the time limit of the
# current step (see hbt_cut!). Returns a minimal ordering, nothing if the
# time runs out, or missing if no triangulation has bags of weight at most
# cap.
#
# Every width below the treewidth costs an exhaustive search. A
# triangulation is useful only if its bags weigh at most target, so the
# search starts there, and stops at the first one that does: the result is
# optimal if no triangulation meets the target, and meets the target
# otherwise. lb is a known lower bound on the bag weight.
#
# The search runs on the pieces of a kernel of the local graph. The kernel
# is computed by compressreduce: true twins are merged into weighted
# vertices, and the safe reduction rules (simplicial, almost simplicial)
# eliminate vertices. The rules get the lower bound only, not the target, so
# they preserve the treewidth: they return a bound lb' <= tw such that an
# elimination of the kernel with bags of weight at most t extends to one of
# the local graph with bags of weight at most max(lb', t). Local graphs are
# full of twins and almost simplicial vertices (every component B of G - U
# turns N(B) into a clique). The kernel is then split further by the safe
# reduction rules and the safe separators (hbt_safe).
function hbt_exa(srh::NamedTuple, weights::AbstractVector{W}, H::AbstractGraph, target::W; lb::W = zero(W), cap::W = typemax(W)) where {W}
    lb = max(lb, convert(W, lowerbound(weights, H, DEFAULT_LOWER_BOUND_ALGORITHM)))
    lb > cap && return missing
    dl = min(srh.deadline[], time() + srh.cutoff[])
    kerweights, kergraph, result = compressreduce(HBT_RULES, weights, H, lb)

    # one task: the search may have workers of its own
    kerorder, status = hbt_safe(kerweights, kergraph, max(lb, result.width), max(lb, result.width, target), cap, dl)
    status == BT_ABOVE && return missing

    if status == BT_TIMEOUT
        srh.ntimeout[] += 1
        return nothing
    end

    srh.nexact[] += 1
    order = rule_pull(result, kerorder)
    order, _ = permutation(weights, H, MinimalChordal(order))
    return order
end

# Exact if the local graph has at most limit vertices and the search
# finishes in time, greedy otherwise; missing as in hbt_exa.
function hbt_tri(srh::NamedTuple, weights::AbstractVector{W}, H::AbstractGraph, target::W, limit::Int; lb::W = zero(W), cap::W = typemax(W)) where {W}
    order = nv(H) <= limit ? hbt_exa(srh, weights, H, target; lb, cap) : nothing
    ismissing(order) && return missing
    return isnothing(order) ? hbt_grd(srh, weights, H) : order
end

# The maximal cliques of the triangulation of HS (a minimal triangulation of
# the local graph L) by order, as sorted local lists appended to the lists
# (cptr, ctgt), except those heavier than k (they cannot be bags of a
# decomposition of width at most k) and the neighborhoods of components of
# G - U. Returns the weight of the heaviest clique.
function hbt_cls!(cptr::Vector{Int}, ctgt::Vector{Int}, srh::NamedTuple, L::NamedTuple, HS::AbstractGraph, order::AbstractVector, k)
    weights = L.wgt; W = eltype(weights); srh.nlocal[] += 1
    label, tree = cliquetree(weights, HS, order); width = zero(W); js = Int[]

    for (j, clique) in enumerate(tree)
        w = zero(W)

        for v in clique
            w += weights[label[v]]
        end

        width = max(width, w)
        w > k && continue
        lo = length(ctgt) + 1

        for v in clique
            push!(ctgt, label[v])
        end

        K = view(ctgt, lo:length(ctgt)); sort!(K; alg = QuickSort)

        if hbt_isnbr(L, K)
            resize!(ctgt, lo - 1)
        else
            push!(cptr, length(ctgt) + 1); push!(js, j)
        end
    end

    # (the components of a few cliques are found faster one at a time)
    if L.fast && length(js) >= HBT_LFC
        for i in eachindex(js)
            push!(L.fcl, length(L.fptr) + i - 1)
        end

        hbt_lfc!(L, label, tree, js)
    else
        for _ in js
            push!(L.fcl, 0)
        end
    end

    return width
end

# Add the PMCs U[K] of G, for the cliques K of the local graph, to the
# solution; a PMC that the solution side has is copied from it.
function hbt_adc!(srh::NamedTuple, st::NamedTuple, L::NamedTuple, cptr::Vector{Int}, ctgt::Vector{Int}, side::Union{Nothing, NamedTuple} = nothing)
    ctx = srh.ctx; sol = st.sol; U = L.U; X = L.X

    for c in oneto(length(cptr) - 1)
        # each PMC costs a search of the local graph, which can be most of the
        # graph: stop at the deadline (a subset of the PMCs of a triangulation
        # is still a set of PMCs, and the solution keeps its own)
        hbt_expired(srh) && break
        K = view(ctgt, cptr[c]:cptr[c + 1] - 1)

        if !isnothing(side)
            empty!(X)

            for k in K
                push!(X, U[k])
            end

            q = bt_tab_find(side.sol.pmcs, X)

            if ispositive(q)
                hbt_cpy!(sol, side.sol, q); continue
            end
        end

        hbt_lft!(ctx, sol, L, c, K)
    end

    return
end

# Triangulate the local graph on U and add the resulting PMCs of G to the
# solution, if the triangulation is no wider than the solution.
function hbt_lcl!(srh::NamedTuple, st::NamedTuple, side::NamedTuple, U::Vector{Int})
    L, cptr, ctgt = hbt_lcc(srh, U, st.width[][1], hbt_limit(srh, st.maxbag[]))
    hbt_adc!(srh, st, L, cptr, ctgt, side)
    return
end

# The local graph on U and the cliques of its triangulation (exact if it has
# at most limit vertices), or no cliques if the triangulation is wider than k.
function hbt_lcc(srh::NamedTuple, U::Vector{Int}, k, limit::Int)
    L = hbt_loc(srh.ctx, U); H = L.graph; weights = L.wgt
    cptr = Int[1]; ctgt = Int[]
    order = hbt_tri(srh, weights, H, k, limit; cap = k)

    if !ismissing(order) && hbt_cls!(cptr, ctgt, srh, L, H, order, typemax(eltype(weights))) > k
        resize!(cptr, 1); empty!(ctgt)
    end

    return L, cptr, ctgt
end

# The cliques (cptr, ctgt) of the local graph L lifted to PMCs of G apart from
# any solution, for hbt_put!: the PMCs that hbt_adc! would add to a solution
# without them, in a solution of their own whose blocks are named in a context
# of their own (it shares the scratch space of ctx), with the blocks and
# separators of that context. Most of the cost of adding a clique to a solution
# is its lifting, which a worker does this way for the main task.
function hbt_lif(srh::NamedTuple, L::NamedTuple, cptr::Vector{Int}, ctgt::Vector{Int})
    W = eltype(srh.ctx.wgt); sol = hbt_sol(W)
    ctx = merge(srh.ctx, (; seps = bt_tab(), swgt = W[], blks = bt_tab(), bmin = Int[], bsep = Int[], bsize = Int[], bwgt = W[]))

    for c in oneto(length(cptr) - 1)
        hbt_expired(srh) && break
        hbt_lft!(ctx, sol, L, c, view(ctgt, cptr[c]:cptr[c + 1] - 1))
    end

    # (and the hashes of the PMCs, made here and not by the main task)
    hsh = [bt_tab_hsh(hbt_pmc(sol, p)) for p in oneto(hbt_len(sol))]
    return (; sol, hsh, ctx.seps, ctx.bmin, ctx.bsep, ctx.bwgt, ctx.bsize)
end

# Add the PMCs of r (see hbt_lif) to the solution, as hbt_adc! adds the cliques
# they were lifted from: the same PMCs, in the same order, with the same blocks.
# (A block is new to the solution's context only in a PMC new to the solution,
# and the first such PMC is the one that r has it from.) The blocks and
# separators of r are renamed (bmap, smap) as they are first met.
function hbt_put!(srh::NamedTuple, st::NamedTuple, r::NamedTuple, side::Union{Nothing, NamedTuple} = nothing)
    ctx = srh.ctx; sol = st.sol; src = r.sol
    bmap = zeros(Int, length(r.bmin)); smap = zeros(Int, bt_tab_len(r.seps))

    for p in oneto(hbt_len(src))
        hbt_expired(srh) && break
        X = hbt_pmc(src, p); h = r.hsh[p]

        if !isnothing(side)
            q = bt_tab_find(side.sol.pmcs, X, h)

            if ispositive(q)
                hbt_cpy!(sol, side.sol, q); continue
            end
        end

        last(bt_tab_add!(sol.pmcs, X, h)) || continue
        push!(sol.pwgt, src.pwgt[p])

        @inbounds for c in src.cfst[p]:src.cfst[p + 1] - 1
            push!(sol.csub, hbt_ren!(ctx, r, bmap, smap, src.csub[c]))
            push!(sol.csup, hbt_ren!(ctx, r, bmap, smap, src.csup[c]))

            for i in src.iptr[c]:src.iptr[c + 1] - 1
                push!(sol.itgt, src.itgt[i])
            end

            push!(sol.iptr, length(sol.itgt) + 1)
        end

        push!(sol.cfst, length(sol.csub) + 1)
    end

    return
end

# The block of ctx named b in r.
function hbt_ren!(ctx::NamedTuple, r::NamedTuple, bmap::Vector{Int}, smap::Vector{Int}, b::Int)
    @inbounds if iszero(bmap[b])
        s = r.bsep[b]
        iszero(smap[s]) && (smap[s] = hbt_sep!(ctx, bt_tab_seq(r.seps, s)))
        bmap[b] = hbt_blk!(ctx, r.bmin[b], smap[s], r.bsize[b], r.bwgt[b])
    end

    return @inbounds bmap[b]
end

# ---------------------------------------------------------------------------
# Initial solutions, merging, and steps
# ---------------------------------------------------------------------------

# A new solution: the PMCs of the best of ninit greedy triangulations and,
# for the first solution, of the minimal triangulation by init (an ordering
# or an elimination algorithm), if there is one. If init is a function, then
# every solution gets one, by init(depth) (nothing for none).
function hbt_ini!(srh::NamedTuple, depth::Int)
    ctx = srh.ctx; g = ctx.graph; wgt = ctx.wgt; W = eltype(wgt)
    best = typemax(W); bestlabel = nothing; besttree = nothing
    init = srh.par.init
    init isa Function ? (init = init(depth)) : !iszero(depth) && (init = nothing)

    # (a solution without init has a greedy one at least)
    for i in oneto(isnothing(init) ? max(1, srh.par.ninit) : srh.par.ninit)
        (i > 1 && hbt_expired(srh)) && break
        order = hbt_grd(srh, wgt, g)
        label, tree = cliquetree(wgt, g, order)
        width = hbt_tw(wgt, label, tree)

        if width < best
            best = width; bestlabel = label; besttree = tree
        end
    end

    st = hbt_st(W, depth)
    isnothing(besttree) || hbt_cqs!(ctx, st.sol, bestlabel, besttree)

    if !isnothing(init)
        order = init isa AbstractVector ? init : first(permutation(wgt, g, init))
        order, _ = permutation(wgt, g, MinimalChordal(order))
        label, tree = cliquetree(wgt, g, order)
        hbt_cqs!(ctx, st.sol, label, tree)
    end

    width, root = hbt_dp!(ctx, st.sol)
    hbt_rec!(srh, st, width, root)
    push!(srh.sols, st)
    return st
end

# The largest bag weight of a clique tree.
function hbt_tw(wgt::AbstractVector{W}, label::AbstractVector, tree) where {W}
    width = zero(W)

    for clique in tree
        w = zero(W)

        for v in clique
            w += wgt[label[v]]
        end

        width = max(width, w)
    end

    return width
end

# Add the maximal cliques of a minimal triangulation (its clique tree) to
# a solution: they are PMCs. (The reference version: one search of the whole
# graph per clique, by hbt_add!. hbt_cqs! below gives the same solution.)
function hbt_cqs_bfs!(ctx::NamedTuple, sol::NamedTuple, label::AbstractVector, tree)
    X = Int[]

    for clique in tree
        empty!(X)

        for v in clique
            push!(X, label[v])
        end

        hbt_add!(ctx, sol, sort!(X))
    end

    return
end

# A component of G - K, for a clique K of a clique tree: its smallest vertex m
# and its size and weight (both over vertices of the graph), and its
# separator N(C), as sorted positions in the elimination order.
struct HBTComp{W}
    m::Int
    sz::Int
    w::W
    sep::Vector{Int}
end

@inline function hbt_find!(par::Vector{Int}, x::Int)
    r = x

    @inbounds while par[r] != r
        r = par[r]
    end

    @inbounds while par[x] != r
        y = par[x]; par[x] = r; x = y
    end

    return r
end

# Add the maximal cliques of a minimal triangulation (its clique tree) to
# a solution: they are PMCs. hbt_add! would search the whole graph once per
# clique to find the components of G - K; here they are found for all the
# cliques together, in time about linear in the size of the graph and the
# tree plus that of the output.
#
# Write U(j) for the vertices eliminated in the subtree of node j (a range of
# positions, the tree being in postorder) and S(j) for its separator. Every
# component of G - K(j) lies in one branch of the tree at j (an edge of G
# lies in some clique, and a vertex outside K(j) in the cliques of one
# branch only), so the components are
#
#   - below each child c: the components of G[U(c)], found bottom-up by a
#     union-find over the elimination order, each with its separator (the
#     neighbours of a class that are not yet eliminated);
#   - above j: the components of G[P(j)], P(j) = V - U(j) - S(j), found
#     top-down: P(j) is the union of P(p) for the parent p, of U(c) for the
#     other children c of p, and of K(p) - S(j), and the components of the
#     first two are joined through the vertices of the third.
#
# The forest is made a tree by a virtual root with an empty clique. The
# components are added in the order of their smallest vertices, as hbt_add!
# finds them, so the solution is the same as hbt_cqs_bfs! gives (up to the
# rounding of component weights, which are summed in another order).
#
# The pass finds the components of every clique, and costs about as much
# as a search per clique (hbt_add!): when most cliques are PMCs of the
# solution already, as when RTW maps decompositions between contractions,
# the new ones are added one by one, and with none, nothing is done.
function hbt_cqs!(ctx::NamedTuple, sol::NamedTuple, label::AbstractVector, tree)
    g = ctx.graph; n = nv(g); wgt = ctx.wgt; W = eltype(wgt); nc = length(tree)
    nnew = 0; X = Int[]

    @inbounds for clique in tree
        empty!(X)

        for v in clique
            push!(X, label[v])
        end

        ispositive(bt_tab_find(sol.pmcs, sort!(X))) || (nnew += 1)
    end

    iszero(nnew) && return
    2nnew <= nc && return hbt_cqs_bfs!(ctx, sol, label, tree)
    pos = Vector{Int}(undef, n)

    @inbounds for i in oneto(n)
        pos[label[i]] = i
    end

    # ---- bottom-up: the components of G[U(j)], for every node j
    par = collect(oneto(n)); cm = zeros(Int, n); cs = zeros(Int, n); cw = zeros(W, n)
    up = Vector{Set{Int}}(undef, n); sid = zeros(Int, n); alive = falses(n)
    stk = Int[]; lo = zeros(Int, nc)
    B = Vector{Vector{HBTComp{W}}}(undef, nc)

    @inbounds for j in oneto(nc)
        R = residual(tree, j); l = isempty(R) ? n + 1 : first(R)

        for c in childindices(tree, j)
            l = min(l, lo[c])
        end

        lo[j] = l

        for v in R
            x = label[v]; S = Set{Int}()

            for y in neighbors(g, x)
                u = pos[y]; u > v && push!(S, u)
            end

            cm[v] = x; cs[v] = 1; cw[v] = wgt[x]; up[v] = S; sid[v] = v
            alive[v] = true; push!(stk, v)

            for y in neighbors(g, x)
                u = pos[y]; u < v || continue
                a = hbt_find!(par, u); b = hbt_find!(par, v)
                a == b && continue

                # the larger set survives
                length(up[a]) < length(up[b]) && ((a, b) = (b, a))
                par[b] = a; union!(up[a], up[b]); up[b] = Set{Int}()
                cm[a] = min(cm[a], cm[b]); cs[a] += cs[b]; cw[a] += cw[b]

                if sid[b] < sid[a]
                    alive[sid[a]] = false; sid[a] = sid[b]
                else
                    alive[sid[b]] = false
                end
            end

            delete!(up[hbt_find!(par, v)], v)
        end

        # the live classes inside U(j) are those with a stack id in it; the
        # stack is sorted by id, and the dead ones are dropped
        i0 = length(stk) + 1

        while i0 > 1 && stk[i0 - 1] >= l
            i0 -= 1
        end

        k = i0 - 1; comps = HBTComp{W}[]

        for i in i0:length(stk)
            t = stk[i]; alive[t] || continue
            k += 1; stk[k] = t; r = hbt_find!(par, t)
            push!(comps, HBTComp{W}(cm[r], cs[r], cw[r], sort!(collect(up[r]))))
        end

        resize!(stk, k); B[j] = comps
    end

    # ---- top-down: the components of G[P(j)], for every node j
    A = Vector{Vector{HBTComp{W}}}(undef, nc)
    roots = Int[]

    for j in oneto(nc)
        isnothing(parentindex(tree, j)) && push!(roots, j)
    end

    markS = zeros(Int, n); markY = zeros(Int, n); markK = zeros(Int, n); gmark = zeros(Int, n)
    itemof = zeros(Int, n); stamp = 0; Y = Int[]; EK = Tuple{Int, Int}[]
    upar = Int[]; nm = Int[]; nsz = Int[]; nw = W[]; nsep = Vector{Int}[]; touched = Bool[]

    # the children of node p (p = 0: the virtual root) get their sides; the
    # nodes in reverse postorder, a parent before its children
    @inbounds for p in Iterators.flatten((0, nc:-1:1))
        children = p == 0 ? roots : childindices(tree, p)
        isempty(children) && continue
        K = p == 0 ? Int[] : collect(tree[p])
        stamp += 1; sk = stamp

        @inbounds for v in K
            markK[v] = sk
        end

        # the edges of G inside K(p)
        empty!(EK)

        @inbounds for v in K
            for y in neighbors(g, label[v])
                u = pos[y]
                (u > v && markK[u] == sk) && push!(EK, (v, u))
            end
        end

        above = p == 0 ? HBTComp{W}[] : A[p]

        @inbounds for j in children
            stamp += 1; ss = stamp

            for v in separator(tree, j)
                markS[v] = ss
            end

            stamp += 1; sy = stamp; empty!(Y)

            for v in K
                if markS[v] != ss
                    push!(Y, v); markY[v] = sy; itemof[v] = length(Y)
                end
            end

            ny = length(Y)

            # items: the vertices Y, then the components of the other branches
            comps = HBTComp{W}[]
            append!(comps, above)

            for c in children
                c == j || append!(comps, B[c])
            end

            ni = ny + length(comps)
            resize!(upar, ni)

            for i in oneto(ni)
                upar[i] = i
            end

            for (a, b) in EK
                if markY[a] == sy && markY[b] == sy
                    ra = hbt_find!(upar, itemof[a]); rb = hbt_find!(upar, itemof[b])
                    ra == rb || (upar[rb] = ra)
                end
            end

            resize!(touched, length(comps)); out = HBTComp{W}[]

            for (q, C) in enumerate(comps)
                touched[q] = false

                for s in C.sep
                    if markS[s] != ss
                        # s is in K(p) - S(j)
                        markY[s] == sy || error("hbt_cqs!: a separator leaves its clique")
                        ra = hbt_find!(upar, itemof[s]); rb = hbt_find!(upar, ny + q)
                        ra == rb || (upar[rb] = ra)
                        touched[q] = true
                    end
                end

                touched[q] || push!(out, C)
            end

            # the groups through Y: aggregates indexed by root item
            resize!(nm, ni); resize!(nsz, ni); resize!(nw, ni); resize!(nsep, ni)

            for i in oneto(ny)
                r = hbt_find!(upar, i)

                if r == i
                    nm[r] = typemax(Int); nsz[r] = 0; nw[r] = zero(W); nsep[r] = Int[]
                end
            end

            for i in oneto(ny)
                r = hbt_find!(upar, i); x = label[Y[i]]
                nm[r] = min(nm[r], x); nsz[r] += 1; nw[r] += wgt[x]
            end

            base = stamp; stamp += ni

            for (a, b) in EK
                # an edge from Y to S(j) puts its end in S(j) into the separator
                if markY[a] == sy && markS[b] == ss
                    r = hbt_find!(upar, itemof[a])
                    gmark[b] == base + r || (gmark[b] = base + r; push!(nsep[r], b))
                elseif markY[b] == sy && markS[a] == ss
                    r = hbt_find!(upar, itemof[b])
                    gmark[a] == base + r || (gmark[a] = base + r; push!(nsep[r], a))
                end
            end

            for (q, C) in enumerate(comps)
                touched[q] || continue
                r = hbt_find!(upar, ny + q)
                nm[r] = min(nm[r], C.m); nsz[r] += C.sz; nw[r] += C.w

                for s in C.sep
                    if markS[s] == ss
                        gmark[s] == base + r || (gmark[s] = base + r; push!(nsep[r], s))
                    end
                end
            end

            for i in oneto(ny)
                hbt_find!(upar, i) == i && push!(out, HBTComp{W}(nm[i], nsz[i], nw[i], sort!(nsep[i])))
            end

            A[j] = out
        end
    end

    # ---- add the cliques, in the order of the tree, as hbt_cqs_bfs! does
    X = Int[]; sep = Int[]; cmin = Int[]; csize = Int[]; cwgt = W[]; csep = Int[]

    @inbounds for j in oneto(nc)
        empty!(X)

        for v in tree[j]
            push!(X, label[v])
        end

        sort!(X)
        ispositive(bt_tab_find(sol.pmcs, X)) && continue
        comps = copy(A[j])

        for c in childindices(tree, j)
            append!(comps, B[c])
        end

        sort!(comps; by = C -> C.m)
        empty!(cmin); empty!(csize); empty!(cwgt); empty!(csep)

        for C in comps
            empty!(sep)

            for v in C.sep
                push!(sep, label[v])
            end

            push!(cmin, C.m); push!(csize, C.sz); push!(cwgt, C.w); push!(csep, hbt_sep!(ctx, sort!(sep)))
        end

        hbt_add!(ctx, sol, X, cmin, csize, cwgt, csep)
    end

    return
end

# ---------------------------------------------------------------------------
# Lifting all the cliques of a triangulation of a local graph at once
# ---------------------------------------------------------------------------

# A component of H⁺ - K (see hbt_lfc!): its smallest local vertex mu
# (typemax(Int) if it has none, as the component B of G - U numbered b), its
# smallest vertex mg in G, its size and weight in G, and its separator, as
# sorted positions in the elimination order.
struct HBTLComp{W}
    mu::Int
    mg::Int
    b::Int
    sz::Int
    w::W
    sep::Vector{Int}
end

# the fewest cliques of a triangulation whose components hbt_lfc! finds
const HBT_LFC = 32

# Whether hbt_lfc! can lift the cliques of triangulations of a local graph
# (on U, with nwrd and the weights of the components of G - U): it is given as lists (a large one, where a search for each clique costs
# most), and the sums of its weights do not depend on their order.
function hbt_isfast(nwrd::Int, wgt::AbstractVector, U::AbstractVector{Int}, bwgt::AbstractVector)
    iszero(nwrd) || return false
    t = zero(eltype(wgt))

    @inbounds for v in U
        hbt_isexact(wgt[v]) || return false
        t += wgt[v]
    end

    @inbounds for w in bwgt
        hbt_isexact(w) || return false
        t += w
    end

    return hbt_isexact(t)
end

hbt_isexact(w::Integer) = true
hbt_isexact(w::AbstractFloat) = isinteger(w) && abs(w) < maxintfloat(typeof(w)) / 4
hbt_isexact(w) = false

# The component B of G - U numbered b, as a component of H⁺ - K for a clique
# K that contains N(B) (hbt_lfc!).
function hbt_lfb(L::NamedTuple, pos::Vector{Int}, b::Int)
    W = eltype(L.bwgt); sep = [pos[x] for x in bt_tab_seq(L.nbrs, L.nidx[b])]
    return HBTLComp{W}(typemax(Int), L.bmin[b], b, L.bsize[b], L.bwgt[b], sort!(sep; alg = QuickSort))
end

# Join the classes of the items i and v (hbt_lfc!).
function hbt_lfj!(par::Vector{Int}, up::Vector{Set{Int}}, cmu::Vector{Int}, cmg::Vector{Int}, cb::Vector{Int}, cs::Vector{Int}, cw::Vector, tid::Vector{Int}, alive::BitVector, i::Int, v::Int)
    a = hbt_find!(par, i); r = hbt_find!(par, v)
    a == r && return

    # the larger set survives
    @inbounds begin
        length(up[a]) < length(up[r]) && ((a, r) = (r, a))
        par[r] = a; union!(up[a], up[r]); up[r] = Set{Int}()
        cmu[a] = min(cmu[a], cmu[r]); cmg[a] = min(cmg[a], cmg[r]); cb[a] = min(cb[a], cb[r])
        cs[a] += cs[r]; cw[a] += cw[r]

        if tid[r] < tid[a]
            alive[tid[a]] = false; tid[a] = tid[r]
        else
            alive[tid[r]] = false
        end
    end

    return
end

# The components of G - U[K] for the cliques K = tree[j] (j in js) of a
# triangulation of the local graph L, as hbt_lft! finds them one clique at a
# time (the components Q of H - K, each joined with the components B of G - U
# whose neighborhoods meet it, ordered by their smallest vertices, then the
# components B with N(B) ⊆ K, in order), appended to the lists of L (fptr,
# fmin, fsize, fwgt, fsptr, fstgt: the separators local and sorted).
#
# They are the components of H⁺ - K, where H⁺ is H with a vertex for every
# component B of G - U, adjacent to N(B), and they are found for all the
# cliques together as hbt_cqs! finds the components of G - K: B is a leaf of
# the tree below the node where the first vertex of N(B) is eliminated (that
# node's clique contains the clique N(B)), and is taken before the node's own
# vertices. The weights are summed in another order than by hbt_lft!, so they
# must sum exactly (hbt_isfast).
function hbt_lfc!(L::NamedTuple, label::AbstractVector, tree, js::Vector{Int})
    H = L.graph; nu = nv(H); U = L.U; wgt = L.wgt; W = eltype(wgt); nc = length(tree)
    nb = length(L.bmin); ni = nu + nb
    pos = Vector{Int}(undef, nu); node = Vector{Int}(undef, nu)

    @inbounds for i in oneto(nu)
        pos[label[i]] = i
    end

    @inbounds for j in oneto(nc), p in residual(tree, j)
        node[p] = j
    end

    # the components B of G - U attached to each node (aptr, atgt), and the
    # components whose neighborhoods contain each vertex (xptr, xtgt)
    aptr = zeros(Int, nc + 1); xptr = zeros(Int, nu + 1); bnode = Vector{Int}(undef, nb)

    # (a component with no neighbors, of a graph that is not connected, is a
    # component of G - U[K] for every clique K, and is not attached)
    free = Int[]

    @inbounds for b in oneto(nb)
        nbr = bt_tab_seq(L.nbrs, L.nidx[b]); p = nu + 1

        for x in nbr
            p = min(p, pos[x]); xptr[x + 1] += 1
        end

        if p > nu
            bnode[b] = 0; push!(free, b)
        else
            bnode[b] = node[p]; aptr[node[p] + 1] += 1
        end
    end

    aptr[1] = 1; xptr[1] = 1

    @inbounds for j in oneto(nc)
        aptr[j + 1] += aptr[j]
    end

    @inbounds for x in oneto(nu)
        xptr[x + 1] += xptr[x]
    end

    atgt = Vector{Int}(undef, nb); xtgt = Vector{Int}(undef, xptr[nu + 1] - 1)
    an = aptr[1:nc]; xn = xptr[1:nu]

    @inbounds for b in oneto(nb)
        iszero(bnode[b]) && continue
        atgt[an[bnode[b]]] = b; an[bnode[b]] += 1

        for x in bt_tab_seq(L.nbrs, L.nidx[b])
            xtgt[xn[x]] = b; xn[x] += 1
        end
    end

    # ---- bottom-up: the components of H⁺[U(j)], for every node j. The items
    # are the positions 1:nu and the components nu + b; tid is the order in
    # which they are taken (stack ids)
    par = collect(oneto(ni)); cmu = zeros(Int, ni); cmg = zeros(Int, ni); cb = zeros(Int, ni)
    cs = zeros(Int, ni); cw = zeros(W, ni); up = Vector{Set{Int}}(undef, ni)
    tid = zeros(Int, ni); itm = zeros(Int, ni); alive = falses(ni)
    stk = Int[]; lo = zeros(Int, nc); t = 0
    B = Vector{Vector{HBTLComp{W}}}(undef, nc)

    @inbounds for j in oneto(nc)
        l = t + 1

        for c in childindices(tree, j)
            l = min(l, lo[c])
        end

        lo[j] = l

        # the components attached to j: no neighbor is taken before them
        for a in aptr[j]:aptr[j + 1] - 1
            b = atgt[a]; i = nu + b; t += 1; tid[i] = t; itm[t] = i
            cmu[i] = typemax(Int); cmg[i] = L.bmin[b]; cb[i] = b; cs[i] = L.bsize[b]; cw[i] = L.bwgt[b]
            up[i] = Set{Int}(pos[x] for x in bt_tab_seq(L.nbrs, L.nidx[b]))
            alive[t] = true; push!(stk, t)
        end

        for v in residual(tree, j)
            x = label[v]; S = Set{Int}(); t += 1; tid[v] = t; itm[t] = v

            for y in neighbors(H, x)
                u = pos[y]; u > v && push!(S, u)
            end

            cmu[v] = x; cmg[v] = U[x]; cb[v] = typemax(Int); cs[v] = 1; cw[v] = wgt[x]; up[v] = S
            alive[t] = true; push!(stk, t)

            # the neighbors taken before v: those eliminated before it, and
            # the components whose neighborhoods contain it
            for y in neighbors(H, x)
                u = pos[y]; u < v && hbt_lfj!(par, up, cmu, cmg, cb, cs, cw, tid, alive, u, v)
            end

            for k in xptr[x]:xptr[x + 1] - 1
                hbt_lfj!(par, up, cmu, cmg, cb, cs, cw, tid, alive, nu + xtgt[k], v)
            end

            delete!(up[hbt_find!(par, v)], v)
        end

        # the live classes inside U(j) are those with a stack id in it
        i0 = length(stk) + 1

        while i0 > 1 && stk[i0 - 1] >= l
            i0 -= 1
        end

        k = i0 - 1; comps = HBTLComp{W}[]

        for i in i0:length(stk)
            s = stk[i]; alive[s] || continue
            k += 1; stk[k] = s; r = hbt_find!(par, itm[s])
            push!(comps, HBTLComp{W}(cmu[r], cmg[r], cb[r], cs[r], cw[r], sort!(collect(up[r]); alg = QuickSort)))
        end

        resize!(stk, k); B[j] = comps
    end

    # ---- top-down: the components of H⁺[P(j)], P(j) = V - U(j) - S(j), for
    # every node j: from those of its parent p, of the other branches at p,
    # the vertices of K(p) - S(j), and the components attached to p
    A = Vector{Vector{HBTLComp{W}}}(undef, nc); roots = Int[]

    for j in oneto(nc)
        isnothing(parentindex(tree, j)) && push!(roots, j)
    end

    markS = zeros(Int, nu); markY = zeros(Int, nu); markK = zeros(Int, nu); gmark = zeros(Int, nu)
    itemof = zeros(Int, nu); stamp = 0; Y = Int[]; EK = Tuple{Int, Int}[]
    upar = Int[]; nmu = Int[]; nmg = Int[]; nbb = Int[]; nsz = Int[]; nw = W[]; nsep = Vector{Int}[]
    touched = Bool[]; isroot = Bool[]

    @inbounds for p in Iterators.flatten((0, nc:-1:1))
        children = p == 0 ? roots : childindices(tree, p)
        isempty(children) && continue
        K = p == 0 ? Int[] : collect(tree[p]); stamp += 1; sk = stamp
        na = p == 0 ? 0 : aptr[p + 1] - aptr[p]

        for v in K
            markK[v] = sk
        end

        # the edges of H inside K(p)
        empty!(EK)

        for v in K, y in neighbors(H, label[v])
            u = pos[y]
            (u > v && markK[u] == sk) && push!(EK, (v, u))
        end

        above = p == 0 ? HBTLComp{W}[] : A[p]

        for j in children
            stamp += 1; ss = stamp

            for v in separator(tree, j)
                markS[v] = ss
            end

            stamp += 1; sy = stamp; empty!(Y)

            for v in K
                if markS[v] != ss
                    push!(Y, v); markY[v] = sy; itemof[v] = length(Y)
                end
            end

            # the items: the vertices Y, the components attached to p, then the
            # components of the other branches
            ny = length(Y); comps = HBTLComp{W}[]; append!(comps, above)

            for c in children
                c == j || append!(comps, B[c])
            end

            ni2 = ny + na + length(comps); resize!(upar, ni2)

            for i in oneto(ni2)
                upar[i] = i
            end

            for (a, b) in EK
                if markY[a] == sy && markY[b] == sy
                    ra = hbt_find!(upar, itemof[a]); rb = hbt_find!(upar, itemof[b])
                    ra == rb || (upar[rb] = ra)
                end
            end

            for e in oneto(na), x in bt_tab_seq(L.nbrs, L.nidx[atgt[aptr[p] + e - 1]])
                s = pos[x]

                if markY[s] == sy
                    ra = hbt_find!(upar, itemof[s]); rb = hbt_find!(upar, ny + e)
                    ra == rb || (upar[rb] = ra)
                end
            end

            resize!(touched, length(comps)); out = HBTLComp{W}[]

            for (q, C) in enumerate(comps)
                touched[q] = false

                for s in C.sep
                    if markS[s] != ss
                        # s is in K(p) - S(j)
                        markY[s] == sy || error("hbt_lfc!: a separator leaves its clique")
                        ra = hbt_find!(upar, itemof[s]); rb = hbt_find!(upar, ny + na + q)
                        ra == rb || (upar[rb] = ra)
                        touched[q] = true
                    end
                end

                touched[q] || push!(out, C)
            end

            # the groups of the vertices Y and the attached components:
            # aggregates indexed by root item
            nyb = ny + na
            resize!(nmu, nyb); resize!(nmg, nyb); resize!(nbb, nyb); resize!(nsz, nyb); resize!(nw, nyb)
            resize!(nsep, nyb); resize!(isroot, nyb)

            for i in oneto(nyb)
                r = hbt_find!(upar, i)
                r <= nyb || error("hbt_lfc!: a group without a root item")
                isroot[i] = r == i

                if r == i
                    nmu[r] = typemax(Int); nmg[r] = typemax(Int); nbb[r] = typemax(Int); nsz[r] = 0; nw[r] = zero(W); nsep[r] = Int[]
                end
            end

            for i in oneto(ny)
                r = hbt_find!(upar, i); x = label[Y[i]]
                nmu[r] = min(nmu[r], x); nmg[r] = min(nmg[r], U[x]); nsz[r] += 1; nw[r] += wgt[x]
            end

            base = stamp; stamp += nyb

            for e in oneto(na)
                b = atgt[aptr[p] + e - 1]; r = hbt_find!(upar, ny + e)
                nmg[r] = min(nmg[r], L.bmin[b]); nbb[r] = min(nbb[r], b); nsz[r] += L.bsize[b]; nw[r] += L.bwgt[b]

                for x in bt_tab_seq(L.nbrs, L.nidx[b])
                    s = pos[x]

                    if markS[s] == ss
                        gmark[s] == base + r || (gmark[s] = base + r; push!(nsep[r], s))
                    end
                end
            end

            for (a, b) in EK
                # an edge from Y to S(j) puts its end in S(j) into the separator
                if markY[a] == sy && markS[b] == ss
                    r = hbt_find!(upar, itemof[a])
                    gmark[b] == base + r || (gmark[b] = base + r; push!(nsep[r], b))
                elseif markY[b] == sy && markS[a] == ss
                    r = hbt_find!(upar, itemof[b])
                    gmark[a] == base + r || (gmark[a] = base + r; push!(nsep[r], a))
                end
            end

            for (q, C) in enumerate(comps)
                touched[q] || continue
                r = hbt_find!(upar, ny + na + q)
                nmu[r] = min(nmu[r], C.mu); nmg[r] = min(nmg[r], C.mg); nbb[r] = min(nbb[r], C.b)
                nsz[r] += C.sz; nw[r] += C.w

                for s in C.sep
                    if markS[s] == ss
                        gmark[s] == base + r || (gmark[s] = base + r; push!(nsep[r], s))
                    end
                end
            end

            for i in oneto(nyb)
                isroot[i] && push!(out, HBTLComp{W}(nmu[i], nmg[i], nbb[i], nsz[i], nw[i], sort!(nsep[i]; alg = QuickSort)))
            end

            A[j] = out
        end
    end

    # ---- the components of the cliques js, in the order of hbt_lft!
    sep = Int[]

    @inbounds for j in js
        comps = copy(A[j])

        for c in childindices(tree, j)
            append!(comps, B[c])
        end

        for a in aptr[j]:aptr[j + 1] - 1
            push!(comps, hbt_lfb(L, pos, atgt[a]))
        end

        for b in free
            push!(comps, hbt_lfb(L, pos, b))
        end

        sort!(comps; by = C -> (C.mu, C.b))

        for C in comps
            empty!(sep)

            for s in C.sep
                push!(sep, label[s])
            end

            append!(L.fstgt, sort!(sep; alg = QuickSort)); push!(L.fsptr, length(L.fstgt) + 1)
            push!(L.fmin, C.mg); push!(L.fsize, C.sz); push!(L.fwgt, C.w)
        end

        push!(L.fptr, length(L.fmin) + 1)
    end

    return
end

# The solution merged with st.
function hbt_side!(srh::NamedTuple, st::NamedTuple)
    d = st.depth + 1
    return d < length(srh.sols) ? srh.sols[d + 1] : hbt_ini!(srh, d)
end

# Merge (Tamaki 2022): pairs of PMCs X of st and Y of side, where side has no
# greater width, pick out small focus sets U, whose local graphs are
# triangulated; their PMCs and those of side are added to st.
function hbt_mrg!(srh::NamedTuple, st::NamedTuple, side::NamedTuple, pool = nothing)
    hbt_cut!(srh, st.kstall[])
    ctx = srh.ctx; g = ctx.graph; n = nv(g); sol = st.sol
    mark1 = ctx.mark1; mark2 = ctx.mark2; mark3 = ctx.mark3; queue = ctx.queue
    srh.nstep[] += 1; ntry = srh.par.ntry

    # the ntry smallest focuses, a focus of a size found first before those
    # of its size found later (as a stable sort of all of them would order
    # them), without repeats
    Us = Vector{Int}[]

    # the scope: a random PMC X and the largest component C of G - X
    p = rand(srh.rng, oneto(hbt_len(sol)))
    cs = sol.cfst[p]:sol.cfst[p + 1] - 1

    if !isempty(cs)
        c = argmax(c -> ctx.bsize[sol.csub[c]], cs)
        ss = hbt_stp!(ctx)

        for x in hbt_pmc(sol, p)
            mark1[x] = ss
        end

        v0 = ctx.bmin[sol.csub[c]]
        empty!(queue); push!(queue, v0); mark1[v0] = ss; head = 1

        while head <= length(queue)
            u = queue[head]; head += 1

            for x in neighbors(g, u)
                if mark1[x] != ss
                    mark1[x] = ss; push!(queue, x)
                end
            end
        end

        # the smallest vertex outside the scope
        o = 0

        for v in oneto(n)
            if mark1[v] != ss
                o = v
                break
            end
        end

        if ispositive(o) && ispositive(ntry)
            k = st.width[][1]; U = Int[]

            for q in oneto(hbt_len(side.sol))
                side.sol.pwgt[q] < k || continue
                Y = hbt_pmc(side.sol, q)
                all(y -> mark1[y] == ss, Y) || continue

                # the focus: N[D] ∩ scope for the component D of G - Y containing o
                sy = hbt_stp!(ctx)

                for y in Y
                    mark2[y] = sy
                end

                # (a focus as large as the largest of ntry kept ones is not
                # kept, so its search stops there)
                bound = length(Us) < ntry ? typemax(Int) : length(last(Us))
                empty!(U); empty!(queue); push!(queue, o); mark3[o] = sy; head = 1

                while head <= length(queue) && length(U) < bound
                    u = queue[head]; head += 1
                    mark1[u] == ss && push!(U, u)

                    for x in neighbors(g, u)
                        if mark2[x] == sy
                            if mark3[x] != sy
                                mark3[x] = sy; push!(U, x)
                            end
                        elseif mark3[x] != sy
                            mark3[x] = sy; push!(queue, x)
                        end
                    end
                end

                length(U) < bound || continue
                sort!(U; alg = QuickSort)
                any(F -> F == U, Us) && continue
                i = searchsortedlast(Us, U; by = length)
                insert!(Us, i + 1, copy(U))
                length(Us) > ntry && pop!(Us)
            end
        end
    end

    if isnothing(pool)
        for U in Us
            hbt_expired(srh) && break
            hbt_lcl!(srh, st, side, U)
        end
    else
        hbt_lcp!(srh, st, side, pool, Us)
    end

    for q in oneto(hbt_len(side.sol))
        hbt_cpy!(sol, side.sol, q)
    end

    return hbt_upd!(srh, st)
end

# Merge with an independent solution of no greater width, improving that
# solution first if necessary (recursively, with the same strategies).
function hbt_imm!(srh::NamedTuple, st::NamedTuple, pool = nothing)
    side = hbt_side!(srh, st)

    if side.width[][1] > st.width[][1]
        hbt_imp!(srh, side, pool)
        return false
    else
        return hbt_mrg!(srh, st, side, pool)
    end
end

# One improvement step. With both strategies on, a solution is diversified,
# and merged with an independent solution only once it has gone patience
# steps without progress. In a parallel search (pool), the step adds every
# diversification that the workers have made of the solution since the last
# step, and counts as that many steps.
function hbt_imp!(srh::NamedTuple, st::NamedTuple, pool = nothing)
    k = st.width[][1]; n = 1

    if srh.par.diversify
        rs = hbt_take!(srh, st, pool)

        if isempty(rs)
            hbt_dvs!(srh, st)
        else
            for r in rs
                hbt_put!(srh, st, r)
            end

            n = length(rs)
        end

        improved = hbt_upd!(srh, st)
    else
        improved = false
    end

    improved = hbt_end!(srh, st, k, improved, n, pool)
    hbt_pub!(pool, st, improved)
    return improved
end

# Merge if the solution has stalled, and count the steps without progress.
function hbt_end!(srh::NamedTuple, st::NamedTuple, k, improved::Bool, n::Int = 1, pool = nothing)
    par = srh.par

    if par.merge && !hbt_expired(srh) && (!par.diversify || st.stall[] >= par.patience)
        improved |= hbt_imm!(srh, st, pool)
    end

    st.stall[] = improved ? 0 : st.stall[] + n
    st.kstall[] = st.width[][1] < k ? 0 : st.kstall[] + n
    return improved
end

# ---------------------------------------------------------------------------
# Diversification (Tamaki 2019, Section 5.1)
#
# Pick a largest bag X₀ of the current best decomposition T and a random
# subtree R of T containing it. Let U be the union of the bags of R and H the
# local graph on U; R is a decomposition of H, so a better decomposition of H
# would give a better decomposition of G. To steer away from R, fill a minimal
# separator S of H that crosses X₀ (so X₀ is not a maximal clique of any
# minimal triangulation of H + S) and triangulate H + S; a minimal
# triangulation of H + S is a minimal triangulation of H (Proposition 3.2), so
# its maximal cliques are PMCs of H, hence of G (Proposition 3.6). This is
# repeated for several separators.
# ---------------------------------------------------------------------------

# One diversification of the snapshot snp of a solution: returns the local
# graph and its cliques (cptr, ctgt) to be added to the solution (hbt_adc!).
function hbt_div(srh::NamedTuple, snp::NamedTuple)
    hbt_cut!(srh, snp.kstall)
    ctx = srh.ctx; par = srh.par; g = ctx.graph; rng = srh.rng; wgt = ctx.wgt; W = eltype(wgt)
    bptr = snp.bptr; btgt = snp.btgt; parent = snp.parent; nt = length(parent)
    mark1 = ctx.mark1; mark2 = ctx.mark2; loc = ctx.loc
    srh.nstep[] += 1

    # a random bag among the heaviest: of a weight in the band of width u (the
    # lightest weight) below the largest weight k with probability 1/2, in
    # the band below that with probability 1/4, and so on, down to near
    # bands (re-triangulating near the largest bags makes room for removing
    # them later); with unit weights, the bands are the weights k, k - 1, ...
    k = snp.width[1]; u = minimum(wgt); tol = tolerance(W); delta = 0

    while delta < par.near && rand(rng, Bool)
        delta += 1
    end

    top = Int[]; bw = W[]

    for i in oneto(nt)
        w = zero(W)

        for v in view(btgt, bptr[i]:bptr[i + 1] - 1)
            w += wgt[v]
        end

        push!(bw, w)
    end

    # the bands are anchored at the heaviest bag as summed here, not at k: with
    # float weights the two sums can differ in the last bits, and a bag just
    # heavier than k would fall between two bands (and the walk below would
    # never end). k stays the width of the solution for the rest of the function.
    kmax = maximum(bw)

    while isempty(top) && !isnegative(delta)
        hi = kmax - delta * u; lo = hi - u

        for i in oneto(nt)
            lo + tol <= bw[i] < hi + tol && push!(top, i)
        end

        delta -= 1
    end

    # outside the supported weights (u below the tolerance, or lost to rounding
    # next to kmax) every band can be empty
    isempty(top) && push!(top, argmax(bw))

    i0 = rand(rng, top); X0 = view(btgt, bptr[i0]:bptr[i0 + 1] - 1)

    # the tree, as lists of neighbors
    nptr = zeros(Int, nt + 1)

    for i in oneto(nt)
        p = parent[i]

        if ispositive(p)
            nptr[i + 1] += 1; nptr[p + 1] += 1
        end
    end

    nptr[1] = 1

    for i in oneto(nt)
        nptr[i + 1] += nptr[i]
    end

    ntgt = Vector{Int}(undef, nptr[nt + 1] - 1); pos = nptr[1:nt]

    for i in oneto(nt)
        p = parent[i]

        if ispositive(p)
            ntgt[pos[i]] = p; pos[i] += 1; ntgt[pos[p]] = i; pos[p] += 1
        end
    end

    # a random subtree containing X₀, grown until its vertex set reaches a
    # random target size: with probability pexact, just above |X₀| (up to
    # margin more vertices), where exact triangulation is fast even for wide
    # bags; otherwise log-uniform between |X₀| and dsize, so that every scale
    # gets tried
    lo = length(X0)

    if rand(rng) < par.pexact
        target = lo + rand(rng, 1:max(1, par.margin))
    else
        hi = max(lo, min(par.dsize, nv(g)))
        target = round(Int, exp(log(lo) + rand(rng) * (log(hi) - log(lo))))
    end

    su = hbt_stp!(ctx); sn = hbt_stp!(ctx); U = Int[]

    for v in X0
        mark1[v] = su; push!(U, v)
    end

    # (mark2 is indexed by bag here)
    mark2[i0] = sn; frontier = ntgt[nptr[i0]:nptr[i0 + 1] - 1]

    for i in frontier
        mark2[i] = sn
    end

    while !isempty(frontier) && length(U) < target
        r = rand(rng, eachindex(frontier))
        i = frontier[r]; frontier[r] = frontier[end]; pop!(frontier)
        bag = view(btgt, bptr[i]:bptr[i + 1] - 1); grow = 0

        for v in bag
            mark1[v] == su || (grow += 1)
        end

        length(U) + grow > target && continue

        for v in bag
            if mark1[v] != su
                mark1[v] = su; push!(U, v)
            end
        end

        for j in view(ntgt, nptr[i]:nptr[i + 1] - 1)
            if mark2[j] != sn
                mark2[j] = sn; push!(frontier, j)
            end
        end
    end

    sort!(U; alg = QuickSort)
    L = hbt_loc(ctx, U); H = L.graph; nu = length(U); weights = L.wgt

    # X₀ in local numbering (loc is set by hbt_loc)
    x0 = Int[loc[v] for v in X0]
    lmark = zeros(Int, nu); limit = hbt_limit(srh, length(X0))
    cptr = Int[1]; ctgt = Int[]

    # a region small enough to be solved exactly is also triangulated as it
    # is: septw finds a decomposition of H below the current width if there
    # is one. If there is none, it finds the treewidth of H, which bounds
    # that of H + S below: then septw need not try the widths below it again
    # for H + S (where they cost an exhaustive search each, and fail).
    lbS = zero(W)

    if nu <= limit
        order = hbt_exa(srh, weights, H, hbt_below(k))

        if isnothing(order)
            hbt_cls!(cptr, ctgt, srh, L, H, hbt_grd(srh, weights, H), k)
        elseif !ismissing(order)
            w = hbt_cls!(cptr, ctgt, srh, L, H, order, k)
            w >= k && (lbS = w)
        end
    end

    # minimal separators crossing X₀: for nonadjacent a, b ∈ X₀, the
    # neighborhood of the component of H - N[a] containing b
    tried = bt_tab()

    # H + S, made in the same arrays of the context for every S (it is done
    # with when the next is made)
    ptr = resize!(ctx.hsp, nu + 1); tgt = ctx.hst

    for _ in oneto(par.nsep)
        hbt_expired(srh) && break
        a = 0; b = 0

        for _ in 1:20
            a1 = rand(rng, x0); b1 = rand(rng, x0)

            if a1 != b1 && !insorted(b1, neighbors(H, a1))
                a = a1; b = b1
                break
            end
        end

        iszero(a) && break
        S = hbt_msp(ctx, H, a, b, lmark)
        last(bt_tab_add!(tried, S)) || continue

        # H + S
        empty!(tgt); sizehint!(tgt, de(H) + length(S)^2; shrink = false); ptr[1] = 1
        sm = hbt_stp!(ctx)

        for v in S
            lmark[v] = sm
        end

        for v in oneto(nu)
            start = length(tgt) + 1
            append!(tgt, neighbors(H, v))

            if lmark[v] == sm
                for x in S
                    x == v || push!(tgt, x)
                end

                sort!(view(tgt, start:length(tgt)); alg = QuickSort); m = start - 1

                for i in start:length(tgt)
                    if m < start || tgt[i] != tgt[m]
                        m += 1; tgt[m] = tgt[i]
                    end
                end

                resize!(tgt, m)
            end

            ptr[v + 1] = length(tgt) + 1
        end

        HS = BipartiteGraph{Int, Int}(nu, nu, length(tgt), ptr, tgt)
        order = hbt_tri(srh, weights, HS, hbt_below(k), limit; lb = lbS)
        ismissing(order) || hbt_cls!(cptr, ctgt, srh, L, HS, order, k)
    end

    return L, cptr, ctgt
end

# Diversify the solution once, in this task.
function hbt_dvs!(srh::NamedTuple, st::NamedTuple)
    L, cptr, ctgt = hbt_div(srh, hbt_snp(st))
    hbt_adc!(srh, st, L, cptr, ctgt)
    return
end

# A minimal separator of H separating nonadjacent vertices a and b: the
# neighborhood of the component of H - N[a] that contains b.
function hbt_msp(ctx::NamedTuple, H::AbstractGraph, a::Int, b::Int, lmark::Vector{Int})
    queue = ctx.queue; sa = hbt_stp!(ctx); lmark[a] = sa

    for v in neighbors(H, a)
        lmark[v] = sa
    end

    sb = hbt_stp!(ctx); S = Int[]
    empty!(queue); push!(queue, b); lmark[b] = sb; head = 1

    while head <= length(queue)
        u = queue[head]; head += 1

        for x in neighbors(H, u)
            if lmark[x] == sa
                # in N(a): on the separator
                lmark[x] = sb; push!(S, x)
            elseif lmark[x] != sb
                lmark[x] = sb; push!(queue, x)
            end
        end
    end

    return sort!(S; alg = QuickSort)
end

function hbt_stats(srh::NamedTuple)
    return srh.nstep[], srh.nlocal[], srh.nexact[], srh.ntimeout[]
end

# ---------------------------------------------------------------------------
# Workers
#
# With nt tasks, the main task improves the solution as hbt_imp! does, and
# nt - 1 workers do the costly parts of its steps: they diversify the latest
# snapshot of each solution that the main task is improving (the solution,
# and the solutions merged with it while it is stalled), again and again,
# each with a context and random numbers of its own, and lift the cliques to
# PMCs (hbt_lif). A step of the main task adds all the diversifications made
# since its last step (hbt_put!), and when it merges, the workers triangulate
# the local graphs of the merge with it. At most cap diversifications of a
# solution wait or are being made, so that a main task that falls behind
# holds the workers back instead of a growing heap of stale results. When no
# diversification is waiting and none is being made, the main task makes one
# itself. Returns the statistics of the workers.
# ---------------------------------------------------------------------------

function hbt_par!(srh::NamedTuple, st::NamedTuple, lb, ub, nt::Integer)
    ctx = srh.ctx; pool = hbt_pool(nt)
    seeds = [rand(srh.rng, UInt64) for _ in 2:nt]
    workers = Vector{Any}(nothing, nt - 1)

    # each worker makes its context on its own thread
    tasks = map(eachindex(seeds)) do i
        return Threads.@spawn begin
            w = hbt_srh(hbt_ctx(ctx.graph, ctx.wgt), Xoshiro(seeds[i]), srh.par)
            w.deadline[] = srh.deadline[]; w.t0[] = srh.t0[]; workers[i] = w
            withworkspace(() -> hbt_wrk(w, pool))
        end
    end

    try
        withworkspace() do
            while !hbt_expired(srh) && hbt_going(st, lb, ub) && !pool.stop[]
                @lock pool.lock pool.round[] += 1
                hbt_imp!(srh, st, pool)
            end
        end
    finally
        @lock pool.lock begin
            pool.stop[] = true; notify(pool.wake); notify(pool.ready)
        end

        foreach(wait, tasks)
    end

    stats = (0, 0, 0, 0)

    for w in workers
        stats = stats .+ hbt_stats(w)
    end

    return stats
end

# The pool of a parallel search, guarded by lock: the latest snapshot of each
# solution, by depth (snaps), the main task's last round of steps that wanted
# its diversifications (want), these diversifications (queue), and the number
# being made (busy); the local graphs of the current merge (jobs), the next
# to take, the number done, and their cliques (jres). Workers wait on wake for
# something to do, and the main task on ready for results.
function hbt_pool(nt::Integer)
    lock = ReentrantLock()

    return (;
        lock, wake = Threads.Condition(lock), ready = Threads.Condition(lock),
        cap = 2 * (Int(nt) - 1), stop = Ref(false), round = Ref(0),
        snaps = Any[], want = Int[], queue = Vector{Any}[], busy = Int[],
        jobs = Any[], next = Ref(1), done = Ref(0), jres = Any[],
    )
end

function hbt_wrk(w::NamedTuple, pool::NamedTuple)
    try
        while !hbt_expired(w)
            job = hbt_job!(pool)
            isnothing(job) && break
            i, x = job

            if ispositive(i)
                # the local graph of a merge
                r = nothing

                if !hbt_expired(w)
                    hbt_cut!(w, x.kstall)
                    r = hbt_lif(w, hbt_lcc(w, x.U, x.k, x.limit)...)
                end

                @lock pool.lock begin
                    pool.jres[i] = r; pool.done[] += 1; notify(pool.ready)
                end
            else
                # a diversification of the solution at depth -i - 1
                r = hbt_lif(w, hbt_div(w, x)...)

                @lock pool.lock begin
                    push!(pool.queue[-i], r); pool.busy[-i] -= 1; notify(pool.ready)
                end
            end
        end
    catch
        # the main task waits for no more results
        @lock pool.lock begin
            pool.stop[] = true; notify(pool.wake); notify(pool.ready)
        end

        rethrow()
    end

    return
end

# The next job of a worker, waiting for one: (i, x) for the local graph x of
# the merge, the ith, or (-d, x) for a diversification of the snapshot x of
# the solution at depth d - 1, the one with the fewest diversifications
# waiting or being made, of the solutions wanted in the last two rounds;
# nothing once the search stops.
function hbt_job!(pool::NamedTuple)
    @lock pool.lock begin
        while !pool.stop[]
            i = pool.next[]

            if i <= length(pool.jobs)
                pool.next[] = i + 1
                return (i, pool.jobs[i])
            end

            d = 0; m = pool.cap

            for e in eachindex(pool.snaps)
                if pool.want[e] >= pool.round[] - 1
                    f = length(pool.queue[e]) + pool.busy[e]

                    if f < m
                        d = e; m = f
                    end
                end
            end

            if ispositive(d)
                pool.busy[d] += 1
                return (-d, pool.snaps[d])
            end

            wait(pool.wake)
        end
    end

    return nothing
end

# The diversifications of the solution made by the workers since the last
# step (none: the main task makes one itself), and want them in this round.
function hbt_take!(srh::NamedTuple, st::NamedTuple, pool::NamedTuple)
    d = st.depth + 1

    @lock pool.lock begin
        while length(pool.snaps) < d
            push!(pool.snaps, nothing); push!(pool.want, -1); push!(pool.queue, Any[]); push!(pool.busy, 0)
        end
    end

    isnothing(pool.snaps[d]) && hbt_pub!(pool, st, true)

    @lock pool.lock begin
        fresh = pool.want[d] < pool.round[] - 1; pool.want[d] = pool.round[]
        rs = pool.queue[d]
        isempty(rs) || (pool.queue[d] = Any[])
        (fresh || !isempty(rs)) && notify(pool.wake)
        return rs
    end
end

hbt_take!(srh::NamedTuple, st::NamedTuple, pool::Nothing) = ()

# Publish the snapshot of a solution after a step: its decomposition is copied
# only when it changes.
function hbt_pub!(pool::NamedTuple, st::NamedTuple, improved::Bool)
    d = st.depth + 1
    snap = improved ? hbt_cop(st) : pool.snaps[d]
    snap = merge(snap, (; kstall = st.kstall[]))
    @lock pool.lock pool.snaps[d] = snap
    return
end

hbt_pub!(pool::Nothing, st::NamedTuple, improved::Bool) = nothing

# A snapshot of the decomposition of a solution, apart from it.
hbt_cop(st::NamedTuple) = (; bptr = copy(st.bptr), btgt = copy(st.btgt), parent = copy(st.parent), width = st.width[])

# Triangulate the local graphs on the sets Us, with the workers, and add their
# PMCs to the solution, in order, as hbt_lcl! does.
function hbt_lcp!(srh::NamedTuple, st::NamedTuple, side::NamedTuple, pool::NamedTuple, Us::Vector{Vector{Int}})
    k = st.width[][1]; limit = hbt_limit(srh, st.maxbag[]); kstall = st.kstall[]; m = length(Us)

    @lock pool.lock begin
        resize!(pool.jres, m); fill!(pool.jres, nothing)

        for U in Us
            push!(pool.jobs, (; U, k, limit, kstall))
        end

        notify(pool.wake)
    end

    # this task takes jobs too, then waits for the others
    while true
        i = @lock pool.lock begin
            j = pool.next[]; j <= m && (pool.next[] = j + 1); j
        end

        i > m && break
        r = hbt_expired(srh) ? nothing : hbt_lif(srh, hbt_lcc(srh, Us[i], k, limit)...)

        @lock pool.lock begin
            pool.jres[i] = r; pool.done[] += 1
        end
    end

    rs = @lock pool.lock begin
        while pool.done[] < m && !pool.stop[]
            wait(pool.ready)
        end

        rs = copy(pool.jres); empty!(pool.jobs); empty!(pool.jres); pool.next[] = 1; pool.done[] = 0
        rs
    end

    for r in rs
        isnothing(r) || hbt_put!(srh, st, r, side)
    end

    return
end

# ---------------------------------------------------------------------------
# The anytime search
# ---------------------------------------------------------------------------

# A search for a tree decomposition of a connected graph, with its first
# solution: the PMCs of the best of ninit greedy triangulations. hbtstep!
# improves it once, hbtrun! until the time runs out or the width meets the
# lower bound lb; hbtord gives the best ordering and its width, and hbtpmc
# the PMCs of the solution, as a relation from the PMCs to the vertices.
function hbtnew(weights::AbstractVector, graph::AbstractGraph;
        seed::Integer = 1, alg::EliminationAlgorithm = MF(; strategy = 1), lb::WidthOrAlgorithm = DEFAULT_LOWER_BOUND_ALGORITHM,
        ninit::Integer = 10, base::Integer = 60, ntry::Integer = 50, refined::Bool = true, merge::Bool = true,
        diversify::Bool = true, dsize::Integer = typemax(Int), nsep::Integer = 4, patience::Integer = 100,
        xtime::Real = 1.0, xmin::Real = 0.05, xstep::Integer = 8, near::Integer = 3, margin::Integer = 40,
        pexact::Real = 0.5, init = nothing, deadline::Real = Inf, verbose::Bool = false)
    bgraph, wgt = hbt_graph(weights, graph); W = eltype(wgt)
    par = (; alg, ninit = Int(ninit), base = Int(base), ntry = Int(ntry), refined, merge, diversify,
        dsize = Int(dsize), nsep = Int(nsep), patience = Int(patience), xtime = Float64(xtime),
        xmin = Float64(xmin), xstep = Int(xstep), near = Int(near), margin = Int(margin), pexact = Float64(pexact), init, verbose)
    srh = hbt_srh(hbt_ctx(bgraph, wgt), Xoshiro(seed), par)
    # the deadline (of the caller's budget) cuts the greedy starts short; the
    # first solution is always made
    srh.deadline[] = deadline
    hbt_ini!(srh, 0)
    srh.deadline[] = Inf
    lbw = iszero(nv(bgraph)) ? zero(W) : convert(W, lowerbound(wgt, bgraph, lb))
    verbose && println("hbt: n = $(nv(bgraph)), initial width $(hbt_str(srh.sols[1].width[][1])) at $(round(time() - srh.t0[]; digits = 2)) s")
    return (; srh, lb = lbw)
end

# One improvement step; returns whether the solution improved.
function hbtstep!(hbt::NamedTuple)
    srh = hbt.srh; isinf(srh.deadline[]) || (srh.deadline[] = Inf)
    return hbt_imp!(srh, srh.sols[1])
end

# Improve the solution for at most mt seconds, until its width meets the
# lower bound lb, or until it falls below ub (if ub is not nothing), with nt
# tasks (see hbt_par!); returns hbtord(hbt).
function hbtrun!(hbt::NamedTuple; mt::Real = 10.0, nt::Integer = Threads.nthreads(), lb = hbt.lb, ub = nothing)
    srh = hbt.srh; st = srh.sols[1]; srh.deadline[] = time() + mt
    stats = (0, 0, 0, 0)

    if nt > 1 && srh.par.diversify
        stats = hbt_par!(srh, st, lb, ub, nt)
    else
        withworkspace() do
            while !hbt_expired(srh) && hbt_going(st, lb, ub)
                hbt_imp!(srh, st)
            end
        end
    end

    if srh.par.verbose
        nstep, nlocal, nexact, ntimeout = hbt_stats(srh) .+ stats
        println("hbt: n = $(nv(srh.ctx.graph)), width = $(hbt_str(st.width[][1])) ($(st.width[][2])), steps = $nstep, local = $nlocal, exact = $nexact, timeouts = $ntimeout, pmcs = $(hbt_len(st.sol)), time = $(round(time() - srh.t0[]; digits = 1))")
    end

    srh.deadline[] = Inf
    return hbtord(hbt)
end

# Whether to keep improving a solution: its width is above the lower bound
# lb, and not below ub.
function hbt_going(st::NamedTuple, lb, ub)
    k = st.width[][1]
    return k > lb && (isnothing(ub) || k >= ub)
end

# The best ordering, its inverse, and its width.
function hbtord(hbt::NamedTuple)
    st = hbt.srh.sols[1]; order = copy(st.order)
    return order, invperm(order), st.width[][1]
end

# The PMCs of the solution, as a relation from the PMCs to the vertices.
function hbtpmc(hbt::NamedTuple)
    sol = hbt.srh.sols[1].sol; tab = sol.pmcs; n = nv(hbt.srh.ctx.graph); np = hbt_len(sol)
    return BipartiteGraph{Int, Int}(n, np, length(tab.tgt), copy(tab.ptr), copy(tab.tgt))
end

# ---------------------------------------------------------------------------
# Batches
#
# A batch of graphs (see batch_permutation) is one problem whose width is
# that of its widest graph: the leaves of a safe separator decomposition, or
# the components of a graph. HBT spends one time budget on the batch, on a
# widest graph until it falls below the next widest, as Tamaki's reference
# implementation does with the atoms of a graph (main.UpLow): improving a
# graph that is not the widest does not lower the width. A graph that is
# never the widest keeps its first ordering.
#
# Each graph starts with the best of ninit greedy orderings (and that of
# init, if init is an elimination algorithm), as the first solution of a
# search does. When it is first the widest, a graph with at most base
# vertices is solved exactly, with at most half the remaining time, as a
# small graph is; its width is then a lower bound on that of the batch. Any
# other graph gets an anytime search (hbtnew), which it keeps between turns.
# The search makes its own first solution: seeding it with the graph's
# ordering as well would add a second triangulation to that solution, and
# the components of the bags of a triangulation are the costly part of
# making one. The run stops early once the widest graph meets the largest
# lower bound of a graph.
# ---------------------------------------------------------------------------

function batch_permutation(weights::AbstractVector{<:AbstractVector{W}}, graphs::AbstractVector{<:AbstractGraph{V}}, alg::HBT; verbose::Bool = false) where {W, V}
    dl = time() + alg.time; ngraph = length(graphs)
    orders = Vector{Vector{V}}(undef, ngraph)
    widths = Vector{W}(undef, ngraph)
    lowers = Vector{W}(undef, ngraph)
    inits = Vector{Union{Nothing, Vector{V}}}(nothing, ngraph)
    tried = falses(ngraph)
    searches = Vector{Any}(nothing, ngraph)
    grd = (; rng = Xoshiro(alg.seed), par = (; alg = alg.alg))

    # an ordering given as init is one of a single graph, so it is not used
    ninit = alg.init isa EliminationAlgorithm ? alg.ninit : max(1, alg.ninit)

    # the best of the triangulation by init and ninit greedy ones, as for
    # the first solution of a search
    for i in oneto(ngraph)
        weight = weights[i]; graph = graphs[i]
        lowers[i] = convert(W, lowerbound(weight, graph, alg.lb))
        widths[i] = typemax(W)

        if alg.init isa EliminationAlgorithm
            order, _ = permutation(weight, graph, MinimalChordal(alg.init))
            inits[i] = orders[i] = order
            widths[i] = treewidth(weight, graph, order)
        end

        for k in oneto(ninit)
            widths[i] <= lowers[i] && break
            # every graph gets one greedy ordering, the others only in time
            k > 1 && isassigned(orders, i) && time() > dl && break
            order = hbt_grd(grd, weight, graph)
            width = treewidth(weight, graph, order)

            if width < widths[i]
                orders[i] = order; widths[i] = width
            end
        end
    end

    lb = maximum(lowers; init = zero(W))

    if alg.verbose && ispositive(ngraph)
        n = maximum(nv, graphs)
        println("hbt: $ngraph graphs, the largest with $n vertices; width $(hbt_str(maximum(widths))), lower bound $(hbt_str(lb))")
    end

    while ispositive(ngraph) && time() < dl
        k, i = findmax(widths)
        k <= lb && break

        # the width of the next widest graph
        ub = nothing

        for j in oneto(ngraph)
            j == i || (ub = isnothing(ub) ? widths[j] : max(ub, widths[j]))
        end

        weight = weights[i]; graph = graphs[i]

        if !tried[i] && nv(graph) <= alg.base
            tried[i] = true
            bgraph, wgt = hbt_graph(weight, graph)
            order, status = hbt_safe(wgt, bgraph, lowers[i], lowers[i], typemax(W), time() + (dl - time()) / 2)

            if status == BT_FOUND
                orders[i] = convert(Vector{V}, order)
                widths[i] = lowers[i] = treewidth(weight, graph, orders[i])
                lb = max(lb, lowers[i])
            end
        else
            if isnothing(searches[i])
                searches[i] = hbt_new(weight, graph, alg; lb = lowers[i], init = inits[i], deadline = dl)
            end

            order, _, width = hbtrun!(searches[i]; mt = dl - time(), nt = alg.threads, lb, ub)

            if width < widths[i]
                orders[i] = convert(Vector{V}, order); widths[i] = width
            end
        end
    end

    return orders
end

# An anytime search with the parameters of alg.
function hbt_new(weights::AbstractVector, graph::AbstractGraph, alg::HBT; seed::Integer = alg.seed, lb = alg.lb, init = alg.init, deadline::Real = Inf)
    return hbtnew(weights, graph; seed, alg = alg.alg, lb, ninit = alg.ninit, deadline,
        base = alg.base, ntry = alg.ntry, refined = alg.refined, merge = alg.merge, diversify = alg.diversify,
        dsize = alg.dsize, nsep = alg.nsep, patience = alg.patience, xtime = alg.xtime, xmin = alg.xmin,
        xstep = alg.xstep, near = alg.near, margin = alg.margin, pexact = alg.pexact, init, verbose = alg.verbose)
end
