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

function bt_tab_hsh(x::AbstractVector{Int})
    h = 0xcbf29ce484222325

    @inbounds for v in x
        h = (h ⊻ (v % UInt64)) * 0x00000100000001b3
    end

    return h ⊻ (h >> 29)
end

function bt_tab_eq(tab::NamedTuple, i::Int, x::AbstractVector{Int})
    lo = tab.ptr[i]; hi = tab.ptr[i + 1]
    hi - lo == length(x) || return false

    @inbounds for k in eachindex(x)
        tab.tgt[lo + k - 1] == x[k] || return false
    end

    return true
end

# The index of the sequence x, or zero.
function bt_tab_find(tab::NamedTuple, x::AbstractVector{Int})
    slot = tab.slot; m = length(slot) - 1; s = (bt_tab_hsh(x) % Int) & m

    @inbounds while true
        i = slot[s + 1]
        iszero(i) && return 0
        bt_tab_eq(tab, i, x) && return i
        s = (s + 1) & m
    end
end

# The index of the sequence x, added if it is new, and whether it is.
function bt_tab_add!(tab::NamedTuple, x::AbstractVector{Int})
    i = bt_tab_find(tab, x)
    ispositive(i) && return i, false
    append!(tab.tgt, x); push!(tab.ptr, length(tab.tgt) + 1)
    i = bt_tab_len(tab)

    if 2i > length(tab.slot)
        bt_tab_rhs!(tab, 2length(tab.slot))
    else
        bt_tab_slt!(tab, i)
    end

    return i, true
end

function bt_tab_slt!(tab::NamedTuple, i::Int)
    slot = tab.slot; m = length(slot) - 1; s = (bt_tab_hsh(bt_tab_seq(tab, i)) % Int) & m

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

        sort!(sep)
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
            sort!(view(btgt, lo:length(btgt))); push!(bptr, length(btgt) + 1); push!(parent, pn)
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

        sort!(nb)
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

        ptr = Vector{Int}(undef, nu + 1); tgt = Int[]; ptr[1] = 1

        @inbounds for i in oneto(nu)
            for k in oneto(nwrd)
                word = bt_get(adj, k, i)

                while !iszero(word)
                    push!(tgt, bt_vtx(Int, k, word)); word &= word - one(UInt64)
                end
            end

            ptr[i + 1] = length(tgt) + 1
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
            lo = ptr[i]; hi = ptr[i + 1] - 1; sort!(view(tgt, lo:hi)); ptr[i] = q

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
    )
end

# Is the set K (local, sorted) the neighborhood of a component of G - U?
hbt_isnbr(L::NamedTuple, K::AbstractVector{Int}) = ispositive(bt_tab_find(L.nbrs, K))

# Add the PMC U[K] of G, for a PMC K (sorted) of the local graph, to the
# solution, unless it is there; returns its index. The components of
# G - U[K] are the components Q of H - K, each joined with the components
# B of G - U with ∅ ≠ N(B) - K ⊆ Q, and the components B of G - U with
# N(B) ⊆ K. The separator of the former is N_H(Q) ∩ K = N_H(Q). Linear in
# the size of the local graph.
function hbt_lft!(ctx::NamedTuple, sol::NamedTuple, L::NamedTuple, K::AbstractVector{Int})
    U = L.U; X = L.X; empty!(X)

    @inbounds for k in K
        push!(X, U[k])
    end

    p = bt_tab_find(sol.pmcs, X)
    ispositive(p) && return p

    lmark = L.lmark; lcomp = L.lcomp; cmin = L.cmin; csize = L.csize; cwgt = L.cwgt
    qptr = L.qptr; qtgt = L.qtgt; csep = L.csep
    empty!(cmin); empty!(csize); empty!(cwgt); resize!(qptr, 1); empty!(qtgt); empty!(csep)
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

        sort!(sep)
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
    n = nv(graph); perm = randperm(srh.rng, n); invp = invperm(perm)
    ptr = Vector{Int}(undef, n + 1); tgt = Vector{Int}(undef, de(graph)); ptr[1] = 1

    @inbounds for i in oneto(n)
        k = ptr[i]

        for x in neighbors(graph, perm[i])
            tgt[k] = invp[x]; k += 1
        end

        sort!(view(tgt, ptr[i]:k - 1)); ptr[i + 1] = k
    end

    pgraph = BipartiteGraph{Int, Int}(n, n, ptr[n + 1] - 1, ptr, tgt)
    order, _ = permutation(weights[perm], pgraph, srh.par.alg)
    order, _ = permutation(weights, graph, MinimalChordal(perm[order]))
    return order
end

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
    label, tree = cliquetree(weights, HS, order); width = zero(W)

    for clique in tree
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

        K = view(ctgt, lo:length(ctgt)); sort!(K)

        if hbt_isnbr(L, K)
            resize!(ctgt, lo - 1)
        else
            push!(cptr, length(ctgt) + 1)
        end
    end

    return width
end

# Add the PMCs U[K] of G, for the cliques K of the local graph, to the
# solution; a PMC that the solution side has is copied from it.
function hbt_adc!(srh::NamedTuple, st::NamedTuple, L::NamedTuple, cptr::Vector{Int}, ctgt::Vector{Int}, side::Union{Nothing, NamedTuple} = nothing)
    ctx = srh.ctx; sol = st.sol; U = L.U; X = L.X

    for c in oneto(length(cptr) - 1)
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

        hbt_lft!(ctx, sol, L, K)
    end

    return
end

# Triangulate the local graph on U and add the resulting PMCs of G to the
# solution, if the triangulation is no wider than the solution.
function hbt_lcl!(srh::NamedTuple, st::NamedTuple, side::NamedTuple, U::Vector{Int})
    L = hbt_loc(srh.ctx, U); H = L.graph; weights = L.wgt
    k = st.width[][1]
    order = hbt_tri(srh, weights, H, k, hbt_limit(srh, st.maxbag[]); cap = k)
    ismissing(order) && return
    cptr = Int[1]; ctgt = Int[]
    hbt_cls!(cptr, ctgt, srh, L, H, order, typemax(eltype(weights))) <= k || return
    hbt_adc!(srh, st, L, cptr, ctgt, side)
    return
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
# a solution: they are PMCs.
function hbt_cqs!(ctx::NamedTuple, sol::NamedTuple, label::AbstractVector, tree)
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

# The solution merged with st.
function hbt_side!(srh::NamedTuple, st::NamedTuple)
    d = st.depth + 1
    return d < length(srh.sols) ? srh.sols[d + 1] : hbt_ini!(srh, d)
end

# Merge (Tamaki 2022): pairs of PMCs X of st and Y of side, where side has no
# greater width, pick out small focus sets U, whose local graphs are
# triangulated; their PMCs and those of side are added to st.
function hbt_mrg!(srh::NamedTuple, st::NamedTuple, side::NamedTuple)
    hbt_cut!(srh, st.kstall[])
    ctx = srh.ctx; g = ctx.graph; n = nv(g); sol = st.sol
    mark1 = ctx.mark1; mark2 = ctx.mark2; mark3 = ctx.mark3; queue = ctx.queue
    srh.nstep[] += 1
    focuses = bt_tab()

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

        if ispositive(o)
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

                empty!(U); empty!(queue); push!(queue, o); mark3[o] = sy; head = 1

                while head <= length(queue)
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

                bt_tab_add!(focuses, sort!(U))
            end
        end
    end

    nf = bt_tab_len(focuses)
    list = sortperm(oneto(nf); by = f -> focuses.ptr[f + 1] - focuses.ptr[f])

    for i in oneto(min(nf, srh.par.ntry))
        hbt_expired(srh) && break
        hbt_lcl!(srh, st, side, collect(bt_tab_seq(focuses, list[i])))
    end

    for q in oneto(hbt_len(side.sol))
        hbt_cpy!(sol, side.sol, q)
    end

    return hbt_upd!(srh, st)
end

# Merge with an independent solution of no greater width, improving that
# solution first if necessary (recursively, with the same strategies).
function hbt_imm!(srh::NamedTuple, st::NamedTuple)
    side = hbt_side!(srh, st)

    if side.width[][1] > st.width[][1]
        hbt_imp!(srh, side)
        return false
    else
        return hbt_mrg!(srh, st, side)
    end
end

# One improvement step. With both strategies on, a solution is diversified,
# and merged with an independent solution only once it has gone patience
# steps without progress.
function hbt_imp!(srh::NamedTuple, st::NamedTuple)
    k = st.width[][1]

    if srh.par.diversify
        hbt_dvs!(srh, st)
        improved = hbt_upd!(srh, st)
    else
        improved = false
    end

    return hbt_end!(srh, st, k, improved)
end

# Merge if the solution has stalled, and count the steps without progress.
function hbt_end!(srh::NamedTuple, st::NamedTuple, k, improved::Bool)
    par = srh.par

    if par.merge && !hbt_expired(srh) && (!par.diversify || st.stall[] >= par.patience)
        improved |= hbt_imm!(srh, st)
    end

    st.stall[] = improved ? 0 : st.stall[] + 1
    st.kstall[] = st.width[][1] < k ? 0 : st.kstall[] + 1
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
    k = snp.width[1]; u = minimum(wgt); delta = 0

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

    while isempty(top)
        hi = k - delta * u; lo = hi - u

        for i in oneto(nt)
            lo < bw[i] <= hi && push!(top, i)
        end

        delta -= 1
    end

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

    sort!(U)
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
        ptr = Vector{Int}(undef, nu + 1); tgt = Int[]; ptr[1] = 1
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

                sort!(view(tgt, start:length(tgt))); m = start - 1

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

    return sort!(S)
end

function hbt_stats(srh::NamedTuple)
    return srh.nstep[], srh.nlocal[], srh.nexact[], srh.ntimeout[]
end

# ---------------------------------------------------------------------------
# Workers
#
# With nt tasks, nt - 1 workers diversify the latest snapshot of the
# solution, again and again, each with a context and random numbers of its
# own, and pass the results to this task, which adds them to the solution,
# evaluates it, and merges when it stalls, as hbt_imp! does. When no result
# is waiting, this task diversifies the solution itself. (The time of a
# diversification varies a great deal, so workers that waited for each
# other would mostly wait.) Returns the statistics of the workers.
# ---------------------------------------------------------------------------

function hbt_par!(srh::NamedTuple, st::NamedTuple, lb, nt::Integer)
    ctx = srh.ctx
    workers = [hbt_srh(hbt_ctx(ctx.graph, ctx.wgt), Xoshiro(rand(srh.rng, UInt64)), srh.par) for _ in 2:nt]

    for w in workers
        w.deadline[] = srh.deadline[]; w.t0[] = srh.t0[]
    end

    # the snapshot copies the decomposition only when it changes
    cur = hbt_cop(st); snap = Ref(merge(cur, (; kstall = st.kstall[])))
    lck = ReentrantLock(); stop = Threads.Atomic{Bool}(false); results = Channel{Any}(typemax(Int))
    tasks = [Threads.@spawn hbt_wrk(w, snap, lck, stop, results) for w in workers]

    try
        while !hbt_expired(srh) && st.width[][1] > lb
            k = st.width[][1]

            if isready(results)
                L, cptr, ctgt = take!(results)
            else
                L, cptr, ctgt = hbt_div(srh, hbt_snp(st))
            end

            hbt_adc!(srh, st, L, cptr, ctgt)
            improved = hbt_upd!(srh, st)
            hbt_end!(srh, st, k, improved)
            improved && (cur = hbt_cop(st))
            s = merge(cur, (; kstall = st.kstall[]))
            lock(() -> (snap[] = s), lck)
        end
    finally
        stop[] = true; foreach(wait, tasks)
    end

    stats = (0, 0, 0, 0)

    for w in workers
        stats = stats .+ hbt_stats(w)
    end

    return stats
end

# A snapshot of the decomposition of a solution, apart from it.
hbt_cop(st::NamedTuple) = (; bptr = copy(st.bptr), btgt = copy(st.btgt), parent = copy(st.parent), width = st.width[])

function hbt_wrk(w::NamedTuple, snap::Ref, lck::ReentrantLock, stop::Threads.Atomic{Bool}, results::Channel)
    while !stop[] && !hbt_expired(w)
        s = lock(() -> snap[], lck)
        put!(results, hbt_div(w, s))
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
        pexact::Real = 0.5, init = nothing, verbose::Bool = false)
    bgraph, wgt = hbt_graph(weights, graph); W = eltype(wgt)
    par = (; alg, ninit = Int(ninit), base = Int(base), ntry = Int(ntry), refined, merge, diversify,
        dsize = Int(dsize), nsep = Int(nsep), patience = Int(patience), xtime = Float64(xtime),
        xmin = Float64(xmin), xstep = Int(xstep), near = Int(near), margin = Int(margin), pexact = Float64(pexact), init, verbose)
    srh = hbt_srh(hbt_ctx(bgraph, wgt), Xoshiro(seed), par)
    hbt_ini!(srh, 0)
    lbw = iszero(nv(bgraph)) ? zero(W) : convert(W, lowerbound(wgt, bgraph, lb))
    verbose && println("hbt: n = $(nv(bgraph)), initial width $(hbt_str(srh.sols[1].width[][1])) at $(round(time() - srh.t0[]; digits = 2)) s")
    return (; srh, lb = lbw)
end

# One improvement step; returns whether the solution improved.
function hbtstep!(hbt::NamedTuple)
    srh = hbt.srh; isinf(srh.deadline[]) || (srh.deadline[] = Inf)
    return hbt_imp!(srh, srh.sols[1])
end

# Improve the solution for at most mt seconds, or until its width meets the
# lower bound, with nt tasks (see hbt_par!); returns hbtord(hbt).
function hbtrun!(hbt::NamedTuple; mt::Real = 10.0, nt::Integer = Threads.nthreads())
    srh = hbt.srh; st = srh.sols[1]; srh.deadline[] = time() + mt
    stats = (0, 0, 0, 0)

    if nt > 1 && srh.par.diversify
        stats = hbt_par!(srh, st, hbt.lb, nt)
    else
        while !hbt_expired(srh) && st.width[][1] > hbt.lb
            hbt_imp!(srh, st)
        end
    end

    if srh.par.verbose
        nstep, nlocal, nexact, ntimeout = hbt_stats(srh) .+ stats
        println("hbt: n = $(nv(srh.ctx.graph)), width = $(hbt_str(st.width[][1])) ($(st.width[][2])), steps = $nstep, local = $nlocal, exact = $nexact, timeouts = $ntimeout, pmcs = $(hbt_len(st.sol)), time = $(round(time() - srh.t0[]; digits = 1))")
    end

    srh.deadline[] = Inf
    return hbtord(hbt)
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
