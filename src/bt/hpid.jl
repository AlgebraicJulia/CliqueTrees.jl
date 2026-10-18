#
# Tamaki's heuristic PID (Tamaki 2023, Section 4; PIDIterative of his
# RTW), with vertex weights: a search for a decomposition of a connected
# graph whose bags weigh at most K, by positive-instance driven dynamic
# programming, in an order meant to find one quickly rather than to
# exhaust the search. RTW runs it on each node, between children.
#
# A block is a connected set C whose neighborhood S = N(C) is a minimal
# separator, or the whole vertex set (block 1, with S empty). C is small
# if 2|C| + |S| <= n. The width of a block is the lightest width known
# of a decomposition of G[C ∪ S] with a bag that holds S: at first that
# of the one bag C ∪ S. It is feasible if its width is at most K.
#
# Each PMC X caps one block (Tamaki's rooting): for the separators S of
# the components of G - X, from the largest, the first whose full
# component holding X - S is small and not the largest full component
# of S; if there is none, the whole vertex set. The width of X is the
# largest of its weight and the widths of the components of G - X
# inside the block it caps (its inner blocks), and it bounds the width
# of that block.
#
# The search keeps the feasible blocks in sieves, one for each smallest
# vertex (tries, see below), and a queue of small feasible blocks: fewer
# trials first, then larger blocks. From a block C with separator S, it
# searches the full component of S on the other side (the large one, if
# there is one):
#
#   - for each w in S, the set S ∪ (N(w) ∩ scope), if it is a PMC;
#   - for each feasible block C' in the scope, by increasing smallest
#     vertex, whose separator fits with S into a bag, the union
#     U = S ∪ N(C'): if no component of the rest of the scope is full,
#     U is tried as a PMC (and its inner blocks that are not feasible
#     are filled in by a greedy decomposition of their local graphs);
#     otherwise the search goes on in the full components, from U.
#
# Each search is a tick; hpid_improve! stops after budget ticks, or when the
# whole vertex set is feasible (its width is at most K), or when the
# queue runs dry and no block became feasible. The widths are then
# propagated (hpid_dp!), smallest blocks first.
#
# The sets of the search are packed sets on a pool, as in sepenum; the
# vertex sets of the blocks and the PMCs are kept in tables (bt_tab, as
# words), and the separators of the blocks in a list of words.
#
#   - Tamaki, Hisao. "A contraction-recursive algorithm for treewidth."
#     *18th International Symposium on Parameterized and Exact
#     Computation (IPEC 2023)*. 2023.
#
function hpid(ctx::NamedTuple, K::W) where {W}
    graph = ctx.graph; wgt = ctx.wgt; n = nv(graph); nwrd = cld(n, 64)
    adj = bt_mat(nwrd, n); bt_adj!(adj, graph)

    hp = (;
        ctx, graph, wgt, K, n, nwrd, adj, wmin = minimum(wgt),
        pool = UInt64[],
        # blocks: vertex sets (as words), separators (words), widths, caps,
        # sizes, small, sieved, queued, trials, smallest vertices, margins
        blks = bt_tab(), bsep = UInt64[], bwid = W[], bcap = Vector{Int}[], blen = Int[],
        bsml = Bool[], bsvd = Bool[], bque = Bool[], btri = Int[], bmin = Int[], bmar = W[],
        # PMCs: sets (as words), the block each caps, its inner blocks, its
        # weight, and whether it has been handed out
        pmcs = bt_tab(), pcap = Int[], pibl = Vector{Int}[], pwgt = W[], pout = Bool[],
        # the sets found not to be PMCs, and the weight of every vertex if
        # they all weigh the same (otherwise zero)
        npmc = bt_tab(), unit = all(==(first(wgt)), wgt) ? first(wgt) : zero(W),
        # the sieves: the root of each (0 if empty); the nodes (intervals,
        # entries); the entries of the inner nodes (keys, unions,
        # intersections, as words; margins; nodes); the results of the
        # queries, with the weights of their unions with the separator;
        # the stack of a query
        sroot = zeros(Int, n), nsta = Int[], nsto = Int[], nent = Vector{Int}[],
        ekey = UInt64[], eany = UInt64[], eall = UInt64[], emar = W[], enod = Int[],
        qbuf = Int[], qwu = W[], qstk = Tuple{Int, W}[],
        # the search: the queue (a heap of (trials, -size, block)), the
        # ticks, and the deadline
        heap = Tuple{Int, Int, Int}[], tick = Ref(0), deadline = Ref(Inf),
    )

    # the whole vertex set is block 1
    p = bt_new!(hp.pool, nwrd)

    @inbounds for i in oneto(nwrd)
        hp.pool[p + i] = ~zero(UInt64)
    end

    bt_cap!(hp.pool, n, p, nwrd); hpid_blk!(hp, p); empty!(hp.pool)
    return hp
end

# ---------------------------------------------------------------------------
# Packed sets
# ---------------------------------------------------------------------------

# The set p of the pool as a key of a table.
hpid_key(hp::NamedTuple, p::Int) = reinterpret(Int, view(hp.pool, p + 1:p + hp.nwrd))

# The word i of the set k of the table tab.
@inline hpid_wrd(tab::NamedTuple, k::Int, i::Int) = @inbounds reinterpret(UInt64, tab.tgt[tab.ptr[k] + i - 1])

# Load the separator of the block b onto the pool.
function hpid_lds!(hp::NamedTuple, b::Int)
    nwrd = hp.nwrd; p = bt_new!(hp.pool, nwrd); q = (b - 1) * nwrd

    @inbounds for i in oneto(nwrd)
        hp.pool[p + i] = hp.bsep[q + i]
    end

    return p
end

# A new set on the pool: the complement of the set p.
function hpid_cmpl!(hp::NamedTuple, p::Int)
    pool = hp.pool; q = bt_new!(pool, hp.nwrd)

    @inbounds for i in oneto(hp.nwrd)
        pool[q + i] = ~pool[p + i]
    end

    bt_cap!(pool, hp.n, q, hp.nwrd)
    return q
end

# The components of the set r (destroyed), each followed by its
# neighborhood, as offsets on the pool: the first is lo, then every
# 2nwrd words to the end of the pool.
function hpid_cps!(hp::NamedTuple, r::Int)
    pool = hp.pool; adj = hp.adj; nwrd = hp.nwrd; lo = length(pool)

    @inbounds while true
        v = bt_min(pool, r, nwrd)
        iszero(v) && break
        c = bt_new!(pool, nwrd); s = bt_new!(pool, nwrd)
        bt_cmp!(pool, adj, c, r, v, nwrd); bt_nbr!(pool, adj, s, c, nwrd)

        for i in oneto(nwrd)
            pool[r + i] &= ~pool[c + i]
        end
    end

    return lo
end

# Is the set p of the pool a PMC? Yes if no component of
# G - p is full, and every vertex of p sees the others: by an edge, or
# through the neighborhood of a component (Bouchitté and Todinca).
function hpid_ispmc(hp::NamedTuple, p::Int)
    pool = hp.pool; adj = hp.adj; nwrd = hp.nwrd; top = length(pool)
    lo = hpid_cps!(hp, hpid_cmpl!(hp, p)); hi = length(pool)
    ok = true

    # no full component
    @inbounds for c in lo:2nwrd:hi - 1
        if bt_eq(pool, c + nwrd, p, nwrd)
            ok = false; break
        end
    end

    # every vertex sees the others
    if ok
        s = bt_new!(pool, nwrd)

        @inbounds for i in oneto(nwrd)
            word = pool[p + i]

            while ok && !iszero(word)
                x = bt_vtx(Int, i, word); word &= word - one(UInt64)

                for j in oneto(nwrd)
                    pool[s + j] = bt_get(adj, j, x)
                end

                pool[s + i] |= bt_bit(x)

                for c in lo:2nwrd:hi - 1
                    if !iszero(pool[c + nwrd + i] & bt_bit(x))
                        for j in oneto(nwrd)
                            pool[s + j] |= pool[c + nwrd + j]
                        end
                    end
                end

                ok = bt_sub(pool, p, s, nwrd)
            end

            ok || break
        end
    end

    resize!(pool, top)
    return ok
end

# Compare the sets p and q of the pool as Tamaki does: by size, then as
# binary numbers with the largest vertex first.
function hpid_less(hp::NamedTuple, p::Int, q::Int)
    pool = hp.pool; nwrd = hp.nwrd
    np = bt_len(pool, p, nwrd); nq = bt_len(pool, q, nwrd)
    np == nq || return np < nq

    @inbounds for i in nwrd:-1:1
        pool[p + i] == pool[q + i] || return pool[p + i] < pool[q + i]
    end

    return false
end

# ---------------------------------------------------------------------------
# Blocks and PMCs
# ---------------------------------------------------------------------------

# The block with the vertex set p of the pool, made if it is new.
function hpid_blk!(hp::NamedTuple, p::Int)
    b, new = bt_tab_add!(hp.blks, hpid_key(hp, p))
    new || return b
    pool = hp.pool; nwrd = hp.nwrd; top = length(pool)
    s = bt_new!(pool, nwrd); bt_nbr!(pool, hp.adj, s, p, nwrd)

    @inbounds for i in oneto(nwrd)
        push!(hp.bsep, pool[s + i])
    end

    nc = bt_len(pool, p, nwrd); ns = bt_len(pool, s, nwrd); ws = bt_wgt(pool, s, hp.wgt, nwrd)
    push!(hp.bwid, bt_wgt(pool, p, hp.wgt, nwrd) + ws); push!(hp.bmar, hp.K - ws)
    push!(hp.bcap, Int[]); push!(hp.blen, nc); push!(hp.bsml, 2nc + ns <= hp.n)
    push!(hp.bsvd, false); push!(hp.bque, false); push!(hp.btri, 0); push!(hp.bmin, bt_min(pool, p, nwrd))
    resize!(pool, top)
    return b
end

# The PMC with the vertex set p of the pool (a PMC of the graph), made if
# it is new: its components, its inner blocks, and the block it caps.
function hpid_pmc!(hp::NamedTuple, p::Int)
    x = bt_tab_find(hp.pmcs, hpid_key(hp, p))
    ispositive(x) && return x
    pool = hp.pool; nwrd = hp.nwrd; n = hp.n; top = length(pool)

    # the components of G - X, and their blocks
    lo = hpid_cps!(hp, hpid_cmpl!(hp, p)); hi = length(pool)
    blocks = Int[]

    for c in lo:2nwrd:hi - 1
        push!(blocks, hpid_blk!(hp, c))
    end

    # the block that X caps: the small inner full component of the first
    # separator that has one, from the largest separator (Tamaki's order)
    seps = Int[]

    for c in lo:2nwrd:hi - 1
        any(d -> bt_eq(pool, d, c + nwrd, nwrd), seps) || push!(seps, c + nwrd)
    end

    sort!(seps; lt = (a, b) -> hpid_less(hp, b, a))
    cap = 1

    for s in seps
        mid = length(pool); in = hpid_inner(hp, p, s)

        if ispositive(in)
            cap = hpid_blk!(hp, in); ibl = Int[]

            for (j, c) in enumerate(lo:2nwrd:hi - 1)
                bt_sub(pool, c, in, nwrd) && push!(ibl, blocks[j])
            end

            blocks = ibl; resize!(pool, mid)
            break
        end

        resize!(pool, mid)
    end

    k, _ = bt_tab_add!(hp.pmcs, hpid_key(hp, p))
    push!(hp.pcap, cap); push!(hp.pibl, blocks); push!(hp.pwgt, bt_wgt(pool, p, hp.wgt, nwrd)); push!(hp.pout, false)
    push!(hp.bcap[cap], k)
    resize!(pool, top)
    return k
end

# The PMC with the vertex set p of the pool, made if it is new, or 0 if
# p is not a PMC: the sets found not to be PMCs are kept, so that the
# search tests each once.
function hpid_ifpmc!(hp::NamedTuple, p::Int)
    x = bt_tab_find(hp.pmcs, hpid_key(hp, p)); ispositive(x) && return x
    ispositive(bt_tab_find(hp.npmc, hpid_key(hp, p))) && return 0
    hpid_ispmc(hp, p) && return hpid_pmc!(hp, p)
    bt_tab_add!(hp.npmc, hpid_key(hp, p))
    return 0
end

# The full component of the separator s (a set of the pool) that holds
# X - s, for the PMC X = p, if it is small and not the largest full
# component of s (in Tamaki's order); otherwise 0. It is left on the pool.
function hpid_inner(hp::NamedTuple, p::Int, s::Int)
    pool = hp.pool; nwrd = hp.nwrd; x0 = 0

    @inbounds for i in oneto(nwrd)
        word = pool[p + i] & ~pool[s + i]

        if !iszero(word)
            x0 = bt_vtx(Int, i, word); break
        end
    end

    iszero(x0) && return 0
    lo = hpid_cps!(hp, hpid_cmpl!(hp, s)); hi = length(pool)
    in = 0; big = 0

    for d in lo:2nwrd:hi - 1
        bt_eq(pool, d + nwrd, s, nwrd) || continue
        (iszero(big) || hpid_less(hp, big, d)) && (big = d)
        @inbounds !iszero(pool[d + bt_wrd(x0)] & bt_bit(x0)) && (in = d)
    end

    small = ispositive(in) && 2bt_len(pool, in, nwrd) + bt_len(pool, s, nwrd) <= hp.n
    return small && in != big ? in : 0
end

# The width of the PMC x, which bounds the width of the block it caps.
function hpid_eval!(hp::NamedTuple, x::Int)
    w = hp.pwgt[x]

    @inbounds for b in hp.pibl[x]
        w = max(w, hp.bwid[b])
    end

    c = hp.pcap[x]
    w < hp.bwid[c] && (hp.bwid[c] = w)
    return w
end

function hpid_beval!(hp::NamedTuple, b::Int)
    for x in hp.bcap[b]
        hpid_eval!(hp, x)
    end

    return
end

# Propagate the widths: the small blocks, smallest first, then the whole
# vertex set.
function hpid_dp!(hp::NamedTuple)
    nb = bt_tab_len(hp.blks)

    for b in sort!(filter(b -> hp.bsml[b], collect(oneto(nb))); by = b -> hp.blen[b])
        hpid_beval!(hp, b)
    end

    hpid_beval!(hp, 1)
    return
end

# Add the PMCs (sorted lists of vertices, PMCs of the graph), and
# propagate the widths.
function hpid_import!(hp::NamedTuple, pmcs)
    pool = hp.pool; nwrd = hp.nwrd

    for X in pmcs
        p = bt_new!(pool, nwrd)

        @inbounds for v in X
            pool[p + bt_wrd(v)] |= bt_bit(v)
        end

        hpid_pmc!(hp, p); empty!(pool)
    end

    hpid_dp!(hp)
    return
end

# The PMCs of width at most K not handed out before (Tamaki's
# usefulPMCs), as sorted lists of vertices.
function hpid_export!(hp::NamedTuple)
    out = Vector{Int}[]

    for x in oneto(bt_tab_len(hp.pmcs))
        hp.pout[x] && continue

        if hpid_eval!(hp, x) <= hp.K
            hp.pout[x] = true; X = Int[]

            for i in oneto(hp.nwrd)
                word = hpid_wrd(hp.pmcs, x, i)

                while !iszero(word)
                    push!(X, bt_vtx(Int, i, word)); word &= word - one(UInt64)
                end
            end

            push!(out, X)
        end
    end

    return out
end

hpid_width(hp::NamedTuple) = hp.bwid[1]

# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------

# A heap of (trials, -size, block), smallest first.
function hpid_push!(heap::Vector{Tuple{Int, Int, Int}}, key::Tuple{Int, Int, Int})
    push!(heap, key); i = length(heap)

    @inbounds while i > 1
        j = i >> 1
        heap[j] <= heap[i] && break
        heap[i], heap[j] = heap[j], heap[i]; i = j
    end

    return
end

function hpid_pop!(heap::Vector{Tuple{Int, Int, Int}})
    @inbounds top = heap[1]; last = pop!(heap)

    if !isempty(heap)
        @inbounds heap[1] = last; i = 1; m = length(heap)

        @inbounds while true
            l = 2i; r = l + 1; s = i
            l <= m && heap[l] < heap[s] && (s = l)
            r <= m && heap[r] < heap[s] && (s = r)
            s == i && break
            heap[i], heap[s] = heap[s], heap[i]; i = s
        end
    end

    return top
end

# A feasible block enters the sieve (once) and the queue (if it is small
# and not there).
function hpid_enter!(hp::NamedTuple, b::Int)
    if !hp.bsvd[b]
        hp.bsvd[b] = true
        b != 1 && hpid_sv!(hp, b)
    end

    if !hp.bque[b] && (b == 1 || hp.bsml[b])
        hp.bque[b] = true; hpid_push!(hp.heap, (hp.btri[b], -hp.blen[b], b))
    end

    return
end

function hpid_feasibles(hp::NamedTuple)
    return [b for b in oneto(bt_tab_len(hp.blks)) if hp.bwid[b] <= hp.K]
end

# Search for budget ticks, until the deadline, or until the width is at
# most K.
function hpid_improve!(hp::NamedTuple, budget::Integer, deadline::Float64)
    hp.tick[] = 0; hpid_width(hp) <= hp.K && return
    hp.deadline[] = deadline; empty!(hp.heap); hpid_svclr!(hp)
    fill!(hp.bsvd, false); fill!(hp.bque, false)
    feas = hpid_feasibles(hp); nsf = length(feas)

    for b in feas
        hpid_enter!(hp, b)
    end

    while !isempty(hp.heap)
        b = hpid_pop!(hp.heap)[3]
        b == 1 && break
        hpid_search!(hp, b)
        hpid_width(hp) <= hp.K && break

        # the queue ran dry: again from the blocks that became feasible, if any
        if isempty(hp.heap)
            feas = hpid_feasibles(hp)
            length(feas) == nsf && break
            nsf = length(feas); fill!(hp.bque, false)

            for b in feas
                hpid_enter!(hp, b)
            end
        end

        (hp.tick[] > budget || time() > deadline) && return
    end

    hpid_dp!(hp)
    return
end

# Search from the small feasible block b: the full component of its
# separator on the other side (the large one, if there is one).
function hpid_search!(hp::NamedTuple, b::Int)
    hp.btri[b] += 1
    pool = hp.pool; nwrd = hp.nwrd; n = hp.n; top = length(pool)
    s = hpid_lds!(hp, b); ns = bt_len(pool, s, nwrd)
    lo = hpid_cps!(hp, hpid_cmpl!(hp, s)); hi = length(pool)
    big = 0

    for c in lo:2nwrd:hi - 1
        if bt_eq(pool, c + nwrd, s, nwrd) && 2bt_len(pool, c, nwrd) + ns > n
            big = c; break
        end
    end

    if ispositive(big)
        hpid_srch!(hp, big, s, 1)
    else
        for c in lo:2nwrd:hi - 1
            bt_eq(pool, c + nwrd, s, nwrd) || continue
            hpid_isblk(hp, c, b) && continue
            hpid_srch!(hp, c, s, 1)
        end
    end

    resize!(pool, top)
    return
end

# Is the set c of the pool the vertex set of the block b?
function hpid_isblk(hp::NamedTuple, c::Int, b::Int)
    @inbounds for i in oneto(hp.nwrd)
        hp.pool[c + i] == hpid_wrd(hp.blks, b, i) || return false
    end

    return true
end

# Search the scope (a set of the pool) from the separator sep (a set of
# the pool), with feasible blocks whose smallest vertex is at least vmin.
function hpid_srch!(hp::NamedTuple, scope::Int, sep::Int, vmin::Int)
    hp.tick[] += 1
    (iszero(hp.tick[] & 15) && time() > hp.deadline[]) && return
    pool = hp.pool; adj = hp.adj; nwrd = hp.nwrd; K = hp.K; wgt = hp.wgt; top = length(pool)
    wsep = bt_wgt(pool, sep, wgt, nwrd)

    # S ∪ (N(w) ∩ scope), for each w in S
    @inbounds for i in oneto(nwrd)
        word = pool[sep + i]

        while !iszero(word)
            w = bt_vtx(Int, i, word); word &= word - one(UInt64)
            u = bt_new!(pool, nwrd)

            for j in oneto(nwrd)
                pool[u + j] = pool[sep + j] | (pool[scope + j] & bt_get(adj, j, w))
            end

            if bt_wgt(pool, u, wgt, nwrd) <= K
                x = hpid_ifpmc!(hp, u)

                if ispositive(x)
                    hpid_eval!(hp, x); c = hp.pcap[x]
                    hp.bwid[c] <= K && hpid_enter!(hp, c)
                end
            end

            resize!(pool, top)
        end
    end

    # the feasible blocks C in the scope whose separator fits with S into a
    # bag, by smallest vertex: S ∪ N(C), and the rest of the scope. The
    # results of a query are taken before they are tried, as trying them
    # adds blocks to the sieves.
    @inbounds for i in bt_wrd(vmin):nwrd
        word = pool[scope + i]

        while !iszero(word)
            v0 = bt_vtx(Int, i, word); word &= word - one(UInt64)
            (v0 < vmin || iszero(hp.sroot[v0])) && continue
            lo = length(hp.qbuf); hpid_qry!(hp, v0, scope, sep, wsep); hi = length(hp.qbuf)

            for t in lo + 1:hi
                c = hp.qbuf[t]; wu = hp.qwu[t]; q = (c - 1) * nwrd
                r = bt_new!(pool, nwrd); u = bt_new!(pool, nwrd)

                for j in oneto(nwrd)
                    pool[r + j] = pool[scope + j] & ~hpid_wrd(hp.blks, c, j) & ~hp.bsep[q + j]
                    pool[u + j] = pool[sep + j] | hp.bsep[q + j]
                end

                hpid_try!(hp, r, u, wu, v0)
                resize!(pool, top)
            end

            resize!(hp.qbuf, lo); resize!(hp.qwu, lo)
        end
    end

    return
end

# ---------------------------------------------------------------------------
# The sieves
# ---------------------------------------------------------------------------
#
# The feasible blocks with smallest vertex v are kept in a trie, keyed by
# their closures X = C ∪ N(C), with their margins m = K - w(N(C)) (after
# the sieve of PIDBTLib, which keys partial decompositions by their bags).
# A query from a separator S into a scope (a full component of S) asks
# for the blocks with
#
#   (1) C ⊆ scope, so that X avoids R, the complement of scope ∪ S;
#   (2) w(S ∪ N(C)) <= K, that is, w(S - X) <= m (when C avoids S).
#
# Each node compares the keys below it on an interval of vertices; the
# intervals along a path from the root partition 1:n. The nodes whose
# intervals end at n hold blocks; the others hold entries, each with a
# key that agrees on the interval with the keys below it, their union
# and intersection, the largest margin below it, and its node. The union
# and intersection bound the keys beyond the interval, so that a query
# can skip an entry without visiting its node: if the intersection meets
# R, or if even the union leaves too heavy a part of S. A node with too
# many entries is split: its interval is cut short, and its entries are
# grouped under new nodes by their keys on the shorter interval.

const HPID_NODE = 128

# Word i of the closure of the block b.
@inline function hpid_clo(hp::NamedTuple, b::Int, i::Int)
    return @inbounds hpid_wrd(hp.blks, b, i) | hp.bsep[(b - 1) * hp.nwrd + i]
end

# Word i of the interval a:z.
@inline function hpid_msk(i::Int, a::Int, z::Int)
    lo = 64(i - 1) + 1; hi = 64i
    (z < lo || a > hi) && return zero(UInt64)
    m = ~zero(UInt64)
    a > lo && (m &= ~zero(UInt64) << (a - lo))
    z < hi && (m &= ~zero(UInt64) >> (hi - z))
    return m
end

# The weight of the vertices of word i.
@inline function hpid_wwd(hp::NamedTuple, i::Int, word::UInt64)
    wgt = hp.wgt; ispositive(hp.unit) && return count_ones(word) * hp.unit
    w = zero(eltype(wgt))

    @inbounds while !iszero(word)
        w += wgt[bt_vtx(Int, i, word)]; word &= word - one(UInt64)
    end

    return w
end

function hpid_svclr!(hp::NamedTuple)
    fill!(hp.sroot, 0)

    for x in (hp.nsta, hp.nsto, hp.nent, hp.ekey, hp.eany, hp.eall, hp.emar, hp.enod)
        empty!(x)
    end

    return
end

function hpid_node!(hp::NamedTuple, a::Int, z::Int)
    push!(hp.nsta, a); push!(hp.nsto, z); push!(hp.nent, Int[])
    return length(hp.nsta)
end

# A new entry with the key, union, and intersection of the block x (if
# leaf) or of the entry x, its margin m, and the node v.
function hpid_ent!(hp::NamedTuple, leaf::Bool, x::Int, m, v::Int)
    nwrd = hp.nwrd; q = (x - 1) * nwrd

    @inbounds for i in oneto(nwrd)
        if leaf
            k = hpid_clo(hp, x, i); push!(hp.ekey, k); push!(hp.eany, k); push!(hp.eall, k)
        else
            push!(hp.ekey, hp.ekey[q + i]); push!(hp.eany, hp.eany[q + i]); push!(hp.eall, hp.eall[q + i])
        end
    end

    push!(hp.emar, m); push!(hp.enod, v)
    return length(hp.enod)
end

# Merge the block x (if leaf) or the entry x into the entry e.
function hpid_mrg!(hp::NamedTuple, e::Int, leaf::Bool, x::Int, m)
    nwrd = hp.nwrd; p = (e - 1) * nwrd; q = (x - 1) * nwrd

    @inbounds for i in oneto(nwrd)
        if leaf
            k = hpid_clo(hp, x, i); hp.eany[p + i] |= k; hp.eall[p + i] &= k
        else
            hp.eany[p + i] |= hp.eany[q + i]; hp.eall[p + i] &= hp.eall[q + i]
        end
    end

    hp.emar[e] = max(hp.emar[e], m)
    return
end

# Does the key of the block b (if leaf) or of the entry x agree with the
# key of the entry e on the interval a:z?
function hpid_agr(hp::NamedTuple, e::Int, leaf::Bool, x::Int, a::Int, z::Int)
    nwrd = hp.nwrd; p = (e - 1) * nwrd; q = (x - 1) * nwrd

    @inbounds for i in bt_wrd(a):bt_wrd(z)
        k = leaf ? hpid_clo(hp, x, i) : hp.ekey[q + i]
        iszero((k ⊻ hp.ekey[p + i]) & hpid_msk(i, a, z)) || return false
    end

    return true
end

# Add the feasible block b to the sieve of its smallest vertex.
function hpid_sv!(hp::NamedTuple, b::Int)
    v0 = hp.bmin[b]; m = hp.bmar[b]; n = hp.n
    iszero(hp.sroot[v0]) && (hp.sroot[v0] = hpid_node!(hp, 1, n))
    v = hp.sroot[v0]

    @inbounds while hp.nsto[v] < n
        a = hp.nsta[v]; z = hp.nsto[v]; found = 0

        for e in hp.nent[v]
            if hpid_agr(hp, e, true, b, a, z)
                found = e; break
            end
        end

        if iszero(found)
            w = hpid_node!(hp, z + 1, n); push!(hp.nent[w], b)
            push!(hp.nent[v], hpid_ent!(hp, true, b, m, w))
            hpid_spl!(hp, v)
            return
        end

        hpid_mrg!(hp, found, true, b, m); v = hp.enod[found]
    end

    push!(hp.nent[v], b); hpid_spl!(hp, v)
    return
end

# Split the node v if it has too many entries.
function hpid_spl!(hp::NamedTuple, v::Int)
    a = hp.nsta[v]; z = hp.nsto[v]
    (length(hp.nent[v]) > HPID_NODE && a < z) || return
    leaf = z == hp.n; y = a + (1 << floor(Int, log(z - a))) - 1
    old = hp.nent[v]; hp.nsto[v] = y; hp.nent[v] = Int[]

    @inbounds for x in old
        m = leaf ? hp.bmar[x] : hp.emar[x]; found = 0

        for e in hp.nent[v]
            if hpid_agr(hp, e, leaf, x, a, y)
                found = e; break
            end
        end

        if iszero(found)
            found = hpid_ent!(hp, leaf, x, m, hpid_node!(hp, y + 1, z)); push!(hp.nent[v], found)
        else
            hpid_mrg!(hp, found, leaf, x, m)
        end

        push!(hp.nent[hp.enod[found]], x)
    end

    return
end

# Push onto the results the blocks of the sieve of v0 in the scope whose
# separators fit with sep (of weight wsep) into a bag, with the weights
# of the unions.
function hpid_qry!(hp::NamedTuple, v0::Int, scope::Int, sep::Int, wsep::W) where {W}
    pool = hp.pool; nwrd = hp.nwrd; n = hp.n; K = hp.K; wgt = hp.wgt; stk = hp.qstk
    empty!(stk); push!(stk, (hp.sroot[v0], zero(W)))

    @inbounds while !isempty(stk)
        v, acc = pop!(stk); a = hp.nsta[v]; z = hp.nsto[v]

        if z == n
            for c in hp.nent[v]
                q = (c - 1) * nwrd; fits = true; wu = wsep

                for j in oneto(nwrd)
                    iszero(hpid_wrd(hp.blks, c, j) & ~pool[scope + j]) || (fits = false; break)
                end

                fits || continue

                for j in oneto(nwrd)
                    wu += hpid_wwd(hp, j, hp.bsep[q + j] & ~pool[sep + j])
                end

                if wu <= K
                    push!(hp.qbuf, c); push!(hp.qwu, wu)
                end
            end
        else
            for e in hp.nent[v]
                p = (e - 1) * nwrd; m = hp.emar[e]; ok = true; ii = acc; jj = zero(W)

                for j in oneto(nwrd)
                    R = ~(pool[scope + j] | pool[sep + j]); S = pool[sep + j]; M = hpid_msk(j, a, z)
                    k = hp.ekey[p + j]

                    # the keys below agree with k on the interval, and hold
                    # the intersection
                    if !iszero(k & M & R) || !iszero(hp.eall[p + j] & R)
                        ok = false; break
                    end

                    ii += hpid_wwd(hp, j, S & M & ~k)
                    jj += hpid_wwd(hp, j, S & ~(M | hpid_msk(j, 1, a - 1)) & ~hp.eany[p + j])
                end

                (ok && ii <= m && ii + jj <= m) && push!(stk, (hp.enod[e], ii))
            end
        end
    end

    return
end

# The union U (a set of the pool, of weight wu) of a separator and the
# separator of a feasible block, with the rest r of the scope: if no
# component of r is full, try U as a PMC; otherwise search on.
function hpid_try!(hp::NamedTuple, r::Int, u::Int, wu, vmin::Int)
    pool = hp.pool; nwrd = hp.nwrd; n = hp.n; K = hp.K; top = length(pool)
    nu = bt_len(pool, u, nwrd)
    lo = hpid_cps!(hp, r); hi = length(pool)
    nfull = 0; big = 0

    for c in lo:2nwrd:hi - 1
        if bt_eq(pool, c + nwrd, u, nwrd)
            nfull += 1
            2bt_len(pool, c, nwrd) + nu > n && (big = c)
        end
    end

    if iszero(nfull)
        x = hpid_ifpmc!(hp, u)

        if ispositive(x)
            for b in copy(hp.pibl[x])
                hp.bwid[b] > K && hpid_fil!(hp, b)
            end

            hpid_eval!(hp, x); c = hp.pcap[x]
            hp.bwid[c] <= K && hpid_enter!(hp, c)
        end
    elseif wu + hp.wmin <= K
        for c in lo:2nwrd:hi - 1
            bt_eq(pool, c + nwrd, u, nwrd) || continue
            (iszero(big) || c == big) && hpid_srch!(hp, c, u, vmin + 1)
        end
    end

    resize!(pool, top)
    return
end

# Fill in the block b by a greedy decomposition of its local graph, if
# its bags weigh at most K: their PMCs, and the blocks they cap, smallest
# first.
function hpid_fil!(hp::NamedTuple, b::Int)
    pool = hp.pool; nwrd = hp.nwrd; top = length(pool); K = hp.K

    # the closed neighborhood N[C], sorted
    U = Int[]

    @inbounds for i in oneto(nwrd)
        word = hpid_clo(hp, b, i)

        while !iszero(word)
            push!(U, bt_vtx(Int, i, word)); word &= word - one(UInt64)
        end
    end

    L = hbt_loc(hp.ctx, U)
    label, tree = rtw_min(L.graph, L.wgt, first(permutation(L.wgt, L.graph, MF(; strategy = 1))))

    for clique in tree
        sum(L.wgt[label[v]] for v in clique) <= K || return
    end

    caps = Int[]; Kl = Int[]

    for clique in tree
        empty!(Kl)

        for v in clique
            push!(Kl, label[v])
        end

        sort!(Kl); hbt_isnbr(L, Kl) && continue
        p = bt_new!(pool, nwrd)

        @inbounds for k in Kl
            v = U[k]; pool[p + bt_wrd(v)] |= bt_bit(v)
        end

        x = hpid_ifpmc!(hp, p); ispositive(x) && push!(caps, hp.pcap[x])
        resize!(pool, top)
    end

    for c in sort!(unique!(caps); by = c -> hp.blen[c])
        hpid_beval!(hp, c)
        hp.bwid[c] <= K && hpid_enter!(hp, c)
    end

    return
end
