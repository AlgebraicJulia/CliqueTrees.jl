#
# List the potential maximal cliques of a graph, given all of its
# minimal separators (Bouchitté and Todinca's algorithm, with
# Korhonen's peeling). The vertices v_1, ..., v_n of a component
# are in breadth-first order, so that every graph
# G_k = G[v_1, ..., v_k] is connected, and they are peeled from
# the end. With a = v_k, the minimal separators of G_k fall into
#
#   - T: those that contain a (then S - a is one of G_{k-1});
#   - N: those that do not, and are minimal separators of G_{k-1};
#   - R: the rest.
#
# A PMC of G_k either comes from a PMC of G_{k-1}, or is new,
# and then it is S ∪ {a} for some S in N ∪ R, or S ∪ (T - a) for
# some S in R and T in T. Since a PMC Ω ⊃ S has Ω - S inside one
# full component C of G_k - S, the second kind is kept only when
# T - a ⊄ S and T - a ⊆ S ∪ C for some C (Bouchitté and Todinca's
# S ∪ (T ∩ C)); the survivors repeat, and they are deduplicated
# before the PMC test. The candidates that pass the PMC test
# are lifted to G: going up, a PMC of G_{j-1} gains v_j when it
# has a full component in G_j. The minimal separators of G_{k-1}
# are N and T - a. The PMCs of a disconnected graph are those of
# its components.
#
# No PMC is found twice: the candidates of a step are distinct,
# and none of them, minus a, is a PMC of G_{k-1}, whose lift it
# would repeat. For S ∪ {a}, S has a full component in G_{k-1}.
# For X = S ∪ (T - a), the full component of S in G_k containing
# a breaks when a goes, leaving S one full component C in G_{k-1}.
# If X were a PMC of G_{k-1}, then T - a ⊆ S ∪ C, so each full
# component of T in G_k, being adjacent to a and to T - a - S,
# would meet S - T. These are components of G_{k-1} - (T - a),
# and X - (T - a) lies in only one of them.
#
#   - Bouchitté, Vincent, and Ioan Todinca. "Listing all potential
#     maximal cliques of a graph." Theoretical Computer Science
#     276 (2002).
#   - Korhonen, Tuukka. "Finding optimal tree decompositions."
#     MSc thesis, University of Helsinki, 2020.
#
function pmcenum(graph::AbstractGraph{V}, sep::BipartiteGraph{V}) where {V}
    nvtx = nv(graph); nsep = nv(sep); nwrd = cld(nvtx, 64)

    adj    = bt_adj!(FMatrix{UInt64}(undef, nwrd, nvtx), graph)
    label  = FVector{V}(undef, nvtx)
    order  = FVector{V}(undef, nvtx)
    cmpptr = FVector{V}(undef, nvtx + 1)
    out    = UInt64[]

    ncmp = pmcenum_bfs!(label, order, cmpptr, graph)

    # the nonempty separators of each component
    sepcmp = V[]; sepids = V[]

    @inbounds for s in vertices(sep)
        for x in neighbors(sep, s)
            push!(sepcmp, label[x]); push!(sepids, s)
            break
        end
    end

    lst = BipartiteGraph{V, V}(ncmp, convert(V, length(sepids)), length(sepids), oneto(length(sepids) + 1), sepcmp)
    cmpsep = reverse(lst)

    @inbounds for c in oneto(ncmp)
        pmcenum_run!(out, adj, sep, order, cmpptr[c], cmpptr[c + one(V)] - one(V),
            sepids, neighbors(cmpsep, c), nwrd)
    end

    return bt_out(out, nvtx, nwrd)
end

# Label the components, and list each one in breadth-first order:
# the component c is order[cmpptr[c]:cmpptr[c + 1] - 1].
function pmcenum_bfs!(
        label::AbstractVector{V},
        order::AbstractVector{V},
        cmpptr::AbstractVector{V},
        graph::AbstractGraph{V},
    ) where {V}
    nvtx = nv(graph)

    @assert nvtx <= length(label)
    @assert nvtx <= length(order)
    @assert nvtx < length(cmpptr)

    @inbounds for v in vertices(graph)
        label[v] = zero(V)
    end

    c = zero(V); t = zero(V)

    @inbounds for v in vertices(graph)
        if iszero(label[v])
            c += one(V); cmpptr[c] = h = t + one(V)
            t += one(V); order[t] = v; label[v] = c

            while h <= t
                u = order[h]; h += one(V)

                for x in neighbors(graph, u)
                    if iszero(label[x])
                        t += one(V); order[t] = x; label[x] = c
                    end
                end
            end
        end
    end

    cmpptr[c + one(V)] = t + one(V)
    return c
end

# Peel the component order[lo:hi], whose minimal separators are
# the separators sepids[i] of sep, for i in ids.
function pmcenum_run!(
        out::Vector{UInt64},
        adj::AbstractMatrix{UInt64},
        sep::BipartiteGraph{V},
        order::AbstractVector{V},
        lo::Integer,
        hi::Integer,
        sepids::AbstractVector{V},
        ids::AbstractVector{V},
        nwrd::Int,
    ) where {V}
    nvtx = size(adj, 2); pool = UInt64[]

    # the vertices of G_k, for k = hi - lo + 1
    pm = bt_new!(pool, nwrd)

    @inbounds for i in lo:hi
        v = order[i]; pool[pm + bt_wrd(v)] |= bt_bit(v)
    end

    # the minimal separators of G_k
    cur = UInt64[]

    @inbounds for i in ids
        p = length(cur)

        for _ in oneto(nwrd)
            push!(cur, zero(UInt64))
        end

        for x in neighbors(sep, sepids[i])
            cur[p + bt_wrd(x)] |= bt_bit(x)
        end
    end

    lstT = UInt64[]; lstN = UInt64[]; lstR = UInt64[]
    lstF = UInt64[]; lstX = UInt64[]
    ps = bt_new!(pool, nwrd)    # a separator
    px = bt_new!(pool, nwrd)    # a candidate
    pn = bt_new!(pool, nwrd)    # the vertices of G_{k-1}

    @inbounds for k in hi:-1:lo + 1
        a = order[k]; wa = bt_wrd(a); ba = bt_bit(a)

        for i in oneto(nwrd)
            pool[pn + i] = pool[pm + i]
        end

        pool[pn + wa] &= ~ba
        empty!(lstT); empty!(lstN); empty!(lstR)

        for q in 0:nwrd:length(cur) - 1
            for i in oneto(nwrd)
                pool[ps + i] = cur[q + i]
            end

            if !iszero(pool[ps + wa] & ba)
                pool[ps + wa] &= ~ba
                pmcenum_put!(lstT, pool, ps, nwrd)
            elseif pmcenum_sep(pool, adj, ps, pn, nwrd)
                pmcenum_put!(lstN, pool, ps, nwrd)
            else
                pmcenum_put!(lstR, pool, ps, nwrd)
            end
        end

        # S ∪ {a}, for S in N ∪ R
        for lst in (lstN, lstR), q in 0:nwrd:length(lst) - 1
            for i in oneto(nwrd)
                pool[px + i] = lst[q + i]
            end

            pool[px + wa] |= ba

            if pmcenum_pmc(pool, adj, px, pm, nwrd)
                pmcenum_lft!(out, pool, adj, px, pm, order, k, hi, nwrd)
            end
        end

        # S ∪ (T - a), for S in R and T in T with T - a ⊄ S and
        # T - a ⊆ S ∪ C for some full component C of G_k - S
        empty!(lstX)

        for q in 0:nwrd:length(lstR) - 1
            for i in oneto(nwrd)
                pool[ps + i] = lstR[q + i]
            end

            empty!(lstF)
            pmcenum_ful!(lstF, pool, adj, ps, pm, nwrd)

            for r in 0:nwrd:length(lstT) - 1
                if pmcenum_one(lstT, r, lstR, q, lstF, nwrd)
                    for i in oneto(nwrd)
                        push!(lstX, lstR[q + i] | lstT[r + i])
                    end
                end
            end
        end

        bt_unq!(lstX, nwrd)

        for q in 0:nwrd:length(lstX) - 1
            for i in oneto(nwrd)
                pool[px + i] = lstX[q + i]
            end

            if pmcenum_pmc(pool, adj, px, pm, nwrd)
                pmcenum_lft!(out, pool, adj, px, pm, order, k, hi, nwrd)
            end
        end

        # the minimal separators of G_{k-1}
        empty!(cur); append!(cur, lstT); append!(cur, lstN)
        bt_unq!(cur, nwrd)

        for i in oneto(nwrd)
            pool[pm + i] = pool[pn + i]
        end
    end

    # G_1 has one vertex, which is its only PMC
    @inbounds for i in oneto(nwrd)
        pool[px + i] = zero(UInt64)
    end

    v = order[lo]
    @inbounds pool[px + bt_wrd(v)] = bt_bit(v)
    pmcenum_lft!(out, pool, adj, px, pm, order, lo, hi, nwrd)
    return
end

# Is the set S = ps a minimal separator of the graph induced by
# the set pm? That is, does G[pm] - S have two full components?
function pmcenum_sep(pool::Vector{UInt64}, adj::AbstractMatrix{UInt64}, ps::Int, pm::Int, nwrd::Int)
    top = length(pool); full = 0
    pr = bt_new!(pool, nwrd)
    pc = bt_new!(pool, nwrd)
    pb = bt_new!(pool, nwrd)

    @inbounds for i in oneto(nwrd)
        pool[pr + i] = pool[pm + i] & ~pool[ps + i]
    end

    @inbounds while full < 2
        v = bt_min(pool, pr, nwrd)
        iszero(v) && break
        bt_cmp!(pool, adj, pc, pr, v, nwrd)

        for i in oneto(nwrd)
            pool[pr + i] &= ~pool[pc + i]
            pool[pb + i] = zero(UInt64)
        end

        bt_nbr!(pool, adj, pb, pc, nwrd)

        for i in oneto(nwrd)
            pool[pb + i] &= pool[pm + i]
        end

        bt_eq(pool, pb, ps, nwrd) && (full += 1)
    end

    resize!(pool, top)
    return full >= 2
end

# Is the set X = px a PMC of the graph induced by the set pm? That
# is, G[pm] - X has no full component, and every two vertices of X
# are adjacent or lie together in the neighborhood of a component.
function pmcenum_pmc(pool::Vector{UInt64}, adj::AbstractMatrix{UInt64}, px::Int, pm::Int, nwrd::Int)
    top = length(pool)
    pr = bt_new!(pool, nwrd)
    pc = bt_new!(pool, nwrd)
    nb = length(pool)           # the neighborhoods of the components follow

    @inbounds for i in oneto(nwrd)
        pool[pr + i] = pool[pm + i] & ~pool[px + i]
    end

    @inbounds while true
        v = bt_min(pool, pr, nwrd)
        iszero(v) && break
        bt_cmp!(pool, adj, pc, pr, v, nwrd)
        pb = bt_new!(pool, nwrd)

        for i in oneto(nwrd)
            pool[pr + i] &= ~pool[pc + i]
        end

        bt_nbr!(pool, adj, pb, pc, nwrd)

        for i in oneto(nwrd)
            pool[pb + i] &= pool[pm + i]
        end

        if bt_eq(pool, pb, px, nwrd)
            resize!(pool, top)
            return false
        end
    end

    nn = length(pool); pa = bt_new!(pool, nwrd); clq = true

    @inbounds for i in oneto(nwrd)
        word = pool[px + i]

        while clq && !iszero(word)
            x = bt_vtx(Int, i, word); word &= word - one(UInt64)
            wx = bt_wrd(x); bx = bt_bit(x)

            for j in oneto(nwrd)
                pool[pa + j] = adj[j, x] & pool[pm + j]
            end

            pool[pa + wx] |= bx

            for pb in nb:nwrd:nn - 1
                if !iszero(pool[pb + wx] & bx)
                    for j in oneto(nwrd)
                        pool[pa + j] |= pool[pb + j]
                    end
                end
            end

            clq = bt_sub(pool, px, pa, nwrd)
        end
    end

    resize!(pool, top)
    return clq
end

# Lift the PMC X = px of G_k, whose vertices are the set pm, to G,
# and record it.
function pmcenum_lft!(
        out::Vector{UInt64},
        pool::Vector{UInt64},
        adj::AbstractMatrix{UInt64},
        px::Int,
        pm::Int,
        order::AbstractVector{V},
        k::Integer,
        hi::Integer,
        nwrd::Int,
    ) where {V}
    top = length(pool)
    pl = bt_new!(pool, nwrd)    # the vertices of G_j
    py = bt_new!(pool, nwrd)    # the lifted PMC
    pr = bt_new!(pool, nwrd)
    pc = bt_new!(pool, nwrd)
    pb = bt_new!(pool, nwrd)

    @inbounds for i in oneto(nwrd)
        pool[pl + i] = pool[pm + i]
        pool[py + i] = pool[px + i]
    end

    @inbounds for j in k + 1:hi
        v = order[j]; pool[pl + bt_wrd(v)] |= bt_bit(v)

        # a full component of G_j - Y contains v
        for i in oneto(nwrd)
            pool[pr + i] = pool[pl + i] & ~pool[py + i]
            pool[pb + i] = zero(UInt64)
        end

        bt_cmp!(pool, adj, pc, pr, v, nwrd)
        bt_nbr!(pool, adj, pb, pc, nwrd)

        for i in oneto(nwrd)
            pool[pb + i] &= pool[pl + i]
        end

        if bt_eq(pool, pb, py, nwrd)
            pool[py + bt_wrd(v)] |= bt_bit(v)
        end
    end

    @inbounds for i in oneto(nwrd)
        push!(out, pool[py + i])
    end

    resize!(pool, top)
    return
end

# List S ∪ C, for each full component C of G[pm] - S, S = ps.
function pmcenum_ful!(lst::Vector{UInt64}, pool::Vector{UInt64}, adj::AbstractMatrix{UInt64}, ps::Int, pm::Int, nwrd::Int)
    top = length(pool)
    pr = bt_new!(pool, nwrd)
    pc = bt_new!(pool, nwrd)
    pb = bt_new!(pool, nwrd)

    @inbounds for i in oneto(nwrd)
        pool[pr + i] = pool[pm + i] & ~pool[ps + i]
    end

    @inbounds while true
        v = bt_min(pool, pr, nwrd)
        iszero(v) && break
        bt_cmp!(pool, adj, pc, pr, v, nwrd)

        for i in oneto(nwrd)
            pool[pr + i] &= ~pool[pc + i]
            pool[pb + i] = zero(UInt64)
        end

        bt_nbr!(pool, adj, pb, pc, nwrd)

        for i in oneto(nwrd)
            pool[pb + i] &= pool[pm + i]
        end

        if bt_eq(pool, pb, ps, nwrd)
            for i in oneto(nwrd)
                push!(lst, pool[ps + i] | pool[pc + i])
            end
        end
    end

    resize!(pool, top)
    return
end

# Is T ⊄ S, and T ⊆ U for some set U in lst? Here T = lstT[r + 1:r + nwrd]
# and S = lstR[q + 1:q + nwrd].
function pmcenum_one(lstT::Vector{UInt64}, r::Int, lstR::Vector{UInt64}, q::Int, lst::Vector{UInt64}, nwrd::Int)
    sub = true

    @inbounds for i in oneto(nwrd)
        sub &= iszero(lstT[r + i] & ~lstR[q + i])
    end

    sub && return false

    @inbounds for f in 0:nwrd:length(lst) - 1
        sub = true

        for i in oneto(nwrd)
            sub &= iszero(lstT[r + i] & ~lst[f + i])
        end

        sub && return true
    end

    return false
end

function pmcenum_put!(lst::Vector{UInt64}, pool::Vector{UInt64}, p::Int, nwrd::Int)
    @inbounds for i in oneto(nwrd)
        push!(lst, pool[p + i])
    end

    return
end
