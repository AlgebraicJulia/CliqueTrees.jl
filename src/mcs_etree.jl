function mcs_etree(weights::AbstractVector, graph::AbstractGraph{V}, alg::PermutationOrAlgorithm) where {V}
    order, index = permutation(weights, graph, alg)
    return mcs_etree!(order, index, graph)
end

function mcs_etree!(order::AbstractVector{V}, index::AbstractVector{V}, graph::AbstractGraph{V}) where {V}
    E = etype(graph); m = de(graph); n = nv(graph); ws = workspace()

    begptr = workvector(ws, E, 91, n + one(V))
    endptr = workvector(ws, E, 92, n + one(V))
    target = workvector(ws, V, 93, m)
    tree = Parent(n, workvector(ws, V, 94, n))
    fdesc = workvector(ws, V, 95, n)
    count = workvector(ws, V, 96, n)
    stale = workvector(ws, V, 97, n)
    stamp = workvector(ws, V, 98, n)
    stops = workvector(ws, V, 99, n)
    marks = workvector(ws, V, 100, n)
    block = workvector(ws, V, 101, n)

    work1 = workvector(ws, V, 102, n)
    work2 = workvector(ws, V, 103, n)
    work3 = workvector(ws, V, 104, n)
    work4 = workvector(ws, V, 105, n)
    work5 = workvector(ws, V, 106, n)
    work6 = workvector(ws, V, 107, n)
    work7 = workvector(ws, V, 108, n)
    work8 = workvector(ws, V, 109, n)

    mcs_etree_impl!(work1, work2, work3, work4, work5, work6, work7, work8,
        block, marks, stamp, begptr, endptr, target, tree, fdesc, count,
        stale, stops, graph, order, index)

    return order, index
end

# Fast Computation of Minimal Fill inside a Given Elimination Ordering
# Heggernes and Peyton
# MCS-ETree (blocked implementation)
#
# The time complexity is
#
#     O(mn α(m, n))
#
# m = |E|, n = |V|, and α is the extremely-slow-growing
# inverse of the Ackermann function.
function mcs_etree_impl!(
        work1::AbstractVector{V},
        work2::AbstractVector{V},
        work3::AbstractVector{V},
        work4::AbstractVector{V},
        work5::AbstractVector{V},
        work6::AbstractVector{V},
        work7::AbstractVector{V},
        work8::AbstractVector{V},
        block::AbstractVector{V},
        marks::AbstractVector{V},
        stamp::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        target::AbstractVector{V},
        tree::Parent{V},
        fdesc::AbstractVector{V},
        count::AbstractVector{V},
        stale::AbstractVector{V},
        stops::AbstractVector{V},
        graph::AbstractGraph{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
    ) where {V, E}
    @assert nv(graph) <= length(work1)
    @assert nv(graph) <= length(work2)
    @assert nv(graph) <= length(work3)
    @assert nv(graph) <= length(work4)
    @assert nv(graph) <= length(work5)
    @assert nv(graph) <= length(work6)
    @assert nv(graph) <= length(work7)
    @assert nv(graph) <= length(work8)
    @assert nv(graph) <= length(block)
    @assert nv(graph) <= length(marks)
    @assert nv(graph) <= length(stamp)
    @assert nv(graph) < length(begptr)
    @assert nv(graph) < length(endptr)
    @assert de(graph) <= length(target)
    @assert nv(graph) == length(tree)
    @assert nv(graph) <= length(fdesc)
    @assert nv(graph) <= length(count)
    @assert nv(graph) <= length(stale)
    @assert nv(graph) <= length(stops)
    @assert nv(graph) <= length(order)
    @assert nv(graph) <= length(index)

    # `stops` and `num` form a stack of the roots of the
    # unnumbered elimination subtrees
    #
    # `stale[v]` is positive if the subtree rooted at vertex `v`
    # has changed since its count and skeleton adjacency set were
    # computed. The stale vertices of a subtree form an upper set:
    # a subtree contains a stale vertex if and only if its root is
    # stale.
    #
    # `stamp` holds positive marks (the position of a numbered vertex)
    # for the count updates and negative marks (minus the rotation
    # counter) for the rotations.
    num = zero(V); rot = zero(V)

    # initialize elimination forest (`tree`) and first
    # descendants (`fdesc`)
    etree_impl!(tree, work1, graph, order, index)
    postorder!_impl!(work1, work2, work3, work4, tree)
    firstdescendants_impl!(fdesc, tree, vertices(graph))

    # - permute `order` and `index`
    # - initialize empty skeleton graph
    # - push roots to `stops`
    # - every count is zero, since L is empty
    begptr[one(V)] = p = one(E)
    endptr[one(V)] = p - one(E)

    @inbounds for v in vertices(graph)
        marks[v] = zero(V)
        stale[v] = zero(V)
        stamp[v] = zero(V)
        count[v] = zero(V)
        i = index[v] = work3[index[v]]
        order[i] = v
        begptr[v + one(V)] = p += convert(E, eltypedegree(graph, v))
        endptr[v + one(V)] = p - one(E)

        if isnothing(parentindex(tree, v))
            num += one(V); stops[num] = v
        end
    end

    @inbounds while ispositive(num)
        # get an unnumbered elimination subtree T
        stop = stops[num]
        strt = fdesc[stop]
        num -= one(V)

        # recount the vertices of T whose subtrees have changed
        if ispositive(stale[order[stop]])
            mcs_etree_recount!(work1, work2, work3, work4, work5, work6, work7, marks,
                begptr, endptr, target, order, index, graph, tree, fdesc,
                count, stale, strt, stop)
        end

        # find a vertex `root` in T of maximum cardinality
        root = mcs_etree_findroot(count, fdesc, stop)

        # find a block of vertices to number consecutively
        blck, nanc = mcs_etree_findblock!(marks, block, work1, begptr, endptr, target,
            order, index, graph, fdesc, tree, strt, stop, root)

        if blck + nanc == stop
            # the block contains every ancestor of `root`
            num = mcs_etree_fastnumber!(work1, work2, work3, work4, work5,
                block, order, index, tree, fdesc, count, stops, num, stop, root, blck)
        else
            # change the root of T to `root`
            rot += one(V)

            num = mcs_etree_rotate!(work1, work2, work3, work4, work5, work6,
                work7, work8, block, stamp, order, index, graph, tree, fdesc,
                count, stale, stops, num, strt, stop, root, blck, nanc, -rot)
        end

        # number the block
        mcs_etree_number!(stamp, begptr, endptr, target, order, index,
            graph, tree, count, strt, stop, blck)
    end

    return
end

# Find the lowest vertex of maximum count in T. Counts are monotone up
# the tree, so descend into the leftmost child of maximum count.
function mcs_etree_findroot(
        count::AbstractVector{V},
        fdesc::AbstractVector{V},
        stop::V,
    ) where {V}
    @assert stop <= length(count)
    @assert stop <= length(fdesc)
    @inbounds maxcnt = count[stop]; root = stop; next = stop

    @inbounds while !iszero(next)
        root = next; next = zero(V)
        c = root - one(V)

        while c >= fdesc[root]
            if count[c] == maxcnt
                next = c
            end

            c = fdesc[c] - one(V)
        end
    end

    return root
end

# Number the vertices at positions `blck:stop`: add them to the skeleton
# graph, and update the counts of the unnumbered vertices. A vertex x
# gains one for every numbered vertex b adjacent to T[x]: walk up from
# each neighbor of b, stopping at vertices that have already been
# visited for b.
function mcs_etree_number!(
        stamp::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        target::AbstractVector{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
        graph::AbstractGraph{V},
        tree::Parent{V},
        count::AbstractVector{V},
        strt::V,
        stop::V,
        blck::V,
    ) where {V, E}
    @assert nv(graph) <= length(stamp)
    @assert nv(graph) < length(begptr)
    @assert nv(graph) < length(endptr)
    @assert de(graph) <= length(target)
    @assert nv(graph) <= length(order)
    @assert nv(graph) <= length(index)
    @assert nv(graph) == length(tree)
    @assert nv(graph) <= length(count)
    @assert nv(graph) >= stop >= blck >= strt >= one(V)
    parent = tree.prnt

    @inbounds for b in blck:stop
        v = order[b]

        for w in neighbors(graph, v)
            j = index[w]

            if j < blck
                p = endptr[w] += one(E); target[p] = b
                x = j

                while stamp[x] != b
                    stamp[x] = b; count[x] += one(V)
                    y = parent[x]
                    zero(V) < y < blck || break
                    x = y
                end
            end
        end
    end

    return
end

# Recount the stale vertices of T. The stale vertices form an upper set
# of T, and the other vertices form unchanged subtrees whose counts and
# skeleton adjacency sets are valid. Each unchanged subtree is contracted
# to a single item, and the counts are computed with the column-count
# algorithm on the contracted tree. The skeleton adjacency set of each
# stale vertex is rebuilt.
#
# Items are identified by the positions of their tops; the stale
# vertices are stored as positive integers and the tops of the unchanged
# subtrees as negative integers.
function mcs_etree_recount!(
        wt::AbstractVector{V},
        ufp::AbstractVector{V},
        rank::AbstractVector{V},
        rep::AbstractVector{V},
        items::AbstractVector{V},
        touched::AbstractVector{V},
        prev_p::AbstractVector{V},
        prev_nbr::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        target::AbstractVector{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
        graph::AbstractGraph{V},
        tree::Parent{V},
        fdesc::AbstractVector{V},
        count::AbstractVector{V},
        stale::AbstractVector{V},
        strt::V,
        stop::V,
    ) where {V, E}
    @assert nv(graph) <= length(wt)
    @assert nv(graph) <= length(ufp)
    @assert nv(graph) <= length(rank)
    @assert nv(graph) <= length(rep)
    @assert nv(graph) <= length(items)
    @assert nv(graph) <= length(touched)
    @assert nv(graph) <= length(prev_p)
    @assert nv(graph) <= length(prev_nbr)
    @assert nv(graph) < length(begptr)
    @assert nv(graph) < length(endptr)
    @assert de(graph) <= length(target)
    @assert nv(graph) <= length(order)
    @assert nv(graph) <= length(index)
    @assert nv(graph) == length(tree)
    @assert nv(graph) <= length(fdesc)
    @assert nv(graph) <= length(count)
    @assert nv(graph) <= length(stale)
    @assert nv(graph) >= stop >= strt >= one(V)
    parent = tree.prnt

    # `prev_nbr` is all-zero on entry, and restored before returning.
    # `prev_p[u]` is only read after `prev_nbr[u]` is set, so it needs no reset.
    nitm = ntch = zero(V); x = stop

    # enumerate the items in decreasing order
    @inbounds while x >= strt
        nitm += one(V); wt[x] = rank[x] = zero(V); ufp[x] = rep[x] = x

        if ispositive(stale[order[x]])
            items[nitm] = x; x -= one(V)
        else
            items[nitm] = -x; x = fdesc[x] - one(V)
        end
    end

    # process the items in postorder
    @inbounds for k in reverse(oneto(nitm))
        t = items[k]

        if isnegative(t)
            # an unchanged subtree: visit the skeleton
            # adjacency sets of its vertices
            t = -t

            for i in fdesc[t]:t
                v = order[i]

                for p in begptr[v]:endptr[v]
                    _, ntch = mcs_etree_visit!(wt, ufp, rep, prev_p, prev_nbr,
                        touched, fdesc, ntch, t, target[p])
                end
            end
        else
            # a stale vertex: rebuild its skeleton adjacency set
            v = order[t]; stale[v] = zero(V)
            endptr[v] = begptr[v] - one(E)

            for w in neighbors(graph, v)
                u = index[w]

                if u > stop
                    isleaf, ntch = mcs_etree_visit!(wt, ufp, rep, prev_p, prev_nbr,
                        touched, fdesc, ntch, t, u)

                    if isleaf
                        target[endptr[v] += one(E)] = u
                    end
                end
            end
        end

        if t < stop
            mcs_etree_link!(ufp, rank, rep, t, parent[t])
        end
    end

    @inbounds for k in reverse(oneto(nitm))
        t = items[k]

        if isnegative(t)
            t = -t
        else
            count[t] = wt[t]
        end

        if t < stop
            wt[parent[t]] += wt[t]
        end
    end

    @inbounds for k in oneto(ntch)
        prev_nbr[touched[k]] = zero(V)
    end

    return
end

# visit a numbered neighbor `u` of the item `t`; returns `true`
# if `t` is a leaf of the row subtree of `u`
@inline function mcs_etree_visit!(
        wt::AbstractVector{V},
        ufp::AbstractVector{V},
        rep::AbstractVector{V},
        prev_p::AbstractVector{V},
        prev_nbr::AbstractVector{V},
        touched::AbstractVector{V},
        fdesc::AbstractVector{V},
        ntch::V,
        t::V,
        u::V,
    ) where {V}
    @inbounds pnbr = prev_nbr[u]
    @inbounds isleaf = iszero(pnbr) || pnbr < fdesc[t]

    @inbounds if isleaf
        wt[t] += one(V)

        if iszero(pnbr)
            ntch += one(V); touched[ntch] = u
        else
            q = rep[mcs_etree_root!(ufp, prev_p[u])]
            wt[q] -= one(V)
        end

        prev_p[u] = t
    end

    @inbounds prev_nbr[u] = t
    return isleaf, ntch
end

# Disjoint sets with union by rank and path compression. The roots `r`
# satisfy `prnt[r] == r`, and the set rooted at `r` is named `rep[r]`.
@inline function mcs_etree_root!(prnt::AbstractVector{V}, x::V) where {V}
    r = x

    @inbounds while prnt[r] != r
        r = prnt[r]
    end

    @inbounds while prnt[x] != r
        y = prnt[x]; prnt[x] = r; x = y
    end

    return r
end

# merge the sets containing `x` and `y`, and name the result
# after the set containing `y`
@inline function mcs_etree_link!(
        prnt::AbstractVector{V},
        rank::AbstractVector{V},
        rep::AbstractVector{V},
        x::V,
        y::V,
    ) where {V}
    rx = mcs_etree_root!(prnt, x)
    ry = mcs_etree_root!(prnt, y)

    @inbounds if rx != ry
        name = rep[ry]

        if rank[rx] > rank[ry]
            rx, ry = ry, rx
        elseif rank[rx] == rank[ry]
            rank[ry] += one(V)
        end

        prnt[rx] = ry; rep[ry] = name
    end

    return
end

# Find a block of vertices that can be numbered together with `node`
# (Heggernes and Peyton, Lemmas 5.2 and 5.3). The subtree T[`node`]
# occupies the positions fdesc[`node`]:`node`, which need not start at `strt`.
#
# On return, `block[blck]`, ..., `block[stop]` are the positions of the
# block, where `block[stop] = node` and the others are ancestors of `node`
# in ascending order. The second return value is the number of ancestors
# of `node`.
function mcs_etree_findblock!(
        marks::AbstractVector{V},
        block::AbstractVector{V},
        owner::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        target::AbstractVector{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
        graph::AbstractGraph{V},
        fdesc::AbstractVector{V},
        tree::Parent{V},
        strt::V,
        stop::V,
        node::V,
    ) where {V, E}
    @assert nv(graph) <= length(marks)
    @assert nv(graph) <= length(block)
    @assert nv(graph) <= length(owner)
    @assert nv(graph) < length(begptr)
    @assert nv(graph) < length(endptr)
    @assert de(graph) <= length(target)
    @assert nv(graph) <= length(order)
    @assert nv(graph) <= length(index)
    @assert nv(graph) <= length(fdesc)
    @assert nv(graph) == length(tree)
    @assert nv(graph) >= stop >= node >= strt

    # `marks` is all-zero on entry; it is restored before returning
    ndeg = nanc = zero(V); blck = stop

    @inbounds lo = fdesc[node]
    @inbounds block[blck] = node

    if lo < node
        # store the children of `node` in increasing order in
        # `block[strt]`, ..., `block[strt + nchd - 1]`, using the
        # sibling walk c -> fdesc[c] - 1. These entries do not overlap
        # the block, since strt + nchd <= node <= stop - nanc <= blck.
        nchd = zero(V); c = node - one(V)

        @inbounds while c >= lo
            nchd += one(V); c = fdesc[c] - one(V)
        end

        k = strt + nchd; c = node - one(V)

        @inbounds while c >= lo
            k -= one(V); block[k] = c; c = fdesc[c] - one(V)
        end

        # `owner[j]` is the child of `node` whose subtree contains `j`
        @inbounds for k in strt:strt + nchd - one(V)
            c = block[k]

            for j in fdesc[c]:c
                owner[j] = c
            end
        end

        @inbounds for p in begptr[order[node]]:endptr[order[node]]
            i = target[p]
            marks[i] = node; ndeg += one(V)
        end

        i = node

        @inbounds while i < stop
            i = parentindex(tree, i)::V
            ideg = ichd = zero(V); nanc += one(V)

            for v in neighbors(graph, order[i])
                j = index[v]

                if stop < j
                    if zero(V) < marks[j] < i
                        marks[j] = i; ideg += one(V)
                    end
                elseif lo <= j < node
                    k = owner[j]

                    if marks[k] < i
                        marks[k] = i; ichd += one(V)
                    end
                end
            end

            if ideg == ndeg && ichd == nchd
                blck -= one(V); block[blck] = i
            end
        end

        # restore `marks`: the only entries written are the
        # children of `node` and its skeleton neighbors
        @inbounds for k in strt:strt + nchd - one(V)
            marks[block[k]] = zero(V)
        end

        @inbounds for p in begptr[order[node]]:endptr[order[node]]
            marks[target[p]] = zero(V)
        end
    else
        @inbounds for v in neighbors(graph, order[node])
            i = index[v]
            marks[i] = strt; ndeg += one(V)
        end

        i = node

        @inbounds while i < stop
            i = parentindex(tree, i)::V
            nanc += one(V)

            if ispositive(marks[i])
                ideg = one(V)

                for v in neighbors(graph, order[i])
                    j = index[v]

                    if zero(V) < marks[j] < i
                        marks[j] = i; ideg += one(V)
                    end
                end

                if ideg == ndeg
                    blck -= one(V); block[blck] = i
                end
            end
        end

        # restore `marks`: the only entries written
        # are the neighbors of `node`
        @inbounds for v in neighbors(graph, order[node])
            marks[index[v]] = zero(V)
        end
    end

    return blck, nanc
end

# The block consists of `root` and all of its ancestors, so Change_Root
# is the identity on the remaining vertices, and their subtrees are
# unchanged. Stable-partition the positions `root:stop` so that the
# remaining vertices keep their relative order (each of their subtrees
# shifts as a unit) and `block[k]` moves to position `k`. Then push the
# roots of the remaining subtrees: the children of the block. The cost
# is O(stop - root + c), where c is the number of children of the block.
function mcs_etree_fastnumber!(
        newpos::AbstractVector{V},
        neworder::AbstractVector{V},
        newprnt::AbstractVector{V},
        newfdesc::AbstractVector{V},
        newcount::AbstractVector{V},
        block::AbstractVector{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
        tree::Parent{V},
        fdesc::AbstractVector{V},
        count::AbstractVector{V},
        stops::AbstractVector{V},
        num::V,
        stop::V,
        root::V,
        blck::V,
    ) where {V}
    @assert length(tree) <= length(newpos)
    @assert length(tree) <= length(neworder)
    @assert length(tree) <= length(newprnt)
    @assert length(tree) <= length(newfdesc)
    @assert length(tree) <= length(newcount)
    @assert length(tree) <= length(block)
    @assert length(tree) <= length(order)
    @assert length(tree) <= length(index)
    @assert length(tree) <= length(fdesc)
    @assert length(tree) <= length(count)
    @assert length(tree) <= length(stops)
    @assert length(tree) >= stop >= blck >= root >= one(V)
    parent = tree.prnt

    # `newpos[i]` is the new position of position `i`,
    # negated if `i` is in the block
    @inbounds for i in root:stop
        newpos[i] = zero(V)
    end

    @inbounds for k in blck:stop
        newpos[block[k]] = -k
    end

    shift = zero(V)

    @inbounds for i in root:stop
        if isnegative(newpos[i])
            shift += one(V)
        else
            newpos[i] = i - shift
        end
    end

    # push the children of the block that are not in the block
    @inbounds for k in blck:stop
        b = block[k]; c = b - one(V)

        while c >= fdesc[b]
            if c < root
                num += one(V); stops[num] = c; parent[c] = zero(V)
            elseif ispositive(newpos[c])
                num += one(V); stops[num] = newpos[c]
            end

            c = fdesc[c] - one(V)
        end
    end

    # permute `order`, `index`, `tree`, `fdesc`, and `count` on
    # `root:stop`; the positions `strt:root - 1` do not move
    @inbounds for i in root:stop
        p = newpos[i]

        if isnegative(p)
            p = -p
            neworder[p] = order[i]
            newprnt[p] = zero(V)
            newfdesc[p] = p
            newcount[p] = zero(V)
        else
            # `i` is not `stop`, which is in the block
            j = parent[i]
            neworder[p] = order[i]
            newprnt[p] = max(newpos[j], zero(V))
            newfdesc[p] = fdesc[i] - (i - p)
            newcount[p] = count[i]
        end
    end

    @inbounds for p in root:stop
        v = order[p] = neworder[p]
        index[v] = p
        parent[p] = newprnt[p]
        fdesc[p] = newfdesc[p]
        count[p] = newcount[p]
    end

    return num
end

# Change the root of T to `root` and number the block (Change_Root2).
#
# Let c(1) = `root`, c(2), ..., c(nanc + 1) = `stop` be the chain anc[`root`],
# and let R be the chain vertices that are not in the block. The remaining
# vertices of T form unchanged subtrees whose roots are children of chain
# vertices. The level of an unchanged subtree is the chain index of its
# parent, and the level of a vertex in R is the least level (or chain index)
# of a subtree (or chain vertex) that contains one of its neighbors in T.
# The new ordering of T is
#
#     unchanged subtrees, R by decreasing level, the block,
#
# whose elimination forest is computed on "items": the unchanged subtrees
# (leaves) and the vertices of R. Unchanged subtrees keep their relative
# order where possible and move as units, so the cost is proportional to
# the number of items, the degrees of the vertices in R, and the number of
# vertices that move.
function mcs_etree_rotate!(
        s1::AbstractVector{V},
        s2::AbstractVector{V},
        items::AbstractVector{V},
        s4::AbstractVector{V},
        s5::AbstractVector{V},
        s6::AbstractVector{V},
        s7::AbstractVector{V},
        bc::AbstractVector{V},
        block::AbstractVector{V},
        stamp::AbstractVector{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
        graph::AbstractGraph{V},
        tree::Parent{V},
        fdesc::AbstractVector{V},
        count::AbstractVector{V},
        stale::AbstractVector{V},
        stops::AbstractVector{V},
        num::V,
        strt::V,
        stop::V,
        root::V,
        blck::V,
        nanc::V,
        mark::V,
    ) where {V}
    @assert nv(graph) <= length(s1)
    @assert nv(graph) <= length(s2)
    @assert nv(graph) <= length(items)
    @assert nv(graph) <= length(s4)
    @assert nv(graph) <= length(s5)
    @assert nv(graph) <= length(s6)
    @assert nv(graph) <= length(s7)
    @assert nv(graph) <= length(bc)
    @assert nv(graph) <= length(block)
    @assert nv(graph) <= length(stamp)
    @assert nv(graph) <= length(order)
    @assert nv(graph) <= length(index)
    @assert nv(graph) == length(tree)
    @assert nv(graph) <= length(fdesc)
    @assert nv(graph) <= length(count)
    @assert nv(graph) <= length(stale)
    @assert nv(graph) <= length(stops)
    @assert nv(graph) >= stop >= blck > root >= strt >= one(V)
    @assert isnegative(mark)
    parent = tree.prnt

    # the work arrays are reused in phases:
    #
    #            phases 1-5     phases 6-7   phase 8
    #     s1     lzanc          child        bo (order)
    #     s2     level          brother      bp (parent)
    #     s4     ianc           stack        bf (fdesc)
    #     s5     head/iprnt     iprnt        iprnt
    #     s6     next/irep      newpos       newpos
    #     s7     rord           sublo        sublo
    #     bc     irank          -            bc (count)
    #
    # A position x is "stamped" if stamp[x] == mark. A stamped position
    # with lzanc[x] < 0 is the chain vertex c(-lzanc[x]). A stamped
    # position with lzanc[x] > 0 lies in the unchanged subtree whose
    # root is at position lzanc[x].
    lzanc = s1; level = s2; ianc = s4; iprnt = s5; head = s5; next = s6; rord = s7
    irep = s6; irank = bc

    # 1. stamp the chain; level[c] = 0 for the block and -1 for R
    i = root; h = one(V)

    @inbounds while true
        stamp[i] = mark; lzanc[i] = -h; level[i] = -one(V)
        i == stop && break
        i = parent[i]; h += one(V)
    end

    @inbounds for k in blck:stop
        level[block[k]] = zero(V)
    end

    # 2. enumerate the items in decreasing order of position;
    #    the vertices of R are stored as negative integers
    nitm = zero(V); x = stop

    @inbounds while x >= strt
        if stamp[x] == mark
            # a chain vertex
            if !iszero(level[x])
                nitm += one(V); items[nitm] = -x
            end

            x -= one(V)
        else
            # the root of an unchanged subtree
            nitm += one(V); items[nitm] = x
            stamp[x] = mark; lzanc[x] = x
            level[x] = -lzanc[parent[x]]
            x = fdesc[x] - one(V)
        end
    end

    # 3. compute the levels of the vertices in R
    @inbounds for k in oneto(nitm)
        r = items[k]

        if isnegative(r)
            r = -r; lev = nanc

            for w in neighbors(graph, order[r])
                y = index[w]

                if y <= stop && y != r
                    if stamp[y] == mark && isnegative(lzanc[y])
                        lev = min(lev, -lzanc[y])
                    else
                        lev = min(lev, level[mcs_etree_findtop!(stamp, lzanc, parent, mark, y)])
                    end
                end
            end

            level[r] = lev
        end
    end

    # 4. sort R by decreasing level; ties are broken by position.
    #    Mark the vertices of R as stale.
    @inbounds for lev in oneto(nanc)
        head[lev] = zero(V)
    end

    @inbounds for k in oneto(nitm)
        r = items[k]

        if isnegative(r)
            r = -r; lev = level[r]
            next[r] = head[lev]; head[lev] = r
            stale[order[r]] = one(V)
        end
    end

    nrem = zero(V)

    @inbounds for lev in reverse(oneto(nanc))
        r = head[lev]

        while ispositive(r)
            nrem += one(V); rord[nrem] = r; r = next[r]
        end
    end

    # 5. compute the elimination forest of the items. Before a vertex r
    #    of R is processed, iprnt[r] = -1.
    @inbounds for k in oneto(nitm)
        t = items[k]

        if isnegative(t)
            iprnt[-t] = -one(V)
        else
            iprnt[t] = zero(V); ianc[t] = irep[t] = t; irank[t] = zero(V)
        end
    end

    @inbounds for j in oneto(nrem)
        r = rord[j]; iprnt[r] = zero(V); ianc[r] = irep[r] = r; irank[r] = zero(V)

        for w in neighbors(graph, order[r])
            y = index[w]

            if y <= stop && y != r
                if stamp[y] == mark && isnegative(lzanc[y])
                    # skip the block and the unprocessed vertices of R
                    (iszero(level[y]) || isnegative(iprnt[y])) && continue
                    s = y
                else
                    s = mcs_etree_findtop!(stamp, lzanc, parent, mark, y)
                end

                t = irep[mcs_etree_root!(ianc, s)]

                if t != r
                    mcs_etree_link!(ianc, irank, irep, s, r); iprnt[t] = r
                end
            end
        end
    end

    # 6. sort the children of every item by position
    child = s1; brother = s2; roots = zero(V)

    @inbounds for k in oneto(nitm)
        child[abs(items[k])] = zero(V)
    end

    @inbounds for k in oneto(nitm)
        t = abs(items[k]); p = iprnt[t]

        if iszero(p)
            brother[t] = roots; roots = t
        else
            brother[t] = child[p]; child[p] = t
        end
    end

    # 7. postorder the items. `newpos[t]` is the new position of the
    #    top of item `t`, and `sublo[t]` is its new first descendant
    stack = s4; newpos = s6; sublo = s7
    n = strt; r = roots

    @inbounds while ispositive(r)
        nstk = one(V); stack[nstk] = r; sublo[r] = n

        while ispositive(nstk)
            t = stack[nstk]; c = child[t]

            if ispositive(c)
                child[t] = brother[c]
                nstk += one(V); stack[nstk] = c; sublo[c] = n
            else
                nstk -= one(V)

                if !ispositive(stale[order[t]])
                    n += t - fdesc[t]
                end

                newpos[t] = n; n += one(V)
            end
        end

        r = brother[r]
    end

    @assert n == blck

    # push the roots of the new subtrees
    r = roots

    @inbounds while ispositive(r)
        num += one(V); stops[num] = newpos[r]; r = brother[r]
    end

    # 8. move the vertices: gather into buffers indexed by new
    #    position, then scatter
    bo = s1; bp = s2; bf = s4

    @inbounds for k in oneto(nitm)
        t = items[k]

        if isnegative(t)
            t = -t; d = newpos[t]; p = iprnt[t]
            bo[d] = order[t]
            bp[d] = iszero(p) ? zero(V) : newpos[p]
            bf[d] = sublo[t]
            bc[d] = zero(V)
        else
            d = newpos[t]; p = iprnt[t]
            np = iszero(p) ? zero(V) : newpos[p]
            off = d - t

            if iszero(off)
                # the subtree does not move: only the parent of its root changes
                parent[t] = np
            else
                for i in fdesc[t]:t - one(V)
                    bo[i + off] = order[i]
                    bp[i + off] = parent[i] + off
                    bf[i + off] = fdesc[i] + off
                    bc[i + off] = count[i]
                end

                bo[d] = order[t]; bp[d] = np; bf[d] = fdesc[t] + off; bc[d] = count[t]
            end
        end
    end

    @inbounds for k in blck:stop
        bo[k] = order[block[k]]; bp[k] = zero(V); bf[k] = k; bc[k] = zero(V)
    end

    @inbounds for k in oneto(nitm)
        t = items[k]

        if isnegative(t)
            lo = hi = newpos[-t]
        else
            hi = newpos[t]
            hi == t && continue
            lo = bf[hi]
        end

        for d in lo:hi
            v = order[d] = bo[d]; index[v] = d
            parent[d] = bp[d]; fdesc[d] = bf[d]; count[d] = bc[d]
        end
    end

    @inbounds for d in blck:stop
        v = order[d] = bo[d]; index[v] = d
        parent[d] = bp[d]; fdesc[d] = bf[d]; count[d] = bc[d]
    end

    return num
end

# find the root of the unchanged subtree containing the position `y`,
# which is not on the chain; compress the path
@inline function mcs_etree_findtop!(
        stamp::AbstractVector{V},
        lzanc::AbstractVector{V},
        parent::AbstractVector{V},
        mark::V,
        y::V,
    ) where {V}
    x = y

    @inbounds while stamp[x] != mark
        x = parent[x]
    end

    @inbounds top = lzanc[x]

    @inbounds while stamp[y] != mark
        z = parent[y]; stamp[y] = mark; lzanc[y] = top; y = z
    end

    return top
end
