function mcs_etree(weights::AbstractVector, graph::AbstractGraph{V}, alg::PermutationOrAlgorithm) where {V}
    order, index = permutation(weights, graph, alg)
    return mcs_etree!(order, index, graph)
end

function mcs_etree!(order::AbstractVector{V}, index::AbstractVector{V}, graph::AbstractGraph{V}) where {V}
    E = etype(graph); m = de(graph); n = nv(graph)
    
    begptr = FVector{E}(undef, n + one(V))
    endptr = FVector{E}(undef, n + one(V))
    target = FVector{V}(undef, m)
    tree = Parent{V}(n)
    sets = UnionFind{V}(n)
    fdesc = FVector{V}(undef, n)
    stale = FVector{V}(undef, n)
    stops = FVector{V}(undef, n)
    count = FVector{V}(undef, n)
    
    work1 = FVector{V}(undef, n)
    work2 = FVector{V}(undef, n)
    work3 = FVector{V}(undef, n)
    work4 = FVector{V}(undef, n)

    marks = FVector{V}(undef, n)

    mcs_etree_impl!(work1, work2, work3, work4, marks,
        begptr, endptr, target, tree, sets, fdesc, stale, stops, count,
        graph, order, index)

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
        marks::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        target::AbstractVector{V},
        tree::Parent{V},
        sets::UnionFind{V},
        fdesc::AbstractVector{V},
        stale::AbstractVector{V},
        stops::AbstractVector{V},
        count::AbstractVector{V},
        graph::AbstractGraph{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
    ) where {V, E}
    @assert nv(graph) <= length(work1)
    @assert nv(graph) <= length(work2)
    @assert nv(graph) <= length(work3)
    @assert nv(graph) <= length(work4)
    @assert nv(graph) <= length(marks)
    @assert nv(graph) < length(begptr)
    @assert nv(graph) < length(endptr)
    @assert de(graph) <= length(target)
    @assert nv(graph) == length(tree)
    @assert nv(graph) == length(sets)
    @assert nv(graph) <= length(fdesc)
    @assert nv(graph) <= length(stale)
    @assert nv(graph) <= length(stops)
    @assert nv(graph) <= length(count)
    @assert nv(graph) <= length(order)
    @assert nv(graph) <= length(index)

    # `stops` and `num` form a stack of the roots of the
    # unnumbered elimination subtrees
    #
    # `stale[v]` is positive if the subtree rooted at vertex `v`
    # has changed since its skeleton adjacency set was computed.
    # This is tracked per vertex rather than as a range of
    # positions, because `mcs_etree_postorder!` can interleave
    # the former ancestors of `root` with the unchanged subtrees.
    num = zero(V)

    # initialize elimination forest (`tree`) and first
    # descendants (`fdesc`)
    etree_impl!(tree, work1, graph, order, index)
    postorder!_impl!(work1, work2, work3, work4, tree)
    firstdescendants_impl!(fdesc, tree, vertices(graph))

    # - permute `order` and `index`
    # - initialize empty skeleton graph
    # - push roots to `stops`
    # - mark all skeletons as up to date (they are empty)
    begptr[one(V)] = p = one(E)
    endptr[one(V)] = p - one(E)

    @inbounds for v in vertices(graph)
        marks[v] = zero(V)
        stale[v] = zero(V)
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

        # compute higher degrees and adjust skeleton graph 
        mcs_etree_supcnt!(count, begptr, endptr, target, work1, work2, fdesc,
            work3, marks, sets, order, index, graph, tree, strt, stop, stale)

        # find a special vertex `root` in T of maximum cardinality
        root = stop; maxcnt = count[stop]
    
        for i in strt:stop
            work1[i] = -one(V)
        end
    
        for i in strt:stop - one(V)
            cum = work1[i]
            cnt = count[i]
            
            if cum < cnt == maxcnt
                root = i
                break
            else
                j = parentindex(tree, i)::V
                work1[j] = max(work1[j], cum, cnt)
            end
        end

        # find a block of vertices to number consecutively:
        # `block[blck]`, ..., `block[stop]`, where `block[stop] = root`
        # and the others are ancestors of `root`
        blck, nanc = mcs_etree_findblock!(marks, work2, begptr, endptr, target,
            order, index, graph, fdesc, tree, strt, stop, root)

        if blck + nanc == stop
            # the block contains every ancestor of `root`, so Change_Root
            # leaves the remaining subtrees untouched: move the block to
            # the end and store the remaining subtrees for future processing
            num = mcs_etree_fastnumber!(work1, work3, work4, count, work2,
                order, index, tree, fdesc, stops, num, stop, root, blck)
        else
            # ensure that the vertices in anc[`root`] are numbered before their siblings
            mcs_etree_prescribed!(work3, fdesc, tree, strt, stop, root)
            mcs_etree_invpermute!(work1, order, index, work3, strt, stop)
            mcs_etree_invpermute!(work1, tree, work3, strt, stop)
            mcs_etree_firstdescendants!(fdesc, tree, strt, stop)

            for k in blck:stop
                work2[k] = work3[work2[k]]
            end

            # reorder the subtree and change the root to `root`
            fanc = mcs_etree_changeroot!(work1, work2, work3, stale, order, index, graph, fdesc, tree, strt, stop, blck)
            mcs_etree_invpermute!(work1, order, index, work3, strt, stop)
            mcs_etree_invpermute!(work1, tree, work3, strt, stop)

            # compute the elimination subtree for the new reordering
            mcs_etree_etree!(tree, order, index, work1, graph, strt, stop, fanc)
            mcs_etree_postorder!(work1, work2, work3, work4, tree, strt, stop)
            mcs_etree_firstdescendants!(fdesc, tree, strt, stop)
            mcs_etree_invpermute!(work1, order, index, work3, strt, stop)

            # store the unnumbered subtrees for future processing
            for i in strt:blck - one(V)
                j = parentindex(tree, i)::V

                if j >= blck
                    num += one(V); stops[num] = i
                end
            end
        end

        # number the block and add it to the skeleton graph
        for i in blck:stop
            v = order[i]

            for w in neighbors(graph, v)
                j = index[w]

                if j < stop
                    p = endptr[w] += one(E); target[p] = i
                end
            end
        end
    end

    return
end

function mcs_etree_prescribed!(
        index::AbstractVector{V},
        fdesc::AbstractVector{V},
        tree::AbstractVector{V},
        strt::V,
        stop::V,
        root::V,
    ) where {V}
    @assert length(tree) <= length(index)
    @assert length(tree) <= length(fdesc)
    @assert length(tree) >= stop >= root >= strt

    @inbounds n = strt - one(V); curstrt = fdesc[root]; curstop = root

    @inbounds for j in curstrt:curstop
        index[j] = n += one(V)
    end

    i = root; root = n

    @inbounds while i < stop
        i = parentindex(tree, i)::V
        prvstrt, prvstop = curstrt, curstop
        curstrt, curstop = fdesc[i], i

        for j in curstrt:prvstrt - one(V)
            index[j] = n += one(V)
        end

        for j in prvstop + one(V):curstop
            index[j] = n += one(V)
        end
    end

    return root
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
                    # binary search for the child `k` whose subtree contains `j`
                    l = strt; h = strt + nchd - one(V)

                    while l < h
                        m = l + (h - l) ÷ convert(V, 2)

                        if block[m] < j
                            l = m + one(V)
                        else
                            h = m
                        end
                    end

                    k = block[l]
                    @assert fdesc[k] <= j <= k

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
        block::AbstractVector{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
        tree::Parent{V},
        fdesc::AbstractVector{V},
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
    @assert length(tree) <= length(block)
    @assert length(tree) <= length(order)
    @assert length(tree) <= length(index)
    @assert length(tree) <= length(fdesc)
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
                num += one(V); stops[num] = c
            elseif ispositive(newpos[c])
                num += one(V); stops[num] = newpos[c]
            end

            c = fdesc[c] - one(V)
        end
    end

    # permute `order`, `index`, `tree`, and `fdesc` on `root:stop`;
    # the positions `strt:root - 1` do not move
    @inbounds for i in root:stop
        p = newpos[i]

        if isnegative(p)
            p = -p
            neworder[p] = order[i]
            newprnt[p] = zero(V)
            newfdesc[p] = p
        else
            # `i` is not `stop`, which is in the block
            j = parent[i]
            neworder[p] = order[i]
            newprnt[p] = max(newpos[j], zero(V))
            newfdesc[p] = fdesc[i] - (i - p)
        end
    end

    @inbounds for p in root:stop
        v = order[p] = neworder[p]
        index[v] = p
        parent[p] = newprnt[p]
        fdesc[p] = newfdesc[p]
    end

    return num
end

# Change_Root2
function mcs_etree_changeroot!(
        head::AbstractVector{V},
        next::AbstractVector{V},
        alpha::AbstractVector{V},
        stale::AbstractVector{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
        graph::AbstractGraph{V},
        fdesc::AbstractVector{V},
        tree::Parent{V},
        strt::V,
        stop::V,
        blck::V,
    ) where {V}
    @assert nv(graph) <= length(head)
    @assert nv(graph) <= length(next)
    @assert nv(graph) <= length(alpha)
    @assert nv(graph) <= length(stale)
    @assert nv(graph) <= length(order)
    @assert nv(graph) <= length(index)
    @assert nv(graph) <= length(fdesc)
    @assert nv(graph) <= length(tree)
    @assert nv(graph) >= stop >= blck >= strt >= one(V)

    function set(i::V)
        @inbounds h = view(head, i)
        return SinglyLinkedList(h, next)
    end

    @inbounds node = next[stop]; n = strt

    @inbounds for i in strt:stop
        empty!(set(i))

        if i < node || node < fdesc[i]
            alpha[i] = n; n += one(V)
        else
            alpha[i] = zero(V)
        end
    end

    fanc = n

    @inbounds for n in blck:stop
        i = next[n]; alpha[i] = n
    end

    i = node

    @inbounds while i < stop
        i = parentindex(tree, i)::V

        if iszero(alpha[i])
            # `i` is an ancestor of `node` that is not in the
            # block: its subtree is about to change
            v = order[i]; n = stop; stale[v] = one(V)

            for w in neighbors(graph, v)
                if v != w
                    n = min(n, index[w])
                end
            end

            pushfirst!(set(n), i)
        end
    end

    n = blck - one(V)

    @inbounds for m in strt:stop, i in set(m)
        alpha[i] = n; n -= one(V)
    end

    return fanc
end

function mcs_etree_etree!(
        tree::Parent{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
        ancestor::AbstractVector{V},
        graph::AbstractGraph{V},
        strt::V,
        stop::V,
        fanc::V,
    ) where {V}
    @assert nv(graph) == length(tree)
    @assert nv(graph) <= length(order)
    @assert nv(graph) <= length(index)
    @assert nv(graph) <= length(ancestor)
    @assert nv(graph) >= stop >= fanc >= strt >= one(V)
    n = nv(graph); parent = tree.prnt

    @inbounds for i in reverse(strt:fanc - one(V))
        j = parent[i]

        if fanc <= j
            ancestor[i] = -i
        else
            ancestor[i] = abs(ancestor[j])
        end
    end

    @inbounds for i in fanc:stop
        v = order[i]; ancestor[i] = zero(V)

        for w in neighbors(graph, v)
            k = index[w]

            if k < i
                r = k

                while ispositive(ancestor[r]) && ancestor[r] != i
                    t = ancestor[r]
                    ancestor[r] = i
                    r = t
                end

                if !ispositive(ancestor[r])
                    ancestor[r] = i
                    parent[r] = i
                end
            end
        end
    end

    return
end

function mcs_etree_lcrs!(
        brother::AbstractVector{V},
        child::AbstractVector{V},
        tree::Parent{V},
        strt::V,
        stop::V,
    ) where {V}
    @assert length(tree) <= length(brother)
    @assert length(tree) <= length(child)
    @assert length(tree) >= stop >= strt >= one(V)

    @inbounds for i in strt:stop
        child[i] = zero(V)
    end

    @inbounds brother[stop] = zero(V)

    @inbounds for i in reverse(strt:stop - one(V))
        j = parentindex(tree, i)::V
        brother[i] = child[j]; child[j] = i
    end

    return
end

function mcs_etree_postorder!(
        brother::AbstractVector{V},
        child::AbstractVector{V},
        index::AbstractVector{V},
        stack::AbstractVector{V},
        tree::Parent{V},
        strt::V,
        stop::V,
    ) where {V}
    @assert length(tree) <= length(brother)
    @assert length(tree) <= length(child)
    @assert length(tree) <= length(index)
    @assert length(tree) <= length(stack)
    @assert length(tree) >= stop >= strt >= one(V)

    mcs_etree_lcrs!(brother, child, tree, strt, stop)
    
    function brothers(i::V)
        @inbounds head = view(child, i)
        return SinglyLinkedList(head, brother)
    end

    @inbounds num = one(V); stack[num] = stop

    @inbounds for i in strt:stop
        j = stack[num]; num -= one(V)

        while !isempty(brothers(j))
            num += one(V); stack[num] = j
            j = popfirst!(brothers(j))
        end

        index[j] = i
    end

    mcs_etree_invpermute!(stack, tree, index, strt, stop)
    return
end

function mcs_etree_invpermute!(
        work::AbstractVector{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
        alpha::AbstractVector{V},
        strt::V,
        stop::V,
    ) where {V}
    @assert stop <= length(order)
    @assert stop <= length(index)
    @assert stop <= length(alpha)
    @assert stop >= strt

    @inbounds for i in strt:stop
        work[alpha[i]] = order[i]
    end

    @inbounds for i in strt:stop
        v = order[i] = work[i]
        index[v] = i
    end

    return
end

function mcs_etree_invpermute!(
        parent::AbstractVector{V},
        tree::Parent{V},
        index::AbstractVector{V},
        strt::V,
        stop::V,
    ) where {V}
    @assert length(tree) <= length(parent)
    @assert length(tree) <= length(index)
    @assert length(tree) >= stop >= strt >= one(V)

    @inbounds for i in strt:stop - one(V)
        j = parentindex(tree, i)::V
        parent[index[i]] = index[j]
    end

    @inbounds for i in strt:stop - one(V)
        tree.prnt[i] = parent[i]
    end

    return
end

function mcs_etree_firstdescendants!(
        fdesc::AbstractVector{V},
        tree::Parent{V},
        strt::V,
        stop::V,
    ) where {V}
    @assert length(tree) <= length(fdesc)
    @assert length(tree) >= stop >= strt >= one(V)

    @inbounds for i in strt:stop
        fdesc[i] = i
    end

    @inbounds for i in strt:stop - one(V)
        j = parentindex(tree, i)::V
        fdesc[j] = min(fdesc[i], fdesc[j])
    end

    return
end

function mcs_etree_supcnt!(
        wt::AbstractVector{V},
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        target::AbstractVector{V},
        map1::AbstractVector{V},
        inv1::AbstractVector{V},
        fdesc::AbstractVector{V},
        prev_p::AbstractVector{V},
        prev_nbr::AbstractVector{V},
        sets::UnionFind{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
        graph::AbstractGraph{V},
        tree::Parent{V},
        strt::V,
        stop::V,
        stale::AbstractVector{V},
    ) where {V, E}
    @assert nv(graph) <= length(wt)
    @assert nv(graph) < length(begptr)
    @assert nv(graph) < length(endptr)
    @assert de(graph) <= length(target)
    @assert nv(graph) <= length(map1)
    @assert nv(graph) <= length(inv1)
    @assert nv(graph) <= length(fdesc)
    @assert nv(graph) <= length(prev_p)
    @assert nv(graph) <= length(prev_nbr)
    @assert nv(graph) <= length(order)
    @assert nv(graph) <= length(index)
    @assert nv(graph) <= length(tree)
    @assert nv(graph) <= length(stale)
    @assert nv(graph) >= stop >= strt >= one(V)

    @inbounds for p in strt:stop
        wt[p] = zero(V)
        map1[p] = p
        inv1[p] = p
        sets.rank[p] = zero(V)
        sets.parent[p] = zero(V)
    end

    # `prev_nbr` is all-zero on entry, and restored before returning.
    # `prev_p[u]` is only read after `prev_nbr[u]` is set, so it needs no reset.

    function find(u::V)
        v = @inbounds inv1[sets[u]]
        return v
    end

    function union(u::V, v::V)
        @inbounds uu = map1[u]
        @inbounds vv = map1[v]
        @inbounds vv = map1[v] = union!(sets, uu, vv)
        @inbounds inv1[vv] = v
        return
    end

    # visit a numbered neighbor `u` of the vertex at position `p`;
    # returns `true` if `p` is a leaf of the row subtree of `u`
    function visit(p::V, u::V)
        @inbounds pnbr = prev_nbr[u]
        isleaf = iszero(pnbr) || pnbr < fdesc[p]

        @inbounds if isleaf
            wt[p] += one(V)

            if !iszero(pnbr)
                q = find(prev_p[u])
                wt[q] -= one(V)
            end

            prev_p[u] = p
        end

        @inbounds prev_nbr[u] = p
        return isleaf
    end

    @inbounds for p in strt:stop
        v = order[p]

        if ispositive(stale[v])
            # the subtree rooted at `v` has changed: rebuild
            # its skeleton adjacency set from its full adjacency set
            stale[v] = zero(V)
            endptr[v] = begptr[v] - one(E)

            for w in neighbors(graph, v)
                u = index[w]

                if u > stop && visit(p, u)
                    target[endptr[v] += one(E)] = u
                end
            end
        else
            # the subtree rooted at `v` is unchanged: its
            # skeleton adjacency set is still valid
            for t in begptr[v]:endptr[v]
                visit(p, target[t])
            end
        end

        if p < stop
            r = parentindex(tree, p)::V
            union(p, r)
        end
    end

    @inbounds for p in strt:stop - one(V)
        r = parentindex(tree, p)::V
        wt[r] += wt[p]
    end

    # every entry of `prev_nbr` written above is in the skeleton of T:
    # the first visit to `u` always appends it (or it was already there)
    @inbounds for p in strt:stop
        v = order[p]

        for t in begptr[v]:endptr[v]
            prev_nbr[target[t]] = zero(V)
        end
    end

    return
end
