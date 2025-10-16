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
    fancs = FVector{V}(undef, n)
    stops = FVector{V}(undef, n)
    count = FVector{V}(undef, n)
    
    work1 = FVector{V}(undef, n)
    work2 = FVector{V}(undef, n)
    work3 = FVector{V}(undef, n)
    work4 = FVector{V}(undef, n)

    mcs_etree_impl!(work1, work2, work3, work4, begptr, endptr,
        target, tree, sets, fdesc, fancs, stops, count,
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
        begptr::AbstractVector{E},
        endptr::AbstractVector{E},
        target::AbstractVector{V},
        tree::Parent{V},
        sets::UnionFind{V},
        fdesc::AbstractVector{V},
        fancs::AbstractVector{V},
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
    @assert nv(graph) < length(begptr)
    @assert nv(graph) < length(endptr)
    @assert de(graph) <= length(target)
    @assert nv(graph) == length(tree)
    @assert nv(graph) == length(sets)
    @assert nv(graph) <= length(fdesc)
    @assert nv(graph) <= length(fancs)
    @assert nv(graph) <= length(stops)
    @assert nv(graph) <= length(count)
    @assert nv(graph) <= length(order)
    @assert nv(graph) <= length(index)

    # `fancs`, `stops`, and `num` form a stack
    #  - `stops[num]` is the root of the current elimination tree
    #  - `fancs[num]` ... `stops[num] - 1` were ancestors of
    #    `stops[num]` before the previous rotation
    num = zero(V)

    # initialize elimination forest (`tree`) and first
    # descendants (`fdesc`)
    etree_impl!(tree, work1, graph, order, index)
    postorder!_impl!(work1, work2, work3, work4, tree)
    firstdescendants_impl!(fdesc, tree, vertices(graph))

    # - permute `order` and `index`
    # - initialize empty skeleton graph
    # - push roots to `stops`
    begptr[one(V)] = p = one(E)
    endptr[one(V)] = p - one(E)

    @inbounds for v in vertices(graph)
        i = index[v] = work3[index[v]]
        order[i] = v
        begptr[v + one(V)] = p += convert(E, eltypedegree(graph, v))
        endptr[v + one(V)] = p - one(E)

        if isnothing(parentindex(tree, v))
            num += one(V); fancs[num] = v + one(V); stops[num] = v
        end
    end

    @inbounds while ispositive(num)
        # get an unnumbered elimination subtree T
        fanc = fancs[num]
        stop = stops[num]
        strt = fdesc[stop]
        num -= one(V)

        # compute higher degrees and adjust skeleton graph 
        mcs_etree_supcnt!(count, begptr, endptr, target, work1, work2, fdesc,
            work3, work4, sets, order, index, graph, tree, strt, stop, fanc)

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

        # ensure that the vertices in anc[`root`] are numbered before their siblings
        root = mcs_etree_prescribed!(work3, fdesc, tree, strt, stop, root)
        mcs_etree_invpermute!(work1, order, index, work3, strt, stop)
        mcs_etree_invpermute!(work1, tree, work3, strt, stop)
        mcs_etree_firstdescendants!(fdesc, tree, strt, stop)

        # find a block of vertices to number consecutively
        blck = mcs_etree_findblock!(work1, work2, begptr, endptr, target,
            order, index, graph, fdesc, tree, strt, stop, root)

        # reorder the subtree and change the root to `root`
        fanc = mcs_etree_changeroot!(work1, work2, work3, order, index, graph, fdesc, tree, strt, stop, blck)
        mcs_etree_invpermute!(work1, order, index, work3, strt, stop)
        mcs_etree_invpermute!(work1, tree, work3, strt, stop)    

        # compute the elimination subtree for the new reordering
        mcs_etree_etree!(tree, order, index, work1, graph, strt, stop, fanc)
        mcs_etree_postorder!(work1, work2, work3, work4, tree, strt, stop)
        mcs_etree_firstdescendants!(fdesc, tree, strt, stop)
        mcs_etree_invpermute!(work1, order, index, work3, strt, stop)

        # add `root` to the skeleton graph
        for i in blck:stop
            v = order[i]

            for w in neighbors(graph, v)
                j = index[w]

                if j < stop
                    p = endptr[w] += one(V); target[p] = i
                end
            end
        end

        # number `root` and store the unnumbered subtrees for future processing 
        for i in strt:blck - one(V)
            j = parentindex(tree, i)::V

            if j >= blck
                num += one(V); stops[num] = i

                if fdesc[i] <= fanc <= i
                    fancs[num] = fanc
                else
                    fancs[num] = i + one(V)
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

    ndeg = zero(V); blck = stop

    @inbounds block[blck] = node

    @inbounds for i in strt:stop
        marks[i] = zero(V)
    end

    @inbounds for i in ancestorindices(tree, stop)
        marks[i] = zero(V)
    end

    if fdesc[node] < node
        nchd = zero(V)

        @inbounds for i in fdesc[node]:node - one(V)
            if parentindex(tree, i) == node
                nchd += one(V); block[strt + nchd - one(V)] = i
            end
        end

        @inbounds for p in begptr[order[node]]:endptr[order[node]]
            i = target[p]
            marks[i] = node; ndeg += one(V)
        end

        i = node

        @inbounds while i < stop
            i = parentindex(tree, i)::V
            ideg = ichd = zero(V)

            for v in neighbors(graph, order[i])
                j = index[v]

                if stop < j
                    if zero(V) < marks[j] < i
                        marks[j] = i; ideg += one(V)
                    end
                elseif j < node
                    for k in view(block, strt:strt + nchd - one(V))
                        if j <= k
                            @assert fdesc[k] <= j

                            if marks[k] < i
                                marks[k] = i; ichd += one(V)
                            end

                            break
                        end
                    end
                end
            end

            if ideg == ndeg && ichd == nchd
                blck -= one(V); block[blck] = i
            end
        end
    else
        @inbounds for v in neighbors(graph, order[node])
            i = index[v]
            marks[i] = strt; ndeg += one(V)
        end

        i = node

        @inbounds while i < stop
            i = parentindex(tree, i)::V

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
    end

    return blck
end

# Change_Root2
function mcs_etree_changeroot!(
        head::AbstractVector{V},
        next::AbstractVector{V},
        alpha::AbstractVector{V},
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
            v = order[i]; n = stop

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

#=
# Change_Root2
function mcs_etree_changeroot!(
        head::AbstractVector{V},
        next::AbstractVector{V},
        alpha::AbstractVector{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
        graph::AbstractGraph{V},
        fdesc::AbstractVector{V},
        tree::Parent{V},
        strt::V,
        stop::V,
        node::V,
    ) where {V}
    @assert nv(graph) <= length(head)
    @assert nv(graph) <= length(next)
    @assert nv(graph) <= length(alpha)
    @assert nv(graph) <= length(order)
    @assert nv(graph) <= length(index)
    @assert nv(graph) <= length(fdesc)
    @assert nv(graph) <= length(tree)
    @assert nv(graph) >= stop >= node >= strt >= one(V)

    function set(i::V)
        @inbounds h = view(head, i)
        return SinglyLinkedList(h, next)
    end

    n = strt

    @inbounds for i in strt:stop
        empty!(set(i))

        if i < node || node < fdesc[i]
            alpha[i] = n; n += one(V)
        end
    end

    fanc = n; i = node

    @inbounds while i < stop
        i = parentindex(tree, i)::V
        v = order[i]; n = stop

        for w in neighbors(graph, v)
            if v != w
                n = min(n, index[w])
            end
        end

        pushfirst!(set(n), i)
    end

    n = stop

    @inbounds alpha[node] = n; n -= one(V)

    @inbounds for m in strt:stop, i in set(m)
        alpha[i] = n; n -= one(V)
    end

    return fanc
end
=#

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
        fanc::V,
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
    @assert nv(graph) >= stop >= strt >= one(V)
    @assert stop + one(V) >= fanc >= strt

    @inbounds for p in fanc:stop
        v = order[p]
        endptr[v] = begptr[v] - one(E)
    end

    @inbounds for p in strt:stop
        wt[p] = zero(V)
        map1[p] = p
        inv1[p] = p
        sets.rank[p] = zero(V)
        sets.parent[p] = zero(V)
    end

    @inbounds for p in ancestorindices(tree, stop)
        prev_p[p] = zero(V)
        prev_nbr[p] = zero(V)
    end

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

    @inbounds for p in strt:fanc - one(V)
        v = order[p]

        for t in begptr[v]:endptr[v]
            u = target[t]

            if iszero(prev_nbr[u]) || prev_nbr[u] < fdesc[p]
                wt[p] += one(V)            
                pp = prev_p[u]

                if !iszero(pp)
                    q = find(pp)
                    wt[q] -= one(V)
                end

                prev_p[u] = p
            end

            prev_nbr[u] = p
        end

        if p < stop
            r = parentindex(tree, p)::V
            union(p, r)
        end
    end

    @inbounds for p in fanc:stop
        v = order[p]

        for w in neighbors(graph, v)
            u = index[w]

            if u > stop
                if iszero(prev_nbr[u]) || prev_nbr[u] < fdesc[p]
                    wt[p] += one(V)            
                    pp = prev_p[u]
    
                    if !iszero(pp)
                        q = find(pp)
                        wt[q] -= one(V)
                    end
    
                    prev_p[u] = p
                    target[endptr[v] += one(E)] = u
                end
    
                prev_nbr[u] = p
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

    return
end
