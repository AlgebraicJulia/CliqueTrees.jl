# Fast Computation of Minimal Fill inside a Given Elimination Ordering
# Heggernes and Peyton
# MCS-ETree (enhanced implementation)
#
# The time complexity is
#
#     O(mn α(m, n))
#
# m = |E|, n = |V|, and α is the extremely-slow-growing
# inverse of the Ackermann function.
function mcs_etree(weights::AbstractVector, graph::AbstractGraph{V}, alg::PermutationOrAlgorithm) where {V}
    E = etype(graph); m = de(graph); n = nv(graph)
    
    begptr = FVector{E}(undef, n + one(V))
    endptr = FVector{E}(undef, n + one(V))
    target = FVector{V}(undef, m)
    tree = Parent{V}(n)
    sets = UnionFind{V}(n)
    fdesc = FVector{V}(undef, n)
    fancs = FVector{V}(undef, n)
    stack = FVector{V}(undef, n)
    count = FVector{V}(undef, n)
    
    work1 = FVector{V}(undef, n)
    work2 = FVector{V}(undef, n)
    work3 = FVector{V}(undef, n)
    work4 = FVector{V}(undef, n)

    order, index = permutation(weights, graph, alg)

    mcs_etree_impl!(work1, work2, work3, work4, begptr, endptr,
        target, tree, sets, fdesc, fancs, stack, count,
        graph, order, index)

    return order, index
end

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
        stack::AbstractVector{V},
        count::AbstractVector{V},
        graph::AbstractGraph{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
    ) where {V, E}
    @argcheck nv(graph) <= length(work1)
    @argcheck nv(graph) <= length(work2)
    @argcheck nv(graph) <= length(work3)
    @argcheck nv(graph) <= length(work4)
    @argcheck nv(graph) < length(begptr)
    @argcheck nv(graph) < length(endptr)
    @argcheck de(graph) <= length(target)
    @argcheck nv(graph) == length(tree)
    @argcheck nv(graph) == length(sets)
    @argcheck nv(graph) <= length(fdesc)
    @argcheck nv(graph) <= length(fancs)
    @argcheck nv(graph) <= length(stack)
    @argcheck nv(graph) <= length(count)
    @argcheck nv(graph) <= length(order)
    @argcheck nv(graph) <= length(index)
    
    num = zero(V)

    upper = sympermute!_impl!(endptr, target, graph, index, Forward)
    etree_impl!(tree, work1, upper)
    postorder!_impl!(work1, work2, work3, work4, tree)
    firstdescendants_impl!(fdesc, tree, vertices(graph))

    begptr[one(V)] = p = one(E)
    endptr[one(V)] = p - one(E)

    @inbounds for v in vertices(graph)
        i = index[v] = work3[index[v]]
        order[i] = v
        begptr[v + one(V)] = p += convert(E, eltypedegree(graph, v))
        endptr[v + one(V)] = p - one(E)

        if isnothing(parentindex(tree, v))
            num += one(V); fancs[num] = v + one(V); stack[num] = v
        end
    end

    @inbounds while ispositive(num)
        # get an unnumbered elimination subtree T
        fanc = fancs[num]
        stop = stack[num]
        strt = fdesc[stop]
        num -= one(V)
    
        # find a special vertex `root` in T of maximum cardinality
        mcs_etree_supcnt!(count, begptr, endptr, target, work1, work2, fdesc,
            work3, work4, sets, order, index, graph, tree, strt, stop, fanc)

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

        # ensure that the vertices in anc[`root`] are numered before their siblings
        root = mcs_etree_prescribed!(work3, fdesc, tree, strt, stop, root)
        mcs_etree_invpermute!(work1, order, index, work3, strt, stop)
        mcs_etree_invpermute!(work1, tree, work3, strt, stop)
        mcs_etree_firstdescendants!(fdesc, tree, strt, stop)

        # reorder the subtree and change the root to `root`
        fanc = mcs_etree_changeroot!(work1, work2, work3, order, index, graph, fdesc, tree, strt, stop, root)
        mcs_etree_invpermute!(work1, order, index, work3, strt, stop)
        mcs_etree_invpermute!(work1, tree, work3, strt, stop)    

        # compute the elimination subtree for the new reordering
        mcs_etree_etree!(tree, order, index, work1, graph, strt, stop, fanc)
        mcs_etree_postorder!(work1, work2, work3, work4, tree, strt, stop)
        mcs_etree_firstdescendants!(fdesc, tree, strt, stop)
        mcs_etree_invpermute!(work1, order, index, work3, strt, stop)

        for v in neighbors(graph, order[stop])
            i = index[v]

            if i < stop
                target[endptr[v] += one(E)] = stop
            end
        end
        
        # number `root` and store the unnumbered subtrees for future processing 
        for i in strt:stop - one(V)
            j = parentindex(tree, i)::V

            if j == stop
                num += one(V); stack[num] = i

                if fanc <= j
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
    @argcheck length(tree) <= length(index)
    @argcheck length(tree) <= length(fdesc)
    @argcheck length(tree) >= stop >= root >= strt

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
    @argcheck nv(graph) <= length(head)
    @argcheck nv(graph) <= length(next)
    @argcheck nv(graph) <= length(alpha)
    @argcheck nv(graph) <= length(order)
    @argcheck nv(graph) <= length(index)
    @argcheck nv(graph) <= length(fdesc)
    @argcheck nv(graph) <= length(tree)
    @argcheck nv(graph) >= stop >= node >= strt >= one(V)

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
    @argcheck nv(graph) == length(tree)
    @argcheck nv(graph) <= length(order)
    @argcheck nv(graph) <= length(index)
    @argcheck nv(graph) <= length(ancestor)
    @argcheck nv(graph) >= stop >= fanc >= strt >= one(V)
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
    @argcheck length(tree) <= length(brother)
    @argcheck length(tree) <= length(child)
    @argcheck length(tree) >= stop >= strt >= one(V)

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
    @argcheck length(tree) <= length(brother)
    @argcheck length(tree) <= length(child)
    @argcheck length(tree) <= length(index)
    @argcheck length(tree) <= length(stack)
    @argcheck length(tree) >= stop >= strt >= one(V)

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
    @argcheck stop <= length(order)
    @argcheck stop <= length(index)
    @argcheck stop <= length(alpha)
    @argcheck stop >= strt

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
    @argcheck length(tree) <= length(parent)
    @argcheck length(tree) <= length(index)
    @argcheck length(tree) >= stop >= strt >= one(V)

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
    @argcheck length(tree) <= length(fdesc)
    @argcheck length(tree) >= stop >= strt >= one(V)

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
    @argcheck nv(graph) <= length(wt)
    @argcheck nv(graph) < length(begptr)
    @argcheck nv(graph) < length(endptr)
    @argcheck de(graph) <= length(target)
    @argcheck nv(graph) <= length(map1)
    @argcheck nv(graph) <= length(inv1)
    @argcheck nv(graph) <= length(fdesc)
    @argcheck nv(graph) <= length(prev_p)
    @argcheck nv(graph) <= length(prev_nbr)
    @argcheck nv(graph) <= length(order)
    @argcheck nv(graph) <= length(index)
    @argcheck nv(graph) <= length(tree)
    @argcheck nv(graph) >= stop >= strt >= one(V)
    @argcheck fanc >= strt

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

    @inbounds for p in strt:stop
        v = order[p]

        if p < fanc
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
        else
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
