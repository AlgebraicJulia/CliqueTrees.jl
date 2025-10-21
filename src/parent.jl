"""
    Parent{V, Prnt} <: AbstractUnitRange{V}

A rooted forest T = (V, E) with edges oriented from
leaf to root.
"""
struct Parent{V <: Integer, Prnt <: AbstractVector{V}} <: AbstractUnitRange{V}
    """
    V is equal to the set

        V := {1, 2, ..., nv}

    """
    nv::V

    """
    E is equal to the set

        E := {(v, prnt[v]) : v ∈ V and prnt[v] > 0}

    In particular, v is a root if an only if prnt[v] is nonpositive.
    """
    prnt::Prnt

    function Parent{V, Prnt}(nv::Integer, prnt::AbstractVector) where {V <: Integer, Prnt <: AbstractVector{V}}
        @assert !isnegative(nv)
        @assert nv <= length(prnt)
        return new{V, Prnt}(nv, prnt)
    end
end

const FParent{V} = Parent{V, FVector{V}}

function Parent{V, Prnt}(nv::Integer) where {V, Prnt}
    prnt = Prnt(undef, nv)
    return Parent(convert(V, nv), prnt)
end

function Parent(nv::V, prnt::Prnt) where {V <: Integer, Prnt <: AbstractVector{V}}
    tree = Parent{V, Prnt}(nv, prnt)
    return tree
end

function Parent{V}(nv::Integer) where {V}
    return Parent{V, FVector{V}}(nv)
end

# A Compact Row Storage Scheme for Cholesky Factors Using Elimination Trees
# Liu
# Algorithm 4.2: Elimination Tree by Path Compression.
#
# Construct the elimination tree of an ordered graph.
# The complexity is O(m log n), where m = |E| and n = |V|.
function etree_impl!(
        tree::Parent{V},
        ancestor::AbstractVector{V},
        upper::AbstractGraph{V},
        order::AbstractVector{V}=vertices(upper),
        index::AbstractVector{V}=vertices(upper),
    ) where {V}
    @assert nv(upper) == length(tree)
    @assert nv(upper) <= length(ancestor)
    @assert nv(upper) <= length(order)
    @assert nv(upper) <= length(index)
    n = nv(upper); parent = tree.prnt

    @inbounds for i in oneto(n)
        parent[i] = zero(V)
        ancestor[i] = zero(V)

        for k in neighbors(upper, order[i])
            r = index[k]

            if r < i
                while !iszero(ancestor[r]) && ancestor[r] != i
                    t = ancestor[r]
                    ancestor[r] = i
                    r = t
                end

                if iszero(ancestor[r])
                    ancestor[r] = i
                    parent[r] = i
                end
            end
        end
    end

    return
end

# Compute the `root`, `child`, and `brother` fields of a forest.
function lcrs_impl!(
        brother::AbstractVector{V},
        child::AbstractVector{V},
        tree::Parent{V},
    ) where {V}
    @assert length(tree) <= length(brother)
    @assert length(tree) <= length(child)
    root = zero(V)

    @inbounds for i in tree
        child[i] = zero(V)
    end

    @inbounds for i in reverse(tree)
        j = parentindex(tree, i)

        if isnothing(j)
            brother[i] = root; root = i
        else
            brother[i] = child[j]; child[j] = i
        end
    end

    return root
end

function lcrs_impl!(
        brother::AbstractVector{V},
        child::AbstractVector{V},
        tree::Parent{V},
        root::V,
    ) where {V}
    @assert length(tree) <= length(brother)
    @assert length(tree) <= length(child)
    @assert length(tree) >= root >= one(V)

    @inbounds for i in tree
        child[i] = zero(V)
    end

    skip = root; root = zero(V)

    @inbounds for i in reverse(tree)
        if i != skip
            j = parentindex(tree, i)

            if isnothing(j)
                brother[i] = root; root = i
            else
                brother[i] = child[j]; child[j] = i
            end
        end
    end

    brother[skip] = root
    return
end

function reverse!_impl!(
        graph::BipartiteGraph{V, V},
        tree::Parent{V},
    ) where {V}
    @assert length(tree) == nv(graph) - one(V)
    @assert length(tree) == nov(graph)
    @assert length(tree) == ne(graph)

    @inbounds for i in vertices(graph)
        ii = i + one(V)
        pointers(graph)[ii] = zero(V)
    end

    @inbounds for i in tree
        j = parentindex(tree, i)
        
        if !isnothing(j)
            jj = j + two(V)
            pointers(graph)[jj] += one(V)
        end
    end
    
    @inbounds pointers(graph)[begin] = p = one(V)

    @inbounds for i in vertices(graph)
        ii = i + one(V)
        pointers(graph)[ii] = p += pointers(graph)[ii]
    end
    
    @inbounds for i in tree
        j = parentindex(tree, i)
                
        if isnothing(j)
            j = nv(graph)
        end

        jj = j + one(V)
        targets(graph)[pointers(graph)[jj]] = i
        pointers(graph)[jj] += one(V)
    end

    return graph
end

# An Efficient Algorithm to Compute Row and Column Counts for Sparse Cholesky Factorization
# Gilbert, Ng, and Peyton
# Figure 3: Implementation of algorithm to compute row and column counts.
#
# Compute the lower and higher degrees of the monotone transitive extesion of an ordered graph.
# The complexity is O(mα(m, n)), where m = |E|, n = |V|, and α is the inverse Ackermann function.
function supcnt_impl!(
        wt::AbstractVector{W},
        map0::AbstractVector{V},
        inv0::AbstractVector{V},
        inv1::AbstractVector{V},
        fdesc::AbstractVector{V},
        prev_p::AbstractVector{V},
        prev_nbr::AbstractVector{V},
        sets::UnionFind{V},
        weights::AbstractVector{W},
        lower::AbstractGraph{V},
        tree::Parent{V}
    ) where {W, V}
    @assert nv(lower) <= length(wt)
    @assert nv(lower) <= length(map0)
    @assert nv(lower) <= length(inv0)
    @assert nv(lower) <= length(inv1)
    @assert nv(lower) <= length(fdesc)
    @assert nv(lower) <= length(prev_p)
    @assert nv(lower) <= length(prev_nbr)
    @assert nv(lower) <= length(weights)
    @assert nv(lower) <= length(tree)

    n = nv(lower)
    postorder_impl!(map0, inv1, inv0, fdesc, tree)
    firstdescendants_impl!(fdesc, tree, inv0)

    @inbounds for p in oneto(n)
        wt[p] = weights[p]
        map0[inv0[p]] = p
        inv0[p] = p
        inv1[p] = p
        prev_p[p] = zero(V)
        prev_nbr[p] = zero(V)
        sets.rank[p] = zero(V)
        sets.parent[p] = zero(V)
    end

    @inbounds for p in oneto(n)
        r = parentindex(tree, p)

        if !isnothing(r)
            wt[r] = zero(V)
        end
    end

    map1 = inv0

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

    @inbounds for i in oneto(n)
        p = map0[i]
        r = parentindex(tree, p)

        for u in neighbors(lower, p)
            if iszero(prev_nbr[u]) || prev_nbr[u] < fdesc[p]
                wt[p] += weights[u]
                pp = prev_p[u]

                if !iszero(pp)
                    q = find(pp)
                    wt[q] -= weights[u]
                end

                prev_p[u] = p
            end

            prev_nbr[u] = i
        end

        if !isnothing(r)
            wt[r] -= weights[p]
            union(p, r)
        end
    end

    @inbounds for p in oneto(n)
        r = parentindex(tree, p)

        if !isnothing(r)
            wt[r] += wt[p]
        end
    end
    
    return
end

function compositerotations(graph, clique::AbstractVector, alg::PermutationOrAlgorithm)
    return compositerotations(BipartiteGraph(graph), clique, alg)
end

function compositerotations(graph::AbstractGraph{V}, clique::AbstractVector, alg::PermutationOrAlgorithm) where {V}
    n = nv(graph)
    order, index = permutation(graph, alg)
    upper = sympermute(graph, index, Forward)
    alpha = compositerotations(upper, index[clique])
    invpermute!(order, alpha)
    return order, invperm(order)
end

function compositerotations(upper::AbstractGraph{V}, clique::AbstractVector{V}) where {V}
    E = etype(upper); n = nv(upper); m = ne(upper)
    alpha = Vector{V}(undef, n)
    order = Vector{V}(undef, n)
    index = Vector{V}(undef, n)
    fdesc = Vector{V}(undef, n)
    lower = BipartiteGraph{V, E}(n, n, m)
    tree = Parent{V}(n)
    compositerotations_impl!(alpha, order, index, fdesc, lower, tree, upper, clique)
    return alpha
end

# Equivalent Sparse Matrix Reorderings by Elimination Tree Rotations
# Liu
# Algorithm 3.2: Composite_Rotations
#
# Fast Computation of Minimal Fill Inside a Given Elimination Ordering
# Heggernes and Peyton
# Change_Root
function compositerotations_impl!(
        alpha::AbstractVector{V},
        order::AbstractVector{V},
        index::AbstractVector{V},
        fdesc::AbstractVector{V},
        lower::BipartiteGraph{V, E},
        tree::Parent{V},
        upper::AbstractGraph{V},
        clique::AbstractVector{V},
    ) where {V, E}
    @assert nv(upper) <= length(alpha)
    @assert nv(upper) <= length(order)
    @assert nv(upper) <= length(index)
    @assert nv(upper) <= length(fdesc)
    @assert nv(upper) == nv(lower)
    n = nv(upper)
    etree_impl!(tree, fdesc, upper)
    reverse!_impl!(lower, upper)
    postorder_impl!(alpha, fdesc, index, order, tree)
    firstdescendants_impl!(fdesc, tree, index)

    @inbounds for v in vertices(lower)
        i = index[v]
        order[i] = v
        alpha[v] = zero(V)
    end

    if ispositive(n)
        y = n

        @inbounds for v in Iterators.reverse(clique)
            alpha[v] = n; n -= one(V)
            y = min(v, y)
        end

        @inbounds xstop = index[y]; xstart = xstop + one(V)

        @inbounds for z in ancestorindices(tree, y)
            ystart = fdesc[y]; ystop = index[y]

            for i in ystart:(xstart - one(V))
                v = order[i]

                for w in neighbors(lower, v)
                    if y < w && iszero(alpha[w])
                        alpha[w] = n; n -= one(V)
                    end
                end
            end

            for i in (xstop + one(V)):ystop
                v = order[i]

                for w in neighbors(lower, v)
                    if y < w && iszero(alpha[w])
                        alpha[w] = n; n -= one(V)
                    end
                end
            end

            xstart, xstop = ystart, ystop
            y = z
        end

        @inbounds for v in reverse(vertices(lower))
            if iszero(alpha[v])
                alpha[v] = n; n -= one(V)
            end
        end
    end

    return
end

"""
    postorder(tree::Parent)

Compute a postordering of a forest.
"""
function postorder(tree::Parent{V}) where {V}
    n = length(tree)
    brother = Vector{V}(undef, n)
    child = Vector{V}(undef, n)
    index = Vector{V}(undef, n)
    stack = Vector{V}(undef, n)
    postorder_impl!(brother, child, index, stack, tree)
    return index
end

function postorder_impl!(
        brother::AbstractVector{V},
        child::AbstractVector{V},
        index::AbstractVector{V},
        stack::AbstractVector{V},
        tree::Parent{V},
    ) where {V}
    @assert length(tree) <= length(brother)
    @assert length(tree) <= length(child)
    @assert length(tree) <= length(index)
    @assert length(tree) <= length(stack)
    num = zero(V)

    root = lcrs_impl!(brother, child, tree)

    function brothers(i::V)
        @inbounds head = view(child, i)
        return SinglyLinkedList(head, brother)
    end

    roots = SinglyLinkedList(Fill(root), brother)

    @inbounds for i in roots
        num += one(V); stack[num] = i
    end

    @inbounds for i in tree
        j = stack[num]; num -= one(V)

        while !isempty(brothers(j))
            num += one(V); stack[num] = j
            j = popfirst!(brothers(j))
        end

        index[j] = i
    end

    return
end

function postorder_impl!(
        brother::AbstractVector{V},
        child::AbstractVector{V},
        index::AbstractVector{V},
        stack::AbstractVector{V},
        tree::Parent{V},
        root::V,
    ) where {V}
    @assert length(tree) <= length(brother)
    @assert length(tree) <= length(child)
    @assert length(tree) <= length(index)
    @assert length(tree) <= length(stack)
    @assert length(tree) >= root >= one(V)
    num = zero(V)

    lcrs_impl!(brother, child, tree, root)

    function brothers(i::V)
        @inbounds head = view(child, i)
        return SinglyLinkedList(head, brother)
    end

    roots = SinglyLinkedList(Fill(root), brother)

    @inbounds for i in roots
        num += one(V); stack[num] = i
    end

    @inbounds for i in tree
        j = stack[num]; num -= one(V)

        while !isempty(brothers(j))
            num += one(V); stack[num] = j
            j = popfirst!(brothers(j))
        end

        index[j] = i
    end

    return
end

"""
    postorder!(tree::Parent)

Postorder a forest.
"""
function postorder!(tree::Parent{V}) where {V}
    n = length(tree)
    brother = Vector{V}(undef, n)
    child = Vector{V}(undef, n)
    index = Vector{V}(undef, n)
    stack = Vector{V}(undef, n)
    postorder!_impl!(brother, child, index, stack, tree)
    return index
end

function postorder!(tree::Parent{V}, root::V) where {V}
    n = length(tree)
    brother = Vector{V}(undef, n)
    child = Vector{V}(undef, n)
    index = Vector{V}(undef, n)
    stack = Vector{V}(undef, n)
    postorder!_impl!(brother, child, index, stack, tree, root::V)
    return index
end

function postorder!_impl!(
        brother::AbstractVector{V},
        child::AbstractVector{V},
        index::AbstractVector{V},
        stack::AbstractVector{V},
        tree::Parent{V},
    ) where {V}
    postorder_impl!(brother, child, index, stack, tree)
    invpermute!_impl!(stack, tree, index)
    return
end

function postorder!_impl!(
        brother::AbstractVector{V},
        child::AbstractVector{V},
        index::AbstractVector{V},
        stack::AbstractVector{V},
        tree::Parent{V},
        root::V,
    ) where {V}
    postorder_impl!(brother, child, index, stack, tree, root)
    invpermute!_impl!(stack, tree, index)
    return
end

function firstdescendants_impl!(
        fdesc::AbstractVector{V},
        tree::Parent{V},
        index::AbstractVector{V}=tree,
    ) where {V}
    @assert length(tree) <= length(fdesc)
    @assert length(tree) <= length(index)

    @inbounds for i in tree
        fdesc[i] = index[i]
    end

    @inbounds for i in tree
        j = parentindex(tree, i)

        if !isnothing(j)
            fdesc[j] = min(fdesc[i], fdesc[j])
        end
    end

    return
end

function invpermute!_impl!(
        parent::AbstractVector{V},
        tree::Parent{V},
        index::AbstractVector{V},
    ) where {V}
    @assert length(tree) <= length(parent)
    @assert length(tree) <= length(index)

    @inbounds for i in tree
        j = parentindex(tree, i)

        if isnothing(j)
            parent[index[i]] = zero(V)
        else
            parent[index[i]] = index[j]
        end
    end

    @inbounds for i in tree
        tree.prnt[i] = parent[i]
    end

    return
end

function setrootindex!(tree::Parent{I}, root::Integer) where {I <: Integer}
    return setrootindex!(tree, convert(I, root))
end

function setrootindex!(tree::Parent{I}, root::I) where {I <: Integer}
    @assert root in tree
    parent = tree.prnt; i = zero(I); j = root

    @inbounds while ispositive(j)
        k = parent[j]; parent[j] = i; i = j; j = k
    end

    return tree
end

function Base.copy(tree::Parent)
    return Parent(tree.nv, copy(tree.prnt))
end

function Base.copy!(dst::Parent, src::Parent)
    @assert length(dst) == length(src)
    copyto!(dst.prnt, 1, src.prnt, 1, length(dst))
    return dst
end

##########################
# Indexed Tree Interface #
##########################

@propagate_inbounds function AbstractTrees.parentindex(tree::Parent, i::Integer)
    @boundscheck checkbounds(tree, i)
    @inbounds j = tree.prnt[i]

    if !ispositive(j)
        j = nothing
    end

    return j
end

@propagate_inbounds function ancestorindices(tree::Parent, i::Integer)
    @boundscheck checkbounds(tree, i)
    @inbounds head = view(tree.prnt, i)
    return SinglyLinkedList(head, tree.prnt)
end

#################################
# Abstract Unit Range Interface #
#################################

function Base.first(tree::Parent{V}) where {V}
    return one(V)
end

function Base.last(tree::Parent{V}) where {V}
    return tree.nv
end
