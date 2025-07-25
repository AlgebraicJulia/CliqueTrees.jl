"""
    Tree{V} <: AbstractUnitRange{V}

    Tree(tree::AbstractTree)

    Tree{V}(tree::AbstractTree) where V

A rooted forest with vertices of type `V`.
This type implements the [indexed tree interface](https://juliacollections.github.io/AbstractTrees.jl/stable/#The-Indexed-Tree-Interface).
"""
struct Tree{
        V <: Integer,
        Parent <: AbstractVector{V},
        Root <: AbstractScalar{V},
        Child <: AbstractVector{V},
        Brother <: AbstractVector{V},
    } <: AbstractUnitRange{V}
    last::V
    parent::Parent   # vector of parents
    root::Root       # root
    child::Child     # vector of left-children
    brother::Brother # vector of right-siblings
end

function Tree{V}(parent::AbstractVector) where {V}
    n = length(parent)
    last = convert(V, n)
    root = Scalar{V}(undef)
    child = Vector{V}(undef, n)
    brother = Vector{V}(undef, n)
    tree = Tree(last, parent, root, child, brother)
    return lcrs!(tree)
end

function Tree(parent::AbstractVector{V}) where {V}
    return Tree{V}(parent)
end

function Tree{V}(n::Integer) where {V}
    last = convert(V, n)
    parent = Vector{V}(undef, n)
    root = Scalar{V}(undef)
    child = Vector{V}(undef, n)
    brother = Vector{V}(undef, n)
    return Tree(last, parent, root, child, brother)
end

function Tree(tree::Tree)
    return Tree(tree.last, tree.parent, tree.root, tree.child, tree.brother)
end

function Tree(tree::Parent)
    return Tree(tree.prnt)
end

"""
    eliminationtree([weights, ]graph;
        alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM)

Construct a [tree-depth decomposition](https://en.wikipedia.org/wiki/Tr%C3%A9maux_tree) of a simple graph.

```jldoctest
julia> using CliqueTrees

julia> graph = [
           0 1 0 0 0 0 0 0
           1 0 1 0 0 1 0 0
           0 1 0 1 0 1 1 1
           0 0 1 0 0 0 0 0
           0 0 0 0 0 1 1 0
           0 1 1 0 1 0 0 0
           0 0 1 0 1 0 0 1
           0 0 1 0 0 0 1 0
       ];

julia> label, tree = eliminationtree(graph);

julia> tree
8-element Tree{Int64, Vector{Int64}, Array{Int64, 0}, Vector{Int64}, Vector{Int64}}:
 8
 └─ 7
    ├─ 5
    └─ 6
       ├─ 1
       ├─ 3
       │  └─ 2
       └─ 4
```
"""
function eliminationtree(graph; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM)
    label, tree, upper = eliminationtree(graph, alg)
    return label, tree
end

function eliminationtree(weights::AbstractVector, graph; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM)
    label, tree, upper = eliminationtree(weights, graph, alg)
    return label, tree
end

function eliminationtree(graph, alg::PermutationOrAlgorithm)
    return eliminationtree(graph, permutation(graph, alg)...)
end

function eliminationtree(weights::AbstractVector, graph, alg::PermutationOrAlgorithm)
    return eliminationtree(graph, permutation(weights, graph, alg)...)
end

function eliminationtree(graph, label, index)
    upper = sympermute(graph, index, Forward)
    return label, etree(upper), upper
end

function supcnt(lower::AbstractGraph{V}, tree::AbstractParent{V}) where {V}
    n = nv(lower); weights = Ones{V}(n)
    return supcnt(weights, lower, tree)
end

function supcnt(weights::AbstractVector{W}, lower::AbstractGraph{V}, tree::AbstractParent{V}) where {W, V}
    n = nv(lower)
    
    wt = FixedSizeVector{W}(undef, n)
    map0 = FixedSizeVector{V}(undef, n)
    inv0 = FixedSizeVector{V}(undef, n)
    inv1 = FixedSizeVector{V}(undef, n)
    fdesc = FixedSizeVector{V}(undef, n)
    prev_p = FixedSizeVector{V}(undef, n)
    prev_nbr = FixedSizeVector{V}(undef, n)

    rank = FixedSizeVector{V}(undef, n)
    parent = FixedSizeVector{V}(undef, n)
    stack = FixedSizeVector{V}(undef, n)
    sets = UnionFind(rank, parent, stack)
   
    supcnt_impl!(wt, map0, inv0, inv1, fdesc,
        prev_p, prev_nbr, sets, weights, lower, tree)
    
    return wt
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
        tree::AbstractParent{V}
    ) where {W, V}
    @argcheck nv(lower) <= length(wt)
    @argcheck nv(lower) <= length(map0)
    @argcheck nv(lower) <= length(inv0)
    @argcheck nv(lower) <= length(inv1)
    @argcheck nv(lower) <= length(fdesc)
    @argcheck nv(lower) <= length(prev_p)
    @argcheck nv(lower) <= length(prev_nbr)
    @argcheck nv(lower) <= length(weights)
    @argcheck nv(lower) <= length(tree)
    
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
        vv = @inbounds find!(sets, u)
        v = @inbounds inv1[vv]
        return v
    end

    function union(u::V, v::V)
        @inbounds uu = map1[u]
        @inbounds vv = map1[v]
        @inbounds vv = map1[v] = rootunion!(sets, uu, vv)
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
    count = Vector{E}(undef, n)
    lower = BipartiteGraph{V, E}(n, n, m)
    tree = Parent{V}(n)
    compositerotations_impl!(alpha, order, index, fdesc, count, lower, tree, upper, clique)
    return alpha
end

function compositerotations_impl!(
        alpha::AbstractVector{I},
        order::AbstractVector{I},
        index::AbstractVector{I},
        fdesc::AbstractVector{I},
        lower::BipartiteGraph{I, I},
        tree::Parent{I},
        upper::AbstractGraph{I},
        clique::AbstractVector{I},
    ) where {I}
    count = order
    return compositerotations_impl!(alpha, order, index, fdesc, count, lower, tree, upper, clique)
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
        count::AbstractVector{E},
        lower::BipartiteGraph{V, E},
        tree::Parent{V},
        upper::AbstractGraph{V},
        clique::AbstractVector{V},
    ) where {V, E}
    @argcheck nv(upper) <= length(alpha)
    @argcheck nv(upper) <= length(order)
    @argcheck nv(upper) <= length(index)
    @argcheck nv(upper) <= length(fdesc)
    @argcheck nv(upper) == nv(lower)
    n = nv(upper)
    etree_impl!(tree, fdesc, upper)
    reverse!_impl!(count, lower, upper)
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
    postorder(tree::AbstractParent)

Compute a postordering of a forest.
"""
function postorder(tree::AbstractParent{V}) where {V}
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
        tree::AbstractParent{V},
    ) where {V}
    @argcheck length(tree) <= length(brother)
    @argcheck length(tree) <= length(child)
    @argcheck length(tree) <= length(index)
    @argcheck length(tree) <= length(stack)
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

function postorder!(tree::Tree{V}) where {V}
    parent = Parent(last(tree), tree.parent)
    index = postorder!(parent)
    lcrs!(tree)
    return index
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

"""
    firstdescendants(tree::Tree, index::AbstractVector)

Get the first descendant of every vertex of a topologically ordered forest.
"""
function firstdescendants(tree::Tree{V}, index::AbstractVector{V}) where {V}
    n = length(tree); fdesc = Vector{V}(undef, n)
    firstdescendants_impl!(fdesc, tree, index)
    return fdesc
end

function firstdescendants_impl!(
        fdesc::AbstractVector{V},
        tree::AbstractParent{V},
        index::AbstractVector{V},
    ) where {V}
    @argcheck length(tree) <= length(fdesc)
    @argcheck length(tree) <= length(index)

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

# Compute the `root`, `child`, and `brother` fields of a forest.
function lcrs!(tree::Tree{V}) where {V}
    tree.root[] = zero(V)

    @inbounds for i in tree
        tree.child[i] = zero(V)
    end

    @inbounds for i in reverse(tree)
        j = parentindex(tree, i)

        if isnothing(j)
            tree.brother[i] = tree.root[]
            tree.root[] = i
        else
            tree.brother[i] = tree.child[j]
            tree.child[j] = i
        end
    end

    return tree
end

"""
    invpermute!(tree::Tree, index::AbstractVector)

Permute the vertices of a forest.
"""
function Base.invpermute!(tree::Parent{V}, index::AbstractVector{V}) where {V}
    n = length(tree)
    parent = Vector{V}(undef, n)
    invpermute!_impl!(parent, tree, index)
    return tree
end

function invpermute!_impl!(
        parent::AbstractVector{V},
        tree::Parent{V},
        index::AbstractVector{V},
    ) where {V}
    @argcheck length(tree) <= length(parent)
    @argcheck length(tree) <= length(index)

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

function Base.isequal(left::Tree, right::Tree)
    return isequal(left.parent, right.parent) &&
        isequal(left.root, right.root) &&
        isequal(left.child, right.child) &&
        isequal(left.brother, right.brother)
end

function Base.copy(tree::Tree)
    return Tree(
        last(tree),
        copy(tree.parent),
        copy(tree.root),
        copy(tree.child),
        copy(tree.brother),
    )
end

function Base.copy!(dst::Tree, src::Tree)
    @argcheck length(dst) == length(src)
    n = length(dst)
    copyto!(dst.parent, 1, src.parent, 1, n)
    dst.root[] = src.root[]
    copyto!(dst.child, 1, src.child, 1, n)
    copyto!(dst.brother, 1, src.brother, 1, n)
    return dst
end

function Base.:(==)(left::Tree, right::Tree)
    return left.parent == right.parent && left.root == right.root
end

##########################
# Indexed Tree Interface #
##########################

function AbstractTrees.rootindex(tree::Tree)
    j = tree.root[]

    if !iszero(j)
        return j
    end
end

@propagate_inbounds function AbstractTrees.parentindex(tree::Tree, i::Integer)
    @boundscheck checkbounds(tree.parent, i)
    @inbounds j = tree.parent[i]

    if !iszero(j)
        return j
    end
end

@propagate_inbounds function firstchildindex(tree::Tree, i::Integer)
    @boundscheck checkbounds(tree.child, i)
    @inbounds j = tree.child[i]

    if !iszero(j)
        return j
    end
end

@propagate_inbounds function AbstractTrees.nextsiblingindex(tree::Tree, i::Integer)
    @boundscheck checkbounds(tree.brother, i)
    @inbounds j = tree.brother[i]

    if !iszero(j)
        return j
    end
end

function rootindices(tree::Tree)
    return SinglyLinkedList(tree.root, tree.brother)
end

@propagate_inbounds function AbstractTrees.childindices(tree::Tree, i::Integer)
    @boundscheck checkbounds(tree.child, i)
    @inbounds head = @view tree.child[i]
    return SinglyLinkedList(head, tree.brother)
end

@propagate_inbounds function ancestorindices(tree::Tree, i::Integer)
    @boundscheck checkbounds(tree.parent, i)
    @inbounds head = @view tree.parent[i]
    return SinglyLinkedList(head, tree.parent)
end

function setrootindex!(tree::Tree{V}, i::Integer) where {V}
    # validate arguments
    @argcheck i in tree

    # run algorithm
    j = zero(V)
    k = convert(V, i)

    for l in ancestorindices(tree, k)
        tree.parent[k] = j
        j, k = k, l
    end

    tree.parent[k] = j
    return lcrs!(tree)
end

#################################
# Abstract Unit Range Interface #
#################################

function Base.first(tree::Tree{V}) where {V}
    return one(V)
end

function Base.last(tree::Tree{V}) where {V}
    return tree.last
end
