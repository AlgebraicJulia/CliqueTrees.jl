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

# A Compact Row Storage Scheme for Cholesky Factors Using Elimination Trees
# Liu
# Algorithm 4.2: Elimination Tree by Path Compression.
#
# Construct the elimination tree of an ordered graph.
# The complexity is O(mlogn), where m = |E| and n = |V|.
function etree(upper::AbstractGraph{V}) where {V}
    n = nv(upper)
    tree = Tree{V}(n)
    ancestor = Vector{V}(undef, n)
    etree_impl!(tree, ancestor, upper)
    return tree
end

function etree_impl!(
        tree::Tree{V},
        ancestor::AbstractVector{V},
        upper::AbstractGraph{V},
    ) where {V}
    @argcheck nv(upper) == length(tree)
    @argcheck nv(upper) <= length(ancestor)
    parent = tree.parent

    @inbounds for i in vertices(upper)
        parent[i] = zero(V)
        ancestor[i] = zero(V)

        for k in neighbors(upper, i)
            r = k

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

    lcrs!(tree)
    return
end

# An Efficient Algorithm to Compute Row and Column Counts for Sparse Cholesky Factorization
# Gilbert, Ng, and Peyton
# Figure 3: Implementation of algorithm to compute row and column counts.
#
# Compute the lower and higher degrees of the monotone transitive extesion of an ordered graph.
# The complexity is O(mα(m, n)), where m = |E|, n = |V|, and α is the inverse Ackermann function.
function supcnt(lower::AbstractGraph{V}, tree::Tree{V}) where {V}
    # validate arguments
    @argcheck vertices(lower) == tree

    # find postordering, first descendants, and levels
    index = postorder(tree)
    order = Perm(Forward, index)
    fdesc = firstdescendants(tree, order)
    level = levels(tree)

    # construct disjoint set forest
    sets = UnionFind{V}(length(tree))
    root = Vector{V}(tree)
    repr = Vector{V}(tree)

    function find(u)
        v = @inbounds repr[find!(sets, u)]
        return v
    end

    function union(u, v)
        @inbounds root[v] = rootunion!(sets, root[u], root[v])
        @inbounds repr[root[v]] = v
        return nothing
    end

    # run algorithm
    prev_p = zeros(V, length(tree))
    prev_nbr = zeros(V, length(tree))
    rc = ones(V, length(tree))
    wt = ones(V, length(tree))

    @inbounds for p in tree
        r = parentindex(tree, p)

        if !isnothing(r)
            wt[r] = zero(V)
        end
    end

    @inbounds for p in invperm(index)
        r = parentindex(tree, p)

        for u in neighbors(lower, p)
            if iszero(prev_nbr[u]) || lt(order, prev_nbr[u], fdesc[p])
                wt[p] += one(V)
                pp = prev_p[u]

                if iszero(pp)
                    rc[u] += level[p] - level[u]
                else
                    q = find(pp)
                    rc[u] += level[p] - level[q]
                    wt[q] -= one(V)
                end

                prev_p[u] = p
            end

            prev_nbr[u] = p
        end

        if !isnothing(r)
            wt[r] -= one(V)
            union(p, r)
        end
    end

    cc = wt

    @inbounds for p in tree
        r = parentindex(tree, p)

        if !isnothing(r)
            cc[r] += cc[p]
        end
    end

    return rc, cc
end

function compositerotations(graph, clique::AbstractVector, alg::PermutationOrAlgorithm)
    return compositerotations(BipartiteGraph(graph), clique, alg)
end

function compositerotations(graph::AbstractGraph{V}, clique::AbstractVector, alg::PermutationOrAlgorithm) where {V}
    clique::AbstractVector{V} = clique
    return compositerotations(graph, clique, alg)
end

function compositerotations(graph::AbstractGraph{V}, clique::AbstractVector{V}, alg::PermutationOrAlgorithm) where {V}
    order, alpha = permutation(graph, alg)
    upper = sympermute(graph, alpha, Forward)
    E = etype(upper); n = nv(upper); m = ne(upper)
    index = Vector{V}(undef, n)
    fdesc = Vector{V}(undef, n)
    count = Vector{E}(undef, n)
    lower = BipartiteGraph{V, E}(n, n, m)
    tree = Tree{V}(n)
    compositerotations_impl!(alpha, order, index, fdesc, count, lower, tree, upper, clique)
    order[alpha] = oneto(n)
    return order, alpha
end

function compositerotations_impl!(
        alpha::AbstractVector{I},
        order::AbstractVector{I},
        index::AbstractVector{I},
        fdesc::AbstractVector{I},
        lower::BipartiteGraph{I, I},
        tree::Tree{I},
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
        tree::Tree{V},
        upper::AbstractGraph{V},
        clique::AbstractVector{V},
    ) where {V, E}
    @argcheck nv(upper) <= length(alpha)
    @argcheck nv(upper) <= length(order)
    @argcheck nv(upper) <= length(index)
    @argcheck nv(upper) <= length(fdesc)
    @argcheck nv(upper) == nv(lower)
    n = nv(upper)
    etree_impl!(tree, alpha, upper)
    reverse!_impl!(count, lower, upper)
    postorder_impl!(index, alpha, order, tree)
    firstdescendants_impl!(fdesc, tree, Perm(Forward, index))

    @inbounds for v in vertices(lower)
        order[index[v]] = v
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
            ystart = index[fdesc[y]]; ystop = index[y]

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
    postorder(tree::Tree)

Compute a postordering of a forest.
"""
function postorder(tree::Tree{V}) where {V}
    n = length(tree)
    index = Vector{V}(undef, n)
    child = Vector{V}(undef, n)
    stack = Vector{V}(undef, n)
    postorder_impl!(index, child, stack, tree)
    return index
end

function postorder_impl!(
        index::AbstractVector{V},
        child::AbstractVector{V},
        stack::AbstractVector{V},
        tree::Tree{V},
    ) where {V}
    @argcheck length(tree) <= length(index)
    @argcheck length(tree) <= length(child)
    @argcheck length(tree) <= length(stack) 
    n = length(tree); num = zero(V)

    function set(i)
        @inbounds head = @view child[i]
        return SinglyLinkedList(head, tree.brother)
    end

    if child !== tree.child
        copyto!(child, 1, tree.child, 1, n)
    end

    @inbounds for j in rootindices(tree)
        num += one(V); stack[num] = j
    end

    @inbounds for i in tree
        j = stack[num]; num -= one(V)

        while !isempty(set(j))
            num += one(V); stack[num] = j
            j = popfirst!(set(j))
        end

        index[j] = i
    end

    return
end

"""
    postorder!(tree::Tree)

Postorder a forest.
"""
function postorder!(tree::Tree{V}) where {V}
    n = length(tree)
    index = Vector{V}(undef, n)
    stack = Vector{V}(undef, n)
    postorder!_impl!(index, stack, tree)
    return index
end

function postorder!_impl!(
        index::AbstractVector{V},
        stack::AbstractVector{V},
        tree::Tree{V},
    ) where {V}
    child = tree.child; postorder_impl!(index, child, stack, tree)
    parent = stack; invpermute!_impl!(parent, tree, index)
    return tree
end

"""
    levels(tree::Tree)

Get the level of every vertex of a toplogically ordered forest.
"""
function levels(tree::Tree{V}) where {V}
    n = length(tree); level = Vector{V}(undef, n)
    levels_impl!(level, tree)
    return level
end

function levels_impl!(
        level::AbstractVector{V},
        tree::Tree{V},
    ) where {V}
    @argcheck length(tree) <= length(level)

    @inbounds for i in reverse(tree)
        j = parentindex(tree, i)

        if isnothing(j)
            level[i] = zero(V)
        else
            level[i] = level[j] + one(V)
        end
    end

    return
end

"""
    firstdescendants(tree::Tree[, ordering::Ordering])

Get the first descendant of every vertex of atopologically ordered forest.
"""
function firstdescendants(tree::Tree{V}, order::Ordering = Forward) where {V}
    n = length(tree); fdesc = Vector{V}(undef, n)
    firstdescendants_impl!(fdesc, tree, order)
    return fdesc
end

function firstdescendants_impl!(
        fdesc::AbstractVector{V},
        tree::Tree{V},
        order::Ordering,
    ) where {V}
    @argcheck length(tree) <= length(tree)

    @inbounds for j in tree
        jj = j

        for i in childindices(tree, j)
            ii = fdesc[i]

            if lt(order, ii, jj)
                jj = ii
            end
        end

        fdesc[j] = jj
    end

    return
end

# Compute the `root`, `child`, and `brother` fields of a forest.
function lcrs!(tree::Tree{V}) where {V}
    fill!(tree.root, zero(V))
    fill!(tree.child, zero(V))

    for i in reverse(tree)
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
function Base.invpermute!(tree::Tree{V}, index::AbstractVector{V}) where {V}
    n = length(tree); parent = Vector{V}(undef, n)
    invpermute!_impl!(parent, tree, index)
    return tree
end

function invpermute!_impl!(
        parent::AbstractVector{V},
        tree::Tree{V},
        index::AbstractVector{V},
    ) where {V}
    @argcheck length(tree) <= length(parent)
    @argcheck length(tree) <= length(index)
    n = length(tree)

    @inbounds for i in tree
        j = parentindex(tree, i)

        if isnothing(j)
            parent[index[i]] = zero(V)
        else
            parent[index[i]] = index[j]
        end
    end

    copyto!(tree.parent, 1, parent, 1, n)
    lcrs!(tree)
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
