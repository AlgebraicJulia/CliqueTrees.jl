"""
    Tree{V} <: AbstractUnitRange{V}

    Tree(tree::AbstractTree)

    Tree{V}(tree::AbstractTree) where V

A rooted forest with vertices of type `V`.
This type implements the [indexed tree interface](https://juliacollections.github.io/AbstractTrees.jl/stable/#The-Indexed-Tree-Interface).
"""
struct Tree{V <: Signed} <: AbstractUnitRange{V}
    parent::Vector{V}  # vector of parents
    root::Scalar{V}    # root
    child::Vector{V}   # vector of left-children
    brother::Vector{V} # vector of right-siblings
end

function Tree{V}(parent::AbstractVector) where {V}
    root = Scalar{V}(undef)
    child = Vector{V}(undef, length(parent))
    brother = Vector{V}(undef, length(parent))
    tree = Tree{V}(parent, root, child, brother)
    return lcrs!(tree)
end

function Tree(parent::AbstractVector{V}) where {V}
    return Tree{V}(parent)
end

function Tree{V}(tree::Tree) where {V}
    return Tree{V}(tree.parent, tree.root, tree.child, tree.brother)
end

function Tree(tree::Tree)
    return Tree(tree.parent, tree.root, tree.child, tree.brother)
end

"""
    eliminationtree(graph;
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
8-element Tree{Int64}:
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

function eliminationtree(graph, alg::PermutationOrAlgorithm)
    label, index = permutation(graph, alg)
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
    parent = Vector{V}(undef, nv(upper))
    ancestor = Vector{V}(undef, nv(upper))

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

    return Tree(parent)
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
    sets = IntDisjointSets{V}(length(tree))
    root = Vector{V}(tree)
    repr = Vector{V}(tree)

    function find(u)
        v = @inbounds repr[find_root!(sets, u)]
        return v
    end

    function union(u, v)
        @inbounds root[v] = root_union!(sets, root[u], root[v])
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

# Equivalent Sparse Matrix Reorderings by Elimination Tree Rotations
# Liu
# Algorithm 3.2: Composite_Rotations
#
# Fast Computation of Minimal Fill Inside a Given Elimination Ordering
# Heggernes and Peyton
# Change_Root
function compositerotations(
        graph::AbstractGraph{V}, tree::Tree{V}, clique::AbstractVector{V}
    ) where {V}
    index = postorder(tree)
    order = invperm(index)
    fdesc = firstdescendants(tree, Perm(Forward, index))

    n = nv(graph)
    alpha = zeros(V, n)

    if ispositive(n)
        y = n

        @inbounds for v in reverse(clique)
            alpha[v] = n
            n -= one(V)

            if v < y
                y = v
            end
        end

        @inbounds xstop = index[y]
        xstart = xstop + one(V)

        @inbounds for z in ancestorindices(tree, y)
            ystart = index[fdesc[y]]
            ystop = index[y]

            for i in ystart:(xstart - one(V))
                v = order[i]

                for w in neighbors(graph, v)
                    if y < w && iszero(alpha[w])
                        alpha[w] = n
                        n -= one(V)
                    end
                end
            end

            for i in (xstop + one(V)):ystop
                v = order[i]

                for w in neighbors(graph, v)
                    if y < w && iszero(alpha[w])
                        alpha[w] = n
                        n -= one(V)
                    end
                end
            end

            xstart, xstop = ystart, ystop
            y = z
        end

        @inbounds for v in reverse(vertices(graph))
            if iszero(alpha[v])
                alpha[v] = n
                n -= one(V)
            end
        end
    end

    return alpha
end

# Compute a postordering of a forest.
function postorder(tree::Tree{V}) where {V}
    index = Vector{V}(undef, length(tree))

    # construct bucket queue data structure
    child = copy(tree.child)

    function set(i)
        @inbounds head = @view child[i]
        return SinglyLinkedList(head, tree.brother)
    end

    # construct stack data structure
    n = zero(V)
    stack = Vector{V}(undef, length(tree))

    # run algorithm
    @inbounds for j in rootindices(tree)
        n += one(V)
        stack[n] = j
    end

    @inbounds for i in tree
        j = stack[n]
        n -= one(V)

        while !isempty(set(j))
            n += one(V)
            stack[n] = j
            j = popfirst!(set(j))
        end

        index[j] = i
    end

    return index
end

# postorder a forest
function postorder!(tree::Tree)
    index = postorder(tree)
    invpermute!(tree, index)
    return index
end

# Get the level of every vertex in a topologically ordered tree.
function levels(tree::Tree{V}) where {V}
    level = Vector{V}(undef, length(tree))

    for i in reverse(tree)
        j = parentindex(tree, i)
        level[i] = isnothing(j) ? zero(V) : level[j] + one(V)
    end

    return level
end

# Get the first descendant of every vertex in a topologically ordered forest.
function firstdescendants(tree::Tree{V}, order::Ordering = Forward) where {V}
    fdesc = Vector{V}(undef, length(tree))

    @inbounds for j in tree
        v = j

        for i in childindices(tree, j)
            u = fdesc[i]

            if lt(order, u, v)
                v = u
            end
        end

        fdesc[j] = v
    end

    return fdesc
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

# Permute the vertices of a forest.
function Base.invpermute!(tree::Tree{V}, index::AbstractVector{V}) where {V}
    # validate arguments
    @argcheck tree == eachindex(index)

    # run algorithm
    tree.parent[index] = map(tree.parent) do i
        iszero(i) ? i : index[i]
    end

    return lcrs!(tree)
end

function Base.isequal(left::Tree, right::Tree)
    return isequal(left.parent, right.parent) &&
        isequal(left.root, right.root) &&
        isequal(left.child, right.child) &&
        isequal(left.brother, right.brother)
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
    n::V = length(tree.parent)
    return n
end
