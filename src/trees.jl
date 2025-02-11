"""
    Tree{V} <: AbstractUnitRange{V}

    Tree(tree::AbstractTree)

    Tree{V}(tree::AbstractTree) where V

A rooted forest with vertices of type `V`.
This type implements the [indexed tree interface](https://juliacollections.github.io/AbstractTrees.jl/stable/#The-Indexed-Tree-Interface).
"""
struct Tree{V<:Signed} <: AbstractUnitRange{V}
    parent::Vector{V}  # vector of parents
    root::Scalar{V}    # root
    child::Vector{V}   # vector of left-children
    brother::Vector{V} # vector of right-siblings

    function Tree{V}(parent::AbstractVector) where {V}
        root = Scalar{V}(undef)
        child = Vector{V}(undef, length(parent))
        brother = Vector{V}(undef, length(parent))
        tree = new{V}(parent, root, child, brother)
        return lcrs!(tree)
    end
end

function Tree(parent::AbstractVector{V}) where {V}
    return Tree{V}(parent)
end

"""
    eliminationtree(graph;
        alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM)

Construct a [tree-depth decomposition](https://en.wikipedia.org/wiki/Tr%C3%A9maux_tree) of a simple graph.

```julia
julia> using CliqueTrees

julia> graph = [
           0 1 1 0 0 0 0 0
           1 0 1 0 0 1 0 0
           1 1 0 1 1 0 0 0
           0 0 1 0 1 0 0 0
           0 0 1 1 0 0 1 1
           0 1 0 0 0 0 1 0
           0 0 0 0 1 1 0 1
           0 0 0 0 1 0 1 0
       ];

julia> label, tree = eliminationtree(graph);

julia> tree
8-element Tree{Int64}:
 8
 └─ 7
    ├─ 5
    │  ├─ 1
    │  └─ 4
    │     └─ 3
    │        └─ 2
    └─ 6
```
"""
function eliminationtree(graph; alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM)
    label, tree, upper = eliminationtree(graph, alg)
    return label, tree
end

function eliminationtree(graph, alg::PermutationOrAlgorithm)
    label, index = permutation(graph, alg)
    upper = sympermute(graph, index, ForwardOrdering())
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
        parent[i] = 0
        ancestor[i] = 0

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
    vertices(lower) != tree && throw(ArgumentError("vertices(lower) != tree"))

    # find postordering, first descendants, and levels
    index = postorder(tree)
    order = Perm(ForwardOrdering(), index)
    fdesc = firstdescendants(tree, order)
    level = levels(tree)

    # construct disjoint set forest
    sets = IntDisjointSets{V}(length(tree))
    root::Vector{V} = tree
    repr::Vector{V} = tree

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
            wt[r] = 0
        end
    end

    @inbounds for p in invperm(index)
        r = parentindex(tree, p)

        for u in neighbors(lower, p)
            if iszero(prev_nbr[u]) || lt(order, prev_nbr[u], fdesc[p])
                wt[p] += 1
                pp = prev_p[u]

                if iszero(pp)
                    rc[u] += level[p] - level[u]
                else
                    q = find(pp)
                    rc[u] += level[p] - level[q]
                    wt[q] -= 1
                end

                prev_p[u] = p
            end

            prev_nbr[u] = p
        end

        if !isnothing(r)
            wt[r] -= 1
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

# Compute a postordering of a forest.
function postorder(tree::Tree{V}) where {V}
    index = Vector{V}(undef, length(tree))

    # construct disjoint sets data structure
    child = copy(tree.child)

    function set(i)
        head = @view child[i]
        return SinglyLinkedList(head, tree.brother)
    end

    # construct stack data structure 
    n::V = 0
    stack = Vector{V}(undef, length(tree))

    # run algorithm
    for j in rootindices(tree)
        n += 1
        stack[n] = j
    end

    for i in tree
        j = stack[n]
        n -= 1

        while !isempty(set(j))
            n += 1
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
        level[i] = isnothing(j) ? 0 : level[j] + 1
    end

    return level
end

# Get the first descendant of every vertex in a topologically ordered forest.
function firstdescendants(tree::Tree{V}, order::Ordering=ForwardOrdering()) where {V}
    fdesc = Vector{V}(undef, length(tree))

    for j in tree
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
    fill!(tree.root, 0)
    fill!(tree.child, 0)

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
    tree != eachindex(index) && throw(ArgumentError("tree != eachindex(index)"))

    # run algorithm
    tree.parent[index] = map(tree.parent) do i
        iszero(i) ? i : index[i]
    end

    return lcrs!(tree)
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
    i ∉ tree && throw(ArgumentError("i ∉ tree"))

    # run algorithm
    j::V = 0
    k::V = i

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
