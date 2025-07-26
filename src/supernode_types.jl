"""
    SupernodeType

A type of supernode partition. The options are

| type                  | name                            |
|:--------------------- |:------------------------------- |
| [`Nodal`](@ref)       | nodal supernode partition       |
| [`Maximal`](@ref)     | maximal supernode partition     |
| [`Fundamental`](@ref) | fundamental supernode partition |
"""
abstract type SupernodeType end

"""
    Nodal <: SupernodeType

A nodal  supernode partition.
"""
struct Nodal <: SupernodeType end

"""
    Maximal <: SupernodeType

A maximal supernode partition.
"""
struct Maximal <: SupernodeType end

"""
    Fundamental <: SupernodeType

A fundamental supernode partition.
"""
struct Fundamental <: SupernodeType end

# Compact Clique Tree Data Structures in Sparse Matrix Factorizations
# Pothen and Sun
# Figure 4: The Clique Tree Algorithm 2
#
# Compute the maximal supernode partition of the montone transitive extension of an ordered graph.
# The complexity is O(n), where n = |V|.
function stree_impl!(
        new::AbstractVector{V},
        parent::AbstractVector{V},
        ancestor::AbstractVector{V},
        new_in_clique::AbstractVector{V},
        tree::Tree{V},
        colcount::AbstractVector{V},
        snd::Maximal,
    ) where {V}
    @argcheck length(tree) <= length(new)
    @argcheck length(tree) <= length(parent)
    @argcheck length(tree) <= length(ancestor)
    @argcheck length(tree) <= length(new_in_clique)
    @argcheck length(tree) <= length(colcount)
    snd = zero(V); n = last(tree)

    for v in tree
        u = zero(V); children = childindices(tree, v)

        for s in children
            if colcount[s] == colcount[v] + one(V)
                u = s
                break
            end
        end

        if ispositive(u)
            new_in_clique[v] = new_in_clique[u]

            for s in children
                if s != u
                    parent[new_in_clique[s]] = new_in_clique[v]
                    ancestor[new_in_clique[s]] = v
                end
            end
        else
            new_in_clique[v] = snd += one(V); new[snd] = v
            parent[snd] = ancestor[snd] = zero(V)

            for s in children
                parent[new_in_clique[s]] = new_in_clique[v]
                ancestor[new_in_clique[s]] = v
            end
        end
    end

    return Parent(snd, parent)
end

# Compute the fundamental supernode partition of the montone transitive extension of an ordered graph.
# The complexity is O(n), where n = |V|.
function stree_impl!(
        new::AbstractVector{V},
        parent::AbstractVector{V},
        ancestor::AbstractVector{V},
        new_in_clique::AbstractVector{V},
        tree::Tree{V},
        colcount::AbstractVector{V},
        snd::Fundamental,
    ) where {V}
    @argcheck length(tree) <= length(new)
    @argcheck length(tree) <= length(parent)
    @argcheck length(tree) <= length(ancestor)
    @argcheck length(tree) <= length(new_in_clique)
    @argcheck length(tree) <= length(colcount)
    snd = zero(V); n = last(tree)

    for v in tree
        u = zero(V); children = childindices(tree, v)

        if isone(length(children))
            u = only(children)
        end

        if ispositive(u) && colcount[u] == colcount[v] + one(V)
            new_in_clique[v] = new_in_clique[u]
        else
            new_in_clique[v] = snd += one(V); new[snd] = v
            parent[snd] = ancestor[snd] = zero(V)

            for s in children
                parent[new_in_clique[s]] = new_in_clique[v]
                ancestor[new_in_clique[s]] = v
            end
        end
    end

    return Parent(snd, parent)
end

# Compute the nodal supernode partition of the montone transitive extension of an ordered graph.
# The complexity is O(n), where n = |V|.
function stree_impl!(
        new::AbstractVector{V},
        parent::AbstractVector{V},
        ancestor::AbstractVector{V},
        new_in_clique::AbstractVector{V},
        tree::Tree{V},
        colcount::AbstractVector{V},
        snd::Nodal,
    ) where {V}
    @argcheck length(tree) <= length(new)
    @argcheck length(tree) <= length(parent)
    @argcheck length(tree) <= length(ancestor)
    n = last(tree)

    for i in tree
        j = parentindex(tree, i)

        if isnothing(j)
           j = zero(V)
        end 

        new[i] = i
        parent[i] = j
        ancestor[i] = j
    end

    return Parent(n, parent)
end

"""
    DEFAULT_SUPERNODE_TYPE = Maximal()

The default supernode partition.
"""
const DEFAULT_SUPERNODE_TYPE = Maximal()
