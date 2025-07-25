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
function stree(tree::Tree{V}, colcount::AbstractVector{V}, snd::Maximal) where {V}
    @argcheck length(tree) <= length(colcount)

    n = last(tree)
    new = sizehint!(V[], n)
    parent = sizehint!(V[], n)
    ancestor = sizehint!(V[], n)
    new_in_clique = Vector{V}(undef, n)

    @inbounds for v in tree
        u = nothing

        for s in childindices(tree, v)
            if colcount[s] == colcount[v] + one(V)
                u = s
                break
            end
        end

        if !isnothing(u)
            new_in_clique[v] = new_in_clique[u]

            for s in childindices(tree, v)
                if s !== u
                    parent[new_in_clique[s]] = new_in_clique[v]
                    ancestor[new_in_clique[s]] = v
                end
            end
        else
            push!(new, v)
            push!(parent, zero(V))
            push!(ancestor, zero(V))
            new_in_clique[v] = length(new)

            for s in childindices(tree, v)
                parent[new_in_clique[s]] = new_in_clique[v]
                ancestor[new_in_clique[s]] = v
            end
        end
    end

    m = convert(V, length(new))
    return new, ancestor, Parent(m, parent)
end

# Compute the fundamental supernode partition of the montone transitive extension of an ordered graph.
# The complexity is O(n), where n = |V|.
function stree(tree::Tree{V}, colcount::AbstractVector{V}, snd::Fundamental) where {V}
    @argcheck length(tree) <= length(colcount)

    # run algorithm
    n = last(tree)
    new = sizehint!(V[], n)
    parent = sizehint!(V[], n)
    ancestor = sizehint!(V[], n)
    new_in_clique = Vector{V}(undef, n)

    @inbounds for v in tree
        u = firstchildindex(tree, v)

        if !isnothing(u) &&
                colcount[u] == colcount[v] + one(V) &&
                isnothing(nextsiblingindex(tree, u))
            new_in_clique[v] = new_in_clique[u]
        else
            push!(new, v)
            push!(parent, zero(V))
            push!(ancestor, zero(V))
            new_in_clique[v] = length(new)

            for s in childindices(tree, v)
                parent[new_in_clique[s]] = new_in_clique[v]
                ancestor[new_in_clique[s]] = v
            end
        end
    end

    m = convert(V, length(new))
    return new, ancestor, Parent(m, parent)
end

"""
    DEFAULT_SUPERNODE_TYPE = Maximal()

The default supernode partition.
"""
const DEFAULT_SUPERNODE_TYPE = Maximal()
