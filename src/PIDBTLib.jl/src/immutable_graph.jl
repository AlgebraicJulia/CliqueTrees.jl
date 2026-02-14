# CachedGraph: an immutable snapshot of a Graph for the DP algorithm.
# Shares the same vertex indices as the source Graph (no reindexing).
# Uses AbstractPackedSet for bitset neighborhoods (immutable, functional style).

# ==================== Struct ====================

@enum SeparatorType Neither Cliquish PotentialMaximalClique

struct CachedGraph{PSet <: AbstractPackedSet}
    graph::Graph{PSet}                      # the underlying graph
    cache::Dict{PSet, SeparatorType}        # cache for is_csh! / is_pmc!
end

vertices(g::CachedGraph) = vertices(g.graph)
nv(g::CachedGraph) = nv(g.graph)
neighbors(g::CachedGraph, v::Int) = neighbors(g.graph, v)

# ==================== Constructor from Graph ====================

"""
    CachedGraph(g::Graph{PSet})

Build a CachedGraph from a Graph. Copies the neighborhoods since
the source Graph may be mutated afterward.
"""
function CachedGraph(g::Graph{PSet}) where {PSet}
    graph_copy = Graph{PSet}(copy(g.neighbors), vertices(g))
    return CachedGraph{PSet}(graph_copy, Dict{PSet, SeparatorType}())
end

# ==================== Components and Neighbors ====================

"""
    components(g, S)

Find connected components in G \\ S via BFS.
Returns a lazy iterator of `(component, neighbors_in_separator)` tuples.
"""
function components(g::CachedGraph{PSet}, S::PSet) where {PSet}
    return ComponentsIterator(g.graph, S)
end

# ==================== Neighbors ====================

"""
    neighbors(g, vertex_set)

Open neighborhood of a vertex set: the union of N(v) for v in vertex_set, minus vertex_set itself.
"""
function neighbors(g::CachedGraph{PSet}, vertex_set::PSet) where {PSet}
    result = PSet()

    for v in vertex_set
        result = result ∪ neighbors(g, v)
    end

    return setdiff(result, vertex_set)
end

# ==================== Minimal Separator ====================

"""
    is_minimal_separator(g, separator)

Returns `(is_minimal::Bool, components::Vector{PSet})`.
"""
function is_minimal_separator(g::CachedGraph{PSet}, separator::PSet) where {PSet}
    cmps = PSet[]; count = 0

    for (component, nbrs) in components(g, separator)
        push!(cmps, component)

        if nbrs == separator
            count += 1
        end
    end
    return (count >= 2, cmps)
end

# Definiton 5
#
# A vertex subset K ⊆ V(G) is *cliquish* if for each
# pair of distinct, nonadjacent vertices u, v ∈ K, there
# exists a path from u to v that does not lead through
# other vertices in K.
function is_csh!(work::Vector{PSet}, graph::CachedGraph{PSet}, K::PSet) where {PSet <: AbstractPackedSet}
    return septype!(work, graph, K) ≥ Cliquish
end

# Lemma 6
#
# A vertex subset K ⊆ V(G) is a potential maximal clique if
# and only if the following conditions hold.
#
#   1. K is cliquish.
#   2. K has no full components.
#
function is_pmc!(work::Vector{PSet}, graph::CachedGraph{PSet}, K::PSet) where {PSet <: AbstractPackedSet}
    return septype!(work, graph, K) == PotentialMaximalClique
end

function septype!(work::Vector{PSet}, graph::CachedGraph{PSet}, K::PSet) where {PSet <: AbstractPackedSet}
    return get!(graph.cache, K) do
        csh = 1
        pmc = 1

        @inbounds for v in K
            work[v] = neighbors(graph, v) ∪ v
        end

        for (_, N) in components(graph, K)
            pmc &= N != K

            @inbounds for v in N
                work[v] = work[v] ∪ N
            end
        end

        @inbounds for v in K
            csh &= K ⊆ work[v]
        end

        pmc &= csh
        return SeparatorType(csh + pmc)
    end
end

# ==================== Outlet ====================

"""
    outlet(g, bag, vertices)

Compute the outlet: vertices in `bag` that have neighbors outside `vertices`.

Equivalent to: let external = N(bag) \\ vertices; return N(external) ∩ bag.
Returns an empty bitset if external is empty.
"""
function outlet(g::CachedGraph{PSet}, B::PSet, V::PSet) where {PSet}
    external = setdiff(neighbors(g, B), V)
    return neighbors(g, external) ∩ B
end

