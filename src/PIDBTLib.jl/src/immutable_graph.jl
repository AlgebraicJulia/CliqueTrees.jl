# CachedGraph: an immutable snapshot of a Graph for the DP algorithm.
# Shares the same vertex indices as the source Graph (no reindexing).
# Uses AbstractPackedSet for bitset neighborhoods (immutable, functional style).

# ==================== Struct ====================

@enum SeparatorType Neither Cliquish PotentialMaximalClique

struct CachedGraph{PSet <: AbstractPackedSet}
    graph::Graph{PSet}                      # the underlying graph
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
    return CachedGraph{PSet}(graph_copy)
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
    is_minimal_separator(g, separator, cmps=PSet[])

Returns `(is_minimal::Bool, components::Vector{PSet})`.

The components are written into `cmps`, which is emptied first and returned.
Callers that pass a reused buffer must be done with the components before
the next call.
"""
function is_minimal_separator(g::CachedGraph{PSet}, separator::PSet, cmps::Vector{PSet}=PSet[]) where {PSet}
    empty!(cmps); count = 0

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

# Caching the result (as the C# does) does not pay: about half of all calls
# are misses, and the cache grows to millions of entries.
#
# If K has a full component C, then any two vertices of K are joined by a path
# through C, so K is cliquish but not a PMC. We return as soon as one is found.
# (The components are enumerated starting from the smallest vertex outside K,
# which usually lies in the full component, if there is one.)
function septype!(work::Vector{PSet}, graph::CachedGraph{PSet}, K::PSet) where {PSet <: AbstractPackedSet}
    @inbounds for v in K
        work[v] = neighbors(graph, v) ∪ v
    end

    for (_, N) in components(graph, K)
        N == K && return Cliquish

        @inbounds for v in N
            work[v] = work[v] ∪ N
        end
    end

    @inbounds for v in K
        K ⊆ work[v] || return Neither
    end

    return PotentialMaximalClique
end

# ==================== Outlet ====================

"""
    outlet(g, bag, vertices)

Compute the outlet: vertices in `bag` that have neighbors outside `vertices`.

Equivalent to: let external = N(bag) \\ vertices; return N(external) ∩ bag.
Returns an empty bitset if external is empty.
"""
function outlet(g::CachedGraph{PSet}, B::PSet, V::PSet) where {PSet}
    S = PSet()

    for v in B
        if !(neighbors(g, v) ⊆ V)
            S = S ∪ v
        end
    end

    return S
end

