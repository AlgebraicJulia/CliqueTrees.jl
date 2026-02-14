# Graph: a mutable graph with bitset-backed neighborhoods.
# Faithful transpilation of the C# Graph.cs class, using 1-based vertex indexing
# and immutable AbstractPackedSet values in place of the C# BitSet.

"""
    wt(weights, set)

Sum of vertex weights for all vertices in `set`.
"""
function wt(weights::Vector{Int}, set::PSet) where {PSet <: AbstractPackedSet}
    s = 0

    for v in set
        s += weights[v]
    end
    return s
end

# ==================== Graph ====================

"""
    Graph{PSet}

A mutable graph with logically removable vertices.  `vertices(g)` returns the
packed set of vertices still present; `nv(g)` returns its size.

Type parameter `PSet <: AbstractPackedSet` is the bitset representation used for
neighborhoods and vertex sets.
"""
struct Graph{PSet <: AbstractPackedSet}
    neighbors::Vector{PSet}   # N(v) as bitset
    V::PSet                   # bitset of vertices still present
end

vertices(g::Graph) = g.V
nv(g::Graph) = length(vertices(g))
neighbors(g::Graph, v::Int) = g.neighbors[v]

# ==================== Constructors ====================

"""
    Graph(V::PSet)

Construct a Graph with vertex set V and uninitialized domain-sized neighborhood array.
"""
function Graph(V::PSet) where {PSet <: AbstractPackedSet}
    d = domain(PSet)
    neighbors = Vector{PSet}(undef, d)
    return Graph{PSet}(neighbors, V)
end

"""
    Graph{PSet}(n::Int)

Construct a Graph with vertices 1:n and uninitialized domain-sized neighborhood array.
"""
function Graph{PSet}(n::Int) where {PSet <: AbstractPackedSet}
    return Graph(packedset(PSet, oneto(n)))
end

# ==================== Mutation: add_edge!, make_into_clique! ====================

"""
    add_edge!(g, u, v)

Add an edge between vertices `u` and `v`.  Does nothing if the edge already exists.
"""
function add_edge!(g::Graph{PSet}, u::Int, v::Int) where {PSet}
    if v in neighbors(g, u)
        return nothing
    end
    g.neighbors[u] = neighbors(g, u) ∪ v
    g.neighbors[v] = neighbors(g, v) ∪ u
    return nothing
end


"""
    make_into_clique!(g, vertices)

Add all missing edges among the given `vertices`, making them a clique.
"""
function make_into_clique!(g::Graph{PSet}, V::PSet) where {PSet}
    remaining = V

    while !isempty(remaining)
        u, remaining = popfirst_nonempty(remaining)
        rest = remaining

        while !isempty(rest)
            v, rest = popfirst_nonempty(rest)
            add_edge!(g, u, v)
        end
    end
    return nothing
end

# ==================== Queries ====================

"""
    neighbors(g, vertex_set)

Return N(vertex_set) \\ vertex_set: the union of open neighborhoods of all elements
of `vertex_set`, with the set itself removed.
"""
function neighbors(g::Graph{PSet}, vertex_set::PSet) where {PSet}
    result = PSet()

    for v in vertex_set
        result = result ∪ neighbors(g, v)
    end
    return setdiff(result, vertex_set)
end

"""
    ComponentsIterator{PSet}

Lazy BFS iterator over connected components in G \\ S.
Each iteration yields `(component, separator_neighbors)`.
"""
struct ComponentsIterator{PSet <: AbstractPackedSet}
    graph::Graph{PSet}
    S::PSet
end

function Base.eltype(::Type{ComponentsIterator{PSet}}) where {PSet}
    return Tuple{PSet, PSet}
end

function Base.IteratorSize(::Type{C}) where {C <: ComponentsIterator}
    return Base.SizeUnknown()
end

function Base.iterate(iter::ComponentsIterator{PSet}, V::PSet=setdiff(vertices(iter.graph), iter.S)) where {PSet}
    isempty(V) && return nothing

    v = first_nonempty(V)
    C = F = packedset(PSet, v)

    while !isempty(F)
        N = neighbors(iter.graph, F)
        F = setdiff(N, C ∪ iter.S)
        C = C ∪ N
    end

    return ((setdiff(C, iter.S), C ∩ iter.S), setdiff(V, C))
end

"""
    components(g, S)

Find connected components in G \\ S via BFS.
Returns a lazy iterator of `(component, neighbors_in_separator)` tuples.
"""
function components(g::Graph{PSet}, S::PSet) where {PSet}
    return ComponentsIterator(g, S)
end
