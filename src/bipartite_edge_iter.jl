struct BipartiteEdgeIter{V, E, Ptr, Tgt} <: AbstractEdgeIter
    graph::BipartiteGraph{V, E, Ptr, Tgt}
end

function Base.show(io::IO, iter::I) where {I <: BipartiteEdgeIter}
    n = length(iter)
    println(io, "$n-element $I:")

    for (i, edge) in enumerate(take(iter, MAX_ITEMS_PRINTED + 1))
        if i <= MAX_ITEMS_PRINTED
            println(io, " $edge")
        else
            println(io, " ⋮")
        end
    end
    return
end

############################
# Abstract Graph Interface #
############################

function Graphs.edgetype(::BipartiteGraph{V}) where {V}
    return SimpleEdge{V}
end

function Graphs.edges(graph::BipartiteGraph)
    return BipartiteEdgeIter(graph)
end

#######################
# Iteration Interface #
#######################

function Base.iterate(
        iter::BipartiteEdgeIter{V, E}, (i, p)::Tuple{V, E} = (one(V), one(E))
    ) where {V, E}
    return if p <= length(iter)
        edge = SimpleEdge{V}(i, targets(iter.graph)[p])

        j = i + one(V)
        q = p + one(E)
        state = q < pointers(iter.graph)[j] ? (i, q) : (j, q)

        edge, state
    end
end

function Base.length(iter::BipartiteEdgeIter)
    return ne(iter.graph)
end

function Base.eltype(::Type{<:BipartiteEdgeIter{V}}) where {V}
    return SimpleEdge{V}
end

function Base.in(edge, iter::BipartiteEdgeIter)
    return has_edge(iter.graph, edge)
end
