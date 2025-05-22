struct BipartiteEdgeIter{V, E, Ptr, Tgt} <: AbstractEdgeIter
    graph::BipartiteGraph{V, E, Ptr, Tgt}
end

function Base.show(io::IO, iter::BipartiteEdgeIter)
    printiterator(io, iter)
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

function Base.iterate(iter::BipartiteEdgeIter{V, E}, (i, p)::Tuple{V, E} = (one(V), one(E))) where {V, E}
    graph = iter.graph; ii = i + one(V); pp = p + one(E)
    result = nothing

    if p <= ne(graph)
        @inbounds j = targets(graph)[p]
        @inbounds qq = pointers(graph)[ii]
        edge = SimpleEdge{V}(i, j)

        if pp < qq
            state = (i, pp)
        else
            state = (ii, pp)
        end

        result = (edge, state)
    end

    return result
end

function Base.length(iter::BipartiteEdgeIter)
    m::Int = ne(iter.graph)
    return m
end

function Base.eltype(::Type{<:BipartiteEdgeIter{V}}) where {V}
    return SimpleEdge{V}
end

function Base.in(edge, iter::BipartiteEdgeIter)
    return has_edge(iter.graph, edge)
end
