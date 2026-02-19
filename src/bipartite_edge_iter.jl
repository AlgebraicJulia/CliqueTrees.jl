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
    graph = iter.graph; ip1 = i + one(V); pp1 = p + one(E)
    result = nothing

    if p <= ne(graph)
        @inbounds j = targets(graph)[p]
        @inbounds pnext = pointers(graph)[ip1]

        @inbounds while p ≥ pnext
            i = ip1; ip1 = i + one(V); pnext = pointers(graph)[ip1]
        end

        edge = SimpleEdge{V}(i, j)

        if pp1 ≥ pnext
            i = ip1
        end

        result = (edge, (i, pp1))
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
