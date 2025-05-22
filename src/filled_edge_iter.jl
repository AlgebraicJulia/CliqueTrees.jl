struct FilledEdgeIter{V, E} <: AbstractEdgeIter
    graph::FilledGraph{V, E}
end

function Base.show(io::IO, iter::FilledEdgeIter)
    printiterator(io, iter)
    return
end

############################
# Abstract Graph Interface #
############################

function Graphs.edgetype(::FilledGraph{V}) where {V}
    return SimpleEdge{V}
end

function Graphs.edges(graph::FilledGraph)
    return FilledEdgeIter(graph)
end

#######################
# Iteration Interface #
#######################

function Base.iterate(iter::FilledEdgeIter{V}, (i, p)::Tuple{V, V} = (one(V), one(V))) where {V}
    graph = iter.graph; ii = i + one(V); pp = p + one(V)
    result = nothing

    if i <= nv(graph)
        @inbounds clique = neighbors(graph, i)
        @inbounds degree = eltypedegree(graph, i)

        if p <= degree
            @inbounds j = clique[p]
            edge = SimpleEdge(i, j)

            if p < degree
                state = (i, pp)
            else
                state = (ii, one(V))
            end

            result = (edge, state)
        else
            state = (ii, one(V))
            result = iterate(iter, state)
        end
    end

    return result
end

function Base.length(iter::FilledEdgeIter)
    m::Int = ne(iter.graph)
    return m
end

function Base.eltype(::Type{<:FilledEdgeIter{V}}) where {V}
    return SimpleEdge{V}
end

function Base.in(edge, iter::FilledEdgeIter)
    return has_edge(iter.graph, edge)
end

function Base.hasfastin(::Type{<:FilledEdgeIter})
    return true
end
