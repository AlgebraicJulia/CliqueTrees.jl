struct FilledEdgeIter{V,E} <: AbstractEdgeIter
    graph::FilledGraph{V,E}
end

function Base.show(io::IO, iter::I) where {I<:FilledEdgeIter}
    n = length(iter)
    println(io, "$n-element $I:")

    for (i, edge) in enumerate(take(iter, MAX_ITEMS_PRINTED + 1))
        if i <= MAX_ITEMS_PRINTED
            println(io, " $edge")
        else
            println(io, " ⋮")
        end
    end
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

function Base.iterate(
    iter::FilledEdgeIter{V}, (i, p)::Tuple{V,V}=(one(V), one(V))
) where {V}
    if i <= nv(iter.graph)
        clique = neighbors(iter.graph, i)

        if p <= length(clique)
            edge = SimpleEdge(i, clique[p])
            state = p < length(clique) ? (i, p + one(V)) : (i + one(V), one(V))
            return edge, state
        else
            state = (i + one(V), one(V))
            return iterate(iter, state)
        end
    end
end

function Base.length(iter::FilledEdgeIter)
    return ne(iter.graph)
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
