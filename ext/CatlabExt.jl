module CatlabExt

using CliqueTrees
using Graphs

import Catlab

struct CatlabGraph{G <: Catlab.HasGraph} <: AbstractGraph{Int}
    graph::G
end

function (::Type{G})(graph::BipartiteGraph) where {G <: Catlab.HasGraph}
    m::Int = ne(graph)
    n::Int = nv(graph)
    tgt::Vector{Int} = targets(graph)

    result = G(n)
    Catlab.add_parts!(result, :E, m; tgt = tgt)

    for v::Int in vertices(graph)
        i::Int = pointers(graph)[v]
        j::Int = pointers(graph)[v + 1] - 1
        Catlab.set_subpart!(result, i:j, :src, v)
    end

    return result
end

function CliqueTrees.BipartiteGraph{V, E}(graph::Catlab.HasGraph) where {V, E}
    return BipartiteGraph{V, E}(CatlabGraph(graph))
end

function CliqueTrees.BipartiteGraph(graph::Catlab.HasGraph)
    return BipartiteGraph(CatlabGraph(graph))
end

############################
# Abstract Graph Interface #
############################

function Graphs.is_directed(graph::CatlabGraph)
    return true
end

function Graphs.ne(graph::CatlabGraph)
    return Catlab.ne(graph.graph)
end

function Graphs.nv(graph::CatlabGraph)
    return Catlab.nv(graph.graph)
end

function Graphs.vertices(graph::CatlabGraph)
    return Catlab.vertices(graph.graph)
end

function Graphs.outneighbors(graph::CatlabGraph, i::Integer)
    tgt = Catlab.tgt(graph.graph)::Vector{Int} # type instability
    return @view tgt[Catlab.incident(graph.graph, Int(i), :src)]
end

function Graphs.outdegree(graph::CatlabGraph, i::Integer)
    return length(Catlab.incident(graph.graph, Int(i), :src))
end

end
