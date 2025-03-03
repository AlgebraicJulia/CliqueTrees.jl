module CatlabExt

using Base.Order
using Base.Sort: Algorithm as SortingAlgorithm
using Catlab: Catlab
using CliqueTrees
using CliqueTrees: sympermute, bfs, mcs, lexbfs, rcmmd, rcmgl, mcsm, lexm
using Graphs
using SparseArrays

struct CatlabGraph{G<:Catlab.HasGraph} <: AbstractGraph{Int}
    graph::G
end

function (::Type{G})(graph::BipartiteGraph) where {G<:Catlab.HasGraph}
    m::Int = ne(graph)
    n::Int = nv(graph)
    tgt::Vector{Int} = targets(graph)

    result = G(n)
    Catlab.add_parts!(result, :E, m; tgt=tgt)

    for v::Int in vertices(graph)
        i::Int = pointers(graph)[v]
        j::Int = pointers(graph)[v + 1] - 1
        Catlab.set_subpart!(result, i:j, :src, v)
    end

    return result
end

function CliqueTrees.sympermute(
    graph::Catlab.HasGraph, index::AbstractVector, order::Ordering
)
    return sympermute(CatlabGraph(graph), index, order)
end

function CliqueTrees.bfs(graph::Catlab.HasGraph)
    return bfs(CatlabGraph(graph))
end

function CliqueTrees.mcs(graph::Catlab.HasGraph, clique::AbstractVector)
    return mcs(CatlabGraph(graph), clique)
end

function CliqueTrees.rcmmd(graph::Catlab.HasGraph, alg::SortingAlgorithm)
    return rcmmd(CatlabGraph(graph))
end

function CliqueTrees.rcmgl(graph::Catlab.HasGraph, alg::SortingAlgorithm)
    return rcmgl(CatlabGraph(graph))
end

function CliqueTrees.lexbfs(graph::Catlab.HasGraph)
    return lexbfs(CatlabGraph(graph))
end

function CliqueTrees.lexm(graph::Catlab.HasGraph)
    return lexm(CatlabGraph(graph))
end

function CliqueTrees.mcsm(graph::Catlab.HasGraph)
    return mcsm(CatlabGraph(graph))
end

function CliqueTrees.permutation(graph::Catlab.HasGraph, alg::AbstractVector)
    return permutation(CatlabGraph(graph), alg)
end

function CliqueTrees.permutation(graph::Catlab.HasGraph, alg::Union{AMD,SymAMD})
    return permutation(CatlabGraph(graph), alg)
end

function CliqueTrees.permutation(graph::Catlab.HasGraph, alg::METIS)
    return permutation(CatlabGraph(graph), alg)
end

function CliqueTrees.permutation(graph::Catlab.HasGraph, alg::Spectral)
    return permutation(CatlabGraph(graph), alg)
end

function CliqueTrees.permutation(graph::Catlab.HasGraph, alg::BT)
    return permutation(CatlabGraph(graph), alg)
end

function CliqueTrees.isperfect(
    graph::Catlab.HasGraph, order::AbstractVector, index::AbstractVector
)
    return isperfect(CatlabGraph(graph), order, index)
end

function CliqueTrees.BipartiteGraph{V,E}(graph::Catlab.HasGraph) where {V,E}
    return BipartiteGraph{V,E}(CatlabGraph(graph))
end

function CliqueTrees.BipartiteGraph{V}(graph::Catlab.HasGraph) where {V}
    return BipartiteGraph{V}(CatlabGraph(graph))
end

function CliqueTrees.BipartiteGraph(graph::Catlab.HasGraph)
    return BipartiteGraph(CatlabGraph(graph))
end

############################
# Abstract Graph Interface #
############################

function SparseArrays.sparse(graph::CatlabGraph)
    return sparse(BipartiteGraph(graph))
end

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
    @view tgt[Catlab.incident(graph.graph, Int(i), :src)]
end

function Graphs.outdegree(graph::CatlabGraph, i::Integer)
    return length(Catlab.incident(graph.graph, Int(i), :src))
end

end
