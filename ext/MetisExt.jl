module MetisExt

using CliqueTrees
using Graphs
using Metis: Metis
using SparseArrays

function CliqueTrees.permutation(graph, alg::METIS)
    return permutation(BipartiteGraph(graph), alg)
end

function CliqueTrees.permutation(graph::AbstractGraph{V}, alg::METIS) where {V}
    order::Vector{V}, index::Vector{V} = permutation(sparse(graph), alg)
    return order, index
end

function CliqueTrees.permutation(matrix::SparseMatrixCSC{T,I}, alg::METIS) where {T,I}
    order::Vector{I}, index::Vector{I} = permutation(
        SparseMatrixCSC{T,Metis.idx_t}(matrix), alg
    )
    return order, index
end

function CliqueTrees.permutation(matrix::SparseMatrixCSC{<:Any,Metis.idx_t}, alg::METIS)
    options = Vector{Metis.idx_t}(undef, Metis.METIS_NOPTIONS)
    options .= -1 # null
    options[Metis.METIS_OPTION_CTYPE + 1] = alg.ctype
    options[Metis.METIS_OPTION_RTYPE + 1] = alg.rtype
    options[Metis.METIS_OPTION_NSEPS + 1] = alg.nseps
    options[Metis.METIS_OPTION_NUMBERING + 1] = 1
    options[Metis.METIS_OPTION_NITER + 1] = alg.niter
    options[Metis.METIS_OPTION_SEED + 1] = alg.seed
    options[Metis.METIS_OPTION_COMPRESS + 1] = alg.compress
    options[Metis.METIS_OPTION_CCORDER + 1] = alg.ccorder
    options[Metis.METIS_OPTION_PFACTOR + 1] = alg.pfactor
    options[Metis.METIS_OPTION_UFACTOR + 1] = alg.ufactor

    graph = Metis.graph(matrix; check_hermitian=false)
    order = Vector{Metis.idx_t}(undef, graph.nvtxs)
    index = Vector{Metis.idx_t}(undef, graph.nvtxs)
    Metis.@check Metis.METIS_NodeND(
        Ref{Metis.idx_t}(graph.nvtxs),
        graph.xadj,
        graph.adjncy,
        graph.vwgt,
        options,
        order,
        index,
    )

    return order, index
end

end
