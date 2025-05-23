module MetisExt

using ArgCheck
using CliqueTrees
using CliqueTrees: Scalar, View, simplegraph
using Graphs

import Metis

const INT = Metis.idx_t

function CliqueTrees.permutation(weights::AbstractVector, graph, alg::METIS)
    return permutation(weights, BipartiteGraph(graph), alg)
end

function CliqueTrees.permutation(graph, alg::METIS)
    return permutation(BipartiteGraph(graph), alg)
end

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::METIS) where {V}
    n = nv(graph)

    # construct options
    options = Vector{INT}(undef, Metis.METIS_NOPTIONS)
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

    # construct METIS graph
    simple = simplegraph(INT, INT, graph)
    xadj = pointers(simple)
    adjncy = targets(simple)
    vwght = trunc.(INT, weights)

    # construct permutation
    metisorder = Vector{INT}(undef, n)
    metisindex = Vector{INT}(undef, n)

    Metis.@check Metis.METIS_NodeND(
        Ref{INT}(n),
        xadj,
        adjncy,
        vwght,
        options,
        metisorder,
        metisindex,
    )

    # restore vertex type
    order::Vector{V} = metisorder
    index::Vector{V} = metisindex
    return order, index
end

function CliqueTrees.permutation(graph::AbstractGraph{V}, alg::METIS) where {V}
    n = nv(graph)

    # construct options
    options = Vector{INT}(undef, Metis.METIS_NOPTIONS)
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

    # construct METIS graph
    simple = simplegraph(INT, INT, graph)
    xadj = pointers(simple)
    adjncy = targets(simple)

    # construct permutation
    metisorder = Vector{INT}(undef, n)
    metisindex = Vector{INT}(undef, n)

    Metis.@check Metis.METIS_NodeND(
        Ref{INT}(n),
        xadj,
        adjncy,
        C_NULL,
        options,
        metisorder,
        metisindex,
    )

    # restore vertex type
    order::Vector{V} = metisorder
    index::Vector{V} = metisindex
    return order, index
end

function CliqueTrees.separator!(sepsize::Scalar{INT}, part::Vector{INT}, weights::AbstractVector, graph::BipartiteGraph{INT, INT}, alg::METISND)
    return CliqueTrees.separator!(sepsize, part, trunc.(INT, weights), graph, alg)
end

function CliqueTrees.separator!(sepsize::Scalar{INT}, part::Vector{INT}, weights::Union{View{INT}, Vector{INT}}, graph::BipartiteGraph{INT, INT}, alg::METISND)
    m = ne(graph); n = nv(graph); nn = n + one(INT)
    @argcheck n <= length(part)
    @argcheck n <= length(weights)

    # construct options
    options = Vector{INT}(undef, Metis.METIS_NOPTIONS)
    options .= -1 # null
    options[Metis.METIS_OPTION_NUMBERING + 1] = 1
    options[Metis.METIS_OPTION_SEED + 1] = alg.seed
    options[Metis.METIS_OPTION_UFACTOR + 1] = alg.ufactor

    # construct METIS graph
    xadj = pointers(graph); xadj .-= one(INT)
    adjncy = targets(graph); adjncy .-= one(INT)
    vwght = weights

    # construct separator
    Metis.@check Metis.METIS_ComputeVertexSeparator(
        Ref{INT}(n),
        xadj,
        adjncy,
        vwght,
        options,
        sepsize,
        part,
    )

    xadj .+= one(INT)
    adjncy .+= one(INT)
    return
end

end
