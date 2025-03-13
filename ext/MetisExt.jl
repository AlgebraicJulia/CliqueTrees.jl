module MetisExt

using CliqueTrees
using Graphs
using Metis: Metis

const IDX = Metis.idx_t

function CliqueTrees.permutation(graph, alg::METIS)
    return permutation(BipartiteGraph(graph), alg)
end

function CliqueTrees.permutation(graph::AbstractGraph{V}, alg::METIS) where V
    # construct options
    options = Vector{IDX}(undef, Metis.METIS_NOPTIONS)
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
    xadj = Vector{IDX}(undef, nv(graph) + 1)
    adjncy = Vector{IDX}(undef,  ne(graph) * (2 - is_directed(graph)))
    xadj[begin] = p = one(IDX)

    @inbounds for j in vertices(graph)
        for i in neighbors(graph, j)
            if i != j
                adjncy[p] = i
                p += one(IDX)
            end
        end

        xadj[j + one(V)] = p
    end

    resize!(adjncy, p - one(IDX))

    # construct permutation
    metisorder = Vector{IDX}(undef, nv(graph))
    metisindex = Vector{IDX}(undef, nv(graph))

    Metis.@check Metis.METIS_NodeND(
        Ref{IDX}(nv(graph)),
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

end
