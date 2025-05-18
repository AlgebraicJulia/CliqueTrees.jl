module MetisExt

using Base: oneto
using CliqueTrees
using CliqueTrees: EliminationAlgorithm
using CliqueTrees.Utilities
using Graphs
using Metis: Metis

const INT = Metis.idx_t

function CliqueTrees.permutation(weights::AbstractVector, graph, alg::METIS)
    return permutation(weights, BipartiteGraph(graph), alg)
end

function CliqueTrees.permutation(graph, alg::METIS)
    return permutation(BipartiteGraph(graph), alg)
end

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::METIS) where {V}
    m = ne(graph) * (2 - is_directed(graph))
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
    xadj = Vector{INT}(undef, n + 1)
    adjncy = Vector{INT}(undef, m)
    vwght = Vector{INT}(undef, n)
    xadj[begin] = p = one(INT)

    @inbounds for j in vertices(graph)
        vwght[j] = trunc(INT, weights[j])

        for i in neighbors(graph, j)
            if i != j
                adjncy[p] = i
                p += one(INT)
            end
        end

        xadj[j + one(V)] = p
    end

    resize!(adjncy, p - one(INT))

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
    m = ne(graph) * (2 - is_directed(graph))
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
    xadj = Vector{INT}(undef, n + 1)
    adjncy = Vector{INT}(undef, m)
    xadj[begin] = p = one(INT)

    @inbounds for j in vertices(graph)
        for i in neighbors(graph, j)
            if i != j
                adjncy[p] = i
                p += one(INT)
            end
        end

        xadj[j + one(V)] = p
    end

    resize!(adjncy, p - one(INT))

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

function CliqueTrees.permutation(weights::AbstractVector, graph, alg::IND)
    return permutation(weights, BipartiteGraph(graph), alg)
end

function CliqueTrees.permutation(graph, alg::IND)
    return permutation(BipartiteGraph(graph), alg)
end

function CliqueTrees.permutation(graph::AbstractGraph{V}, alg::IND) where {V}
    weights = ones(V, nv(graph))
    return permutation(weights, graph, alg)
end

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::IND) where {V}
    n = nv(graph)
    m = ne(graph) * (2 - is_directed(graph))
    new = BipartiteGraph{INT, INT}(n, n, m)
    pointers(new)[begin] = p = one(INT)

    @inbounds for v in vertices(graph)
        for w in neighbors(graph, v)
            if v != w
                targets(new)[p] = INT(w)
                p += one(INT)
            end
        end

        pointers(new)[v + one(V)] = p
    end

    resize!(targets(new), p - one(INT))
    order::Vector{V} = indissect(vertices(new), oneto(zero(INT)), zero(INT), weights, new, INT(alg.limit), alg.alg)
    return order, invperm(order)
end

function indissect(label::AbstractVector{INT}, clique::AbstractVector{INT}, delta::INT, weights::AbstractVector, graph::AbstractGraph{INT}, limit::INT, alg::EliminationAlgorithm)
    n = nv(graph)

    if n <= limit + delta
        order = first(permutation(weights, graph; alg = CompositeRotations(clique, alg)))
    else
        project = separator(weights, graph)
        project0 = Vector{INT}(undef, n)
        project1 = Vector{INT}(undef, n)
        n0 = n1 = n2 = zero(INT)
        m0 = m1 = zero(INT)

        @inbounds for v in vertices(graph)
            vv = project[v]

            if iszero(vv)
                project0[v] = n0 += one(INT)
                m0 += eltypedegree(graph, v)
            elseif isone(vv)
                project1[v] = n1 += one(INT)
                m1 += eltypedegree(graph, v)
            else
                project0[v] = n0 += one(INT)
                project1[v] = n1 += one(INT)
                n2 += one(INT)

                for w in outneighbors(graph, v)
                    ww = project[w]

                    if iszero(ww)
                        m0 += one(INT)
                    elseif isone(ww)
                        m1 += one(INT)
                    end
                end
            end
        end

        m0 += n2 * n2 - n2
        m1 += n2 * n2 - n2
        label0 = Vector{INT}(undef, n0)
        label1 = Vector{INT}(undef, n1)
        label2 = Vector{INT}(undef, n2)
        t0 = t1 = t2 = zero(INT)

        @inbounds for v in vertices(graph)
            vv = project[v]

            if iszero(vv)
                t0 += one(INT)
                label0[t0] = v
            elseif isone(vv)
                t1 += one(INT)
                label1[t1] = v
            else
                t0 += one(INT)
                t1 += one(INT)
                t2 += one(INT)
                label0[t0] = label1[t1] = label2[t2] = v
            end
        end

        t0 = t1 = one(INT)
        weights0 = Vector{INT}(undef, n0)
        weights1 = Vector{INT}(undef, n1)
        graph0 = BipartiteGraph{INT, INT}(n0, n0, m0)
        graph1 = BipartiteGraph{INT, INT}(n1, n1, m1)
        pointers(graph0)[t0] = u0 = one(INT)
        pointers(graph1)[t1] = u1 = one(INT)

        @inbounds for v in vertices(graph)
            vv = project[v]

            if iszero(vv)
                weights0[t0] = weights[v]

                for w in neighbors(graph, v)
                    targets(graph0)[u0] = project0[w]
                    u0 += one(INT)
                end

                t0 += one(INT)
                pointers(graph0)[t0] = u0
            elseif isone(vv)
                weights1[t1] = weights[v]

                for w in neighbors(graph, v)
                    targets(graph1)[u1] = project1[w]
                    u1 += one(INT)
                end

                t1 += one(INT)
                pointers(graph1)[t1] = u1
            else
                weights0[t0] = weights1[t1] = weights[v]

                for w in neighbors(graph, v)
                    ww = project[w]

                    if iszero(ww)
                        targets(graph0)[u0] = project0[w]
                        u0 += one(INT)
                    elseif isone(ww)
                        targets(graph1)[u1] = project1[w]
                        u1 += one(INT)
                    end
                end

                for w in label2
                    if v != w
                        targets(graph0)[u0] = project0[w]
                        targets(graph1)[u1] = project1[w]
                        u0 += one(INT)
                        u1 += one(INT)
                    end
                end

                t0 += one(INT)
                t1 += one(INT)
                pointers(graph0)[t0] = u0
                pointers(graph1)[t1] = u1
            end
        end

        order0 = indissect(label0, view(project0, label2), n2, weights0, graph0, limit + delta, alg)
        order1 = indissect(label1, view(project1, label2), n2, weights1, graph1, limit + delta, alg)
        order = first(permutation(graph; alg = CompositeRotations(clique, [order0; order1; label2])))
    end

    @inbounds for i in oneto(n - delta)
        order[i] = label[order[i]]
    end

    return resize!(order, n - delta)
end

function separator(weights::AbstractVector, graph::BipartiteGraph{INT, INT, Vector{INT}, Vector{INT}})
    n = nv(graph)

    # construct options
    options = Vector{INT}(undef, Metis.METIS_NOPTIONS)
    options .= -1 # null
    options[Metis.METIS_OPTION_NUMBERING + 1] = 1

    # construct METIS graph
    xadj = pointers(graph) .- one(INT)
    adjncy = targets(graph) .- one(INT)
    vwght = trunc.(INT, weights)

    # construct separator
    part = Vector{INT}(undef, n)
    sepsize = fill(zero(INT), 1)

    Metis.@check Metis.METIS_ComputeVertexSeparator(
        Ref{INT}(n),
        xadj,
        adjncy,
        vwght,
        options,
        sepsize,
        part,
    )

    return part
end

end
