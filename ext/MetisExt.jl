module MetisExt

using Base: oneto
using Base.Order
using CliqueTrees
using CliqueTrees: EliminationAlgorithm, Parent, UnionFind, simplegraph, partition!, sympermute!_impl!, compositerotations_impl!, bestfill_impl!, bestwidth_impl!, nov
using CliqueTrees.MMDLib: mmd_impl!
using CliqueTrees.Utilities
using Graphs

import Metis

const INT = Metis.idx_t
const NOPTIONS = Metis.METIS_NOPTIONS
const OPTION_CTYPE = Metis.METIS_OPTION_CTYPE + one(INT)
const OPTION_RTYPE = Metis.METIS_OPTION_RTYPE + one(INT)
const OPTION_NSEPS = Metis.METIS_OPTION_NSEPS + one(INT)
const OPTION_NUMBERING = Metis.METIS_OPTION_NUMBERING + one(INT)
const OPTION_NITER = Metis.METIS_OPTION_NITER + one(INT)
const OPTION_SEED = Metis.METIS_OPTION_SEED + one(INT)
const OPTION_COMPRESS = Metis.METIS_OPTION_COMPRESS + one(INT)
const OPTION_CCORDER = Metis.METIS_OPTION_CCORDER + one(INT)
const OPTION_PFACTOR = Metis.METIS_OPTION_PFACTOR + one(INT)
const OPTION_UFACTOR = Metis.METIS_OPTION_UFACTOR + one(INT)

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::METIS) where {V}
    simple = simplegraph(INT, INT, graph)
    order::Vector{V}, index::Vector{V} = metis(weights, simple, alg)
    return order, index
end

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph, alg::ND{<:Any, <:EliminationAlgorithm, METISND})
    order = dissect(weights, graph, alg)
    return order, invperm(order)
end

function metis(weights::AbstractVector, graph::BipartiteGraph{INT, INT}, alg::METIS)
    @assert nv(graph) <= length(weights)
    n = nv(graph); new = Vector{INT}(undef, n)

    @inbounds for v in oneto(n)
        new[v] = trunc(INT, weights[v])
    end

    return metis(new, graph, alg)
end

function metis(weights::Vector{INT}, graph::BipartiteGraph{INT, INT}, alg::METIS)
    n = nv(graph)

    # construct options
    options = Vector{INT}(undef, NOPTIONS)
    setoptions!(options, alg)

    # construct METIS graph
    xadj = pointers(graph)
    adjncy = targets(graph)
    vwght = weights

    # construct permutation
    order = Vector{INT}(undef, n)
    index = Vector{INT}(undef, n)

    Metis.@check Metis.METIS_NodeND(
        Ref{INT}(n),
        xadj,
        adjncy,
        vwght,
        options,
        order,
        index,
    )

    return order, index
end

function separator!(options::AbstractVector{INT}, sepsize::AbstractScalar{INT}, part::AbstractVector{INT}, weights::AbstractVector{INT}, graph::BipartiteGraph{INT, INT}, imbalance::INT, alg::METISND)
    @assert NOPTIONS <= length(options)
    @assert nv(graph) <= length(part)
    @assert nv(graph) <= length(weights)
    @assert ispositive(imbalance)
    n = nv(graph); m = ne(graph); nn = n + one(INT)

    # construct options
    setoptions!(options, imbalance, alg)

    # construct METIS graph
    xadj = pointers(graph)
    adjncy = targets(graph)
    vwght = weights

    @inbounds for i in oneto(nn)
        xadj[i] -= one(INT)
    end

    @inbounds for p in oneto(m)
        adjncy[p] -= one(INT)
    end

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

    @inbounds for i in oneto(nn)
        xadj[i] += one(INT)
    end

    @inbounds for p in oneto(m)
        adjncy[p] += one(INT)
    end

    return
end

function dissect(weights::AbstractVector{W}, graph::AbstractGraph, alg::ND) where {W}
    n = nv(graph)
    scale = convert(W, alg.scale)
    intweights = FVector{INT}(undef, n)

    @inbounds for v in oneto(n)
        w = weights[v]

        if W <: AbstractFloat
            w *= scale
        end

        intweights[v] = trunc(INT, w)
    end

    return dissect(intweights, graph, alg)
end

function dissect(weights::FVector{INT}, graph::AbstractGraph{V}, alg::ND) where {V <: Integer}
    simple = simplegraph(INT, INT, graph)
    order = convert(Vector{V}, dissectsimple(weights, simple, alg))
    return order
end

function dissectsimple(weights::AbstractVector{INT}, graph::BipartiteGraph{INT, INT}, alg::ND{S}) where {S}
    n = nv(graph); m = ne(graph); nn = n + one(INT)
    maxlevel = convert(INT, alg.level)
    minwidth = convert(INT, alg.width)
    imbalance = convert(INT, alg.imbalance)

    work00 = FScalar{INT}(undef)
    work01 = Vector{INT}(undef, m)
    work02 = Vector{INT}(undef, m)
    work03 = FVector{INT}(undef, max(n, NOPTIONS))
    work04 = FVector{INT}(undef, n)
    work05 = FVector{INT}(undef, n)
    work06 = FVector{INT}(undef, n)
    work07 = FVector{INT}(undef, n)
    work08 = FVector{INT}(undef, n)
    work09 = FVector{INT}(undef, nn)
    work10 = FVector{INT}(undef, nn)
    work11 = FVector{INT}(undef, n)
    work12 = FVector{INT}(undef, n)
    work13 = FVector{INT}(undef, n)
    work14 = FVector{INT}(undef, n)
    work15 = FVector{INT}(undef, n)
    work16 = FVector{INT}(undef, n)

    parts = FVector{INT}[]
    orders = FVector{INT}[]

    nodes = Tuple{
        BipartiteGraph{INT, INT, FVector{INT}, FVector{INT}}, # graph
        FVector{INT},                                         # weights
        BipartiteGraph{INT, INT, FVector{INT}, FVector{INT}}, # label
        FVector{INT},                                         # clique
        INT,                                                  # level
    }[]

    label = BipartiteGraph{INT, INT}(n, n, n)
    clique = FVector{INT}(undef, zero(INT))
    level = zero(INT)

    @inbounds for v in oneto(n)
        pointers(label)[v] = v
        targets(label)[v] = v
    end

    pointers(label)[nn] = nn
    push!(nodes, (graph, weights, label, clique, level))

    @inbounds while !isempty(nodes)
        graph, weights, label, clique, level = pop!(nodes)
        n = nv(graph); m = ne(graph); l = ne(label); k = convert(INT, length(clique))

        if !isnegative(level) # unprocessed
            isleaf = n <= minwidth || level >= maxlevel || m == n * (n - one(INT))

            if isleaf # leaf
                push!(nodes, (graph, weights, label, clique, -one(INT)))
            else      # branch
                if m > length(work01)
                    resize!(work01, half(m))
                    resize!(work02, half(m))
                end

                part = FVector{INT}(undef, n)
                separator!(work03, work00, part, weights, graph, imbalance, alg.dis)

                child0, child1, order2 = partition!(work00, work03, work04, work07, work08,
                    work11, work12, work13, work09, work10, work01, work02, work05, work06,
                    part, weights, graph)

                push!(
                    nodes,
                    (graph, weights, label, clique, -two(INT)),
                    (child0..., level + one(INT)),
                    (child1..., level + one(INT)),
                )

                push!(parts, part)
                push!(orders, order2)
            end
        else                  # processed
            isleaf = isone(-level)
            iscomplete = m == n * (n - one(INT))

            if iscomplete # complete graph
                for v in oneto(n)
                    work03[v] = v
                end
            else
                tree = Parent(n, work06)
                upper = BipartiteGraph(n, n, half(m), work09, work01)
                lower = BipartiteGraph(n, n, half(m), work10, work02)

                if isleaf # leaf
                    order, index = permutation(weights, graph, alg.alg)
                else      # branch
                    part = pop!(parts)
                    ndsorder = Vector{INT}(undef, n)
                    ndsindex = Vector{INT}(undef, n)
                    i = zero(INT)

                    for v in pop!(orders)
                        if !istwo(part[v])
                            ndsindex[v] = i += one(INT)
                            ndsorder[i] = v
                        end
                    end

                    for v in pop!(orders)
                        if !istwo(part[v])
                            ndsindex[v] = i += one(INT)
                            ndsorder[i] = v
                        end
                    end

                    for v in pop!(orders)
                        ndsindex[v] = i += one(INT)
                        ndsorder[i] = v
                    end 

                    if isone(S) || istwo(S)
                        sets = UnionFind(n, work03, work04, work05)
                        grdorder, grdindex = permutation(weights, graph, alg.alg)

                        if isone(S)
                            best = bestwidth_impl!(lower, upper, tree, sets, work07,
                                work08, work11, work12, work13, work14, work15,
                                work16, weights, graph, (ndsindex, grdindex))
                        else
                            best = bestfill_impl!(lower, upper, tree, sets, work07,
                                work08, work11, work12, work13, work14, work15,
                                work16, weights, graph, (ndsindex, grdindex))
                        end

                        order = (ndsorder, grdorder)[best]
                        index = (ndsindex, grdindex)[best]
                    else
                        order, index = ndsorder, ndsindex
                    end
                end

                sympermute!_impl!(upper, graph, index, Forward)

                for i in oneto(k)
                    clique[i] = index[clique[i]]
                end

                compositerotations_impl!(index, work03, work04,
                    work05, lower, tree, upper, clique)

                for v in oneto(n)
                    i = index[v]; work03[i] = order[v]
                end
            end

            j = zero(INT); outorder = FVector{INT}(undef, l)

            for i in oneto(n)
                v = work03[i]

                for w in neighbors(label, v)
                    j += one(INT); outorder[j] = w
                end
            end

            push!(orders, outorder)
        end
    end

    return only(orders)
end

function setoptions!(options::AbstractVector{INT}, imbalance::INT, alg::METISND)
    for i in oneto(NOPTIONS)
        options[i] = -one(INT) # null
    end

    options[OPTION_NSEPS] = convert(INT, alg.nseps)
    options[OPTION_NUMBERING] = one(INT)
    options[OPTION_SEED] = convert(INT, alg.seed)
    options[OPTION_UFACTOR] = convert(INT, imbalance)
    return
end

function setoptions!(options::AbstractVector{INT}, alg::METIS)
    for i in oneto(NOPTIONS)
        options[i] = -one(INT) # null
    end

    options[OPTION_CTYPE] = convert(INT, alg.ctype)
    options[OPTION_RTYPE] = convert(INT, alg.rtype)
    options[OPTION_NSEPS] = convert(INT, alg.nseps)
    options[OPTION_NUMBERING] = one(INT)
    options[OPTION_NITER] = convert(INT, alg.niter)
    options[OPTION_SEED] = convert(INT, alg.seed)
    options[OPTION_COMPRESS] = convert(INT, alg.compress)
    options[OPTION_CCORDER] = convert(INT, alg.ccorder)
    options[OPTION_PFACTOR] = convert(INT, alg.pfactor)
    options[OPTION_UFACTOR] = convert(INT, alg.ufactor)
    return
end

end
