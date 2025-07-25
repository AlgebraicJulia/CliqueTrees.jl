module MetisExt

using ArgCheck
using Base: oneto
using Base.Order
using CliqueTrees
using CliqueTrees: EliminationAlgorithm, Parent, UnionFind, simplegraph, partition!, sympermute!_impl!, compositerotations_impl!, bestfill_impl!, bestwidth_impl!
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

function CliqueTrees.permutation(graph, alg::METIS)
    return permutation(BipartiteGraph(graph), alg)
end

function CliqueTrees.permutation(graph::AbstractGraph{V}, alg::METIS) where {V}
    simple = simplegraph(INT, INT, graph)
    order::Vector{V}, index::Vector{V} = metis(simple, alg)
    return order, index
end

function CliqueTrees.permutation(weights::AbstractVector, graph, alg::METIS)
    return permutation(weights, BipartiteGraph(graph), alg)
end

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::METIS) where {V}
    simple = simplegraph(INT, INT, graph)
    order::Vector{V}, index::Vector{V} = metis(weights, simple, alg)
    return order, index
end

function CliqueTrees.permutation(graph, alg::ND{<:Any, <:EliminationAlgorithm, METISND})
    order = dissect(graph, alg)
    return order, invperm(order)
end

function CliqueTrees.permutation(weights::AbstractVector, graph, alg::ND{<:Any, <:EliminationAlgorithm, METISND})
    order = dissect(weights, graph, alg)
    return order, invperm(order)
end

function metis(weights::AbstractVector, graph::BipartiteGraph{INT, INT}, alg::METIS)
    @argcheck nv(graph) <= length(weights)
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

function metis(graph::BipartiteGraph{INT, INT}, alg::METIS)
    n = nv(graph)

    # construct options
    options = Vector{INT}(undef, NOPTIONS)
    setoptions!(options, alg)

    # construct METIS graph
    xadj = pointers(graph)
    adjncy = targets(graph)

    # construct permutation
    order = Vector{INT}(undef, n)
    index = Vector{INT}(undef, n)

    Metis.@check Metis.METIS_NodeND(
        Ref{INT}(n),
        xadj,
        adjncy,
        C_NULL,
        options,
        order,
        index,
    )

    return order, index
end

function separator!(options::AbstractVector{INT}, sepsize::AbstractScalar{INT}, part::AbstractVector{INT}, weights::AbstractVector{INT}, graph::BipartiteGraph{INT, INT}, imbalance::INT, alg::METISND)
    @argcheck NOPTIONS <= length(options)
    @argcheck nv(graph) <= length(part)
    @argcheck nv(graph) <= length(weights)
    @argcheck ispositive(imbalance)
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

function dissect(graph, alg::ND)
    return dissect(BipartiteGraph(graph), alg)
end

function dissect(graph::AbstractGraph, alg::ND)
    weights = ones(INT, nv(graph))
    return dissect(weights, graph, alg)
end

function dissect(weights::AbstractVector, graph, alg::ND)
    return dissect(weights, BipartiteGraph(graph), alg)
end

function dissect(weights::AbstractVector, graph::AbstractGraph, alg::ND)
    n = nv(graph); new = Vector{INT}(undef, n)

    @inbounds for v in oneto(n)
        new[v] = trunc(INT, weights[v])
    end

    return dissect(new, graph, alg)
end

function dissect(weights::Vector{INT}, graph::AbstractGraph{V}, alg::ND) where {V <: Integer}
    simple = simplegraph(INT, INT, graph)
    order::Vector{V} = dissectsimple(weights, simple, alg)
    return order
end

function dissectsimple(weights::AbstractVector{INT}, graph::BipartiteGraph{INT, INT}, alg::ND{S}) where {S}
    n = nv(graph); m = ne(graph); nn = n + one(INT); width = zero(INT)
    maxlevel = convert(INT, alg.level)
    minwidth = convert(INT, alg.width)
    imbalance = convert(INT, alg.imbalance)

    @inbounds for v in oneto(n)
        width += weights[v]
    end

    swork = Scalar{INT}(undef)
    vwork1 = Vector{INT}(undef, m)
    vwork2 = Vector{INT}(undef, m)
    vwork3 = Vector{INT}(undef, max(n, NOPTIONS))
    vwork4 = Vector{INT}(undef, n)
    vwork5 = Vector{INT}(undef, n)
    vwork6 = Vector{INT}(undef, n)
    vwork9 = Vector{INT}(undef, n)
    vwork10 = Vector{INT}(undef, n)
    vwork11 = Vector{INT}(undef, nn)
    vwork12 = Vector{INT}(undef, nn)
    vwork13 = Vector{INT}(undef, n)
    vwork14 = Vector{INT}(undef, n)
    vwork15 = Vector{INT}(undef, n)
    vwork16 = Vector{INT}(undef, n)
    vwork17 = Vector{INT}(undef, n)
    vwork18 = Vector{INT}(undef, n)

    orders = Vector{INT}[]

    nodes = Tuple{
        BipartiteGraph{INT, INT, Vector{INT}, Vector{INT}}, # graph
        Vector{INT},                                        # weights
        Vector{INT},                                        # label
        Vector{INT},                                        # clique
        INT,                                                # width
        INT,                                                # level
    }[]

    push!(nodes, (graph, weights, collect(oneto(n)), INT[], width, zero(INT)))

    @inbounds while !isempty(nodes)
        graph, weights, label, clique, width, level = pop!(nodes)
        n = nv(graph); m = ne(graph); k = convert(INT, length(clique))

        if half(m) > length(vwork1)
            resize!(vwork1, half(m))
            resize!(vwork2, half(m))
        end

        if !isnegative(level) # unprocessed
            isleaf = width <= minwidth || level >= maxlevel

            if isleaf # leaf
                push!(nodes, (graph, weights, label, clique, width, -one(INT)))
            else      # branch
                separator!(vwork3, swork, vwork4, weights, graph, imbalance, alg.dis)

                child0, child1, order2 = partition!(
                    vwork4,
                    vwork5,
                    vwork6,
                    weights,
                    graph,
                )

                push!(
                    nodes,
                    (graph, weights, label, clique, width, -two(INT)),
                    (child0..., level + one(INT)),
                    (child1..., level + one(INT)),
                )

                push!(orders, order2)
            end
        else                  # processed
            isleaf = isone(-level)
            tree = Parent(n, vwork6)
            upper = BipartiteGraph(n, n, half(m), vwork11, vwork1)
            lower = BipartiteGraph(n, n, half(m), vwork12, vwork2)

            if isleaf # leaf
                order, index = permutation(weights, graph, alg.alg)
            else      # branch
                order0 = pop!(orders)
                order1 = pop!(orders)
                order2 = pop!(orders)

                order = [
                    order0
                    order1
                    order2
                ]

                index = invperm(order)

                if isone(S) || istwo(S)
                    sets = UnionFind(vwork3, vwork4, vwork5)
                    greedyorder, greedyindex = permutation(weights, graph, alg.alg)

                    if isone(S)
                        best = bestwidth_impl!(lower, upper, tree, sets, vwork9,
                            vwork10, vwork13, vwork14, vwork15, vwork16, vwork17,
                            vwork18, weights, graph, (index, greedyindex))
                    else
                        best = bestfill_impl!(lower, upper, tree, sets, vwork9,
                            vwork10, vwork13, vwork14, vwork15, vwork16, vwork17,
                            vwork18, weights, graph, (index, greedyindex))
                    end

                    if istwo(best)
                        order, index = greedyorder, greedyindex
                    end
                end
            end

            sympermute!_impl!(vwork3, upper, graph, index, Forward)

            for i in oneto(k)
                clique[i] = index[clique[i]]
            end

            compositerotations_impl!(index, vwork3, vwork4,
                vwork5, lower, tree, upper, clique)

            copyto!(vwork3, order)
            resize!(order, n - k)

            for v in oneto(n)
                i = index[v]

                if i <= n - k
                    order[i] = label[vwork3[v]]
                end
            end

            push!(orders, order)
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
