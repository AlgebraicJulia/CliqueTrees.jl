module MetisExt

using Base: oneto
using Base.Order
using CliqueTrees
using CliqueTrees: EliminationAlgorithm, Parent, UnionFind, simplegraph, compresstwins, popelements!, realize!, quotientsplit!, sympermute!_impl!, compositerotations_impl!, bestfill_impl!, bestwidth_impl!
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

    # merge twins up front: every graph in the dissection is then twin-free
    cmpgraph, cmpweights, project = compresstwins(weights, simple)

    order = convert(Vector{V}, dissectsimple(cmpweights, cmpgraph, project, alg))
    return order
end

# Nested dissection on one global quotient graph (see dissection_algorithms.jl):
# a node of the separator tree stores only its vertex set and twin classes, and
# its graph G'[W] is realized, compressed, when the node is split and again at
# its postorder. At most one realized graph is alive at a time.
function dissectsimple(weights::AbstractVector{INT}, graph::BipartiteGraph{INT, INT}, label::BipartiteGraph{INT, INT}, alg::ND{S}) where {S}
    n = nv(graph); m = ne(graph); nn = n + one(INT)
    maxlevel = convert(INT, alg.level)
    minwidth = convert(INT, alg.width)
    imbalance = convert(INT, alg.imbalance)

    # the separators of the ancestors of the current node (a stack)
    nelm = FScalar{INT}(undef); nelm[] = zero(INT)
    elmptr = FVector{INT}(undef, maxlevel + two(INT)); elmptr[begin] = one(INT)
    pinvtx = Vector{INT}(undef, n)
    pinlvl = Vector{INT}(undef, n)
    pinnext = Vector{INT}(undef, n)
    pinhead = FVector{INT}(undef, n)

    # the realization of the current node
    tag = FScalar{Int}(undef); tag[] = 0
    stamp = FVector{Int}(undef, n)
    vclass = FVector{INT}(undef, n)
    marker = FVector{INT}(undef, max(n, maxlevel + two(INT)))
    mask = FVector{UInt64}(undef, n)
    clsptr = FVector{INT}(undef, maxlevel + two(INT))
    clstgt = Vector{INT}(undef, n)
    lblptr = FVector{INT}(undef, nn)
    lbltgt = FVector{INT}(undef, n)
    pointer = FVector{INT}(undef, nn)
    target = Vector{INT}(undef, max(m, one(INT)))

    @inbounds for v in oneto(n)
        pinhead[v] = zero(INT); stamp[v] = 0
    end

    work00 = FScalar{INT}(undef)
    work01 = Vector{INT}(undef, half(m))
    work02 = Vector{INT}(undef, half(m))
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
        FVector{INT}, # vertex set
        FVector{INT}, # classes
        INT,          # number of classes
        INT,          # level (negative: postorder)
    }[]

    vertexset = FVector{INT}(undef, n)
    cls = FVector{INT}(undef, n)

    @inbounds for v in oneto(n)
        vertexset[v] = cls[v] = v
    end

    push!(nodes, (vertexset, cls, n, zero(INT)))

    @inbounds while !isempty(nodes)
        vertexset, cls, nc, level = pop!(nodes)
        unprocessed = !isnegative(level)
        curlevel = unprocessed ? level : -level - one(INT)

        popelements!(nelm, elmptr, pinvtx, pinnext, pinhead, curlevel)

        cmpgraph, cmpweights, cmplabel, clique = realize!(tag, stamp, vclass, marker,
            mask, clsptr, clstgt, lblptr, lbltgt, pointer, target, nelm, elmptr,
            pinvtx, pinlvl, pinnext, pinhead, weights, vertexset, cls, nc,
            curlevel, graph)

        n = nv(cmpgraph); m = ne(cmpgraph); k = convert(INT, length(clique))
        iscomplete = m == n * (n - one(INT))

        if half(m) > length(work01)
            resize!(work01, half(m))
            resize!(work02, half(m))
        end

        # a leaf is processed as soon as it is reached
        isleaf = unprocessed

        if unprocessed && !(n <= minwidth || level >= maxlevel || iscomplete) # branch
            part = FVector{INT}(undef, n)
            separator!(work03, work00, part, cmpweights, cmpgraph, imbalance, alg.dis)

            child0, child1, order2 = quotientsplit!(work04, work05, work06, work07, work08,
                work11, work12, work13, work14, nelm, elmptr, pinvtx, pinlvl, pinnext,
                pinhead, vertexset, cls, part, level, cmpgraph)

            push!(
                nodes,
                (vertexset, cls, nc, -level - one(INT)),
                (child0..., level + one(INT)),
                (child1..., level + one(INT)),
            )

            push!(parts, part)
            push!(orders, order2)
            continue
        end

        if iscomplete # complete graph
            for v in oneto(n)
                work03[v] = v
            end
        else
            tree = Parent(n, work06)
            upper = BipartiteGraph(n, n, half(m), work09, work01)
            lower = BipartiteGraph(n, n, half(m), work10, work02)

            if isleaf # leaf
                order, index = permutation(cmpweights, cmpgraph, alg.alg)
            else      # branch
                part = pop!(parts)
                ndsorder = Vector{INT}(undef, n)
                ndsindex = Vector{INT}(undef, n)
                i = zero(INT)

                # the children's orders list vertices of the
                # quotient graph: read each class at its first vertex
                seen = marker

                for c in oneto(n)
                    seen[c] = zero(INT)
                end

                for pass in oneto(two(INT))
                    for v in pop!(orders)
                        c = vclass[v]

                        if !istwo(part[c]) && seen[c] != pass
                            seen[c] = pass
                            ndsindex[c] = i += one(INT)
                            ndsorder[i] = c
                        end
                    end
                end

                for c in pop!(orders)
                    ndsindex[c] = i += one(INT)
                    ndsorder[i] = c
                end

                if isone(S) || istwo(S)
                    sets = UnionFind(n, work03, work04, work05)
                    grdorder, grdindex = permutation(cmpweights, cmpgraph, alg.alg)

                    if isone(S)
                        best = bestwidth_impl!(lower, upper, tree, sets, work07,
                            work08, work11, work12, work13, work14, work15,
                            work16, cmpweights, cmpgraph, (ndsindex, grdindex))
                    else
                        best = bestfill_impl!(lower, upper, tree, sets, work07,
                            work08, work11, work12, work13, work14, work15,
                            work16, cmpweights, cmpgraph, (ndsindex, grdindex))
                    end

                    order = (ndsorder, grdorder)[best]
                    index = (ndsindex, grdindex)[best]
                else
                    order, index = ndsorder, ndsindex
                end
            end

            sympermute!_impl!(upper, cmpgraph, index, Forward)

            for i in oneto(k)
                clique[i] = index[clique[i]]
            end

            compositerotations_impl!(index, work03, work04,
                work05, lower, tree, upper, clique)

            for v in oneto(n)
                i = index[v]; work03[i] = order[v]
            end
        end

        j = zero(INT); outorder = FVector{INT}(undef, ne(cmplabel))

        for i in oneto(n)
            for v in neighbors(cmplabel, work03[i])
                j += one(INT); outorder[j] = v
            end
        end

        push!(orders, outorder)
    end

    # expand the vertices of the twin-free graph
    j = zero(INT); result = FVector{INT}(undef, ne(label))

    @inbounds for v in only(orders), w in neighbors(label, v)
        j += one(INT); result[j] = w
    end

    return result
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
