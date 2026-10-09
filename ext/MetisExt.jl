module MetisExt

using Base: oneto
using Base.Order
using CliqueTrees
using CliqueTrees: EliminationAlgorithm, Parent, UnionFind, simplegraph, compresstwins, popelements!, realize!, segmentsplit!, classify!, applymerges!, rollback!, mergesorted!, writeresidual!, sympermute!_impl!, compositerotations_impl!, bestfill_impl!, bestwidth_impl!
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

# Nested dissection on one global quotient graph, without a node stack (see
# dissection_algorithms.jl). The state is the twin-free graph, the separators
# of the ancestors of the current node, one array of working orderings, a
# union-find holding the twin classes of the path, and a few scalars per
# level. The graph G'[W] of a node is realized, compressed, when the node is
# split and again when its children have returned; at most one realized graph
# is alive at a time.
function dissectsimple(weights::AbstractVector{INT}, graph::BipartiteGraph{INT, INT}, label::BipartiteGraph{INT, INT}, alg::ND{S}) where {S}
    n = nv(graph); m = ne(graph); nn = n + one(INT); ng = n
    maxlevel = convert(INT, alg.level)
    minwidth = convert(INT, alg.width)
    imbalance = convert(INT, alg.imbalance)
    nl = maxlevel + one(INT)

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

    # the working orderings and the classes of the path (see `segmentsplit!`)
    segment = FVector{INT}(undef, n)
    vertexset = FVector{INT}(undef, n)
    cls = FVector{INT}(undef, n)
    ufp = FVector{INT}(undef, n)
    rmark = FVector{Int}(undef, n)
    rid = FVector{INT}(undef, n)
    mrg = Vector{INT}(undef, n)
    mlog = Vector{INT}(undef, n)
    base = FVector{INT}(undef, nl)
    na = FVector{INT}(undef, nl)
    nb = FVector{INT}(undef, nl)
    side = FVector{INT}(undef, nl)
    logstart = FVector{INT}(undef, nl)
    mrgbase = FVector{INT}(undef, nl)
    mrgstart = FVector{INT}(undef, nl)
    mrgstop = FVector{INT}(undef, nl)

    @inbounds for v in oneto(n)
        pinhead[v] = zero(INT); stamp[v] = 0; rmark[v] = 0
        ufp[v] = vertexset[v] = v
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

    level = zero(INT); nw = n; t = 0; nmrg = zero(INT); nlog = zero(INT)
    @inbounds base[begin] = one(INT)
    descend = true

    @inbounds while true
        l = level + one(INT)

        if descend
            # the node at `level`, with vertex set vertexset[1:nw]
            t += 1; nc = classify!(cls, rmark, rid, ufp, vertexset, nw, t)
            set = view(vertexset, oneto(nw)); setcls = view(cls, oneto(nw))
        else
            # the node at `level` has returned; its residual starts at base[l]
            iszero(level) && break
            level -= one(INT); l = level + one(INT)
            p = base[l]; sstart = elmptr[l]; sstop = elmptr[l + one(INT)] - one(INT)

            if isone(side[l]) # child 1 has returned: start child 0 (A ∪ S)
                nlog = rollback!(ufp, mlog, nlog, logstart[l])
                nlog = applymerges!(ufp, mlog, nlog, mrg, mrgstart[l], mrgstop[l])
                nw = mergesorted!(vertexset, segment, p, p + na[l] - one(INT), pinvtx, sstart, sstop)

                # move the residual of child 1 down, to base[l]
                copyto!(segment, p, segment, p + na[l], nb[l])

                side[l] = zero(INT); level += one(INT)
                base[level + one(INT)] = p + nb[l]
                descend = true
                continue
            end

            # child 0 has returned: join the node at `level`
            nlog = rollback!(ufp, mlog, nlog, logstart[l]); nmrg = mrgbase[l]
            nw = zero(INT)

            for q in p:(p + na[l] + nb[l] - one(INT))
                nw += one(INT); vertexset[nw] = segment[q]
            end

            for q in sstart:sstop
                nw += one(INT); vertexset[nw] = pinvtx[q]
            end

            sort!(view(vertexset, oneto(nw)))
            popelements!(nelm, elmptr, pinvtx, pinnext, pinhead, level)
            t += 1; nc = classify!(cls, rmark, rid, ufp, vertexset, nw, t)
            set = view(vertexset, oneto(nw)); setcls = view(cls, oneto(nw))
        end

        cmpgraph, cmpweights, cmplabel, clique = realize!(tag, stamp, vclass, marker,
            mask, clsptr, clstgt, lblptr, lbltgt, pointer, target, nelm, elmptr,
            pinvtx, pinlvl, pinnext, pinhead, weights, set, setcls, nc, level,
            graph)

        n = nv(cmpgraph); m = ne(cmpgraph); k = convert(INT, length(clique))
        iscomplete = m == n * (n - one(INT))

        if half(m) > length(work01)
            resize!(work01, half(m))
            resize!(work02, half(m))
        end

        # a leaf is processed as soon as it is reached
        isleaf = descend

        if descend && !(n <= minwidth || level >= maxlevel || iscomplete) # branch
            part = FVector{INT}(undef, n)
            separator!(work03, work00, part, cmpweights, cmpgraph, imbalance, alg.dis)

            p = base[l]; mrgbase[l] = nmrg

            nap, nbp, stop1, stop0 = segmentsplit!(work04, work05, work06, work07,
                work08, work11, work12, work13, work14, marker, segment, p, mrg, nmrg,
                ufp, lblptr, lbltgt, nelm, elmptr, pinvtx, pinlvl, pinnext, pinhead,
                set, setcls, part, nc, level, cmpgraph)

            na[l] = nap; nb[l] = nbp; side[l] = one(INT)
            mrgstart[l] = stop1 + one(INT); mrgstop[l] = nmrg = stop0

            # start child 1 (B ∪ S)
            logstart[l] = nlog + one(INT)
            nlog = applymerges!(ufp, mlog, nlog, mrg, mrgbase[l] + one(INT), stop1)
            sstart = elmptr[l]; sstop = elmptr[l + one(INT)] - one(INT)
            nw = mergesorted!(vertexset, segment, p + nap, p + nap + nbp - one(INT), pinvtx, sstart, sstop)

            level += one(INT)
            base[level + one(INT)] = p + nap
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
            else      # branch: child 0, then child 1, then the separator
                p = base[l]
                ndsorder = Vector{INT}(undef, n)
                ndsindex = Vector{INT}(undef, n)
                i = zero(INT)

                # the residuals list vertices of the quotient
                # graph: read each class at its first vertex
                seen = marker

                for c in oneto(n)
                    seen[c] = zero(INT)
                end

                for (qstart, qstop) in ((p + nb[l], p + nb[l] + na[l] - one(INT)), (p, p + nb[l] - one(INT)))
                    for q in qstart:qstop
                        c = vclass[segment[q]]

                        if iszero(seen[c])
                            seen[c] = one(INT)
                            ndsindex[c] = i += one(INT)
                            ndsorder[i] = c
                        end
                    end
                end

                # the classes of the separator, in increasing order
                for c in oneto(n)
                    if iszero(seen[c])
                        ndsindex[c] = i += one(INT)
                        ndsorder[i] = c
                    end
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

        # the residual: the ordering, minus the separator of the parent
        t += 1

        if ispositive(level)
            for q in elmptr[level]:(elmptr[l] - one(INT))
                rmark[pinvtx[q]] = t
            end
        end

        writeresidual!(segment, base[l], work03, n, cmplabel, rmark, t)
        descend = false
    end

    # expand the vertices of the twin-free graph
    j = zero(INT); result = FVector{INT}(undef, ne(label))

    @inbounds for i in oneto(ng), w in neighbors(label, segment[i])
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
