module KaHyParExt

using Base: oneto
using Base.Order
using CliqueTrees
using CliqueTrees: EliminationAlgorithm, Parent, UnionFind, bestfill_impl!, bestwidth_impl!, compositerotations_impl!, connect!, hseparator!, hrealize!, hrecord!, implicitarcs, implicitsegmentsplit!, segmentsplit!, popelements!, prepare!, classify!, applymerges!, rollback!, mergesorted!, writeresidual!, sympermute!_impl!, nov, simplegraph, qcc, compresstwins
using CliqueTrees.Utilities
using Graphs

import KaHyPar

const WINT1 = KaHyPar.kahypar_hypernode_weight_t
const WINT2 = KaHyPar.kahypar_hyperedge_weight_t
const VINT1 = KaHyPar.kahypar_hypernode_id_t
const VINT2 = KaHyPar.kahypar_hyperedge_id_t
const EINT = KaHyPar.Csize_t
const PINT = KaHyPar.kahypar_partition_id_t

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph, alg::ND{<:Any, <:EliminationAlgorithm, <:KaHyParND})
    order = dissect(weights, graph, alg)
    return order, invperm(order)
end

function separator!(sepsize::AbstractScalar{WINT2}, part::AbstractVector{PINT}, vwght::AbstractVector{WINT1}, ewght::AbstractVector{WINT2}, graph::BipartiteGraph{VINT2, EINT}, imbalance::PINT, alg::KaHyParND)
    @assert nov(graph) <= length(part)
    @assert nov(graph) <= length(vwght)
    @assert nv(graph) <= length(ewght)

    m = ne(graph); n = nv(graph); nn = n + one(VINT2)

    @inbounds for v in oneto(nn)
        pointers(graph)[v] -= one(EINT)
    end

    @inbounds for p in oneto(m)
        targets(graph)[p] -= one(VINT2)
    end

    imbalance -= convert(PINT, 100)

    context = KaHyPar.kahypar_context_new()
    KaHyPar.kahypar_configure_context_from_file(context, joinpath(@__DIR__, "config/cut_kKaHyPar_sea20.ini"))

    KaHyPar.kahypar_partition(
        convert(VINT1, nov(graph)),
        nv(graph),
        convert(Cdouble, imbalance) / convert(Cdouble, 1000),
        two(PINT),
        vwght,
        ewght,
        pointers(graph),
        targets(graph),
        sepsize,
        context,
        part,
    )

    @inbounds for v in oneto(nn)
        pointers(graph)[v] += one(EINT)
    end

    @inbounds for p in oneto(m)
        targets(graph)[p] += one(VINT2)
    end

    return
end

function dissect(weights::AbstractVector{W}, graph::AbstractGraph, alg::ND) where {W}
    n = nv(graph)
    scale = convert(W, alg.scale)
    intweights = FVector{PINT}(undef, n)

    @inbounds for v in oneto(n)
        w = weights[v]

        if W <: AbstractFloat
            w *= scale
        end

        intweights[v] = trunc(PINT, w)
    end

    return dissect(intweights, graph, alg)
end

function dissect(weights::FVector{WINT2}, graph::AbstractGraph{V}, alg::ND) where {V}
    simple = simplegraph(PINT, PINT, graph)

    # merge twins up front: every graph in the dissection is then twin-free
    cmpgraph, cmpweights, project = compresstwins(weights, simple)

    cover = qcc(VINT2, EINT, cmpgraph, alg.dis.beta, alg.dis.order)
    order = convert(Vector{V}, dissectsimple(cmpweights, reverse(cover), cmpgraph, project, alg))
    return order
end

# Nested dissection on one global quotient graph, without a node stack, with
# a global clique cover (see dissection_algorithms.jl). The state is the
# twin-free graph and a cover of its edges by cliques, the separators of the
# ancestors of the current node, one array of working orderings, a
# union-find holding the twin classes of the path, a record of the last
# split of each clique, and a few scalars per level. The hypergraph that
# KaHyPar partitions is built from the cover when a node is split, and the
# node is split without realizing its graph G'[W]; the graph is realized,
# compressed, only at leaves and when the children of a node have returned.
function dissectsimple(weights::AbstractVector{WINT2}, cover::BipartiteGraph{VINT2, EINT}, graph::BipartiteGraph{PINT, PINT}, label::BipartiteGraph{PINT, PINT}, alg::ND{S}) where {S}
    h = nov(cover); n = nv(graph); m = ne(graph); nn = n + one(PINT); ng = n
    maxlevel = convert(PINT, alg.level)
    minwidth = convert(WINT2, alg.width)
    imbalance = convert(PINT, alg.imbalance)
    nl = maxlevel + one(PINT); hmax = h + maxlevel + one(PINT)

    # the separators of the ancestors of the current node (a stack)
    nelm = FScalar{PINT}(undef); nelm[] = zero(PINT)
    elmptr = FVector{PINT}(undef, maxlevel + two(PINT)); elmptr[begin] = one(PINT)
    pinvtx = Vector{PINT}(undef, n)
    pinlvl = Vector{PINT}(undef, n)
    pinnext = Vector{PINT}(undef, n)
    pinhead = FVector{PINT}(undef, n)

    # the realization of the current node
    tag = FScalar{Int}(undef); tag[] = 0
    stamp = FVector{Int}(undef, n)
    vclass = FVector{PINT}(undef, n)
    marker = FVector{PINT}(undef, max(n, maxlevel + two(PINT)))
    mask = FVector{UInt64}(undef, n)
    clsptr = FVector{PINT}(undef, maxlevel + two(PINT))
    clstgt = Vector{PINT}(undef, n)
    lblptr = FVector{PINT}(undef, nn)
    lbltgt = FVector{PINT}(undef, n)
    pointer = FVector{PINT}(undef, nn)
    target = Vector{PINT}(undef, max(m, one(PINT)))

    # the implicit split (it needs one bit per level)
    implicit = maxlevel < 64
    gtag = FScalar{Int}(undef); gtag[] = 0
    gmark = FVector{Int}(undef, n)
    gbuf = FVector{PINT}(undef, n)

    # the working orderings and the classes of the path (see `segmentsplit!`)
    segment = FVector{PINT}(undef, n)
    vertexset = FVector{PINT}(undef, n)
    cls = FVector{PINT}(undef, n)
    ufp = FVector{PINT}(undef, n)
    rmark = FVector{Int}(undef, n)
    rid = FVector{PINT}(undef, n)
    mrg = Vector{PINT}(undef, n)
    mlog = Vector{PINT}(undef, n)
    base = FVector{PINT}(undef, nl)
    na = FVector{PINT}(undef, nl)
    nb = FVector{PINT}(undef, nl)
    side = FVector{PINT}(undef, nl)
    logstart = FVector{PINT}(undef, nl)
    mrgbase = FVector{PINT}(undef, nl)
    mrgstart = FVector{PINT}(undef, nl)
    mrgstop = FVector{PINT}(undef, nl)

    # the clique cover (see `hrealize!`)
    nodeepoch = FVector{Int}(undef, nl)
    clev = FVector{PINT}(undef, h)
    cside = FVector{PINT}(undef, h)
    cepoch = FVector{Int}(undef, h)
    slev = FVector{PINT}(undef, nl)
    sside = FVector{PINT}(undef, nl)
    sepoch = FVector{Int}(undef, nl)
    cmark = FVector{Int}(undef, h); ctag = FScalar{Int}(undef); ctag[] = 0
    hmark = FVector{Int}(undef, hmax); htag = FScalar{Int}(undef); htag[] = 0
    hid = FVector{VINT2}(undef, h)
    sid = FVector{VINT2}(undef, nl)
    helm = FVector{PINT}(undef, hmax)
    hpart = FVector{PINT}(undef, hmax)
    hproject0 = FVector{PINT}(undef, hmax)
    hproject1 = FVector{PINT}(undef, hmax)
    hwght = FVector{WINT1}(undef, hmax)
    hptr = FVector{EINT}(undef, nn)
    htgt = Vector{VINT2}(undef, max(ne(cover), one(EINT)) + n)

    @inbounds for v in oneto(n)
        pinhead[v] = zero(PINT); stamp[v] = 0; gmark[v] = 0; rmark[v] = 0
        ufp[v] = vertexset[v] = v
    end

    # every clique of the cover is active at the root
    @inbounds for e in oneto(h)
        clev[e] = zero(PINT); cside[e] = two(PINT); cepoch[e] = 0; cmark[e] = 0
    end

    @inbounds for i in oneto(hmax)
        hmark[i] = 0; hwght[i] = one(WINT1)
    end

    work00 = FScalar{WINT2}(undef)
    work01 = Vector{PINT}(undef, half(m))
    work02 = Vector{PINT}(undef, half(m))
    work03 = FVector{PINT}(undef, n)
    work04 = FVector{PINT}(undef, n)
    work05 = FVector{PINT}(undef, n)
    work06 = FVector{PINT}(undef, n)
    work07 = FVector{WINT2}(undef, n)
    work08 = FVector{WINT2}(undef, n)
    work09 = FVector{PINT}(undef, nn)
    work10 = FVector{PINT}(undef, nn)
    work11 = FVector{PINT}(undef, n)
    work12 = FVector{PINT}(undef, n)
    work13 = FVector{PINT}(undef, n)
    work14 = FVector{PINT}(undef, n)
    work15 = FVector{PINT}(undef, n)
    work16 = FVector{PINT}(undef, n)

    level = zero(PINT); nw = n; t = 0; nmrg = zero(PINT); nlog = zero(PINT); epoch = 0
    @inbounds base[begin] = one(PINT)
    descend = true

    @inbounds while true
        l = level + one(PINT)

        if descend
            # the node at `level`, with vertex set vertexset[1:nw]
            t += 1; nc = classify!(cls, rmark, rid, ufp, vertexset, nw, t)
            set = view(vertexset, oneto(nw)); setcls = view(cls, oneto(nw))
            epoch += 1; nodeepoch[l] = epoch
        else
            # the node at `level` has returned; its residual starts at base[l]
            iszero(level) && break
            level -= one(PINT); l = level + one(PINT)
            p = base[l]; sstart = elmptr[l]; sstop = elmptr[l + one(PINT)] - one(PINT)

            if isone(side[l]) # child 1 has returned: start child 0 (A ∪ S)
                nlog = rollback!(ufp, mlog, nlog, logstart[l])
                nlog = applymerges!(ufp, mlog, nlog, mrg, mrgstart[l], mrgstop[l])
                nw = mergesorted!(vertexset, segment, p, p + na[l] - one(PINT), pinvtx, sstart, sstop)

                # move the residual of child 1 down, to base[l]
                copyto!(segment, p, segment, p + na[l], nb[l])

                # the separator goes to both children: child 1 has
                # overwritten its record
                slev[l] = l; sside[l] = two(PINT); sepoch[l] = nodeepoch[l]

                side[l] = zero(PINT); level += one(PINT)
                base[level + one(PINT)] = p + nb[l]
                descend = true
                continue
            end

            # child 0 has returned: join the node at `level`
            nlog = rollback!(ufp, mlog, nlog, logstart[l]); nmrg = mrgbase[l]
            nw = zero(PINT)

            for q in p:(p + na[l] + nb[l] - one(PINT))
                nw += one(PINT); vertexset[nw] = segment[q]
            end

            for q in sstart:sstop
                nw += one(PINT); vertexset[nw] = pinvtx[q]
            end

            sort!(view(vertexset, oneto(nw)))
            popelements!(nelm, elmptr, pinvtx, pinnext, pinhead, level)
            t += 1; nc = classify!(cls, rmark, rid, ufp, vertexset, nw, t)
            set = view(vertexset, oneto(nw)); setcls = view(cls, oneto(nw))
        end

        cmpweights, cmplabel = prepare!(tag, stamp, vclass, marker, mask, clsptr,
            clstgt, lblptr, lbltgt, nelm, elmptr, pinlvl, pinnext, pinhead,
            weights, set, setcls, nc, level, graph)

        # a node is split without realizing its graph
        if implicit && descend
            n = nc

            m = implicitarcs(gtag, gmark, gbuf, tag, stamp, vclass, mask, lblptr,
                lbltgt, graph, nc)
        else
            cmpgraph, clique = connect!(tag, stamp, vclass, marker, clsptr, clstgt,
                lblptr, lbltgt, pointer, target, elmptr, pinvtx, pinlvl, pinnext,
                pinhead, nc, level, graph)

            n = nv(cmpgraph); m = ne(cmpgraph)
        end

        iscomplete = m == n * (n - one(PINT))

        # a leaf is processed as soon as it is reached
        isleaf = descend

        if descend && !(n <= minwidth || level >= maxlevel || iscomplete) # branch
            # the hypergraph of the node, from the cover
            pepoch = iszero(level) ? 0 : nodeepoch[level]
            pside = iszero(level) ? two(PINT) : side[level]

            hn, np = hrealize!(hptr, htgt, helm, hid, sid, hmark, htag, cmark,
                ctag, clev, cside, cepoch, slev, sside, sepoch, lblptr, lbltgt,
                nelm, pinlvl, pinnext, pinhead, cover, nc, level, pepoch, pside)

            hgraph = BipartiteGraph(convert(VINT2, hn), convert(VINT2, nc), convert(EINT, np), hptr, htgt)
            separator!(work00, hpart, hwght, cmpweights, hgraph, imbalance, alg.dis)

            part = FVector{PINT}(undef, nc)
            hseparator!(hproject0, hproject1, hpart, part, hgraph)
            hrecord!(clev, cside, cepoch, slev, sside, sepoch, helm, hpart, hn, level, nodeepoch[l])

            p = base[l]; mrgbase[l] = nmrg

            if implicit
                nap, nbp, stop1, stop0 = implicitsegmentsplit!(work03, work04, work05,
                    work06, work09, work11, work12, work13, work14, marker, segment, p,
                    mrg, nmrg, ufp, gtag, gmark, gbuf, tag, stamp, vclass, mask, clsptr,
                    clstgt, lblptr, lbltgt, nelm, elmptr, pinvtx, pinlvl, pinnext,
                    pinhead, set, setcls, part, nc, level, graph)
            else
                nap, nbp, stop1, stop0 = segmentsplit!(work03, work04, work05, work06,
                    work09, work11, work12, work13, work14, marker, segment, p, mrg,
                    nmrg, ufp, lblptr, lbltgt, nelm, elmptr, pinvtx, pinlvl, pinnext,
                    pinhead, set, setcls, part, nc, level, cmpgraph)
            end

            # the separator goes to both children
            slev[l] = l; sside[l] = two(PINT); sepoch[l] = nodeepoch[l]

            na[l] = nap; nb[l] = nbp; side[l] = one(PINT)
            mrgstart[l] = stop1 + one(PINT); mrgstop[l] = nmrg = stop0

            # start child 1 (B ∪ S)
            logstart[l] = nlog + one(PINT)
            nlog = applymerges!(ufp, mlog, nlog, mrg, mrgbase[l] + one(PINT), stop1)
            sstart = elmptr[l]; sstop = elmptr[l + one(PINT)] - one(PINT)
            nw = mergesorted!(vertexset, segment, p + nap, p + nap + nbp - one(PINT), pinvtx, sstart, sstop)

            level += one(PINT)
            base[level + one(PINT)] = p + nap
            continue
        end

        if implicit && descend # a leaf needs its graph after all
            cmpgraph, clique = connect!(tag, stamp, vclass, marker, clsptr, clstgt,
                lblptr, lbltgt, pointer, target, elmptr, pinvtx, pinlvl, pinnext,
                pinhead, nc, level, graph)
        end

        k = convert(PINT, length(clique))

        if half(m) > length(work01)
            resize!(work01, half(m))
            resize!(work02, half(m))
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
                ndsorder = Vector{PINT}(undef, n)
                ndsindex = Vector{PINT}(undef, n)
                i = zero(PINT)

                # the residuals list vertices of the quotient
                # graph: read each class at its first vertex
                seen = marker

                for c in oneto(n)
                    seen[c] = zero(PINT)
                end

                for (qstart, qstop) in ((p + nb[l], p + nb[l] + na[l] - one(PINT)), (p, p + nb[l] - one(PINT)))
                    for q in qstart:qstop
                        c = vclass[segment[q]]

                        if iszero(seen[c])
                            seen[c] = one(PINT)
                            ndsindex[c] = i += one(PINT)
                            ndsorder[i] = c
                        end
                    end
                end

                # the classes of the separator, in increasing order
                for c in oneto(n)
                    if iszero(seen[c])
                        ndsindex[c] = i += one(PINT)
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
            for q in elmptr[level]:(elmptr[l] - one(PINT))
                rmark[pinvtx[q]] = t
            end
        end

        writeresidual!(segment, base[l], work03, n, cmplabel, rmark, t)
        descend = false
    end

    # expand the vertices of the twin-free graph
    j = zero(PINT); result = FVector{PINT}(undef, ne(label))

    @inbounds for i in oneto(ng), w in neighbors(label, segment[i])
        j += one(PINT); result[j] = w
    end

    return result
end

end
