module KaHyParExt

using Base: oneto
using Base.Order
using CliqueTrees
using CliqueTrees: EliminationAlgorithm, Parent, UnionFind, bestfill_impl!, bestwidth_impl!, compositerotations_impl!, connect!, hcompresspart, hseparator!, implicitarcs, implicitsplit!, popelements!, prepare!, quotientsplit!, sympermute!_impl!, twinfreelabel, nov, simplegraph, qcc, compresstwins
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

# Nested dissection on one global quotient graph (see dissection_algorithms.jl). A node
# of the separator tree stores its clique cover (the hypergraph that KaHyPar
# partitions, in which the separator S of each ancestor is a single clique),
# its vertex set and its twin classes. A node is split without realizing its
# graph G'[W] (see `implicitsplit!`); the graph is realized, compressed, only at
# leaves and at postorders, and at most one realized graph is alive at a time.
function dissectsimple(weights::AbstractVector{WINT2}, hgraph::BipartiteGraph{VINT2, EINT}, graph::BipartiteGraph{PINT, PINT}, label::BipartiteGraph{PINT, PINT}, alg::ND{S}) where {S}
    h = nov(hgraph); n = nv(graph); m = ne(graph); nn = n + one(PINT)
    maxlevel = convert(PINT, alg.level)
    minwidth = convert(WINT2, alg.width)
    imbalance = convert(PINT, alg.imbalance)

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

    @inbounds for v in oneto(n)
        pinhead[v] = zero(PINT); stamp[v] = 0; gmark[v] = 0
    end

    work00 = FScalar{WINT2}(undef)
    work01 = Vector{PINT}(undef, half(m))
    work02 = Vector{PINT}(undef, half(m))
    work03 = FVector{PINT}(undef, max(h, n))
    work04 = FVector{PINT}(undef, n)
    work05 = FVector{PINT}(undef, max(h, n))
    work06 = FVector{PINT}(undef, max(h, n))
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
    hwght = FVector{WINT1}(undef, h)

    @inbounds for v in oneto(h)
        hwght[v] = one(WINT1)
    end

    parts = FVector{PINT}[]
    orders = FVector{PINT}[]

    HGraph = BipartiteGraph{VINT2, EINT, FVector{EINT}, FVector{VINT2}}

    nodes = Tuple{
        HGraph,        # clique cover (empty at postorder)
        FVector{PINT}, # vertex set
        FVector{PINT}, # classes
        PINT,          # number of classes
        PINT,          # level (negative: postorder)
    }[]

    nohgraph = BipartiteGraph{VINT2, EINT}(zero(VINT2), zero(VINT2), zero(EINT))
    vertexset = FVector{PINT}(undef, n)
    cls = FVector{PINT}(undef, n)

    @inbounds for v in oneto(n)
        vertexset[v] = cls[v] = v
    end

    push!(nodes, (hgraph, vertexset, cls, n, zero(PINT)))

    @inbounds while !isempty(nodes)
        hgraph, vertexset, cls, nc, level = pop!(nodes)
        unprocessed = !isnegative(level)
        curlevel = unprocessed ? level : -level - one(PINT)

        popelements!(nelm, elmptr, pinvtx, pinnext, pinhead, curlevel)

        cmpweights, cmplabel = prepare!(tag, stamp, vclass, marker, mask, clsptr,
            clstgt, lblptr, lbltgt, nelm, elmptr, pinlvl, pinnext, pinhead,
            weights, vertexset, cls, nc, curlevel, graph)

        # a node is split without realizing its graph
        if implicit && unprocessed
            n = nc

            m = implicitarcs(gtag, gmark, gbuf, tag, stamp, vclass, mask, lblptr,
                lbltgt, graph, nc)
        else
            cmpgraph, clique = connect!(tag, stamp, vclass, marker, clsptr, clstgt,
                lblptr, lbltgt, pointer, target, elmptr, pinvtx, pinlvl, pinnext,
                pinhead, nc, curlevel, graph)

            n = nv(cmpgraph); m = ne(cmpgraph)
        end

        iscomplete = m == n * (n - one(PINT))

        # a leaf is processed as soon as it is reached
        isleaf = unprocessed

        if unprocessed && !(n <= minwidth || level >= maxlevel || iscomplete) # branch
            part = FVector{PINT}(undef, n)
            separator!(work00, work03, hwght, cmpweights, hgraph, imbalance, alg.dis)
            h0, h1 = hseparator!(work05, work06, work03, part, hgraph)

            if implicit
                child0, child1, order2 = implicitsplit!(work04, work09, work10, work11,
                    work12, work13, work14, work15, work16, gtag, gmark, gbuf, tag, stamp,
                    vclass, mask, clsptr, clstgt, lblptr, lbltgt, nelm, elmptr, pinvtx,
                    pinlvl, pinnext, pinhead, vertexset, cls, part, nc, level, graph)
            else
                child0, child1, order2 = quotientsplit!(work04, work09, work10, work11,
                    work12, work13, work14, work15, work16, nelm, elmptr, pinvtx, pinlvl,
                    pinnext, pinhead, vertexset, cls, part, level, cmpgraph)
            end

            label0, clique0 = twinfreelabel(work15, part, zero(PINT), child0[3], n)
            label1, clique1 = twinfreelabel(work16, part, one(PINT), child1[3], n)

            htag = one(PINT)
            hgraph0, htag = hcompresspart(h0, htag, hgraph, work05, work03, label0, clique0)
            hgraph1, htag = hcompresspart(h1, htag, hgraph, work06, work03, label1, clique1)

            push!(
                nodes,
                (nohgraph, vertexset, cls, nc, -level - one(PINT)),
                (hgraph0, child0..., level + one(PINT)),
                (hgraph1, child1..., level + one(PINT)),
            )

            push!(parts, part)
            push!(orders, order2)
            continue
        end

        if implicit && unprocessed # a leaf needs its graph after all
            cmpgraph, clique = connect!(tag, stamp, vclass, marker, clsptr, clstgt,
                lblptr, lbltgt, pointer, target, elmptr, pinvtx, pinlvl, pinnext,
                pinhead, nc, curlevel, graph)
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
            else      # branch
                part = pop!(parts)
                ndsorder = Vector{PINT}(undef, n)
                ndsindex = Vector{PINT}(undef, n)
                i = zero(PINT)

                # the children's orders list vertices of the
                # quotient graph: read each class at its first vertex
                seen = marker

                for c in oneto(n)
                    seen[c] = zero(PINT)
                end

                for pass in oneto(two(PINT))
                    for v in pop!(orders)
                        c = vclass[v]

                        if !istwo(part[c]) && seen[c] != pass
                            seen[c] = pass
                            ndsindex[c] = i += one(PINT)
                            ndsorder[i] = c
                        end
                    end
                end

                for c in pop!(orders)
                    ndsindex[c] = i += one(PINT)
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

        j = zero(PINT); outorder = FVector{PINT}(undef, ne(cmplabel))

        for i in oneto(n)
            for v in neighbors(cmplabel, work03[i])
                j += one(PINT); outorder[j] = v
            end
        end

        push!(orders, outorder)
    end

    # expand the vertices of the twin-free graph
    j = zero(PINT); result = FVector{PINT}(undef, ne(label))

    @inbounds for v in only(orders), w in neighbors(label, v)
        j += one(PINT); result[j] = w
    end

    return result
end

end
