module KaHyParExt

using ArgCheck
using Base: oneto
using Base.Order
using CliqueTrees
using CliqueTrees: EliminationAlgorithm, Parent, UnionFind, bestfill_impl!, bestwidth_impl!, compositerotations_impl!, hpartition!, sympermute!_impl!, nov, outvertices, simplegraph, qcc
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
    @argcheck nov(graph) <= length(part)
    @argcheck nov(graph) <= length(vwght)
    @argcheck nv(graph) <= length(ewght)

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
    cover = qcc(VINT2, EINT, simple, alg.dis.beta, alg.dis.order)
    order = convert(Vector{V}, dissectsimple(weights, reverse(cover), simple, alg))
    return order
end

function dissectsimple(weights::AbstractVector{WINT2}, hgraph::BipartiteGraph{VINT2, EINT}, graph::BipartiteGraph{PINT, PINT}, alg::ND{S}) where {S}
    h = nov(hgraph); n = nv(graph); m = ne(graph); nn = n + one(PINT)
    maxlevel = convert(PINT, alg.level)
    minwidth = convert(WINT2, alg.width)
    imbalance = convert(PINT, alg.imbalance)

    work00 = FScalar{WINT2}(undef)
    work01 = Vector{PINT}(undef, m)
    work02 = Vector{PINT}(undef, m)
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

    nodes = Tuple{
        BipartiteGraph{VINT2, EINT, FVector{EINT}, FVector{VINT2}}, # hgraph
        BipartiteGraph{PINT, PINT, FVector{PINT}, FVector{PINT}},   # graph
        FVector{PINT},                                              # weights
        BipartiteGraph{PINT, PINT, FVector{PINT}, FVector{PINT}},   # label
        FVector{PINT},                                              # clique
        PINT,                                                       # level
    }[]

    label = BipartiteGraph{PINT, PINT}(n, n, n)
    clique = FVector{PINT}(undef, zero(PINT))
    level = zero(PINT)

    @inbounds for v in oneto(n)
        pointers(label)[v] = v
        targets(label)[v] = v
    end

    pointers(label)[nn] = nn
    push!(nodes, (hgraph, graph, weights, label, clique, level))

    @inbounds while !isempty(nodes)
        hgraph, graph, weights, label, clique, level = pop!(nodes)
        n = nv(graph); m = ne(graph); l = ne(label); k = convert(PINT, length(clique))

        if !isnegative(level) # unprocessed
            isleaf = n <= minwidth || level >= maxlevel || m == n * (n - one(PINT))

            if isleaf # leaf
                push!(nodes, (hgraph, graph, weights, label, clique, -one(PINT)))
            else      # branch
                if m > length(work01)
                    resize!(work01, half(m))
                    resize!(work02, half(m))
                end

                part = FVector{PINT}(undef, n)
                separator!(work00, work03, hwght, weights, hgraph, imbalance, alg.dis)

                child0, child1, order2 = hpartition!(work00, work14, work04, work07, work08,
                     work11, work12, work13, work09, work10, work01, work02, work15, work16,
                     work05, work06, work03, part, weights, hgraph, graph)

                push!(
                    nodes,
                    (hgraph, graph, weights, label, clique, -two(PINT)),
                    (child0..., level + one(PINT)),
                    (child1..., level + one(PINT)),
                )

                push!(parts, part)
                push!(orders, order2)
            end
        else                  # processed
            isleaf = isone(-level)
            iscomplete = m == n * (n - one(PINT))

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
                    ndsorder = Vector{PINT}(undef, n)
                    ndsindex = Vector{PINT}(undef, n)
                    i = zero(PINT)

                    for v in pop!(orders)
                        if !istwo(part[v])
                            ndsindex[v] = i += one(PINT)
                            ndsorder[i] = v
                        end
                    end

                    for v in pop!(orders)
                        if !istwo(part[v])
                            ndsindex[v] = i += one(PINT)
                            ndsorder[i] = v
                        end
                    end

                    for v in pop!(orders)
                        ndsindex[v] = i += one(PINT)
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

            j = zero(PINT); outorder = FVector{PINT}(undef, l)

            for i in oneto(n)
                v = work03[i]

                for w in neighbors(label, v)
                    j += one(PINT); outorder[j] = w
                end
            end

            push!(orders, outorder)
        end
    end

    return only(orders)
end

end
