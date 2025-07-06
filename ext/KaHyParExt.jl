module KaHyParExt

using ArgCheck
using Base: oneto
using Base.Order
using CliqueTrees
using CliqueTrees: EliminationAlgorithm, UnionFind, bestfill_impl!, bestwidth_impl!, compositerotations_impl!, hpartition!, partition!, sympermute!_impl!, nov, outvertices, simplegraph, qcc
using CliqueTrees.Utilities
using Graphs

import KaHyPar

const WINT1 = KaHyPar.kahypar_hypernode_weight_t
const WINT2 = KaHyPar.kahypar_hyperedge_weight_t
const VINT1 = KaHyPar.kahypar_hypernode_id_t
const VINT2 = KaHyPar.kahypar_hyperedge_id_t
const EINT = KaHyPar.Csize_t
const PINT = KaHyPar.kahypar_partition_id_t

function CliqueTrees.permutation(graph, alg::ND{<:Any, <:EliminationAlgorithm, <:KaHyParND})
    order = dissect(graph, alg)
    return order, invperm(order)
end

function CliqueTrees.permutation(weights::AbstractVector, graph, alg::ND{<:Any, <:EliminationAlgorithm, <:KaHyParND})
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

function dissect(graph, alg::ND)
    return dissect(BipartiteGraph(graph), alg)
end

function dissect(graph::AbstractGraph, alg::ND)
    weights = ones(WINT2, nv(graph))
    return dissect(weights, graph, alg)
end

function dissect(weights::AbstractVector, graph, alg::ND)
    return dissect(weights, BipartiteGraph(graph), alg)
end

function dissect(weights::AbstractVector, graph::AbstractGraph, alg::ND)
    n = nv(graph); new = Vector{WINT2}(undef, n)

    @inbounds for v in oneto(n)
        new[v] = trunc(WINT2, weights[v])
    end

    return dissect(new, graph, alg)
end

function dissect(weights::Vector{WINT2}, graph::AbstractGraph{V}, alg::ND) where {V}
    simple = simplegraph(PINT, PINT, graph)
    cover = qcc(VINT2, EINT, simple, alg.dis.beta, alg.dis.order)
    order::Vector{V} = dissectsimple(weights, reverse(cover), simple, alg)
    return order
end

function dissectsimple(weights::AbstractVector{WINT2}, hgraph::BipartiteGraph{VINT2, EINT}, graph::BipartiteGraph{PINT, PINT}, alg::ND{S}) where {S}
    h = nov(hgraph); n = nv(graph); m = ne(graph); nn = n + one(PINT); width = zero(WINT2)
    maxlevel = convert(PINT, alg.level)
    minwidth = convert(WINT2, alg.width)
    imbalance = convert(PINT, alg.imbalance)

    @inbounds for v in oneto(n)
        width += weights[v]
    end

    swork = Scalar{WINT2}(undef)
    vwork1 = Vector{PINT}(undef, m)
    vwork2 = Vector{PINT}(undef, m)
    vwork3 = Vector{PINT}(undef, max(h, n))
    vwork4 = Vector{PINT}(undef, n)
    vwork5 = Vector{PINT}(undef, max(h, n))
    vwork6 = Vector{PINT}(undef, max(h, n))
    vwork7 = Vector{PINT}(undef, n)
    vwork8 = Vector{PINT}(undef, n)
    vwork9 = Vector{WINT2}(undef, n)
    vwork10 = Vector{WINT2}(undef, n)
    vwork11 = Vector{PINT}(undef, nn)
    vwork12 = Vector{PINT}(undef, nn)
    vwork13 = Vector{PINT}(undef, n)
    vwork14 = Vector{PINT}(undef, n)
    vwork15 = Vector{PINT}(undef, n)
    vwork16 = Vector{PINT}(undef, n)
    vwork17 = Vector{PINT}(undef, n)
    vwork18 = Vector{PINT}(undef, n)
    vwork19 = Vector{PINT}(undef, n)
    hwght = Vector{WINT1}(undef, h)

    @inbounds for v in oneto(h)
        hwght[v] = one(WINT1)
    end

    orders = Vector{PINT}[]

    nodes = Tuple{
        BipartiteGraph{VINT2, EINT, Vector{EINT}, Vector{VINT2}}, # hgraph
        BipartiteGraph{PINT, PINT, Vector{PINT}, Vector{PINT}},   # graph
        Vector{PINT},                                             # weights
        Vector{PINT},                                             # label
        Vector{PINT},                                             # clique
        WINT2,                                                    # width
        PINT,                                                     # level
    }[]

    push!(nodes, (hgraph, graph, weights, collect(oneto(n)), PINT[], width, zero(PINT)))

    @inbounds while !isempty(nodes)
        hgraph, graph, weights, label, clique, width, level = pop!(nodes)
        n = nv(graph); m = ne(graph); k = convert(PINT, length(clique))

        if m > length(vwork1)
            resize!(vwork1, m)
            resize!(vwork2, m)
        end

        if !isnegative(level) # unprocessed
            isleaf = width <= minwidth || level >= maxlevel

            if isleaf # leaf
                push!(nodes, (hgraph, graph, weights, label, clique, width, -one(PINT)))
            else      # branch
                separator!(swork, vwork3, hwght, weights, hgraph, imbalance, alg.dis)

                hchild0, hchild1 = hpartition!(
                    vwork3,
                    vwork4,
                    vwork5,
                    vwork6,
                    hgraph,
                )

                child0, child1, order2 = partition!(
                    vwork4,
                    vwork5,
                    vwork6,
                    weights,
                    graph,
                )

                push!(
                    nodes,
                    (hgraph, graph, weights, label, clique, width, -two(PINT)),
                    (hchild0, child0..., level + one(PINT)),
                    (hchild1, child1..., level + one(PINT)),
                )

                push!(orders, order2)
            end
        else                  # processed
            isleaf = isone(-level)
            tree = Tree(n, vwork6, swork, vwork7, vwork8)
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
                            vwork18, vwork19, weights, graph, (index, greedyindex))
                    else
                        best = bestfill_impl!(lower, upper, tree, sets, vwork9,
                            vwork10, vwork13, vwork14, vwork15, vwork16, vwork17,
                            vwork18, vwork19, weights, graph, (index, greedyindex))
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

end
