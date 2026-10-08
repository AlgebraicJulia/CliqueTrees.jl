module ScotchExt

using Base: oneto
using CliqueTrees
using CliqueTrees: SCOTCH
using CliqueTrees.Utilities
using Graphs

using Scotch: SCOTCH_Num, graph_build, strat_build, block_ordering

const INT = SCOTCH_Num

const VBipartiteGraph{V, E} = BipartiteGraph{V, E, Vector{E}, Vector{V}}

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::SCOTCH) where {V}
    order, index = scotch(weights, graph, alg)
    return order, index
end

function scotch(weights::AbstractVector, graph::AbstractGraph{V}, alg::SCOTCH) where {V}
    vwght = scotch_vwght(weights, graph)
    xadj, adjncy = scotch_graph(graph)

    scotchgraph = graph_build(xadj, adjncy; index_start=one(INT), v_weights=vwght)

    strat = strat_build(:graph_order;
        strategy=alg.strategy, level_nbr=alg.level, imbalance_ratio=alg.imbalance)

    permtab, peritab = block_ordering(scotchgraph, strat;
        permutation=true, inv_permutation=true)

    order = convert(Vector{V}, peritab)
    index = convert(Vector{V}, permtab)
    return order, index
end

function scotch_vwght(weights::AbstractVector, graph::AbstractGraph)
    return scotch_vwght_impl(weights, graph)
end

function scotch_vwght(weights::Vector{INT}, graph::AbstractGraph)
    @assert nv(graph) <= length(weights)
    n = nv(graph)

    if length(weights) != n
        return scotch_vwght_impl(weights, graph)
    end

    return weights
end

function scotch_vwght_impl(weights::AbstractVector, graph::AbstractGraph)
    @assert nv(graph) <= length(weights)
    n = nv(graph)
    vwght = Vector{INT}(undef, n)

    @inbounds for v in oneto(n)
        vwght[v] = trunc(INT, weights[v])
    end

    return vwght
end

function scotch_graph(graph::AbstractGraph)
    return scotch_graph_impl(graph)
end

function scotch_graph(graph::VBipartiteGraph{INT, INT})
    n = nv(graph)
    m = ne(graph)
    xadj = pointers(graph)
    adjncy = targets(graph)

    if length(xadj) != n + one(INT) || length(adjncy) != m
        return scotch_graph_impl(graph)
    end

    @inbounds for j in vertices(graph)
        for i in neighbors(graph, j)
            if i == j
                return scotch_graph_impl(graph)
            end
        end
    end

    return xadj, adjncy
end

function scotch_graph_impl(graph::AbstractGraph{V}) where {V}
    n = nv(graph)
    xadj = Vector{INT}(undef, n + one(V))
    adjncy = Vector{INT}(undef, de(graph))

    p = zero(INT)

    @inbounds for j in oneto(n)
        xadj[j] = p + one(INT)

        for i in neighbors(graph, j)
            if i != j
                p += one(INT)
                adjncy[p] = i
            end
        end
    end

    @inbounds xadj[n + one(V)] = p + one(INT)
    resize!(adjncy, p)
    return xadj, adjncy
end

end
