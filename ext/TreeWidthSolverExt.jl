module TreeWidthSolverExt

using CliqueTrees
using Graphs
using TreeWidthSolver: TreeWidthSolver

function CliqueTrees.permutation(graph, alg::BT)
    return permutation(BipartiteGraph(graph), alg)
end

function CliqueTrees.permutation(graph::AbstractGraph{V}, alg::BT) where {V}
    graph = Graph{Int}(graph)

    for v in vertices(graph)
        rem_edge!(graph, v, v)
    end

    order::Vector{V} = reverse!(reduce(vcat, TreeWidthSolver.elimination_order(graph); init = Int[]))
    return order, invperm(order)
end

function CliqueTrees.permutation(weights::AbstractVector, graph, alg::BT)
    return permutation(weights, BipartiteGraph(graph), alg)
end

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::BT) where {V}
    return permutation(collect(weights), graph, alg)
end

function CliqueTrees.permutation(weights::Vector, graph::AbstractGraph{V}, alg::BT) where {V}
    graph = Graph{Int}(graph)

    for v in vertices(graph)
        rem_edge!(graph, v, v)
    end

    order::Vector{V} = reverse!(reduce(vcat, TreeWidthSolver.elimination_order(graph; weights); init = Int[]))
    return order, invperm(order)
end

end
