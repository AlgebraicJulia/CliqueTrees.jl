module TreeWidthSolverExt

using CliqueTrees
using Graphs
using TreeWidthSolver: TreeWidthSolver

function CliqueTrees.permutation(graph, alg::BT)
    return permutation(BipartiteGraph(graph), alg)
end

function CliqueTrees.permutation(graph::BipartiteGraph{V}, alg::BT) where {V}
    order::Vector{V}, index::Vector{V} = permutation(Graph{Int}(graph), alg)
    return order, index
end

function CliqueTrees.permutation(graph::Graph{Int}, alg::BT)
    order = reverse!(reduce(vcat, TreeWidthSolver.elimination_order(graph); init=Int[]))
    return order, invperm(order)
end

end
