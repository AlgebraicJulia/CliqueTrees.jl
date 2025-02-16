module TreeWidthSolverExt

using CliqueTrees
using Graphs
using TreeWidthSolver

function CliqueTrees.permutation(graph::Graph{Int}, alg::BT)
    order::Vector{Int} = reverse!(reduce(vcat, elimination_order(graph); init=Int[]))
    return order, invperm(order)
end

end
