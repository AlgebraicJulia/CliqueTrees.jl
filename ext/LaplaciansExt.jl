module LaplaciansExt

using CliqueTrees
using Graphs
using Laplacians
using SparseArrays

function CliqueTrees.permutation(graph::BipartiteGraph, alg::Spectral)
    order = spectralorder(graph; tol=alg.tol)
    return order, invperm(order)
end

# A Spectral Algorithm for Envelope Reduction of Sparse Matrices
# Barnard, Pothen, and Simon
# Algorithm 1: Spectral Algorithm
#
# Compute the spectral ordering of a graph.
function spectralorder(graph::BipartiteGraph{V}; tol::Float64=0.0) where {V}
    order = Vector{V}(undef, nv(graph))
    matrix = SparseMatrixCSC{Float64}(graph)
    fill!(nonzeros(fkeep!((i, j, v) -> i != j, matrix)), 1)
    value, vector = fiedler(matrix; tol)
    return sortperm!(order, reshape(vector, size(matrix, 2)))
end

end
