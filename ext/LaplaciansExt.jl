module LaplaciansExt

using CliqueTrees
using CliqueTrees: simplegraph
using Graphs
using Laplacians
using SparseArrays

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph, alg::Spectral)
    order = spectralorder(graph; tol = alg.tol)
    return order, invperm(order)
end

# A Spectral Algorithm for Envelope Reduction of Sparse Matrices
# Barnard, Pothen, and Simon
# Algorithm 1: Spectral Algorithm
#
# Compute the spectral ordering of a graph.
function spectralorder(graph::AbstractGraph{V}; tol::Float64 = 0.0) where {V}
    order = Vector{V}(undef, nv(graph))
    matrix = sparse(Float64, simplegraph(graph))
    value, vector = fiedler(matrix; tol)
    return sortperm!(order, reshape(vector, size(matrix, 2)))
end

end
