module AMDExt

using AMD: AMD as AMDLib
using CliqueTrees
using Graphs
using SparseArrays

function CliqueTrees.permutation(graph, alg::Union{AMD,SymAMD})
    return permutation(BipartiteGraph(graph), alg)
end

function CliqueTrees.permutation(graph::AbstractGraph{V}, alg::Union{AMD,SymAMD}) where {V}
    order::Vector{V}, index::Vector{V} = permutation(sparse(graph), alg)
    return order, index
end

function CliqueTrees.permutation(
    matrix::SparseMatrixCSC{T,I}, alg::Union{AMD,SymAMD}
) where {T,I}
    order::Vector{I}, index::Vector{I} = permutation(SparseMatrixCSC{T,Int}(matrix), alg)
    return order, index
end

function CliqueTrees.permutation(
    matrix::SparseMatrixCSC{<:Any,I}, alg::AMD
) where {I<:Union{Int32,Int64}}
    # set parameters
    meta = AMDLib.Amd()
    meta.control[AMDLib.AMD_DENSE] = alg.dense
    meta.control[AMDLib.AMD_AGGRESSIVE] = alg.aggressive

    # run algorithm
    order::Vector{I} = AMDLib.amd(matrix, meta)
    return order, invperm(order)
end

function CliqueTrees.permutation(
    matrix::SparseMatrixCSC{<:Any,I}, alg::SymAMD
) where {I<:Union{Int32,Int64}}
    # set parameters
    meta = AMDLib.Colamd{I}()
    meta.knobs[AMDLib.COLAMD_DENSE_ROW] = alg.dense_row
    meta.knobs[AMDLib.COLAMD_DENSE_COL] = alg.dense_col
    meta.knobs[AMDLib.COLAMD_AGGRESSIVE] = alg.aggressive

    # run algorithm
    order::Vector{I} = AMDLib.symamd(matrix, meta)
    return order, invperm(order)
end

end
