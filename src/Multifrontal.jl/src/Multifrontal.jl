module Multifrontal

using AbstractTrees
using Base: oneto, OneTo, print_matrix, replace_with_centered_mark, isstored, promote_eltype
using FillArrays: Ones, Zeros, AbstractFill, AbstractZeros, AbstractZerosVector, AbstractZerosMatrix
using Graphs
using LinearAlgebra
using LinearAlgebra: Adjoint, Transpose, AdjOrTrans, HermOrSym, BlasFloat, BlasComplex, Factorization, LAPACK, BLAS, PivotingStrategy, RowMaximum, BlasInt, checksquare, chkstride1, require_one_based_indexing, givensAlgorithm, AbstractTriangular
using Random
using Random: rand!
using SparseArrays

import ..BipartiteGraph, ..CliqueTree, ..FArray, ..FMatrix, ..FScalar, ..FVector, ..Scalar, ..Tree,
    ..incident, ..nov, ..ne, ..nv, ..outvertices, ..vertices, ..neighbors, ..pointers, ..targets,
    ..eltypedegree, ..etype, ..residual, ..half, ..ispositive, ..isnegative, ..two, ..twice,
    ..cliquetree, ..residuals, ..separators, ..childindices

export ChordalSymbolic
export ChordalCholesky, FChordalCholesky
export ChordalLDLt, FChordalLDLt
export ChordalTriangular, FChordalTriangular
export DenseCholesky, DenseLDLt
export DenseCholeskyPivoted, DenseLDLtPivoted
export DynamicRegularization, GMW81, SE99
export Permutation, FPermutation
export symbolic
export chordal
export selinv!

const THRESHOLD = 64
const DEFAULT_UPLO = :L
const TransVec = Transpose{<:Any, <:AbstractVector}
const AdjVec = Adjoint{<:Any, <:AbstractVector}
const IOnes{T} = Ones{T, 1, Tuple{OneTo{Int}}}
const HermSparse{T, I} = Hermitian{T, SparseMatrixCSC{T, I}}
const SymSparse{T, I} = Symmetric{T, SparseMatrixCSC{T, I}}
const HermOrSymSparse{T, I} = Union{HermSparse{T, I}, SymSparse{T, I}}
const MaybeHermOrSymSparse{T, I} = Union{HermOrSymSparse{T, I}, SparseMatrixCSC{T, I}}

include("permutation.jl")
include("chordal_symbolic.jl")
include("abstract_factorization.jl")
include("chordal_factorization.jl")
include("chordal_triangular.jl")
include("regularization.jl")
include("cholesky.jl")
include("cholesky_pivoted.jl")
include("cholesky_se99.jl")
include("cholesky_pivoted_se99.jl")
include("utils.jl")
include("operator.jl")
include("blas.jl")
include("divide.jl")
include("multiply.jl")
include("add.jl")
include("selinv.jl")
include("complete.jl")
include("uncholesky.jl")
include("dense/dense.jl")
include("fisherroot.jl")
include("fisher.jl")
include("cholesky_differential.jl")
include("lowrank.jl")
include("krylov.jl")
include("complete_generic.jl")
if Base.USE_GPL_LIBS
    include("cholmod.jl")
end
include("Differential.jl/src/Differential.jl")

using .Differential

end # module Multifrontal
