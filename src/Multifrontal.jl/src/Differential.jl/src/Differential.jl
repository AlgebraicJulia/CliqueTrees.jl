module Differential

using LinearAlgebra
using LinearAlgebra: lmul!, axpy!, ldiv!, rdiv!, Hermitian, HermOrSym
using SparseArrays: SparseMatrixCSC, rowvals, nonzeros
using Base: AbstractVecOrMat

using ...Multifrontal
using ...Multifrontal: HermOrSymTri, HermOrSymSparse, HermSparse, SymSparse, ChordalCholesky, ChordalTriangular, Permutation
using ...Multifrontal: fisher!, selinv!, triangular, project, diagblock, fronts

export selinv, selaxpy!, symdot

include("utils.jl")
include("selinv.jl")
include("logdet.jl")
include("divide.jl")

end
