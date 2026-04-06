module Differential

using LinearAlgebra
using LinearAlgebra: lmul!, axpy!, Hermitian, HermOrSym
using SparseArrays: SparseMatrixCSC, rowvals, nonzeros

using ...Multifrontal
using ...Multifrontal: HermOrSymTri, HermOrSymSparse, HermSparse, SymSparse, ChordalCholesky, ChordalTriangular, Permutation
using ...Multifrontal: fisher!, selinv!, selupd!, triangular, project, diagblock, fronts

export selinv, selaxpy!, symdot

include("utils.jl")
include("selinv.jl")
include("logdet.jl")

end
