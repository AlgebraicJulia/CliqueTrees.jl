module Differential

using Base: promote_eltype
using ChainRulesCore
using ChainRulesCore: ProjectTo, NoTangent, ZeroTangent, unthunk, @thunk
using LinearAlgebra
using LinearAlgebra: dot, tr, RealOrComplex, UniformScaling, axpy!

using ...Multifrontal
using ...Multifrontal: HermOrSymTri, MaybeHermOrSymTri, ChordalCholesky, ChordalTriangular, ChordalSymbolic, Permutation
using ...Multifrontal: AdjTri, TransTri, AdjOrTransTri, HermTri, SymTri
using ...Multifrontal: cholesky!, complete!, dfcholesky!, fisher!, rmul!, selinv!, selupd!, uncholesky!
using ...Multifrontal: fronts, diagblock, diagind, ndz, nlz, nnz, triangular, checksymbolic
using ...Multifrontal: DEFAULT_UPLO

export selinv, uncholesky, soft, flat, unflattri, unflatsym, ldiv, rdiv

function ChainRulesCore.ProjectTo(X::HermTri)
    return ProjectTo{Hermitian}()
end

function ChainRulesCore.ProjectTo(X::SymTri)
    return ProjectTo{Symmetric}()
end

function (::ChainRulesCore.ProjectTo{Hermitian})(dX::ChordalTriangular{:N, UPLO}) where {UPLO}
    return Hermitian(dX, UPLO)
end

function (::ChainRulesCore.ProjectTo{Symmetric})(dX::ChordalTriangular{:N, UPLO}) where {UPLO}
    return Symmetric(dX, UPLO)
end

function ChainRulesCore.ProjectTo(X::AdjTri)
    return ProjectTo{Adjoint}()
end

function ChainRulesCore.ProjectTo(X::TransTri)
    return ProjectTo{Transpose}()
end

function (::ChainRulesCore.ProjectTo{Adjoint})(dX::ChordalTriangular)
    return dX'
end

function (::ChainRulesCore.ProjectTo{Transpose})(dX::ChordalTriangular)
    return transpose(dX)
end

include("utils.jl")
include("selinv.jl")
include("cholesky.jl")
include("uncholesky.jl")
include("logdet.jl")
include("ldiv.jl")
include("ldivadj.jl")
include("ldivsym.jl")
include("rdiv.jl")
include("rdivadj.jl")
include("rdivsym.jl")
include("lmul.jl")
include("lmuladj.jl")
include("rmul.jl")
include("rmuladj.jl")
include("dot.jl")
include("quad.jl")
include("soft.jl")
include("unflat.jl")
include("flat.jl")
include("lmulsym.jl")
include("rmulsym.jl")
include("adjoint.jl")
include("trace.jl")
include("diag.jl")
include("lmulnum.jl")
include("rmulnum.jl")
include("ldivnum.jl")
include("rdivnum.jl")
include("add.jl")
include("laddnum.jl")
include("raddnum.jl")
include("ambiguities.jl")
include("lmulprm.jl")
include("rmulprm.jl")
include("ldivprm.jl")
include("rdivprm.jl")

end
