module Differential

using Base: promote_eltype
using ChainRulesCore
using ChainRulesCore: ProjectTo, NoTangent, ZeroTangent, unthunk, @thunk
using LinearAlgebra
using LinearAlgebra: dot, tr, diag, Diagonal, RealOrComplex, UniformScaling, HermOrSym, axpy!

using ...Multifrontal
using ...Multifrontal: HermOrSymTri, MaybeHermOrSymTri, ChordalCholesky, ChordalTriangular, ChordalSymbolic, Permutation
using ...Multifrontal: AdjTri, TransTri, AdjOrTransTri, MaybeAdjOrTransTri, HermTri, SymTri
using ...Multifrontal: cholesky!, complete!, dfcholesky!, fisher!, rmul!, selinv!, selupd!, uncholesky!
using ...Multifrontal: fronts, diagblock, diagind, ndz, nlz, nnz, triangular, checksymbolic
using ...Multifrontal: DEFAULT_UPLO, chordal, cong

export selinv, uncholesky, softmax, flat, unflattri, unflatsym, ldiv, rdiv

const CHORDAL_TYPES = (ChordalTriangular{:N}, HermTri, SymTri, AdjTri{:N}, TransTri{:N})

function ChainRulesCore.ProjectTo(X::ChordalTriangular{:N})
    return ProjectTo{ChordalTriangular{:N}}()
end

function ChainRulesCore.ProjectTo(X::HermTri)
    return ProjectTo{HermTri}()
end

function ChainRulesCore.ProjectTo(X::SymTri)
    return ProjectTo{SymTri}()
end

function (::ChainRulesCore.ProjectTo{ChordalTriangular{:N}})(dX::ChordalTriangular{:N})
    return dX
end

function (::ChainRulesCore.ProjectTo{ChordalTriangular{:N}})(dX::UniformScaling)
    return dX
end

function (::ChainRulesCore.ProjectTo{ChordalTriangular{:N}})(dX::Diagonal)
    return dX
end

function (::ChainRulesCore.ProjectTo{HermTri})(dX::ChordalTriangular{:N, UPLO}) where {UPLO}
    return Hermitian(dX, UPLO)
end

function (::ChainRulesCore.ProjectTo{HermTri})(dX::HermTri)
    return dX
end

function (::ChainRulesCore.ProjectTo{HermTri})(dX::UniformScaling)
    return dX
end

function (::ChainRulesCore.ProjectTo{HermTri})(dX::Diagonal)
    return dX
end

function (::ChainRulesCore.ProjectTo{SymTri})(dX::ChordalTriangular{:N, UPLO}) where {UPLO}
    return Symmetric(dX, UPLO)
end

function (::ChainRulesCore.ProjectTo{SymTri})(dX::SymTri)
    return dX
end

function (::ChainRulesCore.ProjectTo{SymTri})(dX::UniformScaling)
    return dX
end

function (::ChainRulesCore.ProjectTo{SymTri})(dX::Diagonal)
    return dX
end

function ChainRulesCore.ProjectTo(X::AdjTri)
    return ProjectTo{AdjTri}()
end

function ChainRulesCore.ProjectTo(X::TransTri)
    return ProjectTo{TransTri}()
end

function (::ChainRulesCore.ProjectTo{AdjTri})(dX::AbstractMatrix)
    return adjoint(dX)
end

function (::ChainRulesCore.ProjectTo{AdjTri})(dX::UniformScaling)
    return dX
end

function (::ChainRulesCore.ProjectTo{TransTri})(dX::AbstractMatrix)
    return transpose(dX)
end

function (::ChainRulesCore.ProjectTo{TransTri})(dX::UniformScaling)
    return dX
end

function (P::ChainRulesCore.ProjectTo{UniformScaling})(dX::ChordalTriangular)
    return UniformScaling(P.λ(tr(dX)))
end

function (P::ChainRulesCore.ProjectTo{Diagonal})(dX::ChordalTriangular)
    return Diagonal(P.diag(diag(dX)))
end

include("utils.jl")
include("selinv.jl")
include("cholesky.jl")
include("uncholesky.jl")
include("logdet.jl")
include("ldiv.jl")
include("ldivsym.jl")
include("rdiv.jl")
include("rdivsym.jl")
include("lmul.jl")
include("rmul.jl")
include("dot.jl")
include("quad.jl")
include("softmax.jl")
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
include("ambiguities.jl")
include("lmulprm.jl")
include("rmulprm.jl")
include("ldivprm.jl")
include("rdivprm.jl")
include("chordal.jl")
include("cong.jl")

end
