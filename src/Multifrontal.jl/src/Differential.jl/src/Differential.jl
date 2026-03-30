module Differential

using Base: promote_eltype
using ChainRulesCore
using ChainRulesCore: ProjectTo, NoTangent, ZeroTangent, unthunk, @thunk, add!!
using FillArrays: Fill
using LinearAlgebra
using LinearAlgebra: dot, tr, diag, Diagonal, UniformScaling, HermOrSym, axpy!

using SparseArrays: SparseMatrixCSC

using ...Multifrontal
using ...Multifrontal: HermOrSymTri, MaybeHermOrSymTri, MaybeHermOrSymSparse, HermSparse, SymSparse, ChordalCholesky, ChordalTriangular, ChordalSymbolic, Permutation
using ...Multifrontal: AdjTri, TransTri, AdjOrTransTri, MaybeAdjOrTransTri, HermTri, SymTri
using ...Multifrontal: cholesky!, complete!, dfcholesky!, fisher!, rmul!, selinv!, selupd!, uncholesky!
using ...Multifrontal: fronts, diagblock, diagind, ndz, nlz, nnz, triangular, checksymbolic
using ...Multifrontal: DEFAULT_UPLO, chordal, cong, project

export selinv, uncholesky, ldivsym, rdivsym

const CHORDAL_TYPES = (ChordalTriangular{:N}, HermTri, SymTri, AdjTri{:N}, TransTri{:N})

function Multifrontal.project(A, ::ZeroTangent)
    return ZeroTangent()
end

function Multifrontal.project(A, ::ZeroTangent, P::Permutation)
    return ZeroTangent()
end

for T in CHORDAL_TYPES
    @eval function ChainRulesCore.ProjectTo(::$T)
        return ProjectTo{$T}()
    end

    @eval function (::ChainRulesCore.ProjectTo{$T})(dX::$T)
        return dX
    end

    @eval function (::ChainRulesCore.ProjectTo{$T})(dX::Diagonal)
        return dX
    end

    @eval function (::ChainRulesCore.ProjectTo{$T})(dX::UniformScaling)
        return dX
    end

    @eval function (P::ChainRulesCore.ProjectTo{UniformScaling})(dX::$T)
        return UniformScaling(P.λ(tr(dX)))
    end

    @eval function (P::ChainRulesCore.ProjectTo{Diagonal})(dX::$T)
        return Diagonal(P.diag(diag(dX)))
    end

    @eval function ChainRulesCore.add!!(X::$T, Y::$T)
        return axpy!(true, Y, X)
    end

    @eval function ChainRulesCore.add!!(X::$T, Y::Diagonal)
        return axpy!(true, Y, X)
    end

    @eval function ChainRulesCore.add!!(X::$T, Y::UniformScaling)
        return axpy!(true, Y, X)
    end

    @eval function Base.:\(::$T, ::ZeroTangent)
        return ZeroTangent()
    end
end

# Generic frule/rrule helpers for *, \, /

function mul_frule_impl(A, B, dA, dB)
    return A * B, dA * B + A * dB
end

function mul_rrule_impl(X::StridedMatrix, A::StridedMatrix, Y::StridedMatrix, ΔY)
    error()
end

function mul_rrule(A, X)
    Y = A * X

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔA, ΔX = mul_rrule_impl(A, X, Y, ΔY)
            return NoTangent(), ΔA, ΔX
        end
    end

    return Y, pullback ∘ unthunk
end

function ldiv_rrule(A, X)
    Y = A \ X

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔA, ΔX = ldiv_rrule_impl(A, X, Y, ΔY)
            return NoTangent(), ΔA, ΔX
        end
    end

    return Y, pullback ∘ unthunk
end

function rdiv_rrule(X, A)
    Y = X / A

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔX, ΔA = rdiv_rrule_impl(X, A, Y, ΔY)
            return NoTangent(), ΔX, ΔA
        end
    end

    return Y, pullback ∘ unthunk
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
include("adjoint.jl")
include("trace.jl")
include("diag.jl")
include("add.jl")
include("chordal.jl")

end
