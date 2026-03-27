module MooncakeExt

using CliqueTrees.Multifrontal: ChordalTriangular, DChordalTriangular, FChordalTriangular,
    ChordalSymbolic, HermOrSymTri, HermTri, SymTri, AdjTri, TransTri, Permutation
using CliqueTrees.Multifrontal.Differential: selinv, complete, uncholesky, soft,
    cholesky_rrule_impl,
    uncholesky_rrule_impl,
    selinv_rrule_impl,
    complete_rrule_impl,
    dot_rrule_impl,
    ldiv_rrule_impl,
    tr_rrule_impl,
    diag_rrule_impl,
    logdet_rrule_impl,
    soft_rrule_impl,
    add_rrule_impl,
    adjoint_rrule_impl,
    transpose_rrule_impl,
    mul_rrule_impl,
    rdiv_rrule_impl
using ChainRulesCore: unthunk
using LinearAlgebra: dot, logdet, tr, diag, cholesky, adjoint, transpose, Hermitian, Symmetric, UniformScaling, I, axpy!

import Mooncake
using Mooncake: @is_primitive, rrule!!, CoDual, primal, tangent,
    tangent_type, NoRData, NoFData, DefaultCtx, MinimalCtx, fdata, rdata,
    increment_rdata!!, set_to_zero!!, MutableTangent, Tangent, NoTangent,
    @zero_adjoint, ReverseMode, increment_and_get_rdata!, RData, MaybeCache

# ===== uniform_rdata =====

function uniform_rdata(Δλ::T) where {T}
    return RData{@NamedTuple{λ::T}}((λ=Δλ,))
end

# ===== tangent_type =====

function Mooncake.tangent_type(::Type{<:ChordalSymbolic})
    return NoTangent
end

function Mooncake.tangent_type(::Type{<:Permutation})
    return NoTangent
end

function Mooncake.tangent_type(::Type{T}) where {T<:ChordalTriangular}
    return T
end

function Mooncake.tangent_type(::Type{H}) where {H<:HermTri}
    return fieldtype(H, :data)
end

function Mooncake.tangent_type(::Type{S}) where {S<:SymTri}
    return fieldtype(S, :data)
end

function Mooncake.tangent_type(::Type{A}) where {A<:AdjTri}
    return fieldtype(A, :parent)
end

function Mooncake.tangent_type(::Type{T}) where {T<:TransTri}
    return fieldtype(T, :parent)
end

# ===== zero_tangent_internal =====
# Must define these to intercept before Mooncake's @generated fallback

function Mooncake.zero_tangent_internal(::ChordalSymbolic, ::MaybeCache)
    return NoTangent()
end

function Mooncake.zero_tangent_internal(::Permutation, ::MaybeCache)
    return NoTangent()
end

function Mooncake.zero_tangent_internal(x::ChordalTriangular, ::MaybeCache)
    return zero(x)
end

function Mooncake.zero_tangent_internal(x::HermTri, ::MaybeCache)
    return zero(parent(x))
end

function Mooncake.zero_tangent_internal(x::SymTri, ::MaybeCache)
    return zero(parent(x))
end

function Mooncake.zero_tangent_internal(x::AdjTri, ::MaybeCache)
    return zero(parent(x))
end

function Mooncake.zero_tangent_internal(x::TransTri, ::MaybeCache)
    return zero(parent(x))
end

# ===== fdata_type =====

function Mooncake.fdata_type(::Type{<:ChordalSymbolic})
    return NoFData
end

function Mooncake.fdata_type(::Type{<:Permutation})
    return NoFData
end

function Mooncake.fdata_type(::Type{T}) where {T<:ChordalTriangular}
    return T
end

function Mooncake.fdata_type(::Type{H}) where {H<:HermTri}
    return fieldtype(H, :data)
end

function Mooncake.fdata_type(::Type{S}) where {S<:SymTri}
    return fieldtype(S, :data)
end

function Mooncake.fdata_type(::Type{A}) where {A<:AdjTri}
    return fieldtype(A, :parent)
end

function Mooncake.fdata_type(::Type{T}) where {T<:TransTri}
    return fieldtype(T, :parent)
end

# ===== rdata_type =====

function Mooncake.rdata_type(::Type{<:ChordalSymbolic})
    return NoRData
end

function Mooncake.rdata_type(::Type{<:Permutation})
    return NoRData
end

function Mooncake.rdata_type(::Type{<:ChordalTriangular})
    return NoRData
end

function Mooncake.rdata_type(::Type{<:HermTri})
    return NoRData
end

function Mooncake.rdata_type(::Type{<:SymTri})
    return NoRData
end

function Mooncake.rdata_type(::Type{<:AdjTri})
    return NoRData
end

function Mooncake.rdata_type(::Type{<:TransTri})
    return NoRData
end

# ===== fdata =====

function Mooncake.fdata(::ChordalSymbolic)
    return NoFData()
end

function Mooncake.fdata(::Permutation)
    return NoFData()
end

function Mooncake.fdata(t::ChordalTriangular)
    return t
end

function Mooncake.fdata(t::HermTri)
    return parent(t)
end

function Mooncake.fdata(t::SymTri)
    return parent(t)
end

function Mooncake.fdata(t::AdjTri)
    return parent(t)
end

function Mooncake.fdata(t::TransTri)
    return parent(t)
end

# ===== rdata =====

function Mooncake.rdata(::ChordalSymbolic)
    return NoRData()
end

function Mooncake.rdata(::Permutation)
    return NoRData()
end

function Mooncake.rdata(::ChordalTriangular)
    return NoRData()
end

function Mooncake.rdata(::HermTri)
    return NoRData()
end

function Mooncake.rdata(::SymTri)
    return NoRData()
end

function Mooncake.rdata(::AdjTri)
    return NoRData()
end

function Mooncake.rdata(::TransTri)
    return NoRData()
end

# ===== tangent =====

function Mooncake.tangent(f::ChordalTriangular, ::NoRData)
    return f
end

# ===== increment_and_get_rdata! =====

function Mooncake.increment_and_get_rdata!(fdata::ChordalTriangular, ::NoRData, t::ChordalTriangular)
    axpy!(1, t, fdata)
    return NoRData()
end

# ===== set_to_zero!! =====

function Mooncake.set_to_zero!!(t::ChordalTriangular)
    fill!(t, 0)
    return t
end

# ===== rrule!! =====
include("logdet.jl")
include("trace.jl")
include("diag.jl")
include("selinv.jl")
include("cholesky.jl")
include("uncholesky.jl")
include("soft.jl")
include("complete.jl")
include("dot.jl")
include("flat.jl")
include("unflat.jl")
include("adjoint.jl")
include("ldiv.jl")
include("ldivadj.jl")
include("lmul.jl")
include("lmuladj.jl")
include("rdiv.jl")
include("rdivadj.jl")
include("rmul.jl")
include("rmuladj.jl")
include("add.jl")
include("raddnum.jl")
include("rmulnum.jl")
include("rdivnum.jl")
include("lmulsym.jl")
include("rmulsym.jl")
include("quad.jl")
include("lmulprm.jl")
include("rmulprm.jl")
include("ldivprm.jl")
include("rdivprm.jl")

end
