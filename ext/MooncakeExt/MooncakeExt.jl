module MooncakeExt

using CliqueTrees.Multifrontal: ChordalTriangular, DChordalTriangular, FChordalTriangular,
    ChordalSymbolic, HermOrSymTri, HermTri, SymTri, AdjTri, TransTri, Permutation,
    chordal, cong
using CliqueTrees.Multifrontal.Differential: selinv, uncholesky, softmax,
    flat, unflattri, unflatsym, ldiv, rdiv
using LinearAlgebra: HermOrSym
using LinearAlgebra: dot, logdet, tr, diag, cholesky, adjoint, transpose, Diagonal, UniformScaling, axpy!

import Mooncake
using Mooncake: @from_chainrules, tangent_type, NoRData, NoFData, DefaultCtx,
    fdata, rdata, set_to_zero!!, NoTangent, increment_and_get_rdata!, MaybeCache

# ===== to_cr_tangent =====
# Bridge Mooncake tangents to ChainRules tangents (identity for our types)

function Mooncake.to_cr_tangent(t::ChordalTriangular)
    return t
end

# ===== tangent_type =====

function Mooncake.tangent_type(::Type{<:ChordalSymbolic})
    return NoTangent
end

function Mooncake.tangent_type(::Type{<:Permutation})
    return NoTangent
end

function Mooncake.tangent_type(::Type{L}) where {L<:ChordalTriangular}
    return L
end

function Mooncake.tangent_type(::Type{HermTri{UPLO, T, I, Dvl, Lvl}}) where {UPLO, T, I, Dvl, Lvl}
    return ChordalTriangular{:N, UPLO, T, I, Dvl, Lvl}
end

function Mooncake.tangent_type(::Type{SymTri{UPLO, T, I, Dvl, Lvl}}) where {UPLO, T, I, Dvl, Lvl}
    return ChordalTriangular{:N, UPLO, T, I, Dvl, Lvl}
end

function Mooncake.tangent_type(::Type{AdjTri{DIAG, UPLO, T, I, Dvl, Lvl}}) where {DIAG, UPLO, T, I, Dvl, Lvl}
    return ChordalTriangular{DIAG, UPLO, T, I, Dvl, Lvl}
end

function Mooncake.tangent_type(::Type{TransTri{DIAG, UPLO, T, I, Dvl, Lvl}}) where {DIAG, UPLO, T, I, Dvl, Lvl}
    return ChordalTriangular{DIAG, UPLO, T, I, Dvl, Lvl}
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

function Mooncake.fdata_type(::Type{L}) where {L<:ChordalTriangular}
    return L
end

function Mooncake.fdata_type(::Type{HermTri{UPLO, T, I, Dvl, Lvl}}) where {UPLO, T, I, Dvl, Lvl}
    return ChordalTriangular{:N, UPLO, T, I, Dvl, Lvl}
end

function Mooncake.fdata_type(::Type{SymTri{UPLO, T, I, Dvl, Lvl}}) where {UPLO, T, I, Dvl, Lvl}
    return ChordalTriangular{:N, UPLO, T, I, Dvl, Lvl}
end

function Mooncake.fdata_type(::Type{AdjTri{DIAG, UPLO, T, I, Dvl, Lvl}}) where {DIAG, UPLO, T, I, Dvl, Lvl}
    return ChordalTriangular{DIAG, UPLO, T, I, Dvl, Lvl}
end

function Mooncake.fdata_type(::Type{TransTri{DIAG, UPLO, T, I, Dvl, Lvl}}) where {DIAG, UPLO, T, I, Dvl, Lvl}
    return ChordalTriangular{DIAG, UPLO, T, I, Dvl, Lvl}
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

# ===== @from_chainrules =====

# logdet.jl
@from_chainrules DefaultCtx Tuple{typeof(logdet), ChordalTriangular{:N}}
@from_chainrules DefaultCtx Tuple{typeof(logdet), HermTri, ChordalTriangular{:N}}
@from_chainrules DefaultCtx Tuple{typeof(logdet), SymTri, ChordalTriangular{:N}}

# trace.jl
@from_chainrules DefaultCtx Tuple{typeof(tr), ChordalTriangular{:N}}
@from_chainrules DefaultCtx Tuple{typeof(tr), HermTri}
@from_chainrules DefaultCtx Tuple{typeof(tr), SymTri}

# diag.jl
@from_chainrules DefaultCtx Tuple{typeof(diag), ChordalTriangular{:N}}
@from_chainrules DefaultCtx Tuple{typeof(diag), HermTri}
@from_chainrules DefaultCtx Tuple{typeof(diag), SymTri}

# selinv.jl
@from_chainrules DefaultCtx Tuple{typeof(selinv), ChordalTriangular{:N}}
@from_chainrules DefaultCtx Tuple{typeof(selinv), HermTri, ChordalTriangular{:N}}
@from_chainrules DefaultCtx Tuple{typeof(selinv), SymTri, ChordalTriangular{:N}}

# cholesky.jl
@from_chainrules DefaultCtx Tuple{typeof(cholesky), HermTri}
@from_chainrules DefaultCtx Tuple{typeof(cholesky), SymTri}

# uncholesky.jl
@from_chainrules DefaultCtx Tuple{typeof(uncholesky), ChordalTriangular{:N}}

# softmax.jl
@from_chainrules DefaultCtx Tuple{typeof(softmax), ChordalTriangular{:N}}

# dot.jl
@from_chainrules DefaultCtx Tuple{typeof(dot), HermTri, HermTri}
@from_chainrules DefaultCtx Tuple{typeof(dot), SymTri, SymTri}
@from_chainrules DefaultCtx Tuple{typeof(dot), ChordalTriangular{:N}, ChordalTriangular{:N}}

# flat.jl
@from_chainrules DefaultCtx Tuple{typeof(flat), ChordalTriangular{:N}}
@from_chainrules DefaultCtx Tuple{typeof(flat), HermTri}
@from_chainrules DefaultCtx Tuple{typeof(flat), SymTri}

# unflat.jl
@from_chainrules DefaultCtx Tuple{typeof(unflattri), AbstractVector, ChordalSymbolic, Val}
@from_chainrules DefaultCtx Tuple{typeof(unflatsym), AbstractVector, ChordalSymbolic, Val}

# adjoint.jl
@from_chainrules DefaultCtx Tuple{typeof(adjoint), ChordalTriangular}
@from_chainrules DefaultCtx Tuple{typeof(adjoint), AdjTri}
@from_chainrules DefaultCtx Tuple{typeof(transpose), ChordalTriangular}
@from_chainrules DefaultCtx Tuple{typeof(transpose), TransTri}

# add.jl
@from_chainrules DefaultCtx Tuple{typeof(+), ChordalTriangular{:N}, ChordalTriangular{:N}}
@from_chainrules DefaultCtx Tuple{typeof(+), HermTri, HermTri}
@from_chainrules DefaultCtx Tuple{typeof(+), SymTri, SymTri}
@from_chainrules DefaultCtx Tuple{typeof(+), UniformScaling, ChordalTriangular}
@from_chainrules DefaultCtx Tuple{typeof(+), UniformScaling, HermTri}
@from_chainrules DefaultCtx Tuple{typeof(+), UniformScaling, SymTri}
@from_chainrules DefaultCtx Tuple{typeof(+), ChordalTriangular, UniformScaling}
@from_chainrules DefaultCtx Tuple{typeof(+), HermTri, UniformScaling}
@from_chainrules DefaultCtx Tuple{typeof(+), SymTri, UniformScaling}
@from_chainrules DefaultCtx Tuple{typeof(+), Diagonal, ChordalTriangular}
@from_chainrules DefaultCtx Tuple{typeof(+), Diagonal, HermTri}
@from_chainrules DefaultCtx Tuple{typeof(+), Diagonal, SymTri}
@from_chainrules DefaultCtx Tuple{typeof(+), ChordalTriangular, Diagonal}
@from_chainrules DefaultCtx Tuple{typeof(+), HermTri, Diagonal}
@from_chainrules DefaultCtx Tuple{typeof(+), SymTri, Diagonal}
@from_chainrules DefaultCtx Tuple{typeof(-), ChordalTriangular, Diagonal}
@from_chainrules DefaultCtx Tuple{typeof(-), HermTri, Diagonal}
@from_chainrules DefaultCtx Tuple{typeof(-), SymTri, Diagonal}

# lmulnum.jl
@from_chainrules DefaultCtx Tuple{typeof(*), Number, ChordalTriangular{:N}}
@from_chainrules DefaultCtx Tuple{typeof(*), Real, HermTri}
@from_chainrules DefaultCtx Tuple{typeof(*), Real, SymTri}

# rmulnum.jl
@from_chainrules DefaultCtx Tuple{typeof(*), ChordalTriangular{:N}, Number}
@from_chainrules DefaultCtx Tuple{typeof(*), HermTri, Real}
@from_chainrules DefaultCtx Tuple{typeof(*), SymTri, Real}

# ldivnum.jl
@from_chainrules DefaultCtx Tuple{typeof(\), Number, ChordalTriangular{:N}}
@from_chainrules DefaultCtx Tuple{typeof(\), Real, HermTri}
@from_chainrules DefaultCtx Tuple{typeof(\), Real, SymTri}

# rdivnum.jl
@from_chainrules DefaultCtx Tuple{typeof(/), ChordalTriangular{:N}, Number}
@from_chainrules DefaultCtx Tuple{typeof(/), HermTri, Real}
@from_chainrules DefaultCtx Tuple{typeof(/), SymTri, Real}

# ldiv.jl
@from_chainrules DefaultCtx Tuple{typeof(\), ChordalTriangular{:N}, AbstractVecOrMat}
@from_chainrules DefaultCtx Tuple{typeof(\), AdjTri{:N}, AbstractVecOrMat}
@from_chainrules DefaultCtx Tuple{typeof(\), TransTri{:N}, AbstractVecOrMat}

# rdiv.jl
@from_chainrules DefaultCtx Tuple{typeof(/), AbstractMatrix, ChordalTriangular{:N}}
@from_chainrules DefaultCtx Tuple{typeof(/), AbstractMatrix, AdjTri{:N}}
@from_chainrules DefaultCtx Tuple{typeof(/), AbstractMatrix, TransTri{:N}}

# ldivsym.jl
@from_chainrules DefaultCtx Tuple{typeof(ldiv), HermTri, ChordalTriangular{:N}, AbstractVecOrMat}
@from_chainrules DefaultCtx Tuple{typeof(ldiv), SymTri, ChordalTriangular{:N}, AbstractVecOrMat}

# rdivsym.jl
@from_chainrules DefaultCtx Tuple{typeof(rdiv), AbstractMatrix, HermTri, ChordalTriangular{:N}}
@from_chainrules DefaultCtx Tuple{typeof(rdiv), AbstractMatrix, SymTri, ChordalTriangular{:N}}

# lmul.jl
@from_chainrules DefaultCtx Tuple{typeof(*), ChordalTriangular{:N}, AbstractVecOrMat}
@from_chainrules DefaultCtx Tuple{typeof(*), AdjTri{:N}, AbstractVecOrMat}
@from_chainrules DefaultCtx Tuple{typeof(*), TransTri{:N}, AbstractVecOrMat}

# rmul.jl
@from_chainrules DefaultCtx Tuple{typeof(*), AbstractMatrix, ChordalTriangular{:N}}
@from_chainrules DefaultCtx Tuple{typeof(*), AbstractMatrix, AdjTri{:N}}
@from_chainrules DefaultCtx Tuple{typeof(*), AbstractMatrix, TransTri{:N}}

# lmulsym.jl
@from_chainrules DefaultCtx Tuple{typeof(*), HermTri, AbstractVecOrMat}
@from_chainrules DefaultCtx Tuple{typeof(*), SymTri, AbstractVecOrMat}

# rmulsym.jl
@from_chainrules DefaultCtx Tuple{typeof(*), AbstractMatrix, HermTri}
@from_chainrules DefaultCtx Tuple{typeof(*), AbstractMatrix, SymTri}

# quad.jl
@from_chainrules DefaultCtx Tuple{typeof(dot), AbstractVecOrMat, HermTri, AbstractVecOrMat}
@from_chainrules DefaultCtx Tuple{typeof(dot), AbstractVecOrMat, SymTri, AbstractVecOrMat}

# lmulprm.jl
@from_chainrules DefaultCtx Tuple{typeof(*), Permutation, AbstractVecOrMat}

# rmulprm.jl
@from_chainrules DefaultCtx Tuple{typeof(*), AbstractVecOrMat, Permutation}

# ldivprm.jl
@from_chainrules DefaultCtx Tuple{typeof(\), Permutation, AbstractVecOrMat}

# rdivprm.jl
@from_chainrules DefaultCtx Tuple{typeof(/), AbstractVecOrMat, Permutation}

# chordal.jl
@from_chainrules DefaultCtx Tuple{typeof(chordal), HermOrSym, Permutation, ChordalSymbolic, Val}
@from_chainrules DefaultCtx Tuple{typeof(chordal), HermOrSym, Permutation, ChordalSymbolic}

# cong.jl
@from_chainrules DefaultCtx Tuple{typeof(cong), AbstractMatrix, Permutation}

end
