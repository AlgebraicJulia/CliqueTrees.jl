# Kernel functions for X * H where H is HermTri or SymTri

# ===== frule =====

function mul_frule_impl(X::AbstractMatrix, H::HermTri{UPLO}, dX::AbstractMatrix, dH::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(H, dH)
    Y = X * H
    dY = dX * H + X * Hermitian(dH, UPLO)
    return Y, dY
end

function mul_frule_impl(X::AbstractMatrix, S::SymTri{UPLO}, dX::AbstractMatrix, dS::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(S, dS)
    Y = X * S
    dY = dX * S + X * Symmetric(dS, UPLO)
    return Y, dY
end

function mul_frule_impl(X::AbstractMatrix, H::HermTri{UPLO}, dX::ZeroTangent, dH::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(H, dH)
    Y = X * H
    dY = X * Hermitian(dH, UPLO)
    return Y, dY
end

function mul_frule_impl(X::AbstractMatrix, S::SymTri{UPLO}, dX::ZeroTangent, dS::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(S, dS)
    Y = X * S
    dY = X * Symmetric(dS, UPLO)
    return Y, dY
end

function mul_frule_impl(X::AbstractMatrix, H::HermOrSymTri, dX::AbstractMatrix, dH::ZeroTangent)
    Y = X * H
    dY = dX * H
    return Y, dY
end

function mul_frule_impl(X::AbstractMatrix, H::HermOrSymTri, dX::ZeroTangent, dH::ZeroTangent)
    return X * H, ZeroTangent()
end

function ChainRulesCore.frule((_, dX, dH)::Tuple, ::typeof(*), X::AbstractMatrix, H::HermTri{UPLO}) where {UPLO}
    return mul_frule_impl(X, H, dX, dH)
end

function ChainRulesCore.frule((_, dX, dS)::Tuple, ::typeof(*), X::AbstractMatrix, S::SymTri{UPLO}) where {UPLO}
    return mul_frule_impl(X, S, dX, dS)
end

# ===== rrule =====

function mul_rrule_impl(X::AbstractMatrix, H::HermTri{UPLO}, Y::AbstractMatrix, ΔY::AbstractMatrix) where {UPLO}
    ΔX = ΔY * H
    ΔH = @thunk begin
        ΔH = similar(parent(H))
        selupd!(ΔH, X', ΔY, 1 / 2, 0)
        selupd!(ΔH, ΔY', X, 1 / 2, 1)
        ΔH
    end
    return ΔX, ΔH
end

function mul_rrule_impl(X::AbstractMatrix, S::SymTri{UPLO}, Y::AbstractMatrix, ΔY::AbstractMatrix) where {UPLO}
    ΔX = ΔY * S
    ΔS = @thunk begin
        ΔS = similar(parent(S))
        selupd!(ΔS, transpose(X), ΔY, 1 / 2, 0)
        selupd!(ΔS, transpose(ΔY), X, 1 / 2, 1)
        ΔS
    end
    return ΔX, ΔS
end

function mul_rrule(X::AbstractMatrix, H::HermOrSymTri)
    Y = X * H

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔX, ΔH = mul_rrule_impl(X, H, Y, ΔY)
            return NoTangent(), ΔX, ΔH
        end
    end

    return Y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(*), X::AbstractMatrix, H::HermTri)
    return mul_rrule(X, H)
end

function ChainRulesCore.rrule(::typeof(*), X::AbstractMatrix, S::SymTri)
    return mul_rrule(X, S)
end

# type ambiguity with ChainRules
function ChainRulesCore.rrule(::typeof(*), X::AbstractMatrix{<:RealOrComplex}, H::HermTri{<:Any, <:RealOrComplex})
    return mul_rrule(X, H)
end

function ChainRulesCore.rrule(::typeof(*), X::AbstractMatrix{<:RealOrComplex}, S::SymTri{<:Any, <:RealOrComplex})
    return mul_rrule(X, S)
end
