# Kernel functions for H * X where H is HermTri or SymTri

# ===== frule =====

function mul_frule_impl(H::HermTri{UPLO}, X::AbstractVecOrMat, dH::ChordalTriangular{:N, UPLO}, dX::AbstractVecOrMat) where {UPLO}
    @assert checksymbolic(H, dH)
    Y = H * X
    dY = Hermitian(dH, UPLO) * X + H * dX
    return Y, dY
end

function mul_frule_impl(S::SymTri{UPLO}, X::AbstractVecOrMat, dS::ChordalTriangular{:N, UPLO}, dX::AbstractVecOrMat) where {UPLO}
    @assert checksymbolic(S, dS)
    Y = S * X
    dY = Symmetric(dS, UPLO) * X + S * dX
    return Y, dY
end

function mul_frule_impl(H::HermOrSymTri, X::AbstractVecOrMat, dH::ZeroTangent, dX::AbstractVecOrMat)
    Y = H * X
    dY = H * dX
    return Y, dY
end

function mul_frule_impl(H::HermTri{UPLO}, X::AbstractVecOrMat, dH::ChordalTriangular{:N, UPLO}, dX::ZeroTangent) where {UPLO}
    @assert checksymbolic(H, dH)
    Y = H * X
    dY = Hermitian(dH, UPLO) * X
    return Y, dY
end

function mul_frule_impl(S::SymTri{UPLO}, X::AbstractVecOrMat, dS::ChordalTriangular{:N, UPLO}, dX::ZeroTangent) where {UPLO}
    @assert checksymbolic(S, dS)
    Y = S * X
    dY = Symmetric(dS, UPLO) * X
    return Y, dY
end

function mul_frule_impl(H::HermOrSymTri, X::AbstractVecOrMat, dH::ZeroTangent, dX::ZeroTangent)
    return H * X, ZeroTangent()
end

function ChainRulesCore.frule((_, dH, dX)::Tuple, ::typeof(*), H::HermTri{UPLO}, X::AbstractVecOrMat) where {UPLO}
    return mul_frule_impl(H, X, dH, dX)
end

function ChainRulesCore.frule((_, dS, dX)::Tuple, ::typeof(*), S::SymTri{UPLO}, X::AbstractVecOrMat) where {UPLO}
    return mul_frule_impl(S, X, dS, dX)
end

# ===== rrule =====

function mul_rrule_impl(H::HermTri{UPLO}, X::AbstractVecOrMat, Y::AbstractVecOrMat, ΔY::AbstractVecOrMat) where {UPLO}
    ΔX = H * ΔY

    ΔH = @thunk begin
        ΔH = similar(parent(H))
        selupd!(ΔH, ΔY, X', 1 / 2, 0)
        selupd!(ΔH, X, ΔY', 1 / 2, 1)
        ΔH
    end

    return ΔH, ΔX
end

function mul_rrule_impl(S::SymTri{UPLO}, X::AbstractVecOrMat, Y::AbstractVecOrMat, ΔY::AbstractVecOrMat) where {UPLO}
    ΔX = S * ΔY

    ΔS = @thunk begin
        ΔS = similar(parent(S))
        selupd!(ΔS, ΔY, transpose(X), 1 / 2, 0)
        selupd!(ΔS, X, transpose(ΔY), 1 / 2, 1)
        ΔS
    end

    return ΔS, ΔX
end

function mul_rrule(H::HermOrSymTri, X::AbstractVecOrMat)
    Y = H * X

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔH, ΔX = mul_rrule_impl(H, X, Y, ΔY)
            return NoTangent(), ΔH, ΔX
        end
    end

    return Y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(*), H::HermTri, X::AbstractVecOrMat)
    return mul_rrule(H, X)
end

function ChainRulesCore.rrule(::typeof(*), S::SymTri, X::AbstractVecOrMat)
    return mul_rrule(S, X)
end

# type ambiguity with ChainRules
function ChainRulesCore.rrule(::typeof(*), H::HermTri{<:Any, <:RealOrComplex}, X::AbstractVecOrMat{<:RealOrComplex})
    return mul_rrule(H, X)
end

function ChainRulesCore.rrule(::typeof(*), S::SymTri{<:Any, <:RealOrComplex}, X::AbstractVecOrMat{<:RealOrComplex})
    return mul_rrule(S, X)
end
