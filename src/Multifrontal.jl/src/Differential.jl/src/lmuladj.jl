# Kernel functions for L' * X and transpose(L) * X

# ===== frule =====

# AdjTri
function mul_frule_impl(A::AdjTri{:N, UPLO}, X::AbstractVecOrMat, dL::ChordalTriangular{:N, UPLO}, dX::AbstractVecOrMat) where {UPLO}
    @assert checksymbolic(A, dL)
    Y = A * X
    dY = dL' * X + A * dX
    return Y, dY
end

function mul_frule_impl(A::AdjTri{:N, UPLO}, X::AbstractVecOrMat, dL::ChordalTriangular{:N, UPLO}, dX::ZeroTangent) where {UPLO}
    @assert checksymbolic(A, dL)
    Y = A * X
    dY = dL' * X
    return Y, dY
end

# TransTri
function mul_frule_impl(A::TransTri{:N, UPLO}, X::AbstractVecOrMat, dL::ChordalTriangular{:N, UPLO}, dX::AbstractVecOrMat) where {UPLO}
    @assert checksymbolic(A, dL)
    Y = A * X
    dY = transpose(dL) * X + A * dX
    return Y, dY
end

# L const
function mul_frule_impl(A::AdjOrTransTri, X::AbstractVecOrMat, dL::ZeroTangent, dX::AbstractVecOrMat)
    Y = A * X
    dY = A * dX
    return Y, dY
end

function mul_frule_impl(A::TransTri{:N, UPLO}, X::AbstractVecOrMat, dL::ChordalTriangular{:N, UPLO}, dX::ZeroTangent) where {UPLO}
    @assert checksymbolic(A, dL)
    Y = A * X
    dY = transpose(dL) * X
    return Y, dY
end

# Both const
function mul_frule_impl(A::AdjOrTransTri, X::AbstractVecOrMat, dL::ZeroTangent, dX::ZeroTangent)
    return A * X, ZeroTangent()
end

function ChainRulesCore.frule((_, dL, dX)::Tuple, ::typeof(*), A::AdjTri{:N}, X::AbstractVecOrMat)
    return mul_frule_impl(A, X, dL, dX)
end

function ChainRulesCore.frule((_, dL, dX)::Tuple, ::typeof(*), A::TransTri{:N}, X::AbstractVecOrMat)
    return mul_frule_impl(A, X, dL, dX)
end

# ===== rrule =====

function mul_rrule_impl(A::AdjTri{:N}, X::AbstractVecOrMat, Y::AbstractVecOrMat, ΔY::AbstractVecOrMat)
    L = parent(A)
    ΔX = L * ΔY

    ΔL = @thunk begin
        ΔL = similar(L)
        selupd!(ΔL, X, ΔY', 1, 0)
        ΔL
    end

    return ΔL, ΔX
end

function mul_rrule_impl(A::TransTri{:N}, X::AbstractVecOrMat, Y::AbstractVecOrMat, ΔY::AbstractVecOrMat)
    L = parent(A)
    ΔX = L * ΔY

    ΔL = @thunk begin
        ΔL = similar(L)
        selupd!(ΔL, X, transpose(ΔY), 1, 0)
        ΔL
    end

    return ΔL, ΔX
end

function mul_rrule(A::AdjOrTransTri{:N}, X::AbstractVecOrMat)
    Y = A * X

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔL, ΔX = mul_rrule_impl(A, X, Y, ΔY)
            return NoTangent(), ΔL, ΔX
        end
    end

    return Y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(*), A::AdjTri{:N}, X::AbstractVecOrMat)
    return mul_rrule(A, X)
end

function ChainRulesCore.rrule(::typeof(*), A::TransTri{:N}, X::AbstractVecOrMat)
    return mul_rrule(A, X)
end

# type ambiguity with ChainRules
function ChainRulesCore.rrule(::typeof(*), A::AdjTri{:N, <:Any, <:RealOrComplex}, X::AbstractVecOrMat{<:RealOrComplex})
    return mul_rrule(A, X)
end

function ChainRulesCore.rrule(::typeof(*), A::TransTri{:N, <:Any, <:RealOrComplex}, X::AbstractVecOrMat{<:RealOrComplex})
    return mul_rrule(A, X)
end
