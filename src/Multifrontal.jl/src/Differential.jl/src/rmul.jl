# Kernel functions for X * L

# ===== frule =====

function mul_frule_impl(X::AbstractMatrix, L::ChordalTriangular{:N, UPLO}, dX::AbstractMatrix, dL::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(L, dL)
    Y = X * L
    dY = dX * L + X * dL
    return Y, dY
end

function mul_frule_impl(X::AbstractMatrix, L::ChordalTriangular{:N, UPLO}, dX::ZeroTangent, dL::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(L, dL)
    Y = X * L
    dY = X * dL
    return Y, dY
end

function mul_frule_impl(X::AbstractMatrix, L::ChordalTriangular{:N, UPLO}, dX::AbstractMatrix, dL::ZeroTangent) where {UPLO}
    Y = X * L
    dY = dX * L
    return Y, dY
end

function mul_frule_impl(X::AbstractMatrix, L::ChordalTriangular, dX::ZeroTangent, dL::ZeroTangent)
    return X * L, ZeroTangent()
end

function ChainRulesCore.frule((_, dX, dL)::Tuple, ::typeof(*), X::AbstractMatrix, L::ChordalTriangular{:N})
    return mul_frule_impl(X, L, dX, dL)
end

# ===== rrule =====

function mul_rrule_impl(X::AbstractMatrix, L::ChordalTriangular{:N}, Y::AbstractMatrix, ΔY::AbstractMatrix)
    ΔX = ΔY * L'
    ΔL = @thunk begin
        ΔL = similar(L)
        selupd!(ΔL, X', ΔY, 1, 0)
        ΔL
    end
    return ΔX, ΔL
end

function mul_rrule(X::AbstractMatrix, L::ChordalTriangular{:N})
    Y = X * L

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔX, ΔL = mul_rrule_impl(X, L, Y, ΔY)
            return NoTangent(), ΔX, ΔL
        end
    end

    return Y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(*), X::AbstractMatrix, L::ChordalTriangular{:N})
    return mul_rrule(X, L)
end

# type ambiguity with ChainRules
function ChainRulesCore.rrule(::typeof(*), X::AbstractMatrix{<:RealOrComplex}, L::ChordalTriangular{:N, <:Any, <:RealOrComplex})
    return mul_rrule(X, L)
end
