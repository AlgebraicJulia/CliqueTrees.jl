# Kernel functions for X * L' and X * transpose(L)

# ===== frule =====

# AdjTri
function mul_frule_impl(X::AbstractMatrix, A::AdjTri{:N, UPLO}, dX::AbstractMatrix, dL::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, dL)
    Y = X * A
    dY = dX * A + X * dL'
    return Y, dY
end

function mul_frule_impl(X::AbstractMatrix, A::AdjTri{:N, UPLO}, dX::ZeroTangent, dL::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, dL)
    Y = X * A
    dY = X * dL'
    return Y, dY
end

# TransTri
function mul_frule_impl(X::AbstractMatrix, A::TransTri{:N, UPLO}, dX::AbstractMatrix, dL::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, dL)
    Y = X * A
    dY = dX * A + X * transpose(dL)
    return Y, dY
end

function mul_frule_impl(X::AbstractMatrix, A::TransTri{:N, UPLO}, dX::ZeroTangent, dL::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, dL)
    Y = X * A
    dY = X * transpose(dL)
    return Y, dY
end

# L const
function mul_frule_impl(X::AbstractMatrix, A::AdjOrTransTri, dX::AbstractMatrix, dL::ZeroTangent)
    Y = X * A
    dY = dX * A
    return Y, dY
end

# Both const
function mul_frule_impl(X::AbstractMatrix, A::AdjOrTransTri, dX::ZeroTangent, dL::ZeroTangent)
    return X * A, ZeroTangent()
end

function ChainRulesCore.frule((_, dX, dL)::Tuple, ::typeof(*), X::AbstractMatrix, A::AdjTri{:N})
    return mul_frule_impl(X, A, dX, dL)
end

function ChainRulesCore.frule((_, dX, dL)::Tuple, ::typeof(*), X::AbstractMatrix, A::TransTri{:N})
    return mul_frule_impl(X, A, dX, dL)
end

# ===== rrule =====

function mul_rrule_impl(X::AbstractMatrix, A::AdjTri{:N}, Y::AbstractMatrix, ΔY::AbstractMatrix)
    L = parent(A)
    ΔX = ΔY * L

    ΔL = @thunk begin
        ΔL = similar(L)
        selupd!(ΔL, ΔY', X, 1, 0)
        ΔL
    end

    return ΔX, ΔL
end

function mul_rrule_impl(X::AbstractMatrix, A::TransTri{:N}, Y::AbstractMatrix, ΔY::AbstractMatrix)
    L = parent(A)
    ΔX = ΔY * L

    ΔL = @thunk begin
        ΔL = similar(L)
        selupd!(ΔL, transpose(ΔY), X, 1, 0)
        ΔL
    end

    return ΔX, ΔL
end

function mul_rrule(X::AbstractMatrix, A::AdjOrTransTri{:N})
    Y = X * A

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔX, ΔL = mul_rrule_impl(X, A, Y, ΔY)
            return NoTangent(), ΔX, ΔL
        end
    end

    return Y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(*), X::AbstractMatrix, A::AdjTri{:N})
    return mul_rrule(X, A)
end

function ChainRulesCore.rrule(::typeof(*), X::AbstractMatrix, A::TransTri{:N})
    return mul_rrule(X, A)
end

# type ambiguity with ChainRules
function ChainRulesCore.rrule(::typeof(*), X::AbstractMatrix{<:RealOrComplex}, A::AdjTri{:N, <:Any, <:RealOrComplex})
    return mul_rrule(X, A)
end

function ChainRulesCore.rrule(::typeof(*), X::AbstractMatrix{<:RealOrComplex}, A::TransTri{:N, <:Any, <:RealOrComplex})
    return mul_rrule(X, A)
end
