# Kernel functions for L' * X and transpose(L) * X

# ===== frule =====

function mul_frule_impl(A::AdjOrTransTri{:N}, X::AbstractVecOrMat, dL, dX)
    if A isa AdjTri
        dAt = adjoint(dL)
    else
        dAt = transpose(dL)
    end

    return A * X, dAt * X + A * dX
end

function ChainRulesCore.frule((_, dL, dX)::Tuple, ::typeof(*), A::AdjTri{:N}, X::AbstractVecOrMat)
    return mul_frule_impl(A, X, dL, dX)
end

function ChainRulesCore.frule((_, dL, dX)::Tuple, ::typeof(*), A::TransTri{:N}, X::AbstractVecOrMat)
    return mul_frule_impl(A, X, dL, dX)
end

# ===== rrule =====

function mul_rrule_impl(A::AdjOrTransTri{:N}, X::AbstractVecOrMat, Y::AbstractVecOrMat, ΔY::AbstractVecOrMat)
    L = parent(A)
    ΔX = L * ΔY

    if A isa AdjTri
        ΔYt = adjoint(ΔY)
    else
        ΔYt = transpose(ΔY)
    end

    ΔL = @thunk begin
        ΔL = similar(L)
        selupd!(ΔL, X, ΔYt, 1, 0)
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
