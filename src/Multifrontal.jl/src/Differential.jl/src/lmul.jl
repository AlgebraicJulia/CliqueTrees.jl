# Kernel functions for L * X

# ===== frule =====

function mul_frule_impl(L::ChordalTriangular{:N}, X::AbstractVecOrMat, dL, dX)
    return L * X, dL * X + L * dX
end

function ChainRulesCore.frule((_, dL, dX)::Tuple, ::typeof(*), L::ChordalTriangular{:N}, X::AbstractVecOrMat)
    return mul_frule_impl(L, X, dL, dX)
end

# ===== rrule =====

function mul_rrule_impl(L::ChordalTriangular{:N}, X::AbstractVecOrMat, Y::AbstractVecOrMat, ΔY::AbstractVecOrMat)
    ΔX = L' * ΔY

    ΔL = @thunk begin
        ΔL = similar(L)
        selupd!(ΔL, ΔY, X', 1, 0)
        ΔL
    end

    return ΔL, ΔX
end

function mul_rrule(L::ChordalTriangular{:N}, X::AbstractVecOrMat)
    Y = L * X

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔL, ΔX = mul_rrule_impl(L, X, Y, ΔY)
            return NoTangent(), ΔL, ΔX
        end
    end

    return Y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(*), L::ChordalTriangular{:N}, X::AbstractVecOrMat)
    return mul_rrule(L, X)
end

# type ambiguity with ChainRules
function ChainRulesCore.rrule(::typeof(*), L::ChordalTriangular{:N, <:Any, <:RealOrComplex}, X::AbstractVecOrMat{<:RealOrComplex})
    return mul_rrule(L, X)
end
