# Kernel functions for X * P where P is a Permutation

# ===== frule =====

function mul_frule_impl(X::AbstractVecOrMat, P::Permutation, dX)
    return X * P, dX * P
end

function ChainRulesCore.frule((_, dX, dP)::Tuple, ::typeof(*), X::AbstractVecOrMat, P::Permutation)
    return mul_frule_impl(X, P, dX)
end

# ===== rrule =====

function mul_rrule(X::AbstractVecOrMat, P::Permutation)
    Y = X * P

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), NoTangent()
        else
            ΔX = ΔY / P
            return NoTangent(), ΔX, NoTangent()
        end
    end

    return Y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(*), X::AbstractVecOrMat, P::Permutation)
    return mul_rrule(X, P)
end

function ChainRulesCore.rrule(::typeof(*), X::AbstractVecOrMat{<:RealOrComplex}, P::Permutation)
    return mul_rrule(X, P)
end
