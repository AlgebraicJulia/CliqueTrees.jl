# Kernel functions for P \ X where P is a Permutation

# ===== frule =====

function ldiv_frule_impl(P::Permutation, X::AbstractVecOrMat, dX)
    return P \ X, P \ dX
end

function ChainRulesCore.frule((_, dP, dX)::Tuple, ::typeof(\), P::Permutation, X::AbstractVecOrMat)
    return ldiv_frule_impl(P, X, dX)
end

# ===== rrule =====

function ldiv_rrule(P::Permutation, X::AbstractVecOrMat)
    Y = P \ X

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), NoTangent(), ZeroTangent()
        else
            ΔX = P * ΔY
            return NoTangent(), NoTangent(), ΔX
        end
    end

    return Y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(\), P::Permutation, X::AbstractVecOrMat)
    return ldiv_rrule(P, X)
end

function ChainRulesCore.rrule(::typeof(\), P::Permutation, X::AbstractVecOrMat{<:RealOrComplex})
    return ldiv_rrule(P, X)
end
