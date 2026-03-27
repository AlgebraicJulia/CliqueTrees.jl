# Kernel functions for ldiv (L \ X)

# ===== frule =====

function ldiv_frule_impl(L::ChordalTriangular{:N}, X::AbstractVecOrMat, dL, dX)
    Y = L \ X
    return Y, L \ (dX - dL * Y)
end

function ChainRulesCore.frule((_, dL, dX)::Tuple, ::typeof(\), L::ChordalTriangular{:N}, X::AbstractVecOrMat)
    return ldiv_frule_impl(L, X, dL, dX)
end

# ===== rrule =====

function ldiv_rrule_impl(L::ChordalTriangular{:N}, X::AbstractVecOrMat, Y::AbstractVecOrMat, ΔY::AbstractVecOrMat)
    ΔX = L' \ ΔY

    ΔL = @thunk begin
        ΔL = similar(L)
        selupd!(ΔL, ΔX, Y', -1, 0)
        ΔL
    end

    return ΔL, ΔX
end

function ldiv_rrule(L::ChordalTriangular{:N}, X::AbstractVecOrMat)
    Y = L \ X

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔL, ΔX = ldiv_rrule_impl(L, X, Y, ΔY)
            return NoTangent(), ΔL, ΔX
        end
    end

    return Y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(\), L::ChordalTriangular{:N}, X::AbstractVecOrMat)
    return ldiv_rrule(L, X)
end

# type ambiguity with ChainRules
function ChainRulesCore.rrule(::typeof(\), L::ChordalTriangular{:N, <:Any, <:Real}, X::AbstractVecOrMat{<:Real})
    return ldiv_rrule(L, X)
end
