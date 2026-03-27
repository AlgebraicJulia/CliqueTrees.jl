# Kernel functions for ldiv with AdjTri (L' \ X) and TransTri (transpose(L) \ X)

# ===== frule =====

function ldiv_frule_impl(A::AdjOrTransTri{:N}, X::AbstractVecOrMat, dL, dX)
    Y = A \ X

    if A isa AdjTri
        dAt = adjoint(dL)
    else
        dAt = transpose(dL)
    end

    return Y, A \ (dX - dAt * Y)
end

function ChainRulesCore.frule((_, dL, dX)::Tuple, ::typeof(\), A::AdjTri{:N}, X::AbstractVecOrMat)
    return ldiv_frule_impl(A, X, dL, dX)
end

function ChainRulesCore.frule((_, dL, dX)::Tuple, ::typeof(\), A::TransTri{:N}, X::AbstractVecOrMat)
    return ldiv_frule_impl(A, X, dL, dX)
end

# ===== rrule =====

function ldiv_rrule_impl(A::AdjOrTransTri{:N}, X::AbstractVecOrMat, Y::AbstractVecOrMat, ΔY::AbstractVecOrMat)
    L = parent(A)
    ΔX = L \ ΔY

    if A isa AdjTri
        ΔXt = adjoint(ΔX)
    else
        ΔXt = transpose(ΔX)
    end

    ΔL = @thunk begin
        ΔL = similar(L)
        selupd!(ΔL, Y, ΔXt, -1, 0)
        ΔL
    end

    return ΔL, ΔX
end

function ldiv_rrule(A::AdjOrTransTri{:N}, X::AbstractVecOrMat)
    Y = A \ X

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔL, ΔX = ldiv_rrule_impl(A, X, Y, ΔY)
            return NoTangent(), ΔL, ΔX
        end
    end

    return Y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(\), A::AdjTri{:N}, X::AbstractVecOrMat)
    return ldiv_rrule(A, X)
end

function ChainRulesCore.rrule(::typeof(\), A::TransTri{:N}, X::AbstractVecOrMat)
    return ldiv_rrule(A, X)
end

# type ambiguity with ChainRules
function ChainRulesCore.rrule(::typeof(\), A::AdjTri{:N, <:Any, <:Real}, X::AbstractVecOrMat{<:Real})
    return ldiv_rrule(A, X)
end

function ChainRulesCore.rrule(::typeof(\), A::TransTri{:N, <:Any, <:Real}, X::AbstractVecOrMat{<:Real})
    return ldiv_rrule(A, X)
end
