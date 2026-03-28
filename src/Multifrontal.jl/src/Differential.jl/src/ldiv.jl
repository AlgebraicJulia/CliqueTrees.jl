# Kernel functions for L \ X, L' \ X, transpose(L) \ X

# ===== frule_impl =====

function ldiv_frule_impl(A::MaybeAdjOrTransTri{:N}, X::AbstractVecOrMat, dL, dX)
    Y = A \ X
    return Y, A \ (dX - ProjectTo(A)(dL) * Y)
end

# ===== rrule_impl =====

function ldiv_rrule_impl(A::MaybeAdjOrTransTri{:N}, X::AbstractVecOrMat, Y::AbstractVecOrMat, ΔY::AbstractVecOrMat)
    ΔX = A' \ ΔY

    ΔL = @thunk begin
        ΔA = ProjectTo(A)(similar(parent(A)))
        selupd!(ΔA, ΔX, Y', -1, 0)
        parent(ΔA)
    end

    return ΔL, ΔX
end

# ===== rrule helper =====

function ldiv_rrule(A, X)
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

# ===== frule / rrule =====

for T in (ChordalTriangular{:N}, AdjTri{:N}, TransTri{:N})
    @eval function ChainRulesCore.frule((_, dL, dX)::Tuple, ::typeof(\), A::$T, X::AbstractVecOrMat)
        return ldiv_frule_impl(A, X, dL, dX)
    end

    @eval function ChainRulesCore.rrule(::typeof(\), A::$T, X::AbstractVecOrMat)
        return ldiv_rrule(A, X)
    end

    # type ambiguity with ChainRules
    @eval function ChainRulesCore.rrule(::typeof(\), A::$T{<:Any, <:Real}, X::AbstractVecOrMat{<:Real})
        return ldiv_rrule(A, X)
    end
end
