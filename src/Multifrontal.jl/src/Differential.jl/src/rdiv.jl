# Kernel functions for X / L, X / L', X / transpose(L)

# ===== frule_impl =====

function rdiv_frule_impl(X::AbstractMatrix, A::MaybeAdjOrTransTri{:N}, dX, dL)
    Y = X / A
    return Y, (dX - Y * ProjectTo(A)(dL)) / A
end

# ===== rrule_impl =====

function rdiv_rrule_impl(X::AbstractMatrix, A::MaybeAdjOrTransTri{:N}, Y::AbstractMatrix, ΔY::AbstractMatrix)
    ΔX = ΔY / A'

    ΔA = @thunk begin
        ΔA = ProjectTo(A)(similar(parent(A)))
        selupd!(ΔA, Y', ΔX, -1, 0)
        parent(ΔA)
    end

    return ΔX, ΔA
end

# ===== rrule helper =====

function rdiv_rrule(X, A)
    Y = X / A

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔX, ΔL = rdiv_rrule_impl(X, A, Y, ΔY)
            return NoTangent(), ΔX, ΔL
        end
    end

    return Y, pullback ∘ unthunk
end

# ===== frule / rrule =====

for T in (ChordalTriangular{:N}, AdjTri{:N}, TransTri{:N})
    @eval function ChainRulesCore.frule((_, dX, dL)::Tuple, ::typeof(/), X::AbstractMatrix, A::$T)
        return rdiv_frule_impl(X, A, dX, dL)
    end

    @eval function ChainRulesCore.rrule(::typeof(/), X::AbstractMatrix, A::$T)
        return rdiv_rrule(X, A)
    end

    # type ambiguity with ChainRules
    @eval function ChainRulesCore.rrule(::typeof(/), X::AbstractMatrix{<:Real}, A::$T{<:Any, <:Real})
        return rdiv_rrule(X, A)
    end
end
