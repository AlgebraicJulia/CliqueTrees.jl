# Kernel functions for X / L' and X / transpose(L)

# ===== frule =====

function rdiv_frule_impl(X::AbstractMatrix, A::AdjOrTransTri{:N}, dX, dL)
    Y = X / A

    if A isa AdjTri
        dAt = adjoint(dL)
    else
        dAt = transpose(dL)
    end

    return Y, (dX - Y * dAt) / A
end

function ChainRulesCore.frule((_, dX, dL)::Tuple, ::typeof(/), X::AbstractMatrix, A::AdjTri{:N})
    return rdiv_frule_impl(X, A, dX, dL)
end

function ChainRulesCore.frule((_, dX, dL)::Tuple, ::typeof(/), X::AbstractMatrix, A::TransTri{:N})
    return rdiv_frule_impl(X, A, dX, dL)
end

# ===== rrule =====

function rdiv_rrule_impl(X::AbstractMatrix, A::AdjOrTransTri{:N}, Y::AbstractMatrix, ΔY::AbstractMatrix)
    L = parent(A)
    ΔX = ΔY / L

    if A isa AdjTri
        ΔXt = adjoint(ΔX)
    else
        ΔXt = transpose(ΔX)
    end

    ΔL = @thunk begin
        ΔL = similar(L)
        selupd!(ΔL, ΔXt, Y, -1, 0)
        ΔL
    end

    return ΔX, ΔL
end

function rdiv_rrule(X::AbstractMatrix, A::AdjOrTransTri{:N})
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

function ChainRulesCore.rrule(::typeof(/), X::AbstractMatrix, A::AdjTri{:N})
    return rdiv_rrule(X, A)
end

function ChainRulesCore.rrule(::typeof(/), X::AbstractMatrix, A::TransTri{:N})
    return rdiv_rrule(X, A)
end

# type ambiguity with ChainRules
function ChainRulesCore.rrule(::typeof(/), X::AbstractMatrix{<:Real}, A::AdjTri{:N, <:Any, <:Real})
    return rdiv_rrule(X, A)
end

function ChainRulesCore.rrule(::typeof(/), X::AbstractMatrix{<:Real}, A::TransTri{:N, <:Any, <:Real})
    return rdiv_rrule(X, A)
end
