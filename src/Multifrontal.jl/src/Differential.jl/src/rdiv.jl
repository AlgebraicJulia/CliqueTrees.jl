# Kernel functions for X / L

# ===== frule =====

function rdiv_frule_impl(X::AbstractMatrix, L::ChordalTriangular{:N}, dX, dL)
    Y = X / L
    return Y, (dX - Y * dL) / L
end

function ChainRulesCore.frule((_, dX, dL)::Tuple, ::typeof(/), X::AbstractMatrix, L::ChordalTriangular{:N})
    return rdiv_frule_impl(X, L, dX, dL)
end

# ===== rrule =====

function rdiv_rrule_impl(X::AbstractMatrix, L::ChordalTriangular{:N}, Y::AbstractMatrix, ΔY::AbstractMatrix)
    ΔX = ΔY / L'

    ΔL = @thunk begin
        ΔL = similar(L)
        selupd!(ΔL, Y', ΔX, -1, 0)
        ΔL
    end

    return ΔX, ΔL
end

function rdiv_rrule(X, L::ChordalTriangular{:N})
    Y = X / L

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔX, ΔL = rdiv_rrule_impl(X, L, Y, ΔY)
            return NoTangent(), ΔX, ΔL
        end
    end

    return Y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(/), X::AbstractMatrix, L::ChordalTriangular{:N})
    return rdiv_rrule(X, L)
end

# type ambiguity with ChainRules
function ChainRulesCore.rrule(::typeof(/), X::AbstractMatrix{<:Real}, L::ChordalTriangular{:N, <:Any, <:Real})
    return rdiv_rrule(X, L)
end
