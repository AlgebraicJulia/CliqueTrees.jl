# Kernel functions for X * L' and X * transpose(L)

# ===== frule =====

function mul_frule_impl(X::AbstractMatrix, A::AdjOrTransTri{:N}, dX, dL)
    if A isa AdjTri
        dAt = adjoint(dL)
    else
        dAt = transpose(dL)
    end

    return X * A, dX * A + X * dAt
end

function ChainRulesCore.frule((_, dX, dL)::Tuple, ::typeof(*), X::AbstractMatrix, A::AdjTri{:N})
    return mul_frule_impl(X, A, dX, dL)
end

function ChainRulesCore.frule((_, dX, dL)::Tuple, ::typeof(*), X::AbstractMatrix, A::TransTri{:N})
    return mul_frule_impl(X, A, dX, dL)
end

# ===== rrule =====

function mul_rrule_impl(X::AbstractMatrix, A::AdjOrTransTri{:N}, Y::AbstractMatrix, ΔY::AbstractMatrix)
    L = parent(A)
    ΔX = ΔY * L

    if A isa AdjTri
        ΔYt = adjoint(ΔY)
    else
        ΔYt = transpose(ΔY)
    end

    ΔL = @thunk begin
        ΔL = similar(L)
        selupd!(ΔL, ΔYt, X, 1, 0)
        ΔL
    end

    return ΔX, ΔL
end

function mul_rrule(X::AbstractMatrix, A::AdjOrTransTri{:N})
    Y = X * A

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔX, ΔL = mul_rrule_impl(X, A, Y, ΔY)
            return NoTangent(), ΔX, ΔL
        end
    end

    return Y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(*), X::AbstractMatrix, A::AdjTri{:N})
    return mul_rrule(X, A)
end

function ChainRulesCore.rrule(::typeof(*), X::AbstractMatrix, A::TransTri{:N})
    return mul_rrule(X, A)
end

# type ambiguity with ChainRules
function ChainRulesCore.rrule(::typeof(*), X::AbstractMatrix{<:RealOrComplex}, A::AdjTri{:N, <:Any, <:RealOrComplex})
    return mul_rrule(X, A)
end

function ChainRulesCore.rrule(::typeof(*), X::AbstractMatrix{<:RealOrComplex}, A::TransTri{:N, <:Any, <:RealOrComplex})
    return mul_rrule(X, A)
end
