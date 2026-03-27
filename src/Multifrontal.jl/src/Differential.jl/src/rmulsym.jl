# Kernel functions for X * H where H is HermTri or SymTri

# ===== frule =====

function mul_frule_impl(X::AbstractMatrix, H::HermOrSymTri, dX, dH)
    Y = X * H
    dY = dX * H + X * ProjectTo(H)(dH)
    return Y, dY
end

function ChainRulesCore.frule((_, dX, dH)::Tuple, ::typeof(*), X::AbstractMatrix, H::HermTri)
    return mul_frule_impl(X, H, dX, dH)
end

function ChainRulesCore.frule((_, dX, dS)::Tuple, ::typeof(*), X::AbstractMatrix, S::SymTri)
    return mul_frule_impl(X, S, dX, dS)
end

# ===== rrule =====

function mul_rrule_impl(X::AbstractMatrix, H::HermOrSymTri, Y::AbstractMatrix, ΔY::AbstractMatrix)
    ΔX = ΔY * H

    if H isa HermTri
        Xt = adjoint(X)
        ΔYt = adjoint(ΔY)
    else
        Xt = transpose(X)
        ΔYt = transpose(ΔY)
    end

    ΔH = @thunk begin
        ΔH = similar(parent(H))
        selupd!(ΔH, Xt, ΔY, 1 / 2, 0)
        selupd!(ΔH, ΔYt, X, 1 / 2, 1)
        ΔH
    end

    return ΔX, ΔH
end

function mul_rrule(X::AbstractMatrix, H::HermOrSymTri)
    Y = X * H

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔX, ΔH = mul_rrule_impl(X, H, Y, ΔY)
            return NoTangent(), ΔX, ΔH
        end
    end

    return Y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(*), X::AbstractMatrix, H::HermTri)
    return mul_rrule(X, H)
end

function ChainRulesCore.rrule(::typeof(*), X::AbstractMatrix, S::SymTri)
    return mul_rrule(X, S)
end

# type ambiguity with ChainRules
function ChainRulesCore.rrule(::typeof(*), X::AbstractMatrix{<:RealOrComplex}, H::HermTri{<:Any, <:RealOrComplex})
    return mul_rrule(X, H)
end

function ChainRulesCore.rrule(::typeof(*), X::AbstractMatrix{<:RealOrComplex}, S::SymTri{<:Any, <:RealOrComplex})
    return mul_rrule(X, S)
end
