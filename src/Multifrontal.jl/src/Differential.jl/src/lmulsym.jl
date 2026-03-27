# Kernel functions for H * X where H is HermTri or SymTri

# ===== frule =====

function mul_frule_impl(H::HermOrSymTri, X::AbstractVecOrMat, dH, dX)
    Y = H * X
    dY = ProjectTo(H)(dH) * X + H * dX
    return Y, dY
end

function ChainRulesCore.frule((_, dH, dX)::Tuple, ::typeof(*), H::HermTri, X::AbstractVecOrMat)
    return mul_frule_impl(H, X, dH, dX)
end

function ChainRulesCore.frule((_, dS, dX)::Tuple, ::typeof(*), S::SymTri, X::AbstractVecOrMat)
    return mul_frule_impl(S, X, dS, dX)
end

# ===== rrule =====

function mul_rrule_impl(H::HermOrSymTri, X::AbstractVecOrMat, Y::AbstractVecOrMat, ΔY::AbstractVecOrMat)
    ΔX = H * ΔY

    if H isa HermTri
        Xt = adjoint(X)
        ΔYt = adjoint(ΔY)
    else
        Xt = transpose(X)
        ΔYt = transpose(ΔY)
    end

    ΔH = @thunk begin
        ΔH = similar(parent(H))
        selupd!(ΔH, ΔY, Xt, 1 / 2, 0)
        selupd!(ΔH, X, ΔYt, 1 / 2, 1)
        ΔH
    end

    return ΔH, ΔX
end

function mul_rrule(H::HermOrSymTri, X::AbstractVecOrMat)
    Y = H * X

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔH, ΔX = mul_rrule_impl(H, X, Y, ΔY)
            return NoTangent(), ΔH, ΔX
        end
    end

    return Y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(*), H::HermTri, X::AbstractVecOrMat)
    return mul_rrule(H, X)
end

function ChainRulesCore.rrule(::typeof(*), S::SymTri, X::AbstractVecOrMat)
    return mul_rrule(S, X)
end

# type ambiguity with ChainRules
function ChainRulesCore.rrule(::typeof(*), H::HermTri{<:Any, <:RealOrComplex}, X::AbstractVecOrMat{<:RealOrComplex})
    return mul_rrule(H, X)
end

function ChainRulesCore.rrule(::typeof(*), S::SymTri{<:Any, <:RealOrComplex}, X::AbstractVecOrMat{<:RealOrComplex})
    return mul_rrule(S, X)
end
