# Kernel functions for dot(X, A, Y) where A is HermTri or SymTri

# ===== frule =====

function dot_frule_impl(X::AbstractVecOrMat, A::HermOrSymTri, Y::AbstractVecOrMat, dX, dA, dY)
    y = dot(X, A, Y)
    dy = dot(dX, A, Y) + dot(X, A, dY) + dot(X, ProjectTo(A)(dA), Y)
    return y, dy
end

function ChainRulesCore.frule((_, dX, dA, dY)::Tuple, ::typeof(dot), X::AbstractVecOrMat, A::HermTri, Y::AbstractVecOrMat)
    return dot_frule_impl(X, A, Y, dX, dA, dY)
end

function ChainRulesCore.frule((_, dX, dA, dY)::Tuple, ::typeof(dot), X::AbstractVecOrMat, A::SymTri, Y::AbstractVecOrMat)
    return dot_frule_impl(X, A, Y, dX, dA, dY)
end

# type ambiguity with ChainRules
function ChainRulesCore.frule((_, dX, dA, dY)::Tuple, ::typeof(dot), X::AbstractVector{<:Number}, A::HermTri{<:Any, <:Number}, Y::AbstractVector{<:Number})
    return dot_frule_impl(X, A, Y, dX, dA, dY)
end

function ChainRulesCore.frule((_, dX, dA, dY)::Tuple, ::typeof(dot), X::AbstractVector{<:Number}, A::SymTri{<:Any, <:Number}, Y::AbstractVector{<:Number})
    return dot_frule_impl(X, A, Y, dX, dA, dY)
end

# ===== rrule =====

function dot_rrule_impl(X::AbstractVecOrMat, A::HermOrSymTri, Y::AbstractVecOrMat, y, AX::AbstractVecOrMat, AY::AbstractVecOrMat, Δy)
    ΔX = @thunk Δy * AY
    ΔY = @thunk Δy * AX

    if A isa HermTri
        Xt = adjoint(X)
        Yt = adjoint(Y)
    else
        Xt = transpose(X)
        Yt = transpose(Y)
    end

    ΔA = @thunk begin
        ΔA = similar(parent(A))
        selupd!(ΔA, X, Yt, Δy / 2, 0)
        selupd!(ΔA, Y, Xt, Δy / 2, 1)
        ΔA
    end

    return ΔX, ΔA, ΔY
end

function dot_rrule(X::AbstractVecOrMat, A::HermOrSymTri, Y::AbstractVecOrMat)
    AY = A * Y
    AX = A * X
    y = dot(X, AY)

    function pullback(Δy)
        if Δy isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔX, ΔA, ΔY = dot_rrule_impl(X, A, Y, y, AX, AY, Δy)
            return NoTangent(), ΔX, ΔA, ΔY
        end
    end

    return y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(dot), X::AbstractVecOrMat, A::HermTri, Y::AbstractVecOrMat)
    return dot_rrule(X, A, Y)
end

function ChainRulesCore.rrule(::typeof(dot), X::AbstractVecOrMat, A::SymTri, Y::AbstractVecOrMat)
    return dot_rrule(X, A, Y)
end

# type ambiguity with ChainRules
function ChainRulesCore.rrule(::typeof(dot), X::AbstractVector{<:Number}, A::HermTri{<:Any, <:Number}, Y::AbstractVector{<:Number})
    return dot_rrule(X, A, Y)
end

function ChainRulesCore.rrule(::typeof(dot), X::AbstractVector{<:Number}, A::SymTri{<:Any, <:Number}, Y::AbstractVector{<:Number})
    return dot_rrule(X, A, Y)
end
