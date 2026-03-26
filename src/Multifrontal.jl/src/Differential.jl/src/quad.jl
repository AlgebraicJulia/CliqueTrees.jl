# Kernel functions for dot(X, A, Y) where A is HermTri or SymTri

# ===== frule =====

function dot_frule_impl(X::AbstractVecOrMat, A::HermTri{UPLO}, Y::AbstractVecOrMat, dX::AbstractVecOrMat, dA::ChordalTriangular{:N, UPLO}, dY::AbstractVecOrMat) where {UPLO}
    @assert checksymbolic(A, dA)
    y = dot(X, A, Y)
    dy = dot(dX, A, Y) + dot(X, A, dY) + dot(X, Hermitian(dA, UPLO), Y)
    return y, dy
end

function dot_frule_impl(X::AbstractVecOrMat, A::SymTri{UPLO}, Y::AbstractVecOrMat, dX::AbstractVecOrMat, dA::ChordalTriangular{:N, UPLO}, dY::AbstractVecOrMat) where {UPLO}
    @assert checksymbolic(A, dA)
    y = dot(X, A, Y)
    dy = dot(dX, A, Y) + dot(X, A, dY) + dot(X, Symmetric(dA, UPLO), Y)
    return y, dy
end

# X const
function dot_frule_impl(X::AbstractVecOrMat, A::HermTri{UPLO}, Y::AbstractVecOrMat, dX::ZeroTangent, dA::ChordalTriangular{:N, UPLO}, dY::AbstractVecOrMat) where {UPLO}
    @assert checksymbolic(A, dA)
    y = dot(X, A, Y)
    dy = dot(X, A, dY) + dot(X, Hermitian(dA, UPLO), Y)
    return y, dy
end

function dot_frule_impl(X::AbstractVecOrMat, A::SymTri{UPLO}, Y::AbstractVecOrMat, dX::ZeroTangent, dA::ChordalTriangular{:N, UPLO}, dY::AbstractVecOrMat) where {UPLO}
    @assert checksymbolic(A, dA)
    y = dot(X, A, Y)
    dy = dot(X, A, dY) + dot(X, Symmetric(dA, UPLO), Y)
    return y, dy
end

# A const
function dot_frule_impl(X::AbstractVecOrMat, A::HermOrSymTri, Y::AbstractVecOrMat, dX::AbstractVecOrMat, dA::ZeroTangent, dY::AbstractVecOrMat)
    y = dot(X, A, Y)
    dy = dot(dX, A, Y) + dot(X, A, dY)
    return y, dy
end

# Y const
function dot_frule_impl(X::AbstractVecOrMat, A::HermTri{UPLO}, Y::AbstractVecOrMat, dX::AbstractVecOrMat, dA::ChordalTriangular{:N, UPLO}, dY::ZeroTangent) where {UPLO}
    @assert checksymbolic(A, dA)
    y = dot(X, A, Y)
    dy = dot(dX, A, Y) + dot(X, Hermitian(dA, UPLO), Y)
    return y, dy
end

function dot_frule_impl(X::AbstractVecOrMat, A::SymTri{UPLO}, Y::AbstractVecOrMat, dX::AbstractVecOrMat, dA::ChordalTriangular{:N, UPLO}, dY::ZeroTangent) where {UPLO}
    @assert checksymbolic(A, dA)
    y = dot(X, A, Y)
    dy = dot(dX, A, Y) + dot(X, Symmetric(dA, UPLO), Y)
    return y, dy
end

# X, A const
function dot_frule_impl(X::AbstractVecOrMat, A::HermOrSymTri, Y::AbstractVecOrMat, dX::ZeroTangent, dA::ZeroTangent, dY::AbstractVecOrMat)
    return dot(X, A, Y), dot(X, A, dY)
end

# X, Y const
function dot_frule_impl(X::AbstractVecOrMat, A::HermTri{UPLO}, Y::AbstractVecOrMat, dX::ZeroTangent, dA::ChordalTriangular{:N, UPLO}, dY::ZeroTangent) where {UPLO}
    @assert checksymbolic(A, dA)
    return dot(X, A, Y), dot(X, Hermitian(dA, UPLO), Y)
end

function dot_frule_impl(X::AbstractVecOrMat, A::SymTri{UPLO}, Y::AbstractVecOrMat, dX::ZeroTangent, dA::ChordalTriangular{:N, UPLO}, dY::ZeroTangent) where {UPLO}
    @assert checksymbolic(A, dA)
    return dot(X, A, Y), dot(X, Symmetric(dA, UPLO), Y)
end

# A, Y const
function dot_frule_impl(X::AbstractVecOrMat, A::HermOrSymTri, Y::AbstractVecOrMat, dX::AbstractVecOrMat, dA::ZeroTangent, dY::ZeroTangent)
    return dot(X, A, Y), dot(dX, A, Y)
end

# X, A, Y const
function dot_frule_impl(X::AbstractVecOrMat, A::HermOrSymTri, Y::AbstractVecOrMat, dX::ZeroTangent, dA::ZeroTangent, dY::ZeroTangent)
    return dot(X, A, Y), ZeroTangent()
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

function dot_rrule_impl(X::AbstractVecOrMat, A::HermTri{UPLO}, Y::AbstractVecOrMat, y, AX::AbstractVecOrMat, AY::AbstractVecOrMat, Δy) where {UPLO}
    ΔX = @thunk Δy * AY
    ΔY = @thunk Δy * AX
    ΔA = @thunk begin
        ΔA = similar(parent(A))
        selupd!(ΔA, X, Y', Δy / 2, 0)
        selupd!(ΔA, Y, X', Δy / 2, 1)
        ΔA
    end
    return ΔX, ΔA, ΔY
end

function dot_rrule_impl(X::AbstractVecOrMat, A::SymTri{UPLO}, Y::AbstractVecOrMat, y, AX::AbstractVecOrMat, AY::AbstractVecOrMat, Δy) where {UPLO}
    ΔX = @thunk Δy * AY
    ΔY = @thunk Δy * AX
    ΔA = @thunk begin
        ΔA = similar(parent(A))
        selupd!(ΔA, X, transpose(Y), Δy / 2, 0)
        selupd!(ΔA, Y, transpose(X), Δy / 2, 1)
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
