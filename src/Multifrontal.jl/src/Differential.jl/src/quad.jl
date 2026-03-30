# ===== frule =====

function dot_frule_impl(X::StridedVecOrMat, A, Y::StridedVecOrMat, dX, dA, dY)
    y = dot(X, A, Y)
    dy = dot(dX, A, Y) + dot(X, A, dY) + dot(X, dA, Y)
    return y, dy
end

# ===== rrule =====

function dot_rrule_impl(X::StridedVecOrMat, A, Y::StridedVecOrMat, y, AX::StridedVecOrMat, AY::StridedVecOrMat, Δy)
    ΔX = @thunk Δy * AY
    ΔY = @thunk Δy * AX

    ΔA = @thunk begin
        ΔA = similar(A)
        selupd!(ΔA, X, Y, Δy / 2, 0)
        ΔA
    end

    return ΔX, ΔA, ΔY
end

function dot_rrule(X::StridedVecOrMat, A, Y::StridedVecOrMat)
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

# ===== dispatches =====

for T in (HermTri, SymTri, HermSparse, SymSparse, SparseMatrixCSC)
    @eval function ChainRulesCore.frule((_, dX, dA, dY)::Tuple, ::typeof(dot), X::StridedVecOrMat, A::$T, Y::StridedVecOrMat)
        return dot_frule_impl(X, A, Y, dX, dA, dY)
    end

    @eval function ChainRulesCore.rrule(::typeof(dot), X::StridedVecOrMat, A::$T, Y::StridedVecOrMat)
        return dot_rrule(X, A, Y)
    end
end
