# Kernel functions for L * X, L' * X, transpose(L) * X, x * A, P * X

# ===== rrule_impl =====

function mul_rrule_impl(A::AbstractMatrix, X::StridedVecOrMat, Y::StridedVecOrMat, ΔY)
    ΔX = A' * ΔY

    if ΔY isa ZeroTangent
        ΔL = ZeroTangent()
    else
        ΔL = @thunk begin
            ΔA = fwrap(similar, A)
            selupd!(ΔA, ΔY, X', 1, 0)
            ΔA
        end
    end

    return ΔL, ΔX
end

function mul_rrule_impl(H::HermOrSym, X::StridedVecOrMat, Y::StridedVecOrMat, ΔY)
    ΔX = H * ΔY

    if ΔY isa ZeroTangent
        ΔH = ZeroTangent()
    else
        ΔH = @thunk begin
            ΔH = similar(H)
            selupd!(ΔH, ΔY, X, 1 / 2, 0)
            ΔH
        end
    end

    return ΔH, ΔX
end

function mul_rrule_impl(x::Number, A::AbstractMatrix, y::AbstractMatrix, Δy)
    Δx = @thunk dot(A, Δy)
    ΔA = @thunk conj(x) * Δy
    return Δx, ΔA
end

function mul_rrule_impl(P::Permutation, X::StridedVecOrMat, Y::StridedVecOrMat, ΔY)
    return NoTangent(), P \ ΔY
end

# ===== frule / rrule =====

# A * X

for T in (ChordalTriangular{:N}, AdjTri{:N}, TransTri{:N}, HermTri, SymTri)
    @eval function ChainRulesCore.frule((_, dA, dX)::Tuple, ::typeof(*), A::$T, X::StridedVecOrMat)
        return mul_frule_impl(A, X, dA, dX)
    end

    @eval function ChainRulesCore.rrule(::typeof(*), A::$T, X::StridedVecOrMat)
        return mul_rrule(A, X)
    end
end

# x * A (scalar)

for T in (ChordalTriangular{:N}, HermTri, SymTri)
    @eval function ChainRulesCore.frule((_, dx, dA)::Tuple, ::typeof(*), x::Number, A::$T)
        return mul_frule_impl(x, A, dx, dA)
    end

    @eval function ChainRulesCore.rrule(::typeof(*), x::Number, A::$T)
        return mul_rrule(x, A)
    end
end

# P * X (Permutation)

function ChainRulesCore.frule((_, dP, dX)::Tuple, ::typeof(*), P::Permutation, X::StridedVecOrMat)
    return mul_frule_impl(P, X, dP, dX)
end

function ChainRulesCore.rrule(::typeof(*), P::Permutation, X::StridedVecOrMat)
    return mul_rrule(P, X)
end
