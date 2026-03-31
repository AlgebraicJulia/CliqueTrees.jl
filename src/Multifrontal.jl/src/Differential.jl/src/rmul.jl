# Kernel functions for X * L, X * L', X * transpose(L), A * x, X * P

# ===== rrule_impl =====

function mul_rrule_impl(X::StridedMatrix, A::AbstractMatrix, Y::StridedMatrix, ΔY)
    ΔX = ΔY * A'

    if ΔY isa ZeroTangent
        ΔL = ZeroTangent()
    else
        ΔL = @thunk begin
            ΔA = fwrap(similar, A)
            selupd!(ΔA, X', ΔY, 1, 0)
            ΔA
        end
    end

    return ΔX, ΔL
end

function mul_rrule_impl(A::AbstractMatrix, x::Number, y::AbstractMatrix, Δy)
    ΔA = @thunk Δy * conj(x)
    Δx = @thunk dot(A, Δy)
    return ΔA, Δx
end

function mul_rrule_impl(X::StridedMatrix, P::Permutation, Y::StridedMatrix, ΔY)
    return ΔY / P, NoTangent()
end

function mul_rrule_impl(X::StridedMatrix, H::HermOrSym, Y::StridedMatrix, ΔY)
    ΔX = ΔY * H

    if ΔY isa ZeroTangent
        ΔH = ZeroTangent()
    else
        ΔH = @thunk begin
            ΔH = similar(H)
            selupd!(ΔH, X', ΔY', 1 / 2, 0)
            ΔH
        end
    end

    return ΔX, ΔH
end

# ===== frule / rrule =====

# X * A

for T in (ChordalTriangular{:N}, AdjTri{:N}, TransTri{:N}, HermTri, SymTri)
    @eval function ChainRulesCore.frule((_, dX, dA)::Tuple, ::typeof(*), X::StridedMatrix, A::$T)
        return mul_frule_impl(X, A, dX, dA)
    end

    @eval function ChainRulesCore.rrule(::typeof(*), X::StridedMatrix, A::$T)
        return mul_rrule(X, A)
    end
end

# A * x (scalar)

for T in (ChordalTriangular{:N}, HermTri, SymTri)
    @eval function ChainRulesCore.frule((_, dA, dx)::Tuple, ::typeof(*), A::$T, x::Number)
        return mul_frule_impl(A, x, dA, dx)
    end

    @eval function ChainRulesCore.rrule(::typeof(*), A::$T, x::Number)
        return mul_rrule(A, x)
    end
end

# X * P (Permutation)

function ChainRulesCore.frule((_, dX, dP)::Tuple, ::typeof(*), X::AbstractVecOrMat, P::Permutation)
    return mul_frule_impl(X, P, dX, dP)
end

function ChainRulesCore.rrule(::typeof(*), X::AbstractVecOrMat, P::Permutation)
    return mul_rrule(X, P)
end
