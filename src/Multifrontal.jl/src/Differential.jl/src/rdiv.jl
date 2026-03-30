# Kernel functions for X / L, X / L', X / transpose(L), A / x, X / P

# ===== frule_impl =====

function rdiv_frule_impl(X, A, dX, dA)
    Y = X / A
    return Y, (dX - Y * dA) / A
end

# ===== rrule_impl =====

function rdiv_rrule_impl(X::StridedMatrix, A::MaybeAdjOrTransTri{:N}, Y::StridedMatrix, ΔY::StridedMatrix)
    ΔX = ΔY / A'

    ΔA = @thunk begin
        ΔA = fwrap(similar, A)
        selupd!(ΔA, Y', ΔX, -1, 0)
        ΔA
    end

    return ΔX, ΔA
end

function rdiv_rrule_impl(A::MaybeHermOrSymTri, x::Number, y::MaybeHermOrSymTri, Δy)
    ΔA = @thunk Δy / conj(x)
    Δx = @thunk -dot(y, Δy) / conj(x)
    return ΔA, Δx
end

function rdiv_rrule_impl(X::StridedVecOrMat, P::Permutation, Y::StridedVecOrMat, ΔY)
    return ΔY * P, NoTangent()
end

# ===== frule / rrule =====

# X / A

for T in (ChordalTriangular{:N}, AdjTri{:N}, TransTri{:N})
    @eval function ChainRulesCore.frule((_, dX, dA)::Tuple, ::typeof(/), X::StridedMatrix, A::$T)
        return rdiv_frule_impl(X, A, dX, dA)
    end

    @eval function ChainRulesCore.rrule(::typeof(/), X::StridedMatrix, A::$T)
        return rdiv_rrule(X, A)
    end
end

# A / x (scalar)

for T in (ChordalTriangular{:N}, HermTri, SymTri)
    @eval function ChainRulesCore.frule((_, dA, dx)::Tuple, ::typeof(/), A::$T, x::Number)
        return rdiv_frule_impl(A, x, dA, dx)
    end

    @eval function ChainRulesCore.rrule(::typeof(/), A::$T, x::Number)
        return rdiv_rrule(A, x)
    end
end

# X / P (Permutation)

function ChainRulesCore.frule((_, dX, dP)::Tuple, ::typeof(/), X::StridedVecOrMat, P::Permutation)
    return rdiv_frule_impl(X, P, dX, dP)
end

function ChainRulesCore.rrule(::typeof(/), X::StridedVecOrMat, P::Permutation)
    return rdiv_rrule(X, P)
end
