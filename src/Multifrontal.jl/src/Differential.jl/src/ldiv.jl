# Kernel functions for L \ X, L' \ X, transpose(L) \ X, x \ A, P \ X

# ===== frule_impl =====

function ldiv_frule_impl(A, X, dA, dX)
    Y = A \ X
    return Y, A \ (dX - dA * Y)
end

# ===== rrule_impl =====

function ldiv_rrule_impl(A::MaybeAdjOrTransTri{:N}, X::StridedVecOrMat, Y::StridedVecOrMat, ΔY::StridedVecOrMat)
    ΔX = A' \ ΔY

    ΔL = @thunk begin
        ΔA = fwrap(similar, A)
        selupd!(ΔA, ΔX, Y', -1, 0)
        ΔA
    end

    return ΔL, ΔX
end

function ldiv_rrule_impl(x::Number, A::MaybeHermOrSymTri, y::MaybeHermOrSymTri, Δy)
    Δx = @thunk -dot(y, Δy) / conj(x)
    ΔA = @thunk conj(x) \ Δy
    return Δx, ΔA
end

function ldiv_rrule_impl(P::Permutation, X::StridedVecOrMat, Y::StridedVecOrMat, ΔY)
    return NoTangent(), P * ΔY
end

# ===== frule / rrule =====

# A \ X

for T in (ChordalTriangular{:N}, AdjTri{:N}, TransTri{:N})
    @eval function ChainRulesCore.frule((_, dA, dX)::Tuple, ::typeof(\), A::$T, X::StridedVecOrMat)
        return ldiv_frule_impl(A, X, dA, dX)
    end

    @eval function ChainRulesCore.rrule(::typeof(\), A::$T, X::StridedVecOrMat)
        return ldiv_rrule(A, X)
    end
end

# x \ A (scalar)

for T in (ChordalTriangular{:N}, HermTri, SymTri)
    @eval function ChainRulesCore.frule((_, dx, dA)::Tuple, ::typeof(\), x::Number, A::$T)
        return ldiv_frule_impl(x, A, dx, dA)
    end

    @eval function ChainRulesCore.rrule(::typeof(\), x::Number, A::$T)
        return ldiv_rrule(x, A)
    end
end

# P \ X (Permutation)

function ChainRulesCore.frule((_, dP, dX)::Tuple, ::typeof(\), P::Permutation, X::StridedVecOrMat)
    return ldiv_frule_impl(P, X, dP, dX)
end

function ChainRulesCore.rrule(::typeof(\), P::Permutation, X::StridedVecOrMat)
    return ldiv_rrule(P, X)
end
