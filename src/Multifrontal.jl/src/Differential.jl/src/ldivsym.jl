# ldivsym: A \ X using precomputed Cholesky L and Permutation P
# Gradients flow through A and X, not L or P

function ldivsym(A::HermOrSymSparse, L::ChordalTriangular{:N, :L}, P::Permutation, X)
    return P \ (L' \ (L \ (P * X)))
end

function ldivsym(A::HermOrSymSparse, U::ChordalTriangular{:N, :U}, P::Permutation, X)
    return P \ (U \ (U' \ (P * X)))
end

# ===== frule =====

function ldivsym_frule_impl(A::HermOrSymSparse, L::ChordalTriangular{:N}, P::Permutation, X::StridedVecOrMat, dA, dX)
    Y = ldivsym(A, L, P, X)
    dY = ldivsym(A, L, P, dX - dA * Y)
    return Y, dY
end

function ChainRulesCore.frule((_, dA, _, _, dX)::Tuple, ::typeof(ldivsym), A::HermOrSymSparse, L::ChordalTriangular{:N}, P::Permutation, X::StridedVecOrMat)
    return ldivsym_frule_impl(A, L, P, X, dA, dX)
end

# ===== rrule =====

function ldivsym_rrule_impl(A::HermOrSymSparse, L::ChordalTriangular{:N}, P::Permutation, X::StridedVecOrMat, Y::StridedVector, ΔY)
    ΔX = ldivsym(A, L, P, ΔY)

    if ΔY isa ZeroTangent
        ΔA = ZeroTangent()
    else
        ΔA = @thunk begin
            ΔA = similar(A)
            selupd!(ΔA, ΔX, Y, -1 / 2, 0)
            ΔA
        end
    end

    return ΔA, ΔX
end

function ChainRulesCore.rrule(::typeof(ldivsym), A::HermOrSymSparse, L::ChordalTriangular{:N}, P::Permutation, X::StridedVecOrMat)
    Y = ldivsym(A, L, P, X)

    function pullback(ΔY)
        ΔA, ΔX = ldivsym_rrule_impl(A, L, P, X, Y, ΔY)
        return NoTangent(), ΔA, NoTangent(), NoTangent(), ΔX
    end

    return Y, pullback ∘ unthunk
end
