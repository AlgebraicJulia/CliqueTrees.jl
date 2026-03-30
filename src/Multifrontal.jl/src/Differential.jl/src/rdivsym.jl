# rdivsym: X / A using precomputed Cholesky L and Permutation P
# Gradients flow through A and X, not L or P

function rdivsym(X, A::MaybeHermOrSymSparse, L::ChordalTriangular{:N, :L}, P::Permutation)
    return ((X / P) / L' / L) * P
end

function rdivsym(X, A::MaybeHermOrSymSparse, U::ChordalTriangular{:N, :U}, P::Permutation)
    return ((X / P) / U / U') * P
end

# ===== frule =====

function rdivsym_frule_impl(X::StridedMatrix, A::MaybeHermOrSymSparse, L::ChordalTriangular{:N}, P::Permutation, dX, dA)
    Y = rdivsym(X, A, L, P)
    dY = rdivsym(dX - Y * dA, A, L, P)
    return Y, dY
end

function ChainRulesCore.frule((_, dX, dA, _, _)::Tuple, ::typeof(rdivsym), X::StridedMatrix, A::MaybeHermOrSymSparse, L::ChordalTriangular{:N}, P::Permutation)
    return rdivsym_frule_impl(X, A, L, P, dX, dA)
end

# ===== rrule =====

function rdivsym_rrule_impl(X::StridedMatrix, A::MaybeHermOrSymSparse, L::ChordalTriangular{:N}, P::Permutation, Y::StridedMatrix, ΔY)
    ΔX = rdivsym(ΔY, A, L, P)

    if ΔY isa ZeroTangent
        ΔA = ZeroTangent()
    else
        ΔA = @thunk begin
            ΔA = similar(A)
            selupd!(ΔA, Y', ΔX', -1 / 2, 0)
            ΔA
        end
    end

    return ΔA, ΔX
end

function ChainRulesCore.rrule(::typeof(rdivsym), X::StridedMatrix, A::MaybeHermOrSymSparse, L::ChordalTriangular{:N}, P::Permutation)
    Y = rdivsym(X, A, L, P)

    function pullback(ΔY)
        ΔA, ΔX = rdivsym_rrule_impl(X, A, L, P, Y, ΔY)
        return NoTangent(), ΔX, ΔA, NoTangent(), NoTangent()
    end

    return Y, pullback ∘ unthunk
end
