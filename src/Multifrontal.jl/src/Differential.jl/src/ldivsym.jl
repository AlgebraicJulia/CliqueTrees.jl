# 3-arg ldiv: P \ x using precomputed Cholesky L
# Gradients flow through P and x, not L

function ldiv(P::HermOrSymTri{:L}, L::ChordalTriangular{:N, :L}, x::AbstractVecOrMat)
    return L' \ (L \ x)
end

function ldiv(P::HermOrSymTri{:U}, L::ChordalTriangular{:N, :U}, x::AbstractVecOrMat)
    return L \ (L' \ x)
end

function ldiv(P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, x::ZeroTangent) where {UPLO}
    return x
end

# ===== frule =====

function ldiv_frule_impl(P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, x::AbstractVecOrMat, dP, dx) where {UPLO}
    y = ldiv(P, L, x)
    dy = ldiv(P, L, dx - ProjectTo(P)(dP) * y)
    return y, dy
end

function ChainRulesCore.frule((_, dP, _, dx)::Tuple, ::typeof(ldiv), P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, x::AbstractVecOrMat) where {UPLO}
    return ldiv_frule_impl(P, L, x, dP, dx)
end

# ===== rrule =====

function ldiv_rrule_impl(P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, x::AbstractVecOrMat, y::AbstractVector, Δy::AbstractVector) where {UPLO}
    Δx = ldiv(P, L, Δy)

    if P isa HermTri
        yt = adjoint(y)
        Δxt = adjoint(Δx)
    else
        yt = transpose(y)
        Δxt = transpose(Δx)
    end

    ΔP = similar(parent(P))
    selupd!(ΔP, Δx, yt, -1 / 2, 0)
    selupd!(ΔP, y, Δxt, -1 / 2, 1)
    return ΔP, Δx
end

function ChainRulesCore.rrule(::typeof(ldiv), P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, x::AbstractVecOrMat) where {UPLO}
    y = ldiv(P, L, x)

    function pullback(Δy)
        if Δy isa ZeroTangent
            ΔP = ZeroTangent()
            Δx = ZeroTangent()
        else
            ΔP, Δx = ldiv_rrule_impl(P, L, x, y, Δy)
        end

        return NoTangent(), ΔP, NoTangent(), Δx
    end

    return y, pullback ∘ unthunk
end
