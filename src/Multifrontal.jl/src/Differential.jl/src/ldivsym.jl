# 3-arg ldiv: P \ x using precomputed Cholesky L
# Gradients flow through P and x, not L

function ldiv(P::HermOrSymTri{:L}, L::ChordalTriangular{:N, :L}, x::AbstractVector)
    return L' \ (L \ x)
end

function ldiv(P::HermOrSymTri{:U}, L::ChordalTriangular{:N, :U}, x::AbstractVector)
    return L \ (L' \ x)
end

# ===== frule =====

function ldiv_frule_impl(P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, x::AbstractVector, dP::ChordalTriangular{:N, UPLO}, dx::AbstractVector) where {UPLO}
    y = ldiv(P, L, x)
    dy = ldiv(P, L, dx - Hermitian(dP, UPLO) * y)
    return y, dy
end

function ldiv_frule_impl(P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, x::AbstractVector, dP::ZeroTangent, dx::AbstractVector) where {UPLO}
    y = ldiv(P, L, x)
    dy = ldiv(P, L, dx)
    return y, dy
end

function ldiv_frule_impl(P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, x::AbstractVector, dP::ChordalTriangular{:N, UPLO}, dx::ZeroTangent) where {UPLO}
    y = ldiv(P, L, x)
    dy = ldiv(P, L, Hermitian(dP, UPLO) * -y)
    return y, dy
end

function ldiv_frule_impl(P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, x::AbstractVector, dP::ZeroTangent, dx::ZeroTangent) where {UPLO}
    return ldiv(P, L, x), ZeroTangent()
end

function ChainRulesCore.frule((_, dP, _, dx)::Tuple, ::typeof(ldiv), P::HermOrSymTri, L::ChordalTriangular{:N}, x::AbstractVector)
    return ldiv_frule_impl(P, L, x, dP, dx)
end

# ===== rrule =====

function ldiv_rrule_impl(P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, x::AbstractVector, y::AbstractVector, Δy::AbstractVector) where {UPLO}
    Δx = ldiv(P, L, Δy)
    ΔP = similar(parent(P))
    selupd!(ΔP, Δx, y', -1 / 2, 0)
    selupd!(ΔP, y, Δx', -1 / 2, 1)
    return ΔP, Δx
end

function ChainRulesCore.rrule(::typeof(ldiv), P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, x::AbstractVector) where {UPLO}
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
