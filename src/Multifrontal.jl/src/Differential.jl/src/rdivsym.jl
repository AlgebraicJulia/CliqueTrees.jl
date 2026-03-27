# 3-arg rdiv: x / P using precomputed Cholesky L
# Gradients flow through P and x, not L

function rdiv(x::AbstractMatrix, P::HermOrSymTri{:L}, L::ChordalTriangular{:N, :L})
    return (x / L') / L
end

function rdiv(x::AbstractMatrix, P::HermOrSymTri{:U}, L::ChordalTriangular{:N, :U})
    return (x / L) / L'
end

# ===== frule =====

function rdiv_frule_impl(x::AbstractMatrix, P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, dx::AbstractMatrix, dP::ChordalTriangular{:N, UPLO}) where {UPLO}
    y = rdiv(x, P, L)
    dy = rdiv(dx - y * Hermitian(dP, UPLO), P, L)
    return y, dy
end

function rdiv_frule_impl(x::AbstractMatrix, P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, dx::ZeroTangent, dP::ChordalTriangular{:N, UPLO}) where {UPLO}
    y = rdiv(x, P, L)
    dy = rdiv(-y * Hermitian(dP, UPLO), P, L)
    return y, dy
end

function rdiv_frule_impl(x::AbstractMatrix, P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, dx::AbstractMatrix, dP::ZeroTangent) where {UPLO}
    y = rdiv(x, P, L)
    dy = rdiv(dx, P, L)
    return y, dy
end

function rdiv_frule_impl(x::AbstractMatrix, P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, dx::ZeroTangent, dP::ZeroTangent) where {UPLO}
    return rdiv(x, P, L), ZeroTangent()
end

function ChainRulesCore.frule((_, dx, dP, _)::Tuple, ::typeof(rdiv), x::AbstractMatrix, P::HermOrSymTri, L::ChordalTriangular{:N})
    return rdiv_frule_impl(x, P, L, dx, dP)
end

# ===== rrule =====

function rdiv_rrule_impl(x::AbstractMatrix, P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, y::AbstractMatrix, Δy::AbstractMatrix) where {UPLO}
    Δx = rdiv(Δy, P, L)
    ΔP = similar(parent(P))
    selupd!(ΔP, y', Δx, -1 / 2, 0)
    selupd!(ΔP, Δx', y, -1 / 2, 1)
    return ΔP, Δx
end

function ChainRulesCore.rrule(::typeof(rdiv), x::AbstractMatrix, P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}) where {UPLO}
    y = rdiv(x, P, L)

    function pullback(Δy)
        if Δy isa ZeroTangent
            ΔP = ZeroTangent()
            Δx = ZeroTangent()
        else
            ΔP, Δx = rdiv_rrule_impl(x, P, L, y, Δy)
        end

        return NoTangent(), Δx, ΔP, NoTangent()
    end

    return y, pullback ∘ unthunk
end
