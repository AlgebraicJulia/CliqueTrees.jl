# 3-arg rdiv: x / P using precomputed Cholesky L
# Gradients flow through P and x, not L

function rdiv(x::AbstractMatrix, P::HermOrSymTri{:L}, L::ChordalTriangular{:N, :L})
    return (x / L') / L
end

function rdiv(x::AbstractMatrix, P::HermOrSymTri{:U}, L::ChordalTriangular{:N, :U})
    return (x / L) / L'
end

function rdiv(x::ZeroTangent, P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}) where {UPLO}
    return x
end

# ===== frule =====

function rdiv_frule_impl(x::AbstractMatrix, P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, dx, dP) where {UPLO}
    y = rdiv(x, P, L)
    dy = rdiv(dx - y * ProjectTo(P)(dP), P, L)
    return y, dy
end

function ChainRulesCore.frule((_, dx, dP, _)::Tuple, ::typeof(rdiv), x::AbstractMatrix, P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}) where {UPLO}
    return rdiv_frule_impl(x, P, L, dx, dP)
end

# ===== rrule =====

function rdiv_rrule_impl(x::AbstractMatrix, P::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, y::AbstractMatrix, Δy::AbstractMatrix) where {UPLO}
    Δx = rdiv(Δy, P, L)

    if P isa HermTri
        yt = adjoint(y)
        Δxt = adjoint(Δx)
    else
        yt = transpose(y)
        Δxt = transpose(Δx)
    end

    ΔP = similar(parent(P))
    selupd!(ΔP, yt, Δx, -1 / 2, 0)
    selupd!(ΔP, Δxt, y, -1 / 2, 1)
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
