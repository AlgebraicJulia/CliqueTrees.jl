# Kernel functions for logdet

function LinearAlgebra.logdet(A::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}) where {UPLO}
    return 2 * logdet(L)
end

# ===== frule =====

function logdet_frule_impl(L::ChordalTriangular{:N, UPLO}, dL::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(L, dL)
    y = logdet(L)
    dy = zero(promote_eltype(L, dL))

    for f in fronts(L)
        dD, _ = diagblock(dL, f)
        D, _ = diagblock(L, f)

        for i in diagind(D)
            dy += dD[i] / D[i]
        end
    end

    return y, dy
end

function logdet_frule_impl(L::ChordalTriangular{:N}, dL::ZeroTangent)
    return logdet(L), ZeroTangent()
end

function logdet_frule_impl(A::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, dA::ChordalTriangular{:N, UPLO}) where {UPLO}
    return logdet(A, L), dot(selinv(L), Hermitian(dA, UPLO))
end

function logdet_frule_impl(A::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, dA::ZeroTangent) where {UPLO}
    return logdet(A, L), ZeroTangent()
end

function ChainRulesCore.frule((_, dL)::Tuple, ::typeof(logdet), L::ChordalTriangular{:N})
    return logdet_frule_impl(L, dL)
end

function ChainRulesCore.frule((_, dA, _)::Tuple, ::typeof(logdet), A::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}) where {UPLO}
    return logdet_frule_impl(A, L, dA)
end

# ===== rrule =====

function logdet_rrule_impl(L::ChordalTriangular{:N}, y::Number, Δy::Number)
    ΔL = zero(L)

    for f in fronts(L)
        ΔD, _ = diagblock(ΔL, f)
        D, _ = diagblock(L, f)

        for i in diagind(D)
            ΔD[i] = Δy / D[i]
        end
    end

    return ΔL
end

function logdet_rrule_impl(A::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, y::Number, Δy::Number) where {UPLO}
    return Δy * parent(selinv(L))
end

function ChainRulesCore.rrule(::typeof(logdet), L::ChordalTriangular{:N})
    y = logdet(L)

    function pullback(Δy)
        if Δy isa ZeroTangent
            return NoTangent(), ZeroTangent()
        else
            return NoTangent(), logdet_rrule_impl(L, y, Δy)
        end
    end

    return y, pullback
end

function ChainRulesCore.rrule(::typeof(logdet), A::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}) where {UPLO}
    y = logdet(A, L)

    function pullback(Δy)
        if Δy isa ZeroTangent
            return NoTangent(), ZeroTangent(), NoTangent()
        else
            return NoTangent(), logdet_rrule_impl(A, L, y, Δy), NoTangent()
        end
    end

    return y, pullback
end
