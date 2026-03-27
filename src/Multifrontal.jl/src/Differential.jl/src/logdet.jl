# Kernel functions for logdet

function LinearAlgebra.logdet(A::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}) where {UPLO}
    return 2 * logdet(L)
end

# ===== frule =====

function logdet_frule_impl(L::ChordalTriangular{:N}, dL)
    y = logdet(L)

    if dL isa ZeroTangent
        dy = ZeroTangent()
    else
        dy = zero(promote_eltype(L, dL))

        for f in fronts(L)
            dD, _ = diagblock(dL, f)
            D, _ = diagblock(L, f)

            for i in diagind(D)
                dy += dD[i] / D[i]
            end
        end
    end

    return y, dy
end

function logdet_frule_impl(A::HermOrSymTri, L::ChordalTriangular{:N}, dA)
    y = logdet(A, L)

    if dA isa ZeroTangent
        dy = ZeroTangent()
    else
        dy = dot(selinv(L), ProjectTo(A)(dA))
    end

    return y, dy
end

function ChainRulesCore.frule((_, dL)::Tuple, ::typeof(logdet), L::ChordalTriangular{:N})
    return logdet_frule_impl(L, dL)
end

function ChainRulesCore.frule((_, dA, _)::Tuple, ::typeof(logdet), A::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}) where {UPLO}
    return logdet_frule_impl(A, L, dA)
end

# ===== rrule =====

function logdet_rrule_impl(L::ChordalTriangular{:N}, y::Number, Δy)
    if Δy isa ZeroTangent
        ΔL = ZeroTangent()
    else
        ΔL = zero(L)

        for f in fronts(L)
            ΔD, _ = diagblock(ΔL, f)
            D, _ = diagblock(L, f)

            for i in diagind(D)
                ΔD[i] = Δy / D[i]
            end
        end
    end

    return ΔL
end

function logdet_rrule_impl(A::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, y::Number, Δy) where {UPLO}
    if Δy isa ZeroTangent
        ΔA = ZeroTangent()
    else
        ΔA = Δy * parent(selinv(L))
    end

    return ΔA
end

function ChainRulesCore.rrule(::typeof(logdet), L::ChordalTriangular{:N})
    y = logdet(L)
    pullback(Δy) = (NoTangent(), logdet_rrule_impl(L, y, Δy))
    return y, pullback
end

function ChainRulesCore.rrule(::typeof(logdet), A::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}) where {UPLO}
    y = logdet(A, L)
    pullback(Δy) = (NoTangent(), logdet_rrule_impl(A, L, y, Δy), NoTangent())
    return y, pullback
end
