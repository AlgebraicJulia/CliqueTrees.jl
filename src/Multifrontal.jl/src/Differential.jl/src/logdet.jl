# Kernel functions for logdet

function LinearAlgebra.logdet(A::MaybeHermOrSymSparse, L::ChordalTriangular{:N}, P::Permutation)
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
            D, res = diagblock(L, f)

            if dL isa UniformScaling
                for i in diagind(D)
                    dy += dL.λ / D[i]
                end
            elseif dL isa Diagonal
                for (j, i) in enumerate(diagind(D))
                    dy += dL.diag[res[j]] / D[i]
                end
            else
                dD, _ = diagblock(dL, f)

                for i in diagind(D)
                    dy += dD[i] / D[i]
                end
            end
        end
    end

    return y, dy
end

function logdet_frule_impl(A::MaybeHermOrSymSparse, L::ChordalTriangular{:N}, P::Permutation, dA)
    y = logdet(A, L, P)

    if dA isa ZeroTangent
        dy = ZeroTangent()
    else
        dy = dot(selinv(A, L, P), dA)
    end

    return y, dy
end

function ChainRulesCore.frule((_, dL)::Tuple, ::typeof(logdet), L::ChordalTriangular{:N})
    return logdet_frule_impl(L, dL)
end

function ChainRulesCore.frule((_, dA, _, _)::Tuple, ::typeof(logdet), A::MaybeHermOrSymSparse, L::ChordalTriangular{:N}, P::Permutation)
    return logdet_frule_impl(A, L, P, dA)
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

function logdet_rrule_impl(A::MaybeHermOrSymSparse, L::ChordalTriangular{:N}, P::Permutation, y::Number, Δy)
    if Δy isa ZeroTangent
        ΔA = ZeroTangent()
    else
        ΔA = Δy * selinv(A, L, P)
    end

    return ΔA
end

function ChainRulesCore.rrule(::typeof(logdet), L::ChordalTriangular{:N})
    y = logdet(L)
    pullback(Δy) = (NoTangent(), logdet_rrule_impl(L, y, Δy))
    return y, pullback
end

function ChainRulesCore.rrule(::typeof(logdet), A::MaybeHermOrSymSparse, L::ChordalTriangular{:N}, P::Permutation)
    y = logdet(A, L, P)
    pullback(Δy) = (NoTangent(), logdet_rrule_impl(A, L, P, y, Δy), NoTangent(), NoTangent())
    return y, pullback
end
