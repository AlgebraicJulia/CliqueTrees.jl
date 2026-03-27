function softmax(L::ChordalTriangular{:N})
    Y = copy(L)

    for f in fronts(Y)
        D, _ = diagblock(Y, f)

        for i in diagind(D)
            D[i] = softplus(D[i])
        end
    end

    return Y
end

# Kernel functions for softmax
function softmax_frule_impl(L::ChordalTriangular{:N}, dL)
    if dL isa ZeroTangent
        dY = ZeroTangent()
    else
        dY = copy(dL)

        for f in fronts(L)
            D, _ = diagblock(L, f)
            dD, _ = diagblock(dY, f)

            for i in diagind(D)
                dD[i] *= sigmoid(D[i])
            end
        end
    end

    return softmax(L), dY
end

function softmax_rrule_impl(L::ChordalTriangular{:N, UPLO}, Y::ChordalTriangular{:N, UPLO}, ΔY) where {UPLO}
    if ΔY isa ZeroTangent
        ΔL = ZeroTangent()
    else
        ΔL = copy(ΔY)

        for f in fronts(L)
            ΔD, _ = diagblock(ΔL, f)
            D, _ = diagblock(L, f)

            for i in diagind(D)
                ΔD[i] *= sigmoid(D[i])
            end
        end
    end

    return ΔL
end

function ChainRulesCore.frule((_, dL)::Tuple, ::typeof(softmax), L::ChordalTriangular{:N})
    return softmax_frule_impl(L, dL)
end

function ChainRulesCore.rrule(::typeof(softmax), L::ChordalTriangular{:N})
    Y = softmax(L)
    pullback(ΔY) = (NoTangent(), softmax_rrule_impl(L, Y, ΔY))
    return Y, pullback ∘ unthunk
end
