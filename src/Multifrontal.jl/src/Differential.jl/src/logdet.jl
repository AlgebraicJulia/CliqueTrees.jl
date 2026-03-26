# Kernel functions for logdet
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

function logdet_rrule_impl(L::ChordalTriangular{:N}, y, Δy)
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

function ChainRulesCore.frule((_, dL)::Tuple, ::typeof(logdet), L::ChordalTriangular{:N})
    return logdet_frule_impl(L, dL)
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
