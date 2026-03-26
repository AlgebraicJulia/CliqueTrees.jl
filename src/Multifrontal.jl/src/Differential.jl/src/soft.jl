function soft(L::ChordalTriangular{:N})
    Y = copy(L)

    for f in fronts(Y)
        D, _ = diagblock(Y, f)

        for i in diagind(D)
            D[i] = softplus(D[i])
        end
    end

    return Y
end

# Kernel functions for soft
function soft_frule_impl(L::ChordalTriangular{:N}, dL::ChordalTriangular{:N})
    @assert checksymbolic(L, dL)
    Y = copy(L)
    dY = copy(dL)

    for f in fronts(Y)
        D, _ = diagblock(Y, f)
        dD, _ = diagblock(dY, f)

        for i in diagind(D)
            dD[i] *= sigmoid(D[i])
            D[i] = softplus(D[i])
        end
    end

    return Y, dY
end

function soft_rrule_impl(L::ChordalTriangular{:N}, Y::ChordalTriangular{:N}, ΔY::ChordalTriangular{:N})
    @assert checksymbolic(L, Y, ΔY)
    ΔL = copy(ΔY)

    for f in fronts(L)
        ΔD, _ = diagblock(ΔL, f)
        D, _ = diagblock(L, f)

        for i in diagind(D)
            ΔD[i] *= sigmoid(D[i])
        end
    end

    return ΔL
end

function ChainRulesCore.frule((_, dL)::Tuple, ::typeof(soft), L::ChordalTriangular{:N})
    return soft_frule_impl(L, dL)
end

function ChainRulesCore.rrule(::typeof(soft), L::ChordalTriangular{:N})
    Y = soft(L)

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent()
        else
            return NoTangent(), soft_rrule_impl(L, Y, ΔY)
        end
    end

    return Y, pullback ∘ unthunk
end
