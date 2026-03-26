# Number * ChordalTriangular/HermOrSymTri

@is_primitive MinimalCtx Tuple{typeof(*), Number, ChordalTriangular}
@is_primitive MinimalCtx Tuple{typeof(*), Real, HermTri}
@is_primitive MinimalCtx Tuple{typeof(*), Real, SymTri}

function Mooncake.rrule!!(
    ::CoDual{typeof(*)},
    x::CoDual{<:Number},
    L::CoDual{<:ChordalTriangular}
)
    px = primal(x)
    pL = primal(L)
    tL = tangent(L)

    Y = px * pL
    dY = zero(Y)

    function pb!!(::NoRData)
        # ΔL = conj(x) * dY, Δx = dot(L, dY)
        axpy!(conj(px), dY, tL)
        return NoRData(), dot(pL, dY), NoRData()
    end

    return CoDual(Y, dY), pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(*)},
    x::CoDual{<:Real},
    H::CoDual{<:HermOrSymTri}
)
    px = primal(x)
    pH = primal(H)
    tH = tangent(H)

    Y = px * pH
    dY = zero(parent(Y))

    function pb!!(::NoRData)
        # ΔH = x * dY, Δx = dot(H, Hermitian(dY))
        axpy!(px, dY, tH)
        return NoRData(), dot(pH, Hermitian(dY, Symbol(pH.uplo))), NoRData()
    end

    return CoDual(Y, dY), pb!!
end
