# Number \ ChordalTriangular/HermOrSymTri

@is_primitive MinimalCtx Tuple{typeof(\), Number, ChordalTriangular}
@is_primitive MinimalCtx Tuple{typeof(\), Real, HermTri}
@is_primitive MinimalCtx Tuple{typeof(\), Real, SymTri}

function Mooncake.rrule!!(
    ::CoDual{typeof(\)},
    x::CoDual{<:Number},
    L::CoDual{<:ChordalTriangular}
)
    px = primal(x)
    pL = primal(L)
    tL = tangent(L)

    Y = px \ pL
    dY = zero(Y)

    function pb!!(::NoRData)
        # Δx = -dot(Y, dY) / conj(x), ΔL = conj(x) \ dY
        axpy!(1 / conj(px), dY, tL)
        return NoRData(), -dot(Y, dY) / conj(px), NoRData()
    end

    return CoDual(Y, dY), pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(\)},
    x::CoDual{<:Real},
    H::CoDual{<:HermOrSymTri}
)
    px = primal(x)
    pH = primal(H)
    tH = tangent(H)

    Y = px \ pH
    dY = zero(parent(Y))

    function pb!!(::NoRData)
        # Δx = -dot(Y, Hermitian(dY)) / x, ΔH = x \ dY
        axpy!(1 / px, dY, tH)
        return NoRData(), -dot(Y, Hermitian(dY, Symbol(pH.uplo))) / px, NoRData()
    end

    return CoDual(Y, dY), pb!!
end
