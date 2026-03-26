# ChordalTriangular/HermOrSymTri / Number

@is_primitive MinimalCtx Tuple{typeof(/), ChordalTriangular, Number}
@is_primitive MinimalCtx Tuple{typeof(/), HermTri, Real}
@is_primitive MinimalCtx Tuple{typeof(/), SymTri, Real}

function Mooncake.rrule!!(
    ::CoDual{typeof(/)},
    L::CoDual{<:ChordalTriangular},
    x::CoDual{<:Number}
)
    pL = primal(L)
    px = primal(x)
    tL = tangent(L)

    Y = pL / px
    dY = zero(Y)

    function pb!!(::NoRData)
        # ΔL = dY / conj(x), Δx = -dot(Y, dY) / conj(x)
        axpy!(1 / conj(px), dY, tL)
        return NoRData(), NoRData(), -dot(Y, dY) / conj(px)
    end

    return CoDual(Y, dY), pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(/)},
    H::CoDual{<:HermOrSymTri},
    x::CoDual{<:Real}
)
    pH = primal(H)
    px = primal(x)
    tH = tangent(H)

    Y = pH / px
    dY = zero(parent(Y))

    function pb!!(::NoRData)
        # ΔH = dY / x, Δx = -dot(Y, Hermitian(dY)) / x
        axpy!(1 / px, dY, tH)
        return NoRData(), NoRData(), -dot(Y, Hermitian(dY, Symbol(pH.uplo))) / px
    end

    return CoDual(Y, dY), pb!!
end
