# diag(L) where L is ChordalTriangular

@is_primitive MinimalCtx Tuple{typeof(diag), ChordalTriangular{:N}}

function Mooncake.rrule!!(
    ::CoDual{typeof(diag)},
    L::CoDual{<:ChordalTriangular{:N}}
)
    pL = primal(L)
    tL = tangent(L)

    y = diag(pL)
    dy = zero(y)

    function pullback!!(::NoRData)
        axpy!(1, diag_rrule_impl(pL, y, dy), tL)
        return NoRData(), NoRData()
    end

    return CoDual(y, dy), pullback!!
end

# diag(H) where H is HermOrSymTri

@is_primitive MinimalCtx Tuple{typeof(diag), HermTri}
@is_primitive MinimalCtx Tuple{typeof(diag), SymTri}

function Mooncake.rrule!!(
    ::CoDual{typeof(diag)},
    H::CoDual{<:HermOrSymTri}
)
    pH = primal(H)
    tH = tangent(H)

    y = diag(pH)
    dy = zero(y)

    function pullback!!(::NoRData)
        axpy!(1, diag_rrule_impl(pH, y, dy), tH)
        return NoRData(), NoRData()
    end

    return CoDual(y, dy), pullback!!
end
