# tr(L) where L is ChordalTriangular

@is_primitive MinimalCtx Tuple{typeof(tr), ChordalTriangular{:N}}

function Mooncake.rrule!!(
    ::CoDual{typeof(tr)},
    L::CoDual{<:ChordalTriangular{:N}}
)
    pL = primal(L)
    tL = tangent(L)

    y = tr(pL)

    function pullback!!(dy)
        if !iszero(dy)
            axpy!(1, tr_rrule_impl(pL, y, dy), tL)
        end
        return NoRData(), NoRData()
    end

    return CoDual(y, NoFData()), pullback!!
end

# tr(H) where H is HermOrSymTri

@is_primitive MinimalCtx Tuple{typeof(tr), HermTri}
@is_primitive MinimalCtx Tuple{typeof(tr), SymTri}

function Mooncake.rrule!!(
    ::CoDual{typeof(tr)},
    H::CoDual{<:HermOrSymTri}
)
    pH = primal(H)
    tH = tangent(H)

    y = tr(pH)

    function pullback!!(dy)
        if !iszero(dy)
            axpy!(1, tr_rrule_impl(pH, y, dy), tH)
        end
        return NoRData(), NoRData()
    end

    return CoDual(y, NoFData()), pullback!!
end
