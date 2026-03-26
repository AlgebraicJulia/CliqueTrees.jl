# logdet(L) where L is ChordalTriangular

@is_primitive MinimalCtx Tuple{typeof(logdet), ChordalTriangular{:N}}

function Mooncake.rrule!!(
    ::CoDual{typeof(logdet)},
    L::CoDual{<:ChordalTriangular{:N}}
)
    pL = primal(L)
    tL = tangent(L)

    y = logdet(pL)

    function pullback!!(dy)
        if !iszero(dy)
            axpy!(1, logdet_rrule_impl(pL, y, dy), tL)
        end
        return NoRData(), NoRData()
    end

    return CoDual(y, NoFData()), pullback!!
end
