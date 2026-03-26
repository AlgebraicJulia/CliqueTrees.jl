# soft(L) where L is ChordalTriangular

@is_primitive MinimalCtx Tuple{typeof(soft), ChordalTriangular{:N}}

function Mooncake.rrule!!(
    ::CoDual{typeof(soft)},
    L::CoDual{<:ChordalTriangular{:N}}
)
    pL = primal(L)
    tL = tangent(L)

    Y = soft(pL)
    dY = zero(Y)

    function pb!!(::NoRData)
        if !iszero(dY)
            axpy!(1, soft_rrule_impl(pL, Y, dY), tL)
        end
        return NoRData(), NoRData()
    end

    return CoDual(Y, dY), pb!!
end
