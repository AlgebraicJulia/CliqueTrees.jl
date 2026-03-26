# complete(H) where H is HermOrSymTri

using CliqueTrees.Multifrontal.Differential: complete_rrule_impl

@is_primitive MinimalCtx Tuple{typeof(complete), HermTri}
@is_primitive MinimalCtx Tuple{typeof(complete), SymTri}

function Mooncake.rrule!!(
    ::CoDual{typeof(complete)},
    H::CoDual{<:HermOrSymTri}
)
    pH = primal(H)
    tH = tangent(H)

    pL = complete(pH)
    tL = zero(pL)

    function pullback!!(::NoRData)
        axpy!(1, complete_rrule_impl(pH, pL, tL), tH)
        return NoRData(), NoRData()
    end

    return CoDual(pL, tL), pullback!!
end
