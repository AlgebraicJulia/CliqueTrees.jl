# uncholesky(L) where L is ChordalTriangular

using CliqueTrees.Multifrontal.Differential: uncholesky_rrule_impl

@is_primitive MinimalCtx Tuple{typeof(uncholesky), ChordalTriangular{:N}}

function Mooncake.rrule!!(
    ::CoDual{typeof(uncholesky)},
    L::CoDual{<:ChordalTriangular{:N}}
)
    pL = primal(L)
    tL = tangent(L)

    pH = uncholesky(pL)
    tH = zero(parent(pH))

    function pullback!!(::NoRData)
        axpy!(1, uncholesky_rrule_impl(pL, pH, tH), tL)
        return NoRData(), NoRData()
    end

    return CoDual(pH, tH), pullback!!
end
