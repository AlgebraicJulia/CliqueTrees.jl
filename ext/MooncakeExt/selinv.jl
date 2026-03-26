# selinv(L) where L is ChordalTriangular

using CliqueTrees.Multifrontal.Differential: selinv_rrule_impl

@is_primitive MinimalCtx Tuple{typeof(selinv), ChordalTriangular{:N}}

function Mooncake.rrule!!(
    ::CoDual{typeof(selinv)},
    L::CoDual{<:ChordalTriangular{:N}}
)
    pL = primal(L)
    tL = tangent(L)

    pH = selinv(pL)
    tH = zero(parent(pH))

    function pullback!!(::NoRData)
        axpy!(1, selinv_rrule_impl(pL, pH, tH), tL)
        return NoRData(), NoRData()
    end

    return CoDual(pH, tH), pullback!!
end
