# cholesky(H) where H is HermOrSymTri

using CliqueTrees.Multifrontal.Differential: cholesky_rrule_impl

@is_primitive MinimalCtx Tuple{typeof(cholesky), HermTri}
@is_primitive MinimalCtx Tuple{typeof(cholesky), SymTri}

function Mooncake.rrule!!(
    ::CoDual{typeof(cholesky)},
    H::CoDual{<:HermOrSymTri}
)
    pH = primal(H)
    tH = tangent(H)

    pL = cholesky(pH)
    tL = zero(pL)

    function pullback!!(::NoRData)
        axpy!(1, cholesky_rrule_impl(pH, pL, tL), tH)
        return NoRData(), NoRData()
    end

    return CoDual(pL, tL), pullback!!
end
