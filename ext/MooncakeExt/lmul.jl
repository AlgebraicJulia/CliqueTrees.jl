# L * x operations

@is_primitive MinimalCtx Tuple{typeof(*), ChordalTriangular{:N}, AbstractVecOrMat}

function Mooncake.rrule!!(
    ::CoDual{typeof(*)},
    L::CoDual{<:ChordalTriangular{:N}},
    x::CoDual{<:AbstractVecOrMat}
)
    pL = primal(L)
    pX = primal(x)
    tL = tangent(L)
    tX = tangent(x)

    y = pL * pX
    dy = zero(y)

    function pullback!!(::NoRData)
        ΔL, Δx = mul_rrule_impl(pL, pX, y, dy)
        axpy!(1, unthunk(ΔL), tL)
        axpy!(1, Δx, tX)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(y, dy), pullback!!
end
