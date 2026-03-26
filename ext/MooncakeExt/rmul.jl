# x * L operations

@is_primitive MinimalCtx Tuple{typeof(*), AbstractMatrix, ChordalTriangular{:N}}

function Mooncake.rrule!!(
    ::CoDual{typeof(*)},
    x::CoDual{<:AbstractMatrix},
    L::CoDual{<:ChordalTriangular{:N}}
)
    pX = primal(x)
    pL = primal(L)
    tX = tangent(x)
    tL = tangent(L)

    y = pX * pL
    dy = zero(y)

    function pullback!!(::NoRData)
        Δx, ΔL = mul_rrule_impl(pX, pL, y, dy)
        axpy!(1, Δx, tX)
        axpy!(1, unthunk(ΔL), tL)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(y, dy), pullback!!
end
