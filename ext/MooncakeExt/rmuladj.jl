# x * L' and x * transpose(L) operations

@is_primitive MinimalCtx Tuple{typeof(*), AbstractMatrix, AdjTri{:N}}
@is_primitive MinimalCtx Tuple{typeof(*), AbstractMatrix, TransTri{:N}}

function Mooncake.rrule!!(
    ::CoDual{typeof(*)},
    x::CoDual{<:AbstractMatrix},
    A::CoDual{<:AdjTri{:N}}
)
    pX = primal(x)
    pA = primal(A)
    tX = tangent(x)
    tL = tangent(A)

    y = pX * pA
    dy = zero(y)

    function pullback!!(::NoRData)
        Δx, ΔL = mul_rrule_impl(pX, pA, y, dy)
        axpy!(1, Δx, tX)
        axpy!(1, unthunk(ΔL), tL)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(y, dy), pullback!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(*)},
    x::CoDual{<:AbstractMatrix},
    A::CoDual{<:TransTri{:N}}
)
    pX = primal(x)
    pA = primal(A)
    tX = tangent(x)
    tL = tangent(A)

    y = pX * pA
    dy = zero(y)

    function pullback!!(::NoRData)
        Δx, ΔL = mul_rrule_impl(pX, pA, y, dy)
        axpy!(1, Δx, tX)
        axpy!(1, unthunk(ΔL), tL)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(y, dy), pullback!!
end
