# L' \ x and transpose(L) \ x operations

@is_primitive MinimalCtx Tuple{typeof(\), AdjTri{:N}, AbstractVecOrMat}
@is_primitive MinimalCtx Tuple{typeof(\), TransTri{:N}, AbstractVecOrMat}

function Mooncake.rrule!!(
    ::CoDual{typeof(\)},
    A::CoDual{<:AdjTri{:N}},
    x::CoDual{<:AbstractVecOrMat}
)
    pA = primal(A)
    pX = primal(x)
    tL = tangent(A)
    tX = tangent(x)

    y = pA \ pX
    dy = zero(y)

    function pullback!!(::NoRData)
        ΔL, Δx = ldiv_rrule_impl(pA, pX, y, dy)
        axpy!(1, unthunk(ΔL), tL)
        axpy!(1, Δx, tX)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(y, dy), pullback!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(\)},
    A::CoDual{<:TransTri{:N}},
    x::CoDual{<:AbstractVecOrMat}
)
    pA = primal(A)
    pX = primal(x)
    tL = tangent(A)
    tX = tangent(x)

    y = pA \ pX
    dy = zero(y)

    function pullback!!(::NoRData)
        ΔL, Δx = ldiv_rrule_impl(pA, pX, y, dy)
        axpy!(1, unthunk(ΔL), tL)
        axpy!(1, Δx, tX)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(y, dy), pullback!!
end
