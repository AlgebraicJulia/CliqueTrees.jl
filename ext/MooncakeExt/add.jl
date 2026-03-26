# Addition operations

@is_primitive MinimalCtx Tuple{typeof(+), ChordalTriangular, ChordalTriangular}

function Mooncake.rrule!!(
    ::CoDual{typeof(+)},
    A::CoDual{<:ChordalTriangular},
    B::CoDual{<:ChordalTriangular}
)
    pA, pB = primal(A), primal(B)
    tA, tB = tangent(A), tangent(B)

    Y = pA + pB
    dY = zero(Y)

    function pb!!(::NoRData)
        ΔA, ΔB = add_rrule_impl(pA, pB, Y, dY)
        axpy!(1, ΔA, tA)
        axpy!(1, ΔB, tB)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(Y, dY), pb!!
end

@is_primitive MinimalCtx Tuple{typeof(+), HermTri, HermTri}
@is_primitive MinimalCtx Tuple{typeof(+), SymTri, SymTri}

function Mooncake.rrule!!(
    ::CoDual{typeof(+)},
    A::CoDual{<:HermOrSymTri},
    B::CoDual{<:HermOrSymTri}
)
    pA, pB = primal(A), primal(B)
    tA, tB = tangent(A), tangent(B)

    Y = pA + pB
    dY = zero(parent(Y))

    function pb!!(::NoRData)
        ΔA, ΔB = add_rrule_impl(pA, pB, Y, dY)
        axpy!(1, ΔA, tA)
        axpy!(1, ΔB, tB)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(Y, dY), pb!!
end
