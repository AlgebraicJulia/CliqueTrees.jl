# dot(A, B) for various types

@is_primitive MinimalCtx Tuple{typeof(dot), HermTri, HermTri}
@is_primitive MinimalCtx Tuple{typeof(dot), SymTri, SymTri}
@is_primitive MinimalCtx Tuple{typeof(dot), ChordalTriangular, ChordalTriangular}

function Mooncake.rrule!!(
    ::CoDual{typeof(dot)},
    A::CoDual{<:HermOrSymTri},
    B::CoDual{<:HermOrSymTri}
)
    pA = primal(A)
    pB = primal(B)
    tA = tangent(A)
    tB = tangent(B)

    y = dot(pA, pB)

    function pullback!!(dy)
        if !iszero(dy)
            ΔA, ΔB = dot_rrule_impl(pA, pB, y, dy)
            axpy!(1, unthunk(ΔA), tA)
            axpy!(1, unthunk(ΔB), tB)
        end
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(y, NoFData()), pullback!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(dot)},
    A::CoDual{<:ChordalTriangular{:N, UPLO}},
    B::CoDual{<:ChordalTriangular{:N, UPLO}}
) where {UPLO}
    pA = primal(A)
    pB = primal(B)
    tA = tangent(A)
    tB = tangent(B)

    y = dot(pA, pB)

    function pullback!!(dy)
        if !iszero(dy)
            ΔA, ΔB = dot_rrule_impl(pA, pB, y, dy)
            axpy!(1, unthunk(ΔA), tA)
            axpy!(1, unthunk(ΔB), tB)
        end
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(y, NoFData()), pullback!!
end
