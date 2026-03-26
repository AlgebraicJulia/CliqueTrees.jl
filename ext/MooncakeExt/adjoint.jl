# adjoint(L) and transpose(L) for ChordalTriangular

@is_primitive MinimalCtx Tuple{typeof(adjoint), ChordalTriangular}
@is_primitive MinimalCtx Tuple{typeof(adjoint), AdjTri}
@is_primitive MinimalCtx Tuple{typeof(transpose), ChordalTriangular}
@is_primitive MinimalCtx Tuple{typeof(transpose), TransTri}

function Mooncake.rrule!!(
    ::CoDual{typeof(adjoint)},
    L::CoDual{<:ChordalTriangular}
)
    Lval = primal(L)
    dL = tangent(L)

    A = adjoint(Lval)
    # Tangent of Adjoint{T, ChordalTriangular} is ChordalTriangular
    dA = zero(Lval)

    function pb!!(::NoRData)
        axpy!(1, adjoint_rrule_impl(Lval, A, dA), dL)
        return NoRData(), NoRData()
    end

    return CoDual(A, dA), pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(adjoint)},
    L::CoDual{<:AdjTri}
)
    Lval = primal(L)
    dL = tangent(L)  # This is the parent ChordalTriangular tangent

    A = adjoint(Lval)  # Back to ChordalTriangular
    dA = zero(A)

    function pb!!(::NoRData)
        axpy!(1, adjoint_rrule_impl(Lval, A, dA), dL)
        return NoRData(), NoRData()
    end

    return CoDual(A, dA), pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(transpose)},
    L::CoDual{<:ChordalTriangular}
)
    Lval = primal(L)
    dL = tangent(L)

    A = transpose(Lval)
    # Tangent of Transpose{T, ChordalTriangular} is ChordalTriangular
    dA = zero(Lval)

    function pb!!(::NoRData)
        axpy!(1, transpose_rrule_impl(Lval, A, dA), dL)
        return NoRData(), NoRData()
    end

    return CoDual(A, dA), pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(transpose)},
    L::CoDual{<:TransTri}
)
    Lval = primal(L)
    dL = tangent(L)

    A = transpose(Lval)
    dA = zero(A)

    function pb!!(::NoRData)
        axpy!(1, transpose_rrule_impl(Lval, A, dA), dL)
        return NoRData(), NoRData()
    end

    return CoDual(A, dA), pb!!
end
