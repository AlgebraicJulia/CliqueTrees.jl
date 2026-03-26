# ChordalTriangular/HermOrSymTri + UniformScaling

@is_primitive MinimalCtx Tuple{typeof(+), ChordalTriangular, UniformScaling}
@is_primitive MinimalCtx Tuple{typeof(+), HermTri, UniformScaling}
@is_primitive MinimalCtx Tuple{typeof(+), SymTri, UniformScaling}

function Mooncake.rrule!!(
    ::CoDual{typeof(+)},
    L::CoDual{<:ChordalTriangular},
    J::CoDual{<:UniformScaling}
)
    pL = primal(L)
    tL = tangent(L)

    Y = pL + primal(J)
    dY = zero(Y)

    function pb!!(::NoRData)
        axpy!(1, dY, tL)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(Y, dY), pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(+)},
    H::CoDual{<:HermOrSymTri{UPLO}},
    J::CoDual{<:UniformScaling}
) where {UPLO}
    pH = primal(H)
    tH = tangent(H)

    Y = pH + primal(J)
    dY = zero(parent(Y))

    function pb!!(::NoRData)
        axpy!(1, dY, tH)
        # Gradient w.r.t. λ in λI is tr(ΔY)
        Δλ = tr(Hermitian(dY, UPLO))
        return NoRData(), NoRData(), uniform_rdata(Δλ)
    end

    return CoDual(Y, dY), pb!!
end
