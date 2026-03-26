# UniformScaling + ChordalTriangular/HermOrSymTri

@is_primitive MinimalCtx Tuple{typeof(+), UniformScaling, ChordalTriangular}
@is_primitive MinimalCtx Tuple{typeof(+), UniformScaling, HermTri}
@is_primitive MinimalCtx Tuple{typeof(+), UniformScaling, SymTri}

function Mooncake.rrule!!(
    ::CoDual{typeof(+)},
    J::CoDual{<:UniformScaling},
    L::CoDual{<:ChordalTriangular}
)
    pL = primal(L)
    tL = tangent(L)

    Y = primal(J) + pL
    dY = zero(Y)

    function pb!!(::NoRData)
        axpy!(1, dY, tL)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(Y, dY), pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(+)},
    J::CoDual{<:UniformScaling},
    H::CoDual{<:HermOrSymTri{UPLO}}
) where {UPLO}
    pH = primal(H)
    tH = tangent(H)

    Y = primal(J) + pH
    dY = zero(parent(Y))

    function pb!!(::NoRData)
        axpy!(1, dY, tH)
        # Gradient w.r.t. λ in λI is tr(ΔY)
        Δλ = tr(Hermitian(dY, UPLO))
        return NoRData(), uniform_rdata(Δλ), NoRData()
    end

    return CoDual(Y, dY), pb!!
end
