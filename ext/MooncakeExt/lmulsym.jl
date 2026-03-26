# HermOrSymTri * AbstractVecOrMat

using CliqueTrees.Multifrontal: selupd!

@is_primitive MinimalCtx Tuple{typeof(*), HermTri, AbstractVecOrMat}
@is_primitive MinimalCtx Tuple{typeof(*), SymTri, AbstractVecOrMat}

function Mooncake.rrule!!(
    ::CoDual{typeof(*)},
    H::CoDual{<:HermTri{UPLO}},
    x::CoDual{<:AbstractVecOrMat}
) where {UPLO}
    pH = primal(H)
    px = primal(x)
    tH = tangent(H)
    tx = tangent(x)

    Y = pH * px
    dY = zero(Y)

    function pb!!(::NoRData)
        # Δx = H * dY
        axpy!(1, pH * dY, tx)
        # ΔH: selupd!(ΔH, dY, x', 1/2, 0); selupd!(ΔH, x, dY', 1/2, 1)
        selupd!(tH, dY, px', 1 // 2, 1)
        selupd!(tH, px, dY', 1 // 2, 1)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(Y, dY), pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(*)},
    S::CoDual{<:SymTri{UPLO}},
    x::CoDual{<:AbstractVecOrMat}
) where {UPLO}
    pS = primal(S)
    px = primal(x)
    tS = tangent(S)
    tx = tangent(x)

    Y = pS * px
    dY = zero(Y)

    function pb!!(::NoRData)
        # Δx = S * dY
        axpy!(1, pS * dY, tx)
        # ΔS: selupd!(ΔS, dY, transpose(x), 1/2, 0); selupd!(ΔS, x, transpose(dY), 1/2, 1)
        selupd!(tS, dY, transpose(px), 1 // 2, 1)
        selupd!(tS, px, transpose(dY), 1 // 2, 1)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(Y, dY), pb!!
end
