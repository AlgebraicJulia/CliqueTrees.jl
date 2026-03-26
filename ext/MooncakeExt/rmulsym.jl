# AbstractMatrix * HermOrSymTri

using CliqueTrees.Multifrontal: selupd!

@is_primitive MinimalCtx Tuple{typeof(*), AbstractMatrix, HermTri}
@is_primitive MinimalCtx Tuple{typeof(*), AbstractMatrix, SymTri}

function Mooncake.rrule!!(
    ::CoDual{typeof(*)},
    x::CoDual{<:AbstractMatrix},
    H::CoDual{<:HermTri{UPLO}}
) where {UPLO}
    px = primal(x)
    pH = primal(H)
    tx = tangent(x)
    tH = tangent(H)

    Y = px * pH
    dY = zero(Y)

    function pb!!(::NoRData)
        # Δx = dY * H
        axpy!(1, dY * pH, tx)
        # ΔH: selupd!(ΔH, x', dY, 1/2, 0); selupd!(ΔH, dY', x, 1/2, 1)
        selupd!(tH, px', dY, 1 // 2, 1)
        selupd!(tH, dY', px, 1 // 2, 1)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(Y, dY), pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(*)},
    x::CoDual{<:AbstractMatrix},
    S::CoDual{<:SymTri{UPLO}}
) where {UPLO}
    px = primal(x)
    pS = primal(S)
    tx = tangent(x)
    tS = tangent(S)

    Y = px * pS
    dY = zero(Y)

    function pb!!(::NoRData)
        # Δx = dY * S
        axpy!(1, dY * pS, tx)
        # ΔS: selupd!(ΔS, transpose(x), dY, 1/2, 0); selupd!(ΔS, transpose(dY), x, 1/2, 1)
        selupd!(tS, transpose(px), dY, 1 // 2, 1)
        selupd!(tS, transpose(dY), px, 1 // 2, 1)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(Y, dY), pb!!
end
