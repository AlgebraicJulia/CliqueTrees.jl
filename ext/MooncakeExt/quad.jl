# dot(X, A, Y) where A is HermOrSymTri

using CliqueTrees.Multifrontal: selupd!

@is_primitive MinimalCtx Tuple{typeof(dot), AbstractVecOrMat, HermTri, AbstractVecOrMat}
@is_primitive MinimalCtx Tuple{typeof(dot), AbstractVecOrMat, SymTri, AbstractVecOrMat}

function Mooncake.rrule!!(
    ::CoDual{typeof(dot)},
    X::CoDual{<:AbstractVecOrMat},
    A::CoDual{<:HermTri{UPLO}},
    Y::CoDual{<:AbstractVecOrMat}
) where {UPLO}
    pX = primal(X)
    pA = primal(A)
    pY = primal(Y)
    tX = tangent(X)
    tA = tangent(A)
    tY = tangent(Y)

    AY = pA * pY
    AX = pA * pX
    y = dot(pX, AY)

    function pb!!(dy::Float64)
        # ΔX = dy * AY, ΔY = dy * AX
        axpy!(dy, AY, tX)
        axpy!(dy, AX, tY)
        # ΔA: selupd!(ΔA, X, Y', dy/2, 0); selupd!(ΔA, Y, X', dy/2, 1)
        selupd!(tA, pX, pY', dy / 2, 1)
        selupd!(tA, pY, pX', dy / 2, 1)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end

    return CoDual(y, NoFData()), pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(dot)},
    X::CoDual{<:AbstractVecOrMat},
    A::CoDual{<:SymTri{UPLO}},
    Y::CoDual{<:AbstractVecOrMat}
) where {UPLO}
    pX = primal(X)
    pA = primal(A)
    pY = primal(Y)
    tX = tangent(X)
    tA = tangent(A)
    tY = tangent(Y)

    AY = pA * pY
    AX = pA * pX
    y = dot(pX, AY)

    function pb!!(dy::Float64)
        # ΔX = dy * AY, ΔY = dy * AX
        axpy!(dy, AY, tX)
        axpy!(dy, AX, tY)
        # ΔA: selupd!(ΔA, X, transpose(Y), dy/2, 0); selupd!(ΔA, Y, transpose(X), dy/2, 1)
        selupd!(tA, pX, transpose(pY), dy / 2, 1)
        selupd!(tA, pY, transpose(pX), dy / 2, 1)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end

    return CoDual(y, NoFData()), pb!!
end
