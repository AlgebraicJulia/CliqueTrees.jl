# ===== ldiv!(F, X) for ChordalCholesky =====

@is_primitive MinimalCtx Tuple{typeof(ldiv!), ChordalCholesky, AbstractVecOrMat}

function Mooncake.rrule!!(
        ::CoDual{typeof(ldiv!)},
        cdF::CoDual{<:ChordalCholesky},
        cdX::CoDual{<:AbstractVecOrMat},
    )
    F = primal(cdF)
    X, dX = primaltangent(cdX)

    # Save X before forward pass (restoration pattern)
    Y = copy(X)

    # Forward pass
    ldiv!(F, X)

    function pullback!!(::NoRData)
        ldiv_rrule_impl!(Y, F, X, dX)
        return NoRData(), NoRData(), NoRData()
    end

    return cdX, pullback!!
end

function Mooncake.frule!!(
        ::Dual{typeof(ldiv!)},
        cdF::Dual{<:ChordalCholesky},
        cdX::Dual{<:AbstractVecOrMat},
    )
    F = primal(cdF)
    X, dX = primaltangent(cdX)

    ldiv_frule_impl!(F, X, dX)

    return cdX
end

# ===== rdiv!(X, F) for ChordalCholesky =====

@is_primitive MinimalCtx Tuple{typeof(rdiv!), AbstractMatrix, ChordalCholesky}

function Mooncake.rrule!!(
        ::CoDual{typeof(rdiv!)},
        cdX::CoDual{<:AbstractMatrix},
        cdF::CoDual{<:ChordalCholesky},
    )
    X, dX = primaltangent(cdX)
    F = primal(cdF)

    # Save X before forward pass (restoration pattern)
    Y = copy(X)

    # Forward pass
    rdiv!(X, F)

    function pullback!!(::NoRData)
        rdiv_rrule_impl!(Y, F, X, dX)
        return NoRData(), NoRData(), NoRData()
    end

    return cdX, pullback!!
end

function Mooncake.frule!!(
        ::Dual{typeof(rdiv!)},
        cdX::Dual{<:AbstractMatrix},
        cdF::Dual{<:ChordalCholesky},
    )
    X, dX = primaltangent(cdX)
    F = primal(cdF)

    rdiv_frule_impl!(F, X, dX)

    return cdX
end
