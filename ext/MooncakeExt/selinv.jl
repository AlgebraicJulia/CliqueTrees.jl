# ===== selinv(A, F) =====

@is_primitive MinimalCtx Tuple{typeof(selinv), HermOrSymSparse, ChordalCholesky}

function Mooncake.rrule!!(
        ::CoDual{typeof(selinv)},
        cdA::CoDual{<:HermOrSymSparse},
        cdF::CoDual{<:ChordalCholesky},
    )
    A, dA = primaltangent(cdA)
    B = selinv(A, primal(cdF))
    dB = zero(B)

    function pullback!!(::NoRData)
        selinv_rrule_impl!(dA, A, primal(cdF), B, dB)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(B, tofdata(B, dB)), pullback!!
end

function Mooncake.frule!!(
        ::Dual{typeof(selinv)},
        cdA::Dual{<:HermOrSymSparse},
        cdF::Dual{<:ChordalCholesky},
    )
    A, dA = primaltangent(cdA)
    Y, dY = selinv_frule_impl(A, primal(cdF), dA)
    return Dual(Y, totangent(Y, dY))
end
