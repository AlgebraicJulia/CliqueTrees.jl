# ===== logdet(A, F) =====

@is_primitive MinimalCtx Tuple{typeof(logdet), HermOrSymSparse, ChordalCholesky}

function Mooncake.rrule!!(
        ::CoDual{typeof(logdet)},
        cdA::CoDual{<:HermOrSymSparse},
        cdF::CoDual{<:ChordalCholesky},
    )
    A, dA = primaltangent(cdA)
    y = logdet(A, primal(cdF))

    function pullback!!(Δy)
        logdet_rrule_impl!(dA, A, primal(cdF), y, Δy)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(y, NoFData()), pullback!!
end

function Mooncake.frule!!(
        ::Dual{typeof(logdet)},
        cdA::Dual{<:HermOrSymSparse},
        cdF::Dual{<:ChordalCholesky},
    )
    A, dA = primaltangent(cdA)
    y, dy = logdet_frule_impl(A, primal(cdF), dA)
    return Dual(y, dy)
end

# ===== logdet_rrule_impl!(ΣA, A, F, y, Δy) =====

@is_primitive MinimalCtx Tuple{typeof(logdet_rrule_impl!), HermSparse, HermSparse, ChordalCholesky, Number, Number}
@is_primitive MinimalCtx Tuple{typeof(logdet_rrule_impl!), SymSparse, SymSparse, ChordalCholesky, Number, Number}

function Mooncake.frule!!(
        ::Dual{typeof(logdet_rrule_impl!)},
        cdΣA::Dual{<:HermOrSymSparse},
        cdA::Dual{<:HermOrSymSparse},
        cdF::Dual{<:ChordalCholesky},
        cdy::Dual{<:Number},
        cdΔy::Dual{<:Number},
    )
    ΣA, dΣA = primaltangent(cdΣA)
     A,  dA = primaltangent(cdA)
    Δy, dΔy = primaltangent(cdΔy)

    logdet_rrule_frule_impl!(ΣA, dΣA, A, dA, primal(cdF), Δy, dΔy)

    return cdΣA
end

function Mooncake.rrule!!(
        ::CoDual{typeof(logdet_rrule_impl!)},
        cdΣA::CoDual{<:HermOrSymSparse},
        cdA::CoDual{<:HermOrSymSparse},
        cdF::CoDual{<:ChordalCholesky},
        cdy::CoDual{<:Number},
        cdΔy::CoDual{<:Number},
    )
    ΣA, dΣA = primaltangent(cdΣA)
     A,  dA = primaltangent(cdA)
    Δy = primal(cdΔy)

    logdet_rrule_impl!(ΣA, A, primal(cdF), primal(cdy), Δy)

    function pullback!!(ΔΣA)
        _, _, ΣΔy = logdet_rrule_rrule_impl!(dΣA, dA, A, primal(cdF), Δy, ΔΣA)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), ΣΔy
    end

    return cdΣA, pullback!!
end
