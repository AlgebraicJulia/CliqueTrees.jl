function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(unflattri)},
        RT::Type,
        x::Duplicated{<:AbstractVector},
        S::Const,
        uplo::Const
    )
    L, dL = unflattri_frule_impl(x.val, S.val, uplo.val, x.dval)
    if needs_primal(config) && needs_shadow(config)
        return Duplicated(L, dL)
    elseif needs_shadow(config)
        return dL
    elseif needs_primal(config)
        return L
    else
        return
    end
end

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1},
        func::Const{typeof(unflattri)},
        RT::Type,
        x::Annotation{<:AbstractVector},
        S::Const,
        uplo::Const
    )
    L = unflattri(x.val, S.val, uplo.val)

    if needs_shadow(config)
        shadow = zero(L)
    else
        shadow = nothing
    end

    _, ox, _, _ = overwritten(config)
    if ox
        cache = (x.val, L)
    else
        cache = nothing
    end

    tape = (cache, shadow)
    return AugmentedReturn(L, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(unflattri)},
        dret,
        tape,
        x::Annotation{<:AbstractVector},
        S::Const,
        uplo::Const
    )
    cache, shadow = tape

    if cache === nothing
        xval = x.val
        L = unflattri(xval, S.val, uplo.val)
    else
        xval, L = cache
    end

    if dret isa Active
        ΔL = dret.val
    else
        ΔL = shadow
    end

    if x isa Duplicated
        axpy!(1, unflattri_rrule_impl(xval, S.val, uplo.val, L, ΔL), x.dval)
    end

    return (nothing, nothing, nothing)
end

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(unflatsym)},
        RT::Type,
        x::Duplicated{<:AbstractVector},
        S::Const,
        uplo::Const{Val{UPLO}}
    ) where {UPLO}
    H, dH = unflatsym_frule_impl(x.val, S.val, uplo.val, x.dval)
    # Enzyme requires shadow to have same type as primal
    if needs_primal(config) && needs_shadow(config)
        return Duplicated(H, Hermitian(dH, UPLO))
    elseif needs_shadow(config)
        return Hermitian(dH, UPLO)
    elseif needs_primal(config)
        return H
    else
        return
    end
end

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1},
        func::Const{typeof(unflatsym)},
        RT::Type,
        x::Annotation{<:AbstractVector},
        S::Const,
        uplo::Const{Val{UPLO}}
    ) where {UPLO}
    H = unflatsym(x.val, S.val, uplo.val)

    if needs_shadow(config)
        shadow = zero(H)
    else
        shadow = nothing
    end

    _, ox, _, _ = overwritten(config)
    if ox
        cache = (x.val, H)
    else
        cache = nothing
    end

    tape = (cache, shadow)
    return AugmentedReturn(H, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(unflatsym)},
        dret,
        tape,
        x::Annotation{<:AbstractVector},
        S::Const,
        uplo::Const{Val{UPLO}}
    ) where {UPLO}
    cache, shadow = tape

    if cache === nothing
        xval = x.val
        H = unflatsym(xval, S.val, uplo.val)
    else
        xval, H = cache
    end

    if dret isa Active
        ΔH = dret.val
    else
        ΔH = shadow
    end

    if x isa Duplicated
        axpy!(1, unflatsym_rrule_impl(xval, S.val, uplo.val, H, parent(ΔH)), x.dval)
    end

    return (nothing, nothing, nothing)
end
