function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(complete)},
        RT::Type,
        H::Duplicated{<:HermOrSymTri}
    )
    L, dL = complete_frule_impl(H.val, parent(H.dval))
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
        func::Const{typeof(complete)},
        RT::Type,
        H::Annotation{<:HermOrSymTri}
    )
    L = complete(H.val)

    if needs_shadow(config)
        shadow = zero(L)
    else
        shadow = nothing
    end

    _, oH = overwritten(config)
    if oH
        cache = (H.val, L)
    else
        cache = nothing
    end

    tape = (cache, shadow)
    return AugmentedReturn(L, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(complete)},
        dret,
        tape,
        H::Annotation{<:HermOrSymTri}
    )
    cache, shadow = tape

    if cache === nothing
        Hval = H.val
        L = complete(Hval)
    else
        Hval, L = cache
    end

    if dret isa Active
        ΔL = dret.val
    else
        ΔL = shadow
    end

    if H isa Duplicated
        axpy!(1, complete_rrule_impl(Hval, L, ΔL), parent(H.dval))
    end

    return (nothing,)
end
