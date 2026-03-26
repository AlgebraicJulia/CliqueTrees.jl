function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(soft)},
        RT::Type,
        L::Duplicated{<:ChordalTriangular{:N}}
    )
    Y, dY = soft_frule_impl(L.val, L.dval)
    if needs_primal(config) && needs_shadow(config)
        return Duplicated(Y, dY)
    elseif needs_shadow(config)
        return dY
    elseif needs_primal(config)
        return Y
    else
        return
    end
end

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1},
        func::Const{typeof(soft)},
        RT::Type,
        L::Annotation{<:ChordalTriangular{:N}}
    )
    Y = soft(L.val)

    if needs_shadow(config)
        shadow = zero(Y)
    else
        shadow = nothing
    end

    _, oL = overwritten(config)
    if oL
        cache = (L.val, Y)
    else
        cache = nothing
    end

    tape = (cache, shadow)
    return AugmentedReturn(Y, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(soft)},
        dret,
        tape,
        L::Annotation{<:ChordalTriangular{:N}}
    )
    cache, shadow = tape

    if cache === nothing
        Lval = L.val
        Y = soft(Lval)
    else
        Lval, Y = cache
    end

    if dret isa Active
        ΔY = dret.val
    else
        ΔY = shadow
    end

    if L isa Duplicated
        axpy!(1, soft_rrule_impl(Lval, Y, ΔY), L.dval)
    end

    return (nothing,)
end
