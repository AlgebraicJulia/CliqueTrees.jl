function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(logdet)},
        RT::Type,
        L::Duplicated{<:ChordalTriangular{:N}}
    )
    y, dy = logdet_frule_impl(L.val, L.dval)
    if needs_primal(config) && needs_shadow(config)
        return Duplicated(y, dy)
    elseif needs_shadow(config)
        return dy
    elseif needs_primal(config)
        return y
    else
        return
    end
end

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1},
        func::Const{typeof(logdet)},
        ::Type{<:Active},
        L::Annotation{<:ChordalTriangular{:N}}
    )
    y = logdet(L.val)

    if needs_primal(config)
        primal = y
    else
        primal = nothing
    end

    _, oL = overwritten(config)
    if oL
        cache = (L.val, y)
    else
        cache = nothing
    end

    return AugmentedReturn(primal, nothing, cache)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(logdet)},
        dret,
        tape,
        L::Annotation{<:ChordalTriangular{:N}}
    )
    if tape === nothing
        Lval = L.val
        y = logdet(Lval)
    else
        Lval, y = tape
    end

    Δy = dret.val

    if L isa Duplicated
        axpy!(1, unthunk(logdet_rrule_impl(Lval, y, Δy)), L.dval)
    end

    return (nothing,)
end
