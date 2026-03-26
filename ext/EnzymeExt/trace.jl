# tr(L) where L is ChordalTriangular

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(tr)},
        RT::Type,
        L::Duplicated{<:ChordalTriangular{:N}}
    )
    y, dy = tr_frule_impl(L.val, L.dval)
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
        func::Const{typeof(tr)},
        ::Type{<:Active},
        L::Annotation{<:ChordalTriangular{:N}}
    )
    y = tr(L.val)

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
        func::Const{typeof(tr)},
        dret,
        tape,
        L::Annotation{<:ChordalTriangular{:N}}
    )
    if tape === nothing
        Lval = L.val
        y = tr(Lval)
    else
        Lval, y = tape
    end

    Δy = dret.val

    if L isa Duplicated
        axpy!(1, tr_rrule_impl(Lval, y, Δy), L.dval)
    end

    return (nothing,)
end

# tr(H) where H is HermOrSymTri

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(tr)},
        RT::Type,
        H::Duplicated{<:HermOrSymTri}
    )
    y, dy = tr_frule_impl(H.val, parent(H.dval))
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
        func::Const{typeof(tr)},
        ::Type{<:Active},
        H::Annotation{<:HermOrSymTri}
    )
    y = tr(H.val)

    if needs_primal(config)
        primal = y
    else
        primal = nothing
    end

    _, oH = overwritten(config)
    if oH
        cache = (H.val, y)
    else
        cache = nothing
    end

    return AugmentedReturn(primal, nothing, cache)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(tr)},
        dret,
        tape,
        H::Annotation{<:HermOrSymTri}
    )
    if tape === nothing
        Hval = H.val
        y = tr(Hval)
    else
        Hval, y = tape
    end

    Δy = dret.val

    if H isa Duplicated
        axpy!(1, tr_rrule_impl(Hval, y, Δy), parent(H.dval))
    end

    return (nothing,)
end
