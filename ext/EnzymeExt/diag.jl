# diag(L) where L is ChordalTriangular

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(diag)},
        RT::Type,
        L::Duplicated{<:ChordalTriangular{:N}}
    )
    y, dy = diag_frule_impl(L.val, L.dval)
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
        func::Const{typeof(diag)},
        RT::Type,
        L::Annotation{<:ChordalTriangular{:N}}
    )
    y = diag(L.val)

    if needs_shadow(config)
        shadow = zero(y)
    else
        shadow = nothing
    end

    _, oL = overwritten(config)
    if oL
        cache = (L.val, y)
    else
        cache = nothing
    end

    tape = (cache, shadow)
    return AugmentedReturn(y, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(diag)},
        dret,
        tape,
        L::Annotation{<:ChordalTriangular{:N}}
    )
    cache, shadow = tape

    if cache === nothing
        Lval = L.val
        y = diag(Lval)
    else
        Lval, y = cache
    end

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    if L isa Duplicated
        axpy!(1, diag_rrule_impl(Lval, y, Δy), L.dval)
    end

    return (nothing,)
end

# diag(H) where H is HermOrSymTri

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(diag)},
        RT::Type,
        H::Duplicated{<:HermOrSymTri}
    )
    y, dy = diag_frule_impl(H.val, parent(H.dval))
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
        func::Const{typeof(diag)},
        RT::Type,
        H::Annotation{<:HermOrSymTri}
    )
    y = diag(H.val)

    if needs_shadow(config)
        shadow = zero(y)
    else
        shadow = nothing
    end

    _, oH = overwritten(config)
    if oH
        cache = (H.val, y)
    else
        cache = nothing
    end

    tape = (cache, shadow)
    return AugmentedReturn(y, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(diag)},
        dret,
        tape,
        H::Annotation{<:HermOrSymTri}
    )
    cache, shadow = tape

    if cache === nothing
        Hval = H.val
        y = diag(Hval)
    else
        Hval, y = cache
    end

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    if H isa Duplicated
        axpy!(1, diag_rrule_impl(Hval, y, Δy), parent(H.dval))
    end

    return (nothing,)
end
