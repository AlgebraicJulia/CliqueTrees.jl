function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(flat)},
        RT::Type,
        L::Duplicated{<:ChordalTriangular{:N}}
    )
    y, dy = flat_frule_impl(L.val, L.dval)
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
        func::Const{typeof(flat)},
        RT::Type,
        L::Annotation{<:ChordalTriangular{:N}}
    )
    y = flat(L.val)
    uplo, S, x = y

    if needs_shadow(config)
        shadow = (uplo, S, zero(x))
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
        func::Const{typeof(flat)},
        dret,
        tape,
        L::Annotation{<:ChordalTriangular{:N}}
    )
    cache, shadow = tape

    if cache === nothing
        Lval = L.val
        y = flat(Lval)
    else
        Lval, y = cache
    end

    if dret isa Active
        _, _, Δy = dret.val
    else
        _, _, Δy = shadow
    end

    if L isa Duplicated
        axpy!(1, flat_rrule_impl(Lval, y, Δy), L.dval)
    end

    return (nothing,)
end

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(flat)},
        RT::Type,
        H::Duplicated{<:HermOrSymTri{UPLO}}
    ) where {UPLO}
    y, dy = flat_frule_impl(H.val, parent(H.dval))
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
        func::Const{typeof(flat)},
        RT::Type,
        H::Annotation{<:HermOrSymTri{UPLO}}
    ) where {UPLO}
    y = flat(H.val)
    uplo, S, x = y

    if needs_shadow(config)
        shadow = (uplo, S, zero(x))
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
        func::Const{typeof(flat)},
        dret,
        tape,
        H::Annotation{<:HermOrSymTri{UPLO}}
    ) where {UPLO}
    cache, shadow = tape

    if cache === nothing
        Hval = H.val
        y = flat(Hval)
    else
        Hval, y = cache
    end

    if dret isa Active
        _, _, Δy = dret.val
    else
        _, _, Δy = shadow
    end

    if H isa Duplicated
        axpy!(1, parent(flat_rrule_impl(Hval, y, Δy)), parent(H.dval))
    end

    return (nothing,)
end
