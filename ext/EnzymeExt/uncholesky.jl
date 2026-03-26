function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(uncholesky)},
        RT::Type,
        L::Duplicated{<:ChordalTriangular{:N, UPLO}}
    ) where {UPLO}
    H, dH = uncholesky_frule_impl(L.val, L.dval)
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
        func::Const{typeof(uncholesky)},
        RT::Type,
        L::Annotation{<:ChordalTriangular{:N}}
    )
    H = uncholesky(L.val)

    if needs_shadow(config)
        shadow = zero(H)
    else
        shadow = nothing
    end

    _, oL = overwritten(config)
    if oL
        cache = (L.val, H)
    else
        cache = nothing
    end

    tape = (cache, shadow)
    return AugmentedReturn(H, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(uncholesky)},
        dret,
        tape,
        L::Annotation{<:ChordalTriangular{:N}}
    )
    cache, shadow = tape

    if cache === nothing
        Lval = L.val
        H = uncholesky(Lval)
    else
        Lval, H = cache
    end

    if dret isa Active
        ΔH = dret.val
    else
        ΔH = shadow
    end

    if L isa Duplicated
        axpy!(1, uncholesky_rrule_impl(Lval, H, parent(ΔH)), L.dval)
    end

    return (nothing,)
end
