function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(\)},
        RT::Type,
        L::Duplicated{<:ChordalTriangular{:N}},
        x::Duplicated{<:AbstractVecOrMat}
    )
    y, dy = ldiv_frule_impl(L.val, x.val, L.dval, x.dval)
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

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(\)},
        RT::Type,
        L::Const{<:ChordalTriangular{:N}},
        x::Duplicated{<:AbstractVecOrMat}
    )
    y, dy = ldiv_frule_impl(L.val, x.val, ZeroTangent(), x.dval)
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

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(\)},
        RT::Type,
        L::Duplicated{<:ChordalTriangular{:N}},
        x::Const{<:AbstractVecOrMat}
    )
    y, dy = ldiv_frule_impl(L.val, x.val, L.dval, ZeroTangent())
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
        func::Const{typeof(\)},
        RT::Type,
        L::Annotation{<:ChordalTriangular{:N}},
        x::Annotation{<:AbstractVecOrMat}
    )
    y = L.val \ x.val

    if needs_shadow(config)
        shadow = zero(y)
    else
        shadow = nothing
    end

    tape = (L.val, x.val, y, shadow)
    return AugmentedReturn(y, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(\)},
        dret,
        tape,
        L::Annotation{<:ChordalTriangular{:N}},
        x::Annotation{<:AbstractVecOrMat}
    )
    Lval, xval, y, shadow = tape

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    ΔL, Δx = ldiv_rrule_impl(Lval, xval, y, Δy)

    if x isa Duplicated
        axpy!(1, Δx, x.dval)
    end

    if L isa Duplicated
        axpy!(1, unthunk(ΔL), L.dval)
    end

    return (nothing, nothing)
end
