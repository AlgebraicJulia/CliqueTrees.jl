function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(*)},
        RT::Type,
        x::Duplicated{<:AbstractMatrix},
        H::Duplicated{<:HermTri}
    )
    y, dy = mul_frule_impl(x.val, H.val, x.dval, parent(H.dval))
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
        func::Const{typeof(*)},
        RT::Type,
        x::Const{<:AbstractMatrix},
        H::Duplicated{<:HermTri}
    )
    y, dy = mul_frule_impl(x.val, H.val, ZeroTangent(), parent(H.dval))
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
        func::Const{typeof(*)},
        RT::Type,
        x::Duplicated{<:AbstractMatrix},
        H::Const{<:HermTri}
    )
    y, dy = mul_frule_impl(x.val, H.val, x.dval, ZeroTangent())
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
        func::Const{typeof(*)},
        RT::Type,
        x::Duplicated{<:AbstractMatrix},
        S::Duplicated{<:SymTri}
    )
    y, dy = mul_frule_impl(x.val, S.val, x.dval, parent(S.dval))
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
        func::Const{typeof(*)},
        RT::Type,
        x::Const{<:AbstractMatrix},
        S::Duplicated{<:SymTri}
    )
    y, dy = mul_frule_impl(x.val, S.val, ZeroTangent(), parent(S.dval))
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
        func::Const{typeof(*)},
        RT::Type,
        x::Duplicated{<:AbstractMatrix},
        S::Const{<:SymTri}
    )
    y, dy = mul_frule_impl(x.val, S.val, x.dval, ZeroTangent())
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
        func::Const{typeof(*)},
        RT::Type,
        x::Annotation{<:AbstractMatrix},
        H::Annotation{<:HermTri}
    )
    y = x.val * H.val

    if needs_shadow(config)
        shadow = zero(y)
    else
        shadow = nothing
    end

    tape = (x.val, y, shadow)
    return AugmentedReturn(y, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(*)},
        dret,
        tape,
        x::Annotation{<:AbstractMatrix},
        H::Annotation{<:HermTri}
    )
    xval, y, shadow = tape

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    Δx, ΔH = mul_rrule_impl(xval, H.val, y, Δy)

    if x isa Duplicated
        axpy!(1, Δx, x.dval)
    end

    if H isa Duplicated
        axpy!(1, unthunk(ΔH), parent(H.dval))
    end

    return (nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1},
        func::Const{typeof(*)},
        RT::Type,
        x::Annotation{<:AbstractMatrix},
        S::Annotation{<:SymTri}
    )
    y = x.val * S.val

    if needs_shadow(config)
        shadow = zero(y)
    else
        shadow = nothing
    end

    tape = (x.val, y, shadow)
    return AugmentedReturn(y, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(*)},
        dret,
        tape,
        x::Annotation{<:AbstractMatrix},
        S::Annotation{<:SymTri}
    )
    xval, y, shadow = tape

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    Δx, ΔS = mul_rrule_impl(xval, S.val, y, Δy)

    if x isa Duplicated
        axpy!(1, Δx, x.dval)
    end

    if S isa Duplicated
        axpy!(1, unthunk(ΔS), parent(S.dval))
    end

    return (nothing, nothing)
end
