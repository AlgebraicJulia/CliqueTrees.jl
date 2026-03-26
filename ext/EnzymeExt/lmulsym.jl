function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(*)},
        RT::Type,
        H::Duplicated{<:HermTri},
        x::Duplicated{<:AbstractVecOrMat}
    )
    y, dy = mul_frule_impl(H.val, x.val, parent(H.dval), x.dval)
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
        H::Const{<:HermTri},
        x::Duplicated{<:AbstractVecOrMat}
    )
    y, dy = mul_frule_impl(H.val, x.val, ZeroTangent(), x.dval)
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
        H::Duplicated{<:HermTri},
        x::Const{<:AbstractVecOrMat}
    )
    y, dy = mul_frule_impl(H.val, x.val, parent(H.dval), ZeroTangent())
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
        S::Duplicated{<:SymTri},
        x::Duplicated{<:AbstractVecOrMat}
    )
    y, dy = mul_frule_impl(S.val, x.val, parent(S.dval), x.dval)
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
        S::Const{<:SymTri},
        x::Duplicated{<:AbstractVecOrMat}
    )
    y, dy = mul_frule_impl(S.val, x.val, ZeroTangent(), x.dval)
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
        S::Duplicated{<:SymTri},
        x::Const{<:AbstractVecOrMat}
    )
    y, dy = mul_frule_impl(S.val, x.val, parent(S.dval), ZeroTangent())
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
        H::Annotation{<:HermTri},
        x::Annotation{<:AbstractVecOrMat}
    )
    y = H.val * x.val

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
        H::Annotation{<:HermTri},
        x::Annotation{<:AbstractVecOrMat}
    )
    xval, y, shadow = tape

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    ΔH, Δx = mul_rrule_impl(H.val, xval, y, Δy)

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
        S::Annotation{<:SymTri},
        x::Annotation{<:AbstractVecOrMat}
    )
    y = S.val * x.val

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
        S::Annotation{<:SymTri},
        x::Annotation{<:AbstractVecOrMat}
    )
    xval, y, shadow = tape

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    ΔS, Δx = mul_rrule_impl(S.val, xval, y, Δy)

    if x isa Duplicated
        axpy!(1, Δx, x.dval)
    end

    if S isa Duplicated
        axpy!(1, unthunk(ΔS), parent(S.dval))
    end

    return (nothing, nothing)
end
