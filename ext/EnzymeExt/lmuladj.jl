function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(*)},
        RT::Type,
        A::Duplicated{<:AdjTri{:N}},
        x::Duplicated{<:AbstractVecOrMat}
    )
    y, dy = mul_frule_impl(A.val, x.val, parent(A.dval), x.dval)
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
        A::Const{<:AdjTri{:N}},
        x::Duplicated{<:AbstractVecOrMat}
    )
    y, dy = mul_frule_impl(A.val, x.val, ZeroTangent(), x.dval)
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
        A::Duplicated{<:AdjTri{:N}},
        x::Const{<:AbstractVecOrMat}
    )
    y, dy = mul_frule_impl(A.val, x.val, parent(A.dval), ZeroTangent())
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
        A::Duplicated{<:TransTri{:N}},
        x::Duplicated{<:AbstractVecOrMat}
    )
    y, dy = mul_frule_impl(A.val, x.val, parent(A.dval), x.dval)
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
        A::Const{<:TransTri{:N}},
        x::Duplicated{<:AbstractVecOrMat}
    )
    y, dy = mul_frule_impl(A.val, x.val, ZeroTangent(), x.dval)
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
        A::Duplicated{<:TransTri{:N}},
        x::Const{<:AbstractVecOrMat}
    )
    y, dy = mul_frule_impl(A.val, x.val, parent(A.dval), ZeroTangent())
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
        A::Annotation{<:AdjTri{:N}},
        x::Annotation{<:AbstractVecOrMat}
    )
    y = A.val * x.val

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
        A::Annotation{<:AdjTri{:N}},
        x::Annotation{<:AbstractVecOrMat}
    )
    xval, y, shadow = tape

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    ΔL, Δx = mul_rrule_impl(A.val, xval, y, Δy)

    if x isa Duplicated
        axpy!(1, Δx, x.dval)
    end

    if A isa Duplicated
        axpy!(1, unthunk(ΔL), parent(A.dval))
    end

    return (nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1},
        func::Const{typeof(*)},
        RT::Type,
        A::Annotation{<:TransTri{:N}},
        x::Annotation{<:AbstractVecOrMat}
    )
    y = A.val * x.val

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
        A::Annotation{<:TransTri{:N}},
        x::Annotation{<:AbstractVecOrMat}
    )
    xval, y, shadow = tape

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    ΔL, Δx = mul_rrule_impl(A.val, xval, y, Δy)

    if x isa Duplicated
        axpy!(1, Δx, x.dval)
    end

    if A isa Duplicated
        axpy!(1, unthunk(ΔL), parent(A.dval))
    end

    return (nothing, nothing)
end
