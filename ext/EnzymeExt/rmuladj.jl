function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(*)},
        RT::Type,
        x::Duplicated{<:AbstractMatrix},
        A::Duplicated{<:AdjTri{:N}}
    )
    y, dy = mul_frule_impl(x.val, A.val, x.dval, parent(A.dval))
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
        A::Duplicated{<:AdjTri{:N}}
    )
    y, dy = mul_frule_impl(x.val, A.val, ZeroTangent(), parent(A.dval))
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
        A::Const{<:AdjTri{:N}}
    )
    y, dy = mul_frule_impl(x.val, A.val, x.dval, ZeroTangent())
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
        A::Duplicated{<:TransTri{:N}}
    )
    y, dy = mul_frule_impl(x.val, A.val, x.dval, parent(A.dval))
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
        A::Duplicated{<:TransTri{:N}}
    )
    y, dy = mul_frule_impl(x.val, A.val, ZeroTangent(), parent(A.dval))
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
        A::Const{<:TransTri{:N}}
    )
    y, dy = mul_frule_impl(x.val, A.val, x.dval, ZeroTangent())
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
        A::Annotation{<:AdjTri{:N}}
    )
    y = x.val * A.val

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
        A::Annotation{<:AdjTri{:N}}
    )
    xval, y, shadow = tape

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    Δx, ΔL = mul_rrule_impl(xval, A.val, y, Δy)

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
        x::Annotation{<:AbstractMatrix},
        A::Annotation{<:TransTri{:N}}
    )
    y = x.val * A.val

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
        A::Annotation{<:TransTri{:N}}
    )
    xval, y, shadow = tape

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    Δx, ΔL = mul_rrule_impl(xval, A.val, y, Δy)

    if x isa Duplicated
        axpy!(1, Δx, x.dval)
    end

    if A isa Duplicated
        axpy!(1, unthunk(ΔL), parent(A.dval))
    end

    return (nothing, nothing)
end
