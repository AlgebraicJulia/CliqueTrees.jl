function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(*)},
        RT::Type,
        x::Duplicated{<:AbstractMatrix},
        L::Duplicated{<:ChordalTriangular{:N}}
    )
    y, dy = mul_frule_impl(x.val, L.val, x.dval, L.dval)
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
        L::Duplicated{<:ChordalTriangular{:N}}
    )
    y, dy = mul_frule_impl(x.val, L.val, ZeroTangent(), L.dval)
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
        L::Const{<:ChordalTriangular{:N}}
    )
    y, dy = mul_frule_impl(x.val, L.val, x.dval, ZeroTangent())
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
        L::Annotation{<:ChordalTriangular{:N}}
    )
    y = x.val * L.val

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
        L::Annotation{<:ChordalTriangular{:N}}
    )
    xval, y, shadow = tape

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    Δx, ΔL = mul_rrule_impl(xval, L.val, y, Δy)

    if x isa Duplicated
        axpy!(1, Δx, x.dval)
    end

    if L isa Duplicated
        axpy!(1, unthunk(ΔL), L.dval)
    end

    return (nothing, nothing)
end
