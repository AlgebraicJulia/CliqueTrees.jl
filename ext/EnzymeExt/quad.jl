function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(dot)},
        RT::Type,
        X::Duplicated{<:AbstractVecOrMat},
        H::Duplicated{<:HermTri},
        Y::Duplicated{<:AbstractVecOrMat}
    )
    y, dy = dot_frule_impl(X.val, H.val, Y.val, X.dval, parent(H.dval), Y.dval)
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
        func::Const{typeof(dot)},
        RT::Type,
        X::Const{<:AbstractVecOrMat},
        H::Duplicated{<:HermTri},
        Y::Duplicated{<:AbstractVecOrMat}
    )
    y, dy = dot_frule_impl(X.val, H.val, Y.val, ZeroTangent(), parent(H.dval), Y.dval)
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
        func::Const{typeof(dot)},
        RT::Type,
        X::Duplicated{<:AbstractVecOrMat},
        H::Const{<:HermTri},
        Y::Duplicated{<:AbstractVecOrMat}
    )
    y, dy = dot_frule_impl(X.val, H.val, Y.val, X.dval, ZeroTangent(), Y.dval)
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
        func::Const{typeof(dot)},
        RT::Type,
        X::Duplicated{<:AbstractVecOrMat},
        H::Duplicated{<:HermTri},
        Y::Const{<:AbstractVecOrMat}
    )
    y, dy = dot_frule_impl(X.val, H.val, Y.val, X.dval, parent(H.dval), ZeroTangent())
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
        func::Const{typeof(dot)},
        RT::Type,
        X::Const{<:AbstractVecOrMat},
        H::Const{<:HermTri},
        Y::Duplicated{<:AbstractVecOrMat}
    )
    y, dy = dot_frule_impl(X.val, H.val, Y.val, ZeroTangent(), ZeroTangent(), Y.dval)
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
        func::Const{typeof(dot)},
        RT::Type,
        X::Const{<:AbstractVecOrMat},
        H::Duplicated{<:HermTri},
        Y::Const{<:AbstractVecOrMat}
    )
    y, dy = dot_frule_impl(X.val, H.val, Y.val, ZeroTangent(), parent(H.dval), ZeroTangent())
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
        func::Const{typeof(dot)},
        RT::Type,
        X::Duplicated{<:AbstractVecOrMat},
        H::Const{<:HermTri},
        Y::Const{<:AbstractVecOrMat}
    )
    y, dy = dot_frule_impl(X.val, H.val, Y.val, X.dval, ZeroTangent(), ZeroTangent())
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
        func::Const{typeof(dot)},
        RT::Type,
        X::Duplicated{<:AbstractVecOrMat},
        S::Duplicated{<:SymTri},
        Y::Duplicated{<:AbstractVecOrMat}
    )
    y, dy = dot_frule_impl(X.val, S.val, Y.val, X.dval, parent(S.dval), Y.dval)
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
        func::Const{typeof(dot)},
        RT::Type,
        X::Const{<:AbstractVecOrMat},
        S::Duplicated{<:SymTri},
        Y::Duplicated{<:AbstractVecOrMat}
    )
    y, dy = dot_frule_impl(X.val, S.val, Y.val, ZeroTangent(), parent(S.dval), Y.dval)
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
        func::Const{typeof(dot)},
        RT::Type,
        X::Duplicated{<:AbstractVecOrMat},
        S::Const{<:SymTri},
        Y::Duplicated{<:AbstractVecOrMat}
    )
    y, dy = dot_frule_impl(X.val, S.val, Y.val, X.dval, ZeroTangent(), Y.dval)
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
        func::Const{typeof(dot)},
        RT::Type,
        X::Duplicated{<:AbstractVecOrMat},
        S::Duplicated{<:SymTri},
        Y::Const{<:AbstractVecOrMat}
    )
    y, dy = dot_frule_impl(X.val, S.val, Y.val, X.dval, parent(S.dval), ZeroTangent())
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
        func::Const{typeof(dot)},
        RT::Type,
        X::Const{<:AbstractVecOrMat},
        S::Const{<:SymTri},
        Y::Duplicated{<:AbstractVecOrMat}
    )
    y, dy = dot_frule_impl(X.val, S.val, Y.val, ZeroTangent(), ZeroTangent(), Y.dval)
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
        func::Const{typeof(dot)},
        RT::Type,
        X::Const{<:AbstractVecOrMat},
        S::Duplicated{<:SymTri},
        Y::Const{<:AbstractVecOrMat}
    )
    y, dy = dot_frule_impl(X.val, S.val, Y.val, ZeroTangent(), parent(S.dval), ZeroTangent())
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
        func::Const{typeof(dot)},
        RT::Type,
        X::Duplicated{<:AbstractVecOrMat},
        S::Const{<:SymTri},
        Y::Const{<:AbstractVecOrMat}
    )
    y, dy = dot_frule_impl(X.val, S.val, Y.val, X.dval, ZeroTangent(), ZeroTangent())
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
        func::Const{typeof(dot)},
        ::Type{<:Active},
        X::Annotation{<:AbstractVecOrMat},
        H::Annotation{<:HermTri},
        Y::Annotation{<:AbstractVecOrMat}
    )
    HY = H.val * Y.val
    HX = H.val * X.val
    y = dot(X.val, HY)
    tape = (X.val, Y.val, y, HX, HY)

    if needs_primal(config)
        primal = y
    else
        primal = nothing
    end

    return AugmentedReturn(primal, nothing, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(dot)},
        dret::Active,
        tape,
        X::Annotation{<:AbstractVecOrMat},
        H::Annotation{<:HermTri},
        Y::Annotation{<:AbstractVecOrMat}
    )
    Δy = dret.val
    Xval, Yval, y, HX, HY = tape

    ΔX, ΔH, ΔY = dot_rrule_impl(Xval, H.val, Yval, y, HX, HY, Δy)

    if X isa Duplicated
        axpy!(1, unthunk(ΔX), X.dval)
    end

    if Y isa Duplicated
        axpy!(1, unthunk(ΔY), Y.dval)
    end

    if H isa Duplicated
        axpy!(1, unthunk(ΔH), parent(H.dval))
    end

    return (nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1},
        func::Const{typeof(dot)},
        ::Type{<:Active},
        X::Annotation{<:AbstractVecOrMat},
        S::Annotation{<:SymTri},
        Y::Annotation{<:AbstractVecOrMat}
    )
    SY = S.val * Y.val
    SX = S.val * X.val
    y = dot(X.val, SY)
    tape = (X.val, Y.val, y, SX, SY)

    if needs_primal(config)
        primal = y
    else
        primal = nothing
    end

    return AugmentedReturn(primal, nothing, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(dot)},
        dret::Active,
        tape,
        X::Annotation{<:AbstractVecOrMat},
        S::Annotation{<:SymTri},
        Y::Annotation{<:AbstractVecOrMat}
    )
    Δy = dret.val
    Xval, Yval, y, SX, SY = tape

    ΔX, ΔS, ΔY = dot_rrule_impl(Xval, S.val, Yval, y, SX, SY, Δy)

    if X isa Duplicated
        axpy!(1, unthunk(ΔX), X.dval)
    end

    if Y isa Duplicated
        axpy!(1, unthunk(ΔY), Y.dval)
    end

    if S isa Duplicated
        axpy!(1, unthunk(ΔS), parent(S.dval))
    end

    return (nothing, nothing, nothing)
end
