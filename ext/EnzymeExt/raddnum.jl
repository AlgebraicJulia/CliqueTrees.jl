# L + x * I where x is a scalar (UniformScaling)

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(+)},
        RT::Type,
        L::Duplicated{<:ChordalTriangular},
        x::Duplicated{<:UniformScaling}
    )
    y, dy = add_frule_impl(L.val, x.val, L.dval, x.dval)
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
        func::Const{typeof(+)},
        RT::Type,
        L::Const{<:ChordalTriangular},
        x::Duplicated{<:UniformScaling}
    )
    y, dy = add_frule_impl(L.val, x.val, ZeroTangent(), x.dval)
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
        func::Const{typeof(+)},
        RT::Type,
        L::Duplicated{<:ChordalTriangular},
        x::Const{<:UniformScaling}
    )
    y, dy = add_frule_impl(L.val, x.val, L.dval, ZeroTangent())
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

# H + x * I where x is a scalar (UniformScaling) and H is HermTri

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(+)},
        RT::Type,
        H::Duplicated{<:HermTri{UPLO}},
        x::Duplicated{<:UniformScaling}
    ) where {UPLO}
    y, dy = add_frule_impl(H.val, x.val, parent(H.dval), x.dval)
    # Enzyme requires shadow to have same type as primal
    if needs_primal(config) && needs_shadow(config)
        return Duplicated(y, Hermitian(dy, UPLO))
    elseif needs_shadow(config)
        return Hermitian(dy, UPLO)
    elseif needs_primal(config)
        return y
    else
        return
    end
end

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(+)},
        RT::Type,
        H::Const{<:HermTri{UPLO}},
        x::Duplicated{<:UniformScaling}
    ) where {UPLO}
    y, dy = add_frule_impl(H.val, x.val, ZeroTangent(), x.dval)
    # Enzyme requires shadow to have same type as primal
    if needs_primal(config) && needs_shadow(config)
        return Duplicated(y, Hermitian(dy, UPLO))
    elseif needs_shadow(config)
        return Hermitian(dy, UPLO)
    elseif needs_primal(config)
        return y
    else
        return
    end
end

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(+)},
        RT::Type,
        H::Duplicated{<:HermTri{UPLO}},
        x::Const{<:UniformScaling}
    ) where {UPLO}
    y, dy = add_frule_impl(H.val, x.val, parent(H.dval), ZeroTangent())
    # Enzyme requires shadow to have same type as primal
    if needs_primal(config) && needs_shadow(config)
        return Duplicated(y, Hermitian(dy, UPLO))
    elseif needs_shadow(config)
        return Hermitian(dy, UPLO)
    elseif needs_primal(config)
        return y
    else
        return
    end
end

# H + x * I where x is a scalar (UniformScaling) and H is SymTri

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(+)},
        RT::Type,
        H::Duplicated{<:SymTri{UPLO}},
        x::Duplicated{<:UniformScaling}
    ) where {UPLO}
    y, dy = add_frule_impl(H.val, x.val, parent(H.dval), x.dval)
    # Enzyme requires shadow to have same type as primal
    if needs_primal(config) && needs_shadow(config)
        return Duplicated(y, Symmetric(dy, UPLO))
    elseif needs_shadow(config)
        return Symmetric(dy, UPLO)
    elseif needs_primal(config)
        return y
    else
        return
    end
end

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(+)},
        RT::Type,
        H::Const{<:SymTri{UPLO}},
        x::Duplicated{<:UniformScaling}
    ) where {UPLO}
    y, dy = add_frule_impl(H.val, x.val, ZeroTangent(), x.dval)
    # Enzyme requires shadow to have same type as primal
    if needs_primal(config) && needs_shadow(config)
        return Duplicated(y, Symmetric(dy, UPLO))
    elseif needs_shadow(config)
        return Symmetric(dy, UPLO)
    elseif needs_primal(config)
        return y
    else
        return
    end
end

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(+)},
        RT::Type,
        H::Duplicated{<:SymTri{UPLO}},
        x::Const{<:UniformScaling}
    ) where {UPLO}
    y, dy = add_frule_impl(H.val, x.val, parent(H.dval), ZeroTangent())
    # Enzyme requires shadow to have same type as primal
    if needs_primal(config) && needs_shadow(config)
        return Duplicated(y, Symmetric(dy, UPLO))
    elseif needs_shadow(config)
        return Symmetric(dy, UPLO)
    elseif needs_primal(config)
        return y
    else
        return
    end
end

# Reverse rules

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1},
        func::Const{typeof(+)},
        RT::Type,
        L::Annotation{<:ChordalTriangular},
        x::Annotation{<:UniformScaling}
    )
    y = L.val + x.val

    if needs_shadow(config)
        shadow = zero(y)
    else
        shadow = nothing
    end

    tape = shadow
    return AugmentedReturn(y, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(+)},
        dret,
        tape,
        L::Annotation{<:ChordalTriangular},
        x::Annotation{<:UniformScaling}
    )
    shadow = tape

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    if L isa Duplicated
        axpy!(1, Δy, L.dval)
    end

    Δx = x isa Active ? tr(Δy) * I : nothing
    return (nothing, Δx)
end

# H + x * I where x is a scalar (UniformScaling) and H is HermTri

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1},
        func::Const{typeof(+)},
        RT::Type,
        H::Annotation{<:HermTri{UPLO}},
        x::Annotation{<:UniformScaling}
    ) where {UPLO}
    y = H.val + x.val

    if needs_shadow(config)
        shadow = zero(y)
    else
        shadow = nothing
    end

    tape = shadow
    return AugmentedReturn(y, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(+)},
        dret,
        tape,
        H::Annotation{<:HermTri{UPLO}},
        x::Annotation{<:UniformScaling}
    ) where {UPLO}
    shadow = tape

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    if H isa Duplicated
        axpy!(1, Δy, H.dval)
    end

    Δx = x isa Active ? tr(Δy) * I : nothing
    return (nothing, Δx)
end

# H + x * I where x is a scalar (UniformScaling) and H is SymTri

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1},
        func::Const{typeof(+)},
        RT::Type,
        H::Annotation{<:SymTri{UPLO}},
        x::Annotation{<:UniformScaling}
    ) where {UPLO}
    y = H.val + x.val

    if needs_shadow(config)
        shadow = zero(y)
    else
        shadow = nothing
    end

    tape = shadow
    return AugmentedReturn(y, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(+)},
        dret,
        tape,
        H::Annotation{<:SymTri{UPLO}},
        x::Annotation{<:UniformScaling}
    ) where {UPLO}
    shadow = tape

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    if H isa Duplicated
        axpy!(1, Δy, H.dval)
    end

    Δx = x isa Active ? tr(Δy) * I : nothing
    return (nothing, Δx)
end
