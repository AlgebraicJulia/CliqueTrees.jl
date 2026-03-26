# x * I + L where x is a scalar (UniformScaling)

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(+)},
        RT::Type,
        x::Duplicated{<:UniformScaling},
        L::Duplicated{<:ChordalTriangular}
    )
    y, dy = add_frule_impl(x.val, L.val, x.dval, L.dval)
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
        x::Const{<:UniformScaling},
        L::Duplicated{<:ChordalTriangular}
    )
    y, dy = add_frule_impl(x.val, L.val, ZeroTangent(), L.dval)
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
        x::Duplicated{<:UniformScaling},
        L::Const{<:ChordalTriangular}
    )
    y, dy = add_frule_impl(x.val, L.val, x.dval, ZeroTangent())
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

# x * I + H where x is a scalar (UniformScaling) and H is HermTri

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(+)},
        RT::Type,
        x::Duplicated{<:UniformScaling},
        H::Duplicated{<:HermTri{UPLO}}
    ) where {UPLO}
    y, dy = add_frule_impl(x.val, H.val, x.dval, parent(H.dval))
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
        x::Const{<:UniformScaling},
        H::Duplicated{<:HermTri{UPLO}}
    ) where {UPLO}
    y, dy = add_frule_impl(x.val, H.val, ZeroTangent(), parent(H.dval))
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
        x::Duplicated{<:UniformScaling},
        H::Const{<:HermTri{UPLO}}
    ) where {UPLO}
    y, dy = add_frule_impl(x.val, H.val, x.dval, ZeroTangent())
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

# x * I + H where x is a scalar (UniformScaling) and H is SymTri

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(+)},
        RT::Type,
        x::Duplicated{<:UniformScaling},
        H::Duplicated{<:SymTri{UPLO}}
    ) where {UPLO}
    y, dy = add_frule_impl(x.val, H.val, x.dval, parent(H.dval))
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
        x::Const{<:UniformScaling},
        H::Duplicated{<:SymTri{UPLO}}
    ) where {UPLO}
    y, dy = add_frule_impl(x.val, H.val, ZeroTangent(), parent(H.dval))
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
        x::Duplicated{<:UniformScaling},
        H::Const{<:SymTri{UPLO}}
    ) where {UPLO}
    y, dy = add_frule_impl(x.val, H.val, x.dval, ZeroTangent())
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
        x::Annotation{<:UniformScaling},
        L::Annotation{<:ChordalTriangular}
    )
    y = x.val + L.val

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
        x::Annotation{<:UniformScaling},
        L::Annotation{<:ChordalTriangular}
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
    return (Δx, nothing)
end

# x * I + H where x is a scalar (UniformScaling) and H is HermTri

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1},
        func::Const{typeof(+)},
        RT::Type,
        x::Annotation{<:UniformScaling},
        H::Annotation{<:HermTri{UPLO}}
    ) where {UPLO}
    y = x.val + H.val

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
        x::Annotation{<:UniformScaling},
        H::Annotation{<:HermTri{UPLO}}
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
    return (Δx, nothing)
end

# x * I + H where x is a scalar (UniformScaling) and H is SymTri

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1},
        func::Const{typeof(+)},
        RT::Type,
        x::Annotation{<:UniformScaling},
        H::Annotation{<:SymTri{UPLO}}
    ) where {UPLO}
    y = x.val + H.val

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
        x::Annotation{<:UniformScaling},
        H::Annotation{<:SymTri{UPLO}}
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
    return (Δx, nothing)
end
