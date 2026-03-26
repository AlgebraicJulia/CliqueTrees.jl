# L + L where L is ChordalTriangular

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(+)},
        RT::Type,
        A::Duplicated{<:ChordalTriangular},
        B::Duplicated{<:ChordalTriangular}
    )
    y, dy = add_frule_impl(A.val, B.val, A.dval, B.dval)
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
        A::Const{<:ChordalTriangular},
        B::Duplicated{<:ChordalTriangular}
    )
    y, dy = add_frule_impl(A.val, B.val, ZeroTangent(), B.dval)
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
        A::Duplicated{<:ChordalTriangular},
        B::Const{<:ChordalTriangular}
    )
    y, dy = add_frule_impl(A.val, B.val, A.dval, ZeroTangent())
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
        func::Const{typeof(+)},
        RT::Type,
        A::Annotation{<:ChordalTriangular},
        B::Annotation{<:ChordalTriangular}
    )
    y = A.val + B.val

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
        A::Annotation{<:ChordalTriangular},
        B::Annotation{<:ChordalTriangular}
    )
    shadow = tape

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    if A isa Duplicated
        axpy!(1, Δy, A.dval)
    end

    if B isa Duplicated
        axpy!(1, Δy, B.dval)
    end

    return (nothing, nothing)
end

# H + H where H is HermTri

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(+)},
        RT::Type,
        A::Duplicated{<:HermTri{UPLO}},
        B::Duplicated{<:HermTri{UPLO}}
    ) where {UPLO}
    y, dy = add_frule_impl(A.val, B.val, parent(A.dval), parent(B.dval))
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
        A::Const{<:HermTri{UPLO}},
        B::Duplicated{<:HermTri{UPLO}}
    ) where {UPLO}
    y, dy = add_frule_impl(A.val, B.val, ZeroTangent(), parent(B.dval))
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
        A::Duplicated{<:HermTri{UPLO}},
        B::Const{<:HermTri{UPLO}}
    ) where {UPLO}
    y, dy = add_frule_impl(A.val, B.val, parent(A.dval), ZeroTangent())
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

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1},
        func::Const{typeof(+)},
        RT::Type,
        A::Annotation{<:HermTri{UPLO}},
        B::Annotation{<:HermTri{UPLO}}
    ) where {UPLO}
    y = A.val + B.val

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
        A::Annotation{<:HermTri{UPLO}},
        B::Annotation{<:HermTri{UPLO}}
    ) where {UPLO}
    shadow = tape

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    if A isa Duplicated
        axpy!(1, Δy, A.dval)
    end

    if B isa Duplicated
        axpy!(1, Δy, B.dval)
    end

    return (nothing, nothing)
end

# H + H where H is SymTri

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(+)},
        RT::Type,
        A::Duplicated{<:SymTri{UPLO}},
        B::Duplicated{<:SymTri{UPLO}}
    ) where {UPLO}
    y, dy = add_frule_impl(A.val, B.val, parent(A.dval), parent(B.dval))
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
        A::Const{<:SymTri{UPLO}},
        B::Duplicated{<:SymTri{UPLO}}
    ) where {UPLO}
    y, dy = add_frule_impl(A.val, B.val, ZeroTangent(), parent(B.dval))
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
        A::Duplicated{<:SymTri{UPLO}},
        B::Const{<:SymTri{UPLO}}
    ) where {UPLO}
    y, dy = add_frule_impl(A.val, B.val, parent(A.dval), ZeroTangent())
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

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1},
        func::Const{typeof(+)},
        RT::Type,
        A::Annotation{<:SymTri{UPLO}},
        B::Annotation{<:SymTri{UPLO}}
    ) where {UPLO}
    y = A.val + B.val

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
        A::Annotation{<:SymTri{UPLO}},
        B::Annotation{<:SymTri{UPLO}}
    ) where {UPLO}
    shadow = tape

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    if A isa Duplicated
        axpy!(1, Δy, A.dval)
    end

    if B isa Duplicated
        axpy!(1, Δy, B.dval)
    end

    return (nothing, nothing)
end
