# x \ L where x is a scalar

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(\)},
        RT::Type,
        x::Duplicated{<:Number},
        L::Duplicated{<:ChordalTriangular}
    )
    y, dy = ldiv_frule_impl(x.val, L.val, x.dval, L.dval)
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
        x::Const{<:Number},
        L::Duplicated{<:ChordalTriangular}
    )
    y, dy = ldiv_frule_impl(x.val, L.val, ZeroTangent(), L.dval)
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
        x::Duplicated{<:Number},
        L::Const{<:ChordalTriangular}
    )
    y, dy = ldiv_frule_impl(x.val, L.val, x.dval, ZeroTangent())
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
        x::Annotation{<:Number},
        L::Annotation{<:ChordalTriangular}
    )
    y = x.val \ L.val

    if needs_shadow(config)
        shadow = zero(y)
    else
        shadow = nothing
    end

    _, ox, oL = overwritten(config)
    if ox || oL
        cache = y
    else
        cache = nothing
    end

    tape = (cache, shadow)
    return AugmentedReturn(y, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(\)},
        dret,
        tape,
        x::Annotation{<:Number},
        L::Annotation{<:ChordalTriangular}
    )
    cache, shadow = tape

    if cache === nothing
        y = x.val \ L.val
    else
        y = cache
    end

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    if L isa Duplicated
        axpy!(inv(conj(x.val)), Δy, L.dval)
    end

    if x isa Duplicated
        x.dval[] += -dot(y, Δy) / conj(x.val)
    end

    if x isa Active
        Δx = -dot(y, Δy) / conj(x.val)
    else
        Δx = nothing
    end
    return (Δx, nothing)
end

# x \ H where x is a scalar and H is HermTri

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(\)},
        RT::Type,
        x::Duplicated{<:Real},
        H::Duplicated{<:HermTri{UPLO}}
    ) where {UPLO}
    y, dy = ldiv_frule_impl(x.val, H.val, x.dval, parent(H.dval))
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
        func::Const{typeof(\)},
        RT::Type,
        x::Const{<:Real},
        H::Duplicated{<:HermTri{UPLO}}
    ) where {UPLO}
    y, dy = ldiv_frule_impl(x.val, H.val, ZeroTangent(), parent(H.dval))
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
        func::Const{typeof(\)},
        RT::Type,
        x::Duplicated{<:Real},
        H::Const{<:HermTri{UPLO}}
    ) where {UPLO}
    y, dy = ldiv_frule_impl(x.val, H.val, x.dval, ZeroTangent())
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
        func::Const{typeof(\)},
        RT::Type,
        x::Annotation{<:Real},
        H::Annotation{<:HermTri{UPLO}}
    ) where {UPLO}
    y = x.val \ H.val

    if needs_shadow(config)
        shadow = zero(y)
    else
        shadow = nothing
    end

    _, ox, oH = overwritten(config)
    if ox || oH
        cache = y
    else
        cache = nothing
    end

    tape = (cache, shadow)
    return AugmentedReturn(y, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(\)},
        dret,
        tape,
        x::Annotation{<:Real},
        H::Annotation{<:HermTri{UPLO}}
    ) where {UPLO}
    cache, shadow = tape

    if cache === nothing
        y = x.val \ H.val
    else
        y = cache
    end

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    if H isa Duplicated
        axpy!(inv(x.val), Δy, H.dval)
    end

    if x isa Active
        Δx = -dot(y, Hermitian(Δy, UPLO)) / x.val
    else
        Δx = nothing
    end
    return (Δx, nothing)
end

# x \ H where x is a scalar and H is SymTri

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(\)},
        RT::Type,
        x::Duplicated{<:Real},
        H::Duplicated{<:SymTri{UPLO}}
    ) where {UPLO}
    y, dy = ldiv_frule_impl(x.val, H.val, x.dval, parent(H.dval))
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
        func::Const{typeof(\)},
        RT::Type,
        x::Const{<:Real},
        H::Duplicated{<:SymTri{UPLO}}
    ) where {UPLO}
    y, dy = ldiv_frule_impl(x.val, H.val, ZeroTangent(), parent(H.dval))
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
        func::Const{typeof(\)},
        RT::Type,
        x::Duplicated{<:Real},
        H::Const{<:SymTri{UPLO}}
    ) where {UPLO}
    y, dy = ldiv_frule_impl(x.val, H.val, x.dval, ZeroTangent())
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
        func::Const{typeof(\)},
        RT::Type,
        x::Annotation{<:Real},
        H::Annotation{<:SymTri{UPLO}}
    ) where {UPLO}
    y = x.val \ H.val

    if needs_shadow(config)
        shadow = zero(y)
    else
        shadow = nothing
    end

    _, ox, oH = overwritten(config)
    if ox || oH
        cache = y
    else
        cache = nothing
    end

    tape = (cache, shadow)
    return AugmentedReturn(y, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(\)},
        dret,
        tape,
        x::Annotation{<:Real},
        H::Annotation{<:SymTri{UPLO}}
    ) where {UPLO}
    cache, shadow = tape

    if cache === nothing
        y = x.val \ H.val
    else
        y = cache
    end

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    if H isa Duplicated
        axpy!(inv(x.val), Δy, H.dval)
    end

    if x isa Active
        Δx = -dot(y, Symmetric(Δy, UPLO)) / x.val
    else
        Δx = nothing
    end
    return (Δx, nothing)
end
