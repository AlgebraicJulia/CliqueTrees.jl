# x * L where x is a scalar

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(*)},
        RT::Type,
        x::Duplicated{<:Number},
        L::Duplicated{<:ChordalTriangular}
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
        x::Duplicated{<:Number},
        L::Const{<:ChordalTriangular}
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

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(*)},
        RT::Type,
        x::Const{<:Number},
        L::Duplicated{<:ChordalTriangular}
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

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1},
        func::Const{typeof(*)},
        RT::Type,
        x::Annotation{<:Number},
        L::Annotation{<:ChordalTriangular}
    )
    y = x.val * L.val

    if needs_shadow(config)
        shadow = zero(y)
    else
        shadow = nothing
    end

    _, ox, oL = overwritten(config)
    if oL
        Lcache = L.val
    else
        Lcache = nothing
    end

    tape = (Lcache, shadow)
    return AugmentedReturn(y, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(*)},
        dret,
        tape,
        x::Annotation{<:Number},
        L::Annotation{<:ChordalTriangular}
    )
    Lcache, shadow = tape

    if Lcache === nothing
        Lval = L.val
    else
        Lval = Lcache
    end

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    if L isa Duplicated
        axpy!(conj(x.val), Δy, L.dval)
    end

    if x isa Duplicated
        x.dval[] += dot(Lval, Δy)
    end

    if x isa Active
        Δx = dot(Lval, Δy)
    else
        Δx = nothing
    end
    return (Δx, nothing)
end

# x * H where x is a scalar and H is HermTri

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(*)},
        RT::Type,
        x::Duplicated{<:Real},
        H::Duplicated{<:HermTri{UPLO}}
    ) where {UPLO}
    y, dy = mul_frule_impl(x.val, H.val, x.dval, parent(H.dval))
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
        func::Const{typeof(*)},
        RT::Type,
        x::Duplicated{<:Real},
        H::Const{<:HermTri{UPLO}}
    ) where {UPLO}
    y, dy = mul_frule_impl(x.val, H.val, x.dval, ZeroTangent())
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
        func::Const{typeof(*)},
        RT::Type,
        x::Const{<:Real},
        H::Duplicated{<:HermTri{UPLO}}
    ) where {UPLO}
    y, dy = mul_frule_impl(x.val, H.val, ZeroTangent(), parent(H.dval))
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
        func::Const{typeof(*)},
        RT::Type,
        x::Annotation{<:Real},
        H::Annotation{<:HermTri{UPLO}}
    ) where {UPLO}
    y = x.val * H.val

    if needs_shadow(config)
        shadow = zero(y)
    else
        shadow = nothing
    end

    _, ox, oH = overwritten(config)
    if oH
        Hcache = H.val
    else
        Hcache = nothing
    end

    tape = (Hcache, shadow)
    return AugmentedReturn(y, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(*)},
        dret,
        tape,
        x::Annotation{<:Real},
        H::Annotation{<:HermTri{UPLO}}
    ) where {UPLO}
    Hcache, shadow = tape

    if Hcache === nothing
        Hval = H.val
    else
        Hval = Hcache
    end

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    if H isa Duplicated
        axpy!(x.val, Δy, H.dval)
    end

    if x isa Active
        Δx = dot(Hval, Hermitian(Δy, UPLO))
    else
        Δx = nothing
    end
    return (Δx, nothing)
end

# x * H where x is a scalar and H is SymTri

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(*)},
        RT::Type,
        x::Duplicated{<:Real},
        H::Duplicated{<:SymTri{UPLO}}
    ) where {UPLO}
    y, dy = mul_frule_impl(x.val, H.val, x.dval, parent(H.dval))
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
        func::Const{typeof(*)},
        RT::Type,
        x::Duplicated{<:Real},
        H::Const{<:SymTri{UPLO}}
    ) where {UPLO}
    y, dy = mul_frule_impl(x.val, H.val, x.dval, ZeroTangent())
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
        func::Const{typeof(*)},
        RT::Type,
        x::Const{<:Real},
        H::Duplicated{<:SymTri{UPLO}}
    ) where {UPLO}
    y, dy = mul_frule_impl(x.val, H.val, ZeroTangent(), parent(H.dval))
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
        func::Const{typeof(*)},
        RT::Type,
        x::Annotation{<:Real},
        H::Annotation{<:SymTri{UPLO}}
    ) where {UPLO}
    y = x.val * H.val

    if needs_shadow(config)
        shadow = zero(y)
    else
        shadow = nothing
    end

    _, ox, oH = overwritten(config)
    if oH
        Hcache = H.val
    else
        Hcache = nothing
    end

    tape = (Hcache, shadow)
    return AugmentedReturn(y, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(*)},
        dret,
        tape,
        x::Annotation{<:Real},
        H::Annotation{<:SymTri{UPLO}}
    ) where {UPLO}
    Hcache, shadow = tape

    if Hcache === nothing
        Hval = H.val
    else
        Hval = Hcache
    end

    if dret isa Active
        Δy = dret.val
    else
        Δy = shadow
    end

    if H isa Duplicated
        axpy!(x.val, Δy, H.dval)
    end

    if x isa Active
        Δx = dot(Hval, Symmetric(Δy, UPLO))
    else
        Δx = nothing
    end
    return (Δx, nothing)
end
