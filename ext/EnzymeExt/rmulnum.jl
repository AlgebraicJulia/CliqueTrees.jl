# L * x where x is a scalar

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(*)},
        RT::Type,
        L::Duplicated{<:ChordalTriangular},
        x::Duplicated{<:Number}
    )
    y, dy = mul_frule_impl(L.val, x.val, L.dval, x.dval)
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
        L::Const{<:ChordalTriangular},
        x::Duplicated{<:Number}
    )
    y, dy = mul_frule_impl(L.val, x.val, ZeroTangent(), x.dval)
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
        L::Duplicated{<:ChordalTriangular},
        x::Const{<:Number}
    )
    y, dy = mul_frule_impl(L.val, x.val, L.dval, ZeroTangent())
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
        L::Annotation{<:ChordalTriangular},
        x::Annotation{<:Number}
    )
    y = L.val * x.val

    if needs_shadow(config)
        shadow = zero(y)
    else
        shadow = nothing
    end

    _, oL, ox = overwritten(config)
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
        L::Annotation{<:ChordalTriangular},
        x::Annotation{<:Number}
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
    return (nothing, Δx)
end

# H * x where x is a scalar and H is HermTri

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(*)},
        RT::Type,
        H::Duplicated{<:HermTri{UPLO}},
        x::Duplicated{<:Real}
    ) where {UPLO}
    y, dy = mul_frule_impl(H.val, x.val, parent(H.dval), x.dval)
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
        H::Const{<:HermTri{UPLO}},
        x::Duplicated{<:Real}
    ) where {UPLO}
    y, dy = mul_frule_impl(H.val, x.val, ZeroTangent(), x.dval)
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
        H::Duplicated{<:HermTri{UPLO}},
        x::Const{<:Real}
    ) where {UPLO}
    y, dy = mul_frule_impl(H.val, x.val, parent(H.dval), ZeroTangent())
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
        H::Annotation{<:HermTri{UPLO}},
        x::Annotation{<:Real}
    ) where {UPLO}
    y = H.val * x.val

    if needs_shadow(config)
        shadow = zero(y)
    else
        shadow = nothing
    end

    _, oH, ox = overwritten(config)
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
        H::Annotation{<:HermTri{UPLO}},
        x::Annotation{<:Real}
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
    return (nothing, Δx)
end

# H * x where x is a scalar and H is SymTri

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(*)},
        RT::Type,
        H::Duplicated{<:SymTri{UPLO}},
        x::Duplicated{<:Real}
    ) where {UPLO}
    y, dy = mul_frule_impl(H.val, x.val, parent(H.dval), x.dval)
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
        H::Const{<:SymTri{UPLO}},
        x::Duplicated{<:Real}
    ) where {UPLO}
    y, dy = mul_frule_impl(H.val, x.val, ZeroTangent(), x.dval)
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
        H::Duplicated{<:SymTri{UPLO}},
        x::Const{<:Real}
    ) where {UPLO}
    y, dy = mul_frule_impl(H.val, x.val, parent(H.dval), ZeroTangent())
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
        H::Annotation{<:SymTri{UPLO}},
        x::Annotation{<:Real}
    ) where {UPLO}
    y = H.val * x.val

    if needs_shadow(config)
        shadow = zero(y)
    else
        shadow = nothing
    end

    _, oH, ox = overwritten(config)
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
        H::Annotation{<:SymTri{UPLO}},
        x::Annotation{<:Real}
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
    return (nothing, Δx)
end
