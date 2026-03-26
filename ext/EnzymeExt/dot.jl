function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(dot)},
        RT::Type,
        A::Duplicated{<:HermTri{UPLO}},
        B::Duplicated{<:HermTri{UPLO}}
    ) where {UPLO}
    y, dy = dot_frule_impl(A.val, B.val, parent(A.dval), parent(B.dval))
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
        A::Const{<:HermTri{UPLO}},
        B::Duplicated{<:HermTri{UPLO}}
    ) where {UPLO}
    y, dy = dot_frule_impl(A.val, B.val, ZeroTangent(), parent(B.dval))
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
        A::Duplicated{<:HermTri{UPLO}},
        B::Const{<:HermTri{UPLO}}
    ) where {UPLO}
    y, dy = dot_frule_impl(A.val, B.val, parent(A.dval), ZeroTangent())
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
        A::Duplicated{<:SymTri{UPLO}},
        B::Duplicated{<:SymTri{UPLO}}
    ) where {UPLO}
    y, dy = dot_frule_impl(A.val, B.val, parent(A.dval), parent(B.dval))
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
        A::Const{<:SymTri{UPLO}},
        B::Duplicated{<:SymTri{UPLO}}
    ) where {UPLO}
    y, dy = dot_frule_impl(A.val, B.val, ZeroTangent(), parent(B.dval))
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
        A::Duplicated{<:SymTri{UPLO}},
        B::Const{<:SymTri{UPLO}}
    ) where {UPLO}
    y, dy = dot_frule_impl(A.val, B.val, parent(A.dval), ZeroTangent())
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
        A::Duplicated{<:ChordalTriangular{:N, UPLO}},
        B::Duplicated{<:ChordalTriangular{:N, UPLO}}
    ) where {UPLO}
    y, dy = dot_frule_impl(A.val, B.val, A.dval, B.dval)
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
        A::Const{<:ChordalTriangular{:N, UPLO}},
        B::Duplicated{<:ChordalTriangular{:N, UPLO}}
    ) where {UPLO}
    y, dy = dot_frule_impl(A.val, B.val, ZeroTangent(), B.dval)
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
        A::Duplicated{<:ChordalTriangular{:N, UPLO}},
        B::Const{<:ChordalTriangular{:N, UPLO}}
    ) where {UPLO}
    y, dy = dot_frule_impl(A.val, B.val, A.dval, ZeroTangent())
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
        A::Annotation{<:HermTri{UPLO}},
        B::Annotation{<:HermTri{UPLO}}
    ) where {UPLO}
    y = dot(A.val, B.val)
    if needs_primal(config)
        primal = y
    else
        primal = nothing
    end

    _, oA, oB = overwritten(config)
    if oA || oB
        cache = (A.val, B.val, y)
    else
        cache = nothing
    end

    return AugmentedReturn(primal, nothing, cache)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(dot)},
        dret,
        tape,
        A::Annotation{<:HermTri{UPLO}},
        B::Annotation{<:HermTri{UPLO}}
    ) where {UPLO}
    if tape === nothing
        Aval = A.val
        Bval = B.val
        y = dot(Aval, Bval)
    else
        Aval, Bval, y = tape
    end

    Δy = dret.val
    ΔA, ΔB = dot_rrule_impl(Aval, Bval, y, Δy)

    if A isa Duplicated
        axpy!(1, unthunk(ΔA), parent(A.dval))
    end

    if B isa Duplicated
        axpy!(1, unthunk(ΔB), parent(B.dval))
    end

    return (nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1},
        func::Const{typeof(dot)},
        ::Type{<:Active},
        A::Annotation{<:SymTri{UPLO}},
        B::Annotation{<:SymTri{UPLO}}
    ) where {UPLO}
    y = dot(A.val, B.val)

    if needs_primal(config)
        primal = y
    else
        primal = nothing
    end

    _, oA, oB = overwritten(config)
    if oA || oB
        cache = (A.val, B.val, y)
    else
        cache = nothing
    end

    return AugmentedReturn(primal, nothing, cache)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(dot)},
        dret,
        tape,
        A::Annotation{<:SymTri{UPLO}},
        B::Annotation{<:SymTri{UPLO}}
    ) where {UPLO}
    if tape === nothing
        Aval = A.val
        Bval = B.val
        y = dot(Aval, Bval)
    else
        Aval, Bval, y = tape
    end

    Δy = dret.val
    ΔA, ΔB = dot_rrule_impl(Aval, Bval, y, Δy)

    if A isa Duplicated
        axpy!(1, unthunk(ΔA), parent(A.dval))
    end

    if B isa Duplicated
        axpy!(1, unthunk(ΔB), parent(B.dval))
    end

    return (nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1},
        func::Const{typeof(dot)},
        ::Type{<:Active},
        A::Annotation{<:ChordalTriangular{:N, UPLO}},
        B::Annotation{<:ChordalTriangular{:N, UPLO}}
    ) where {UPLO}
    y = dot(A.val, B.val)

    if needs_primal(config)
        primal = y
    else
        primal = nothing
    end

    _, oA, oB = overwritten(config)
    if oA || oB
        cache = (A.val, B.val, y)
    else
        cache = nothing
    end

    return AugmentedReturn(primal, nothing, cache)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(dot)},
        dret,
        tape,
        A::Annotation{<:ChordalTriangular{:N, UPLO}},
        B::Annotation{<:ChordalTriangular{:N, UPLO}}
    ) where {UPLO}
    if tape === nothing
        Aval = A.val
        Bval = B.val
        y = dot(Aval, Bval)
    else
        Aval, Bval, y = tape
    end

    Δy = dret.val
    ΔA, ΔB = dot_rrule_impl(Aval, Bval, y, Δy)

    if A isa Duplicated
        axpy!(1, unthunk(ΔA), A.dval)
    end

    if B isa Duplicated
        axpy!(1, unthunk(ΔB), B.dval)
    end

    return (nothing, nothing)
end
