function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(adjoint)},
        RT::Type,
        L::Duplicated{<:Union{AdjTri, ChordalTriangular}}
    )
    A, dA = adjoint_frule_impl(L.val, L.dval)
    # Enzyme requires shadow to have same type as primal
    if needs_primal(config) && needs_shadow(config)
        return Duplicated(A, adjoint(dA))
    elseif needs_shadow(config)
        return adjoint(dA)
    elseif needs_primal(config)
        return A
    else
        return
    end
end

function EnzymeRules.forward(
        config::FwdConfigWidth{1},
        func::Const{typeof(transpose)},
        RT::Type,
        L::Duplicated{<:Union{TransTri, ChordalTriangular}}
    )
    A, dA = transpose_frule_impl(L.val, L.dval)
    # Enzyme requires shadow to have same type as primal
    if needs_primal(config) && needs_shadow(config)
        return Duplicated(A, transpose(dA))
    elseif needs_shadow(config)
        return transpose(dA)
    elseif needs_primal(config)
        return A
    else
        return
    end
end

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1},
        func::Const{typeof(adjoint)},
        RT::Type,
        L::Annotation{<:Union{AdjTri, ChordalTriangular}}
    )
    A = adjoint(L.val)

    if needs_shadow(config)
        shadow = adjoint(zero(parent(A)))
    else
        shadow = nothing
    end

    _, oL = overwritten(config)
    if oL
        cache = (L.val, A)
    else
        cache = nothing
    end

    tape = (cache, shadow)
    return AugmentedReturn(A, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(adjoint)},
        dret,
        tape,
        L::Annotation{<:Union{AdjTri, ChordalTriangular}}
    )
    cache, shadow = tape

    if cache === nothing
        Lval = L.val
        A = adjoint(Lval)
    else
        Lval, A = cache
    end

    if dret isa Active
        ΔA = dret.val
    else
        ΔA = shadow
    end

    if L isa Duplicated
        axpy!(1, adjoint_rrule_impl(Lval, A, parent(ΔA)), L.dval)
    end

    return (nothing,)
end

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1},
        func::Const{typeof(transpose)},
        RT::Type,
        L::Annotation{<:Union{TransTri, ChordalTriangular}}
    )
    A = transpose(L.val)

    if needs_shadow(config)
        shadow = transpose(zero(parent(A)))
    else
        shadow = nothing
    end

    _, oL = overwritten(config)
    if oL
        cache = (L.val, A)
    else
        cache = nothing
    end

    tape = (cache, shadow)
    return AugmentedReturn(A, shadow, tape)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1},
        func::Const{typeof(transpose)},
        dret,
        tape,
        L::Annotation{<:Union{TransTri, ChordalTriangular}}
    )
    cache, shadow = tape

    if cache === nothing
        Lval = L.val
        A = transpose(Lval)
    else
        Lval, A = cache
    end

    if dret isa Active
        ΔA = dret.val
    else
        ΔA = shadow
    end

    if L isa Duplicated
        axpy!(1, transpose_rrule_impl(Lval, A, parent(ΔA)), L.dval)
    end

    return (nothing,)
end
