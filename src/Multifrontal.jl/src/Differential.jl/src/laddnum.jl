# Kernel functions for α * I + A where α is a scalar

# ===== frule =====

function add_frule_impl(X::UniformScaling, A::MaybeHermOrSymTri{UPLO}, dα::Number, dA::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, dA)
    Y = X + A
    dY = copy(dA)

    for j in fronts(dY)
        D, _ = diagblock(dY, j)

        for i in diagind(D)
            D[i] += dα
        end
    end

    return Y, dY
end

function add_frule_impl(X::UniformScaling, A::MaybeHermOrSymTri{UPLO}, dα::ZeroTangent, dA::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, dA)
    return X + A, dA
end

function add_frule_impl(X::UniformScaling, A::MaybeHermOrSymTri{UPLO}, dα::Number, dA::ZeroTangent) where {UPLO}
    Y = X + A
    dY = zero(parent(Y))

    for j in fronts(dY)
        D, _ = diagblock(dY, j)

        for i in diagind(D)
            D[i] = dα
        end
    end

    return Y, dY
end

function add_frule_impl(X::UniformScaling, A::MaybeHermOrSymTri, dα::ZeroTangent, dA::ZeroTangent)
    return X + A, ZeroTangent()
end

function ChainRulesCore.frule((_, dα, dL)::Tuple, ::typeof(+), X::UniformScaling, L::ChordalTriangular{:N})
    return add_frule_impl(X, L, dα, dL)
end

function ChainRulesCore.frule((_, dα, dH)::Tuple, ::typeof(+), X::UniformScaling, H::HermOrSymTri)
    return add_frule_impl(X, H, dα, dH)
end

# ===== rrule =====

function add_rrule_impl(X::UniformScaling, A::MaybeHermOrSymTri{UPLO}, Y::MaybeHermOrSymTri{UPLO}, ΔY::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, Y, ΔY)
    Δα = @thunk tr(ΔY)
    return Δα, ΔY
end

function add_rrule(X::UniformScaling, A::MaybeHermOrSymTri)
    Y = X + A

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            Δα, ΔA = add_rrule_impl(X, A, Y, ΔY)
            return NoTangent(), Δα, ΔA
        end
    end

    return Y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(+), X::UniformScaling, L::ChordalTriangular{:N})
    return add_rrule(X, L)
end

function ChainRulesCore.rrule(::typeof(+), X::UniformScaling, H::HermOrSymTri)
    return add_rrule(X, H)
end
