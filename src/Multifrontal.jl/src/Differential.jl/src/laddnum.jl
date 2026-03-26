# Kernel functions for X * I + A where X is a scalar

# ===== frule =====

function add_frule_impl(X::UniformScaling, A::MaybeHermOrSymTri{UPLO}, dX::UniformScaling, dA::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, dA)
    return X + A, dX + dA
end

function add_frule_impl(X::UniformScaling, A::MaybeHermOrSymTri{UPLO}, dX::ZeroTangent, dA::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, dA)
    return X + A, dA
end

function add_frule_impl(X::UniformScaling, A::MaybeHermOrSymTri{UPLO}, dX::UniformScaling, dA::ZeroTangent) where {UPLO}
    Y = X + A
    dY = zero(parent(Y))

    for j in fronts(dY)
        D, _ = diagblock(dY, j)

        for i in diagind(D)
            D[i] = dX.λ
        end
    end

    return Y, dY
end

function add_frule_impl(X::UniformScaling, A::MaybeHermOrSymTri, dX::ZeroTangent, dA::ZeroTangent)
    return X + A, ZeroTangent()
end

function ChainRulesCore.frule((_, dX, dL)::Tuple, ::typeof(+), X::UniformScaling, L::ChordalTriangular)
    return add_frule_impl(X, L, dX, dL)
end

function ChainRulesCore.frule((_, dX, dH)::Tuple, ::typeof(+), X::UniformScaling, H::HermOrSymTri)
    return add_frule_impl(X, H, dX, dH)
end

# ===== rrule =====

function add_rrule_impl(X::UniformScaling, A::MaybeHermOrSymTri, Y::MaybeHermOrSymTri, ΔY::ChordalTriangular)
    @assert checksymbolic(A, Y, ΔY)
    ΔX = @thunk tr(ΔY) * I
    return ΔX, ΔY
end

function add_rrule(X::UniformScaling, A::MaybeHermOrSymTri)
    Y = X + A

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔX, ΔA = add_rrule_impl(X, A, Y, ΔY)
            return NoTangent(), ΔX, ΔA
        end
    end

    return Y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(+), X::UniformScaling, L::ChordalTriangular)
    return add_rrule(X, L)
end

function ChainRulesCore.rrule(::typeof(+), X::UniformScaling, H::HermOrSymTri)
    return add_rrule(X, H)
end
