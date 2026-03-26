# Kernel functions for A + X * I where X is a scalar

# ===== frule =====

function add_frule_impl(A::MaybeHermOrSymTri{UPLO}, X::UniformScaling, dA::ChordalTriangular{:N, UPLO}, dX::UniformScaling) where {UPLO}
    @assert checksymbolic(A, dA)
    return A + X, dA + dX
end

function add_frule_impl(A::MaybeHermOrSymTri{UPLO}, X::UniformScaling, dA::ChordalTriangular{:N, UPLO}, dX::ZeroTangent) where {UPLO}
    @assert checksymbolic(A, dA)
    return A + X, dA
end

function add_frule_impl(A::MaybeHermOrSymTri{UPLO}, X::UniformScaling, dA::ZeroTangent, dX::UniformScaling) where {UPLO}
    Y = A + X
    dY = zero(parent(Y))

    for j in fronts(dY)
        D, _ = diagblock(dY, j)

        for i in diagind(D)
            D[i] = dX.λ
        end
    end

    return Y, dY
end

function add_frule_impl(A::MaybeHermOrSymTri, X::UniformScaling, dA::ZeroTangent, dX::ZeroTangent)
    return A + X, ZeroTangent()
end

function ChainRulesCore.frule((_, dL, dX)::Tuple, ::typeof(+), L::ChordalTriangular, X::UniformScaling)
    return add_frule_impl(L, X, dL, dX)
end

function ChainRulesCore.frule((_, dH, dX)::Tuple, ::typeof(+), H::HermOrSymTri, X::UniformScaling)
    return add_frule_impl(H, X, dH, dX)
end

# ===== rrule =====

function add_rrule_impl(A::MaybeHermOrSymTri, X::UniformScaling, Y::MaybeHermOrSymTri, ΔY::ChordalTriangular)
    @assert checksymbolic(A, Y, ΔY)
    ΔX = @thunk tr(ΔY) * I
    return ΔY, ΔX
end

function add_rrule(A::MaybeHermOrSymTri, X::UniformScaling)
    Y = A + X

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔA, ΔX = add_rrule_impl(A, X, Y, ΔY)
            return NoTangent(), ΔA, ΔX
        end
    end

    return Y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(+), L::ChordalTriangular, X::UniformScaling)
    return add_rrule(L, X)
end

function ChainRulesCore.rrule(::typeof(+), H::HermOrSymTri, X::UniformScaling)
    return add_rrule(H, X)
end
