# Kernel functions for A + α * I where α is a scalar

# ===== frule =====

function add_frule_impl(A::MaybeHermOrSymTri, X::UniformScaling, dA, dα)
    return A + X, dA + dα * I
end

function ChainRulesCore.frule((_, dL, dα)::Tuple, ::typeof(+), L::ChordalTriangular{:N}, X::UniformScaling)
    return add_frule_impl(L, X, dL, dα)
end

function ChainRulesCore.frule((_, dH, dα)::Tuple, ::typeof(+), H::HermOrSymTri, X::UniformScaling)
    return add_frule_impl(H, X, dH, dα)
end

# ===== rrule =====

function add_rrule_impl(A::MaybeHermOrSymTri{UPLO}, X::UniformScaling, Y::MaybeHermOrSymTri{UPLO}, ΔY::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, Y, ΔY)
    Δα = @thunk tr(ΔY)
    return ΔY, Δα
end

function add_rrule(A::MaybeHermOrSymTri, X::UniformScaling)
    Y = A + X

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔA, Δα = add_rrule_impl(A, X, Y, ΔY)
            return NoTangent(), ΔA, Δα
        end
    end

    return Y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(+), L::ChordalTriangular{:N}, X::UniformScaling)
    return add_rrule(L, X)
end

function ChainRulesCore.rrule(::typeof(+), H::HermOrSymTri, X::UniformScaling)
    return add_rrule(H, X)
end
