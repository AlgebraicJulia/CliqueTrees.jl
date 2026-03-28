# Kernel functions for x * A where x is a scalar

# ===== frule =====

function mul_frule_impl(x::Number, A::MaybeHermOrSymTri, dx, dA)
    return x * A, dx * parent(A) + x * dA
end

function ChainRulesCore.frule((_, dx, dL)::Tuple, ::typeof(*), x::Number, L::ChordalTriangular{:N})
    return mul_frule_impl(x, L, dx, dL)
end

function ChainRulesCore.frule((_, dx, dH)::Tuple, ::typeof(*), x::Real, H::HermOrSymTri)
    return mul_frule_impl(x, H, dx, dH)
end

# ===== rrule =====

function mul_rrule_impl(x::Number, A::MaybeHermOrSymTri, y::MaybeHermOrSymTri, Δy::ChordalTriangular{:N})
    Δx = @thunk dot(A, ProjectTo(A)(Δy))
    ΔA = @thunk conj(x) * Δy
    return Δx, ΔA
end

function ChainRulesCore.rrule(::typeof(*), x::Number, L::ChordalTriangular{:N})
    return mul_rrule(x, L)
end

# type ambiguity with ChainRules
function ChainRulesCore.rrule(::typeof(*), x::RealOrComplex, L::ChordalTriangular{:N, <:Any, <:RealOrComplex})
    return mul_rrule(x, L)
end

function ChainRulesCore.rrule(::typeof(*), x::Real, H::HermTri)
    return mul_rrule(x, H)
end

function ChainRulesCore.rrule(::typeof(*), x::Real, H::SymTri)
    return mul_rrule(x, H)
end
