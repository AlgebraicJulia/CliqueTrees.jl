# Kernel functions for x \ A where x is a scalar

# ===== frule =====

function ldiv_frule_impl(x::Number, A::MaybeHermOrSymTri, dx, dA)
    y = x \ A
    return y, (dA - dx * parent(y)) / x
end

function ChainRulesCore.frule((_, dx, dL)::Tuple, ::typeof(\), x::Number, L::ChordalTriangular{:N})
    return ldiv_frule_impl(x, L, dx, dL)
end

function ChainRulesCore.frule((_, dx, dH)::Tuple, ::typeof(\), x::Real, H::HermOrSymTri)
    return ldiv_frule_impl(x, H, dx, dH)
end

# ===== rrule =====

function ldiv_rrule_impl(x::Number, A::MaybeHermOrSymTri, y::MaybeHermOrSymTri, Δy::ChordalTriangular{:N})
    Δx = @thunk -dot(y, ProjectTo(A)(Δy)) / conj(x)
    ΔA = @thunk conj(x) \ Δy
    return Δx, ΔA
end

function ChainRulesCore.rrule(::typeof(\), x::Number, L::ChordalTriangular{:N})
    return ldiv_rrule(x, L)
end

# type ambiguity with ChainRules
function ChainRulesCore.rrule(::typeof(\), x::RealOrComplex, L::ChordalTriangular{:N, <:Any, <:RealOrComplex})
    return ldiv_rrule(x, L)
end

function ChainRulesCore.rrule(::typeof(\), x::Real, H::HermTri)
    return ldiv_rrule(x, H)
end

function ChainRulesCore.rrule(::typeof(\), x::Real, H::SymTri)
    return ldiv_rrule(x, H)
end
