# Kernel functions for tr

function ttr(A::AbstractMatrix, n::Int)
    return tr(A)
end

function ttr(A::ZeroTangent, n::Int)
    return ZeroTangent()
end

function ttr(A::UniformScaling, n::Int)
    return A.λ * n
end

function tr_frule_impl(A::MaybeHermOrSymTri, dA)
    return tr(A), ttr(dA, size(A, 1))
end

function tr_rrule_impl(A::MaybeHermOrSymTri, y, Δy)
    return Δy * I
end

function ChainRulesCore.frule((_, dL)::Tuple, ::typeof(tr), L::ChordalTriangular{:N})
    return tr_frule_impl(L, dL)
end

function ChainRulesCore.frule((_, dH)::Tuple, ::typeof(tr), H::HermOrSymTri)
    return tr_frule_impl(H, dH)
end

function tr_rrule(A::MaybeHermOrSymTri)
    y = tr(A)
    pullback(Δy) = (NoTangent(), tr_rrule_impl(A, y, Δy))
    return y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(tr), L::ChordalTriangular{:N})
    return tr_rrule(L)
end

function ChainRulesCore.rrule(::typeof(tr), H::HermOrSymTri)
    return tr_rrule(H)
end
