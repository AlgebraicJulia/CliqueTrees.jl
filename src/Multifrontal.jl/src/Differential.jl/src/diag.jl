# Kernel functions for diag

function diag_frule_impl(A::MaybeHermOrSymTri, dA)
    y = diag(A)

    if dA isa ZeroTangent
        dy = ZeroTangent()
    else
        dy = diag(dA)
    end

    return y, dy
end

function diag_rrule_impl(A::MaybeHermOrSymTri, y, Δy)
    if Δy isa ZeroTangent
        ΔA = ZeroTangent()
    else
        ΔA = Diagonal(Δy)
    end

    return ΔA
end

function ChainRulesCore.frule((_, dA)::Tuple, ::typeof(diag), A::ChordalTriangular{:N})
    return diag_frule_impl(A, dA)
end

function ChainRulesCore.frule((_, dH)::Tuple, ::typeof(diag), H::HermOrSymTri)
    return diag_frule_impl(H, dH)
end

function diag_rrule(A::MaybeHermOrSymTri)
    y = diag(A)
    pullback(Δy) = (NoTangent(), diag_rrule_impl(A, y, Δy))
    return y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(diag), A::ChordalTriangular{:N})
    return diag_rrule(A)
end

function ChainRulesCore.rrule(::typeof(diag), H::HermOrSymTri)
    return diag_rrule(H)
end
