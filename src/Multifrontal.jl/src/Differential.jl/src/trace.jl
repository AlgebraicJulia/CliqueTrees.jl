# Kernel functions for tr

function tr_frule_impl(A::MaybeHermOrSymTri, dA)
    y = tr(A)

    if dA isa ZeroTangent
        dy = ZeroTangent()
    else
        dy = tr(dA)
    end

    return y, dy
end

function tr_rrule_impl(A::MaybeHermOrSymTri, y, Δy)
    if Δy isa ZeroTangent
        ΔA = ZeroTangent()
    else
        ΔA = zero(parent(A))

        for j in fronts(ΔA)
            D, _ = diagblock(ΔA, j)

            for i in diagind(D)
                D[i] = Δy
            end
        end
    end

    return ΔA
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
