function cong_frule_impl(J::HermOrSymTri, P::Permutation, dJ)
    if dJ isa ZeroTangent
        dA = ZeroTangent()
    else
        dA = cong(ProjectTo(J)(dJ), P)
    end

    return cong(J, P), dA
end

function cong_frule_impl(A::AbstractMatrix, P::Permutation, dA)
    if dA isa ZeroTangent
        dy = ZeroTangent()
    else
        dy = cong(dA, P)
    end

    return cong(A, P), dy
end

function cong_rrule_impl(J::HermOrSymTri, P::Permutation, A, ΔA)
    if ΔA isa ZeroTangent
        ΔJ = ZeroTangent()
    else
        ΔJ = parent(chordal(ΔA, P, parent(J).S, parent(J).uplo))
    end

    return ΔJ
end

function cong_rrule_impl(A::AbstractMatrix, P::Permutation, Y, ΔY)
    if ΔY isa ZeroTangent
        ΔA = ZeroTangent()
    else
        ΔA = cong(ΔY, inv(P))
    end

    return ΔA
end

function ChainRulesCore.frule((_, dA, _)::Tuple, ::typeof(cong), A::AbstractMatrix, P::Permutation)
    return cong_frule_impl(A, P, dA)
end

function ChainRulesCore.rrule(::typeof(cong), A::AbstractMatrix, P::Permutation)
    Y = cong(A, P)

    function pullback(ΔY)
        ΔA = cong_rrule_impl(A, P, Y, ΔY)
        return NoTangent(), ΔA, NoTangent()
    end

    return Y, pullback ∘ unthunk
end
