function chordal_frule_impl(A::HermOrSym, P::Permutation, S::ChordalSymbolic, dA)
    if dA isa ZeroTangent
        dJ = ZeroTangent()
    else
        dJ = parent(chordal(dA, P, S))
    end

    return chordal(A, P, S), dJ
end

function chordal_frule_impl(A::HermOrSym, P::Permutation, S::ChordalSymbolic, uplo::Val, dA)
    if dA isa ZeroTangent
        dJ = ZeroTangent()
    else
        dJ = parent(chordal(dA, P, S, uplo))
    end

    return chordal(A, P, S, uplo), dJ
end

function chordal_rrule_impl(A::HermOrSym, P::Permutation, S::ChordalSymbolic, uplo::Val, J::HermOrSymTri, ΔJ)
    return chordal_rrule_impl(A, P, S, J, ΔJ)
end

function chordal_rrule_impl(A::HermOrSym, P::Permutation, S::ChordalSymbolic, J::HermOrSymTri, ΔJ)
    if ΔJ isa ZeroTangent
        ΔA = ZeroTangent()
    else
        ΔA = cong(ProjectTo(J)(ΔJ), P)
    end

    return ΔA
end

function ChainRulesCore.frule((_, dA, _, _, _)::Tuple, ::typeof(chordal), A::HermOrSym, P::Permutation, S::ChordalSymbolic, uplo::Val)
    return chordal_frule_impl(A, P, S, uplo, dA)
end

function ChainRulesCore.frule((_, dA, _, _)::Tuple, ::typeof(chordal), A::HermOrSym, P::Permutation, S::ChordalSymbolic)
    return chordal_frule_impl(A, P, S, dA)
end

function ChainRulesCore.rrule(::typeof(chordal), A::HermOrSym, P::Permutation, S::ChordalSymbolic, uplo::Val)
    J = chordal(A, P, S, uplo)

    function pullback(ΔJ)
        ΔA = chordal_rrule_impl(A, P, S, uplo, J, ΔJ)
        return NoTangent(), ΔA, NoTangent(), NoTangent(), NoTangent()
    end

    return J, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(chordal), A::HermOrSym, P::Permutation, S::ChordalSymbolic)
    J = chordal(A, P, S)

    function pullback(ΔJ)
        ΔA = chordal_rrule_impl(A, P, S, J, ΔJ)
        return NoTangent(), ΔA, NoTangent(), NoTangent()
    end

    return J, pullback ∘ unthunk
end
