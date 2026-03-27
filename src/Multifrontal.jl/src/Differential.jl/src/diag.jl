# Kernel functions for diag
function diag_frule_impl(A::ChordalTriangular{:N, UPLO}, dA::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, dA)
    return diag(A), diag(dA)
end

function diag_frule_impl(H::HermOrSymTri{UPLO}, dH::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(H, dH)
    return diag_frule_impl(parent(H), dH)
end

function diag_rrule_impl(A::ChordalTriangular{:N}, y, Δy)
    ΔA = zero(A)

    for f in fronts(ΔA)
        D, res = diagblock(ΔA, f)

        for (i, j) in enumerate(diagind(D))
            D[j] = Δy[res[i]]
        end
    end

    return ΔA
end

function diag_rrule_impl(H::HermOrSymTri, y, Δy)
    return diag_rrule_impl(parent(H), y, Δy)
end

function diag_frule_impl(A::ChordalTriangular{:N}, ::ZeroTangent)
    return diag(A), ZeroTangent()
end

function diag_frule_impl(H::HermOrSymTri, ::ZeroTangent)
    return diag(H), ZeroTangent()
end

function ChainRulesCore.frule((_, dA)::Tuple, ::typeof(diag), A::ChordalTriangular{:N})
    return diag_frule_impl(A, dA)
end

function ChainRulesCore.frule((_, dH)::Tuple, ::typeof(diag), H::HermOrSymTri)
    return diag_frule_impl(H, dH)
end

function diag_rrule(A::MaybeHermOrSymTri)
    y = diag(A)

    function pullback(Δy)
        if Δy isa ZeroTangent
            return NoTangent(), ZeroTangent()
        else
            return NoTangent(), diag_rrule_impl(A, y, Δy)
        end
    end

    return y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(diag), A::ChordalTriangular{:N})
    return diag_rrule(A)
end

function ChainRulesCore.rrule(::typeof(diag), H::HermOrSymTri)
    return diag_rrule(H)
end
