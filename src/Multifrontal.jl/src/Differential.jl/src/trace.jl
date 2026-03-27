# Kernel functions for tr
function tr_frule_impl(L::ChordalTriangular{:N, UPLO}, dL::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(L, dL)
    return tr(L), tr(dL)
end

function tr_frule_impl(H::HermOrSymTri{UPLO}, dH::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(H, dH)
    return tr_frule_impl(parent(H), dH)
end

function tr_rrule_impl(L::ChordalTriangular{:N}, y, Δy)
    ΔL = zero(L)

    for j in fronts(ΔL)
        D, _ = diagblock(ΔL, j)

        for i in diagind(D)
            D[i] = Δy
        end
    end

    return ΔL
end

function tr_rrule_impl(H::HermOrSymTri, y, Δy)
    return tr_rrule_impl(parent(H), y, Δy)
end

function tr_frule_impl(L::ChordalTriangular{:N}, ::ZeroTangent)
    return tr(L), ZeroTangent()
end

function tr_frule_impl(H::HermOrSymTri, ::ZeroTangent)
    return tr(H), ZeroTangent()
end

function ChainRulesCore.frule((_, dL)::Tuple, ::typeof(tr), L::ChordalTriangular{:N})
    return tr_frule_impl(L, dL)
end

function ChainRulesCore.frule((_, dH)::Tuple, ::typeof(tr), H::HermOrSymTri)
    return tr_frule_impl(H, dH)
end

function tr_rrule(A::MaybeHermOrSymTri)
    y = tr(A)

    function pullback(Δy)
        if Δy isa ZeroTangent
            return NoTangent(), ZeroTangent()
        else
            return NoTangent(), tr_rrule_impl(A, y, Δy)
        end
    end

    return y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(tr), L::ChordalTriangular{:N})
    return tr_rrule(L)
end

function ChainRulesCore.rrule(::typeof(tr), H::HermOrSymTri)
    return tr_rrule(H)
end
