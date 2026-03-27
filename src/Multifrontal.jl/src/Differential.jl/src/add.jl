# Kernel functions for A + B

# ===== frule =====

function add_frule_impl(A::MaybeHermOrSymTri, B::MaybeHermOrSymTri, dA, dB)
    return A + B, dA + dB
end

function ChainRulesCore.frule((_, dA, dB)::Tuple, ::typeof(+), A::ChordalTriangular{:N, UPLO}, B::ChordalTriangular{:N, UPLO}) where {UPLO}
    return add_frule_impl(A, B, dA, dB)
end

function ChainRulesCore.frule((_, dA, dB)::Tuple, ::typeof(+), A::HermOrSymTri, B::HermOrSymTri)
    return add_frule_impl(A, B, dA, dB)
end

# ===== rrule =====

function add_rrule_impl(A::MaybeHermOrSymTri{UPLO}, B::MaybeHermOrSymTri{UPLO}, Y::MaybeHermOrSymTri{UPLO}, ΔY::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, B, Y, ΔY)
    return ΔY, ΔY
end

function add_rrule(A::MaybeHermOrSymTri{UPLO}, B::MaybeHermOrSymTri{UPLO}) where {UPLO}
    Y = A + B

    function pullback(ΔY)
        if ΔY isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔA, ΔB = add_rrule_impl(A, B, Y, ΔY)
            return NoTangent(), ΔA, ΔB
        end
    end

    return Y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(+), A::ChordalTriangular{:N, UPLO}, B::ChordalTriangular{:N, UPLO}) where {UPLO}
    return add_rrule(A, B)
end

function ChainRulesCore.rrule(::typeof(+), A::HermOrSymTri{UPLO}, B::HermOrSymTri{UPLO}) where {UPLO}
    return add_rrule(A, B)
end
