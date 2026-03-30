# Kernel functions for dot(A, B)

# ===== frule =====

function dot_frule_impl(A::MaybeHermOrSymTri{UPLO}, B::MaybeHermOrSymTri{UPLO}, dA, dB) where {UPLO}
    return dot(A, B), dot(dA, B) + dot(A, dB)
end

function ChainRulesCore.frule((_, dA, dB)::Tuple, ::typeof(dot), A::HermTri{UPLO}, B::HermTri{UPLO}) where {UPLO}
    return dot_frule_impl(A, B, dA, dB)
end

function ChainRulesCore.frule((_, dA, dB)::Tuple, ::typeof(dot), A::SymTri{UPLO}, B::SymTri{UPLO}) where {UPLO}
    return dot_frule_impl(A, B, dA, dB)
end

function ChainRulesCore.frule((_, dA, dB)::Tuple, ::typeof(dot), A::ChordalTriangular{:N, UPLO}, B::ChordalTriangular{:N, UPLO}) where {UPLO}
    return dot_frule_impl(A, B, dA, dB)
end

# ===== rrule =====

function dot_rrule_impl(A::MaybeHermOrSymTri{UPLO}, B::MaybeHermOrSymTri{UPLO}, y, Δy) where {UPLO}
    ΔA = @thunk Δy * B
    ΔB = @thunk Δy * A
    return ΔA, ΔB
end

function dot_rrule(A::MaybeHermOrSymTri{UPLO}, B::MaybeHermOrSymTri{UPLO}) where {UPLO}
    y = dot(A, B)

    function pullback(Δy)
        ΔA, ΔB = dot_rrule_impl(A, B, y, Δy)
        return NoTangent(), ΔA, ΔB
    end

    return y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(dot), A::HermTri{UPLO}, B::HermTri{UPLO}) where {UPLO}
    return dot_rrule(A, B)
end

function ChainRulesCore.rrule(::typeof(dot), A::SymTri{UPLO}, B::SymTri{UPLO}) where {UPLO}
    return dot_rrule(A, B)
end

function ChainRulesCore.rrule(::typeof(dot), A::ChordalTriangular{:N, UPLO}, B::ChordalTriangular{:N, UPLO}) where {UPLO}
    return dot_rrule(A, B)
end
