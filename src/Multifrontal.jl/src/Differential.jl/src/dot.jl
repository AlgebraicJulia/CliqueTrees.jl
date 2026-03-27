# Kernel functions for dot(A, B)

# ===== frule =====

function dot_frule_impl(A::HermOrSymTri, B::HermOrSymTri, dA, dB)
    y = dot(A, B)
    dy = dot(ProjectTo(A)(dA), B) + dot(A, ProjectTo(B)(dB))
    return y, dy
end

function dot_frule_impl(A::ChordalTriangular{:N, UPLO}, B::ChordalTriangular{:N, UPLO}, dA, dB) where {UPLO}
    y = dot(A, B)
    dy = dot(dA, B) + dot(A, dB)
    return y, dy
end

function ChainRulesCore.frule((_, dA, dB)::Tuple, ::typeof(dot), A::HermTri, B::HermTri)
    return dot_frule_impl(A, B, dA, dB)
end

function ChainRulesCore.frule((_, dA, dB)::Tuple, ::typeof(dot), A::SymTri, B::SymTri)
    return dot_frule_impl(A, B, dA, dB)
end

function ChainRulesCore.frule((_, dA, dB)::Tuple, ::typeof(dot), A::ChordalTriangular{:N, UPLO}, B::ChordalTriangular{:N, UPLO}) where {UPLO}
    return dot_frule_impl(A, B, dA, dB)
end

# ===== rrule =====

function dot_rrule_impl(A::MaybeHermOrSymTri, B::MaybeHermOrSymTri, y, Δy)
    ΔA = @thunk Δy * parent(B)
    ΔB = @thunk Δy * parent(A)
    return ΔA, ΔB
end

function dot_rrule(A::MaybeHermOrSymTri{UPLO}, B::MaybeHermOrSymTri{UPLO}) where {UPLO}
    y = dot(A, B)

    function pullback(Δy)
        if Δy isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔA, ΔB = dot_rrule_impl(A, B, y, Δy)
            return NoTangent(), ΔA, ΔB
        end
    end

    return y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(dot), A::HermTri, B::HermTri)
    return dot_rrule(A, B)
end

function ChainRulesCore.rrule(::typeof(dot), A::SymTri, B::SymTri)
    return dot_rrule(A, B)
end

function ChainRulesCore.rrule(::typeof(dot), A::ChordalTriangular{:N, UPLO}, B::ChordalTriangular{:N, UPLO}) where {UPLO}
    return dot_rrule(A, B)
end
