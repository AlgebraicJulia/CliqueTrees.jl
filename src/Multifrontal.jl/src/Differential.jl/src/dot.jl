# Kernel functions for dot(A, B)

# ===== frule =====

function dot_frule_impl(A::HermTri{UPLO}, B::HermTri{UPLO}, dA::ChordalTriangular{:N, UPLO}, dB::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, B, dA, dB)
    y = dot(A, B)
    dy = dot(Hermitian(dA, UPLO), B) + dot(A, Hermitian(dB, UPLO))
    return y, dy
end

function dot_frule_impl(A::SymTri{UPLO}, B::SymTri{UPLO}, dA::ChordalTriangular{:N, UPLO}, dB::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, B, dA, dB)
    y = dot(A, B)
    dy = dot(Symmetric(dA, UPLO), B) + dot(A, Symmetric(dB, UPLO))
    return y, dy
end

function dot_frule_impl(A::ChordalTriangular{:N, UPLO}, B::ChordalTriangular{:N, UPLO}, dA::ChordalTriangular{:N, UPLO}, dB::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, B, dA, dB)
    y = dot(A, B)
    dy = dot(dA, B) + dot(A, dB)
    return y, dy
end

function dot_frule_impl(A::HermTri{UPLO}, B::HermTri{UPLO}, dA::ZeroTangent, dB::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, B, dB)
    return dot(A, B), dot(A, Hermitian(dB, UPLO))
end

function dot_frule_impl(A::HermTri{UPLO}, B::HermTri{UPLO}, dA::ChordalTriangular{:N, UPLO}, dB::ZeroTangent) where {UPLO}
    @assert checksymbolic(A, B, dA)
    return dot(A, B), dot(Hermitian(dA, UPLO), B)
end

function dot_frule_impl(A::SymTri{UPLO}, B::SymTri{UPLO}, dA::ZeroTangent, dB::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, B, dB)
    return dot(A, B), dot(A, Symmetric(dB, UPLO))
end

function dot_frule_impl(A::SymTri{UPLO}, B::SymTri{UPLO}, dA::ChordalTriangular{:N, UPLO}, dB::ZeroTangent) where {UPLO}
    @assert checksymbolic(A, B, dA)
    return dot(A, B), dot(Symmetric(dA, UPLO), B)
end

function dot_frule_impl(A::ChordalTriangular{:N, UPLO}, B::ChordalTriangular{:N, UPLO}, dA::ZeroTangent, dB::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, B, dB)
    return dot(A, B), dot(A, dB)
end

function dot_frule_impl(A::ChordalTriangular{:N, UPLO}, B::ChordalTriangular{:N, UPLO}, dA::ChordalTriangular{:N, UPLO}, dB::ZeroTangent) where {UPLO}
    @assert checksymbolic(A, B, dA)
    return dot(A, B), dot(dA, B)
end

function dot_frule_impl(A::MaybeHermOrSymTri, B::MaybeHermOrSymTri, dA::ZeroTangent, dB::ZeroTangent)
    return dot(A, B), ZeroTangent()
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

function dot_rrule_impl(A::HermOrSymTri{UPLO}, B::HermOrSymTri{UPLO}, y, Δy) where {UPLO}
    @assert checksymbolic(A, B)
    ΔA = @thunk Δy * parent(B)
    ΔB = @thunk Δy * parent(A)
    return ΔA, ΔB
end

function dot_rrule_impl(A::ChordalTriangular{:N, UPLO}, B::ChordalTriangular{:N, UPLO}, y, Δy) where {UPLO}
    @assert checksymbolic(A, B)
    ΔA = @thunk Δy * B
    ΔB = @thunk Δy * A
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

function ChainRulesCore.rrule(::typeof(dot), A::HermTri{UPLO}, B::HermTri{UPLO}) where {UPLO}
    return dot_rrule(A, B)
end

function ChainRulesCore.rrule(::typeof(dot), A::SymTri{UPLO}, B::SymTri{UPLO}) where {UPLO}
    return dot_rrule(A, B)
end

function ChainRulesCore.rrule(::typeof(dot), A::ChordalTriangular{:N, UPLO}, B::ChordalTriangular{:N, UPLO}) where {UPLO}
    return dot_rrule(A, B)
end
