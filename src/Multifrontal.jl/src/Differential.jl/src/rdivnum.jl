# Kernel functions for A / x where x is a scalar

# ===== frule =====

function rdiv_frule_impl(A::MaybeHermOrSymTri{UPLO}, x::Number, dA::ChordalTriangular{:N, UPLO}, dx::Number) where {UPLO}
    @assert checksymbolic(A, dA)
    y = A / x
    return y, (dA - parent(y) * dx) / x
end

function rdiv_frule_impl(A::MaybeHermOrSymTri{UPLO}, x::Number, dA::ChordalTriangular{:N, UPLO}, dx::ZeroTangent) where {UPLO}
    @assert checksymbolic(A, dA)
    y = A / x
    return y, dA / x
end

function rdiv_frule_impl(A::MaybeHermOrSymTri, x::Number, dA::ZeroTangent, dx::Number)
    y = A / x
    return y, -parent(y) * dx / x
end

function rdiv_frule_impl(A::MaybeHermOrSymTri, x::Number, dA::ZeroTangent, dx::ZeroTangent)
    return A / x, ZeroTangent()
end

function ChainRulesCore.frule((_, dL, dx)::Tuple, ::typeof(/), L::ChordalTriangular{:N}, x::Number)
    return rdiv_frule_impl(L, x, dL, dx)
end

function ChainRulesCore.frule((_, dH, dx)::Tuple, ::typeof(/), H::HermOrSymTri, x::Real)
    return rdiv_frule_impl(H, x, dH, dx)
end

# ===== rrule =====

function rdiv_rrule_impl(L::ChordalTriangular{:N, UPLO}, x::Number, y::ChordalTriangular{:N, UPLO}, Δy::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(L, y, Δy)
    ΔL = @thunk Δy / conj(x)
    Δx = @thunk -dot(y, Δy) / conj(x)
    return ΔL, Δx
end

function rdiv_rrule_impl(H::HermTri{UPLO}, x::Real, y::HermTri{UPLO}, Δy::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(H, y, Δy)
    ΔH = @thunk Δy / x
    Δx = @thunk -dot(y, Hermitian(Δy, UPLO)) / x
    return ΔH, Δx
end

function rdiv_rrule_impl(H::SymTri{UPLO}, x::Real, y::SymTri{UPLO}, Δy::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(H, y, Δy)
    ΔH = @thunk Δy / x
    Δx = @thunk -dot(y, Symmetric(Δy, UPLO)) / x
    return ΔH, Δx
end

function rdiv_rrule(A::MaybeHermOrSymTri, x::Number)
    y = A / x

    function pullback(Δy)
        if Δy isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔA, Δx = rdiv_rrule_impl(A, x, y, Δy)
            return NoTangent(), ΔA, Δx
        end
    end

    return y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(/), L::ChordalTriangular{:N}, x::Number)
    return rdiv_rrule(L, x)
end

# type ambiguity with ChainRules
function ChainRulesCore.rrule(::typeof(/), L::ChordalTriangular{:N, <:Any, <:RealOrComplex}, x::RealOrComplex)
    return rdiv_rrule(L, x)
end

function ChainRulesCore.rrule(::typeof(/), H::HermTri, x::Real)
    return rdiv_rrule(H, x)
end

function ChainRulesCore.rrule(::typeof(/), H::SymTri, x::Real)
    return rdiv_rrule(H, x)
end
