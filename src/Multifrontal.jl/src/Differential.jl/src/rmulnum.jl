# Kernel functions for A * x where x is a scalar

# ===== frule =====

function mul_frule_impl(A::MaybeHermOrSymTri{UPLO}, x::Number, dA::ChordalTriangular{:N, UPLO}, dx::Number) where {UPLO}
    @assert checksymbolic(A, dA)
    return A * x, dA * x + parent(A) * dx
end

function mul_frule_impl(A::MaybeHermOrSymTri{UPLO}, x::Number, dA::ChordalTriangular{:N, UPLO}, dx::ZeroTangent) where {UPLO}
    @assert checksymbolic(A, dA)
    return A * x, dA * x
end

function mul_frule_impl(A::MaybeHermOrSymTri, x::Number, dA::ZeroTangent, dx::Number)
    return A * x, parent(A) * dx
end

function mul_frule_impl(A::MaybeHermOrSymTri, x::Number, dA::ZeroTangent, dx::ZeroTangent)
    return A * x, ZeroTangent()
end

function ChainRulesCore.frule((_, dL, dx)::Tuple, ::typeof(*), L::ChordalTriangular{:N}, x::Number)
    return mul_frule_impl(L, x, dL, dx)
end

function ChainRulesCore.frule((_, dH, dx)::Tuple, ::typeof(*), H::HermOrSymTri, x::Real)
    return mul_frule_impl(H, x, dH, dx)
end

# ===== rrule =====

function mul_rrule_impl(L::ChordalTriangular{:N, UPLO}, x::Number, y::ChordalTriangular{:N, UPLO}, Δy::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(L, y, Δy)
    ΔL = @thunk Δy * conj(x)
    Δx = @thunk dot(L, Δy)
    return ΔL, Δx
end

function mul_rrule_impl(H::HermTri{UPLO}, x::Real, y::HermTri{UPLO}, Δy::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(H, y, Δy)
    ΔH = @thunk Δy * x
    Δx = @thunk dot(H, Hermitian(Δy, UPLO))
    return ΔH, Δx
end

function mul_rrule_impl(H::SymTri{UPLO}, x::Real, y::SymTri{UPLO}, Δy::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(H, y, Δy)
    ΔH = @thunk Δy * x
    Δx = @thunk dot(H, Symmetric(Δy, UPLO))
    return ΔH, Δx
end

function mul_rrule(A::MaybeHermOrSymTri, x::Number)
    y = A * x

    function pullback(Δy)
        if Δy isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            ΔA, Δx = mul_rrule_impl(A, x, y, Δy)
            return NoTangent(), ΔA, Δx
        end
    end

    return y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(*), L::ChordalTriangular{:N}, x::Number)
    return mul_rrule(L, x)
end

# type ambiguity with ChainRules
function ChainRulesCore.rrule(::typeof(*), L::ChordalTriangular{:N, <:Any, <:RealOrComplex}, x::RealOrComplex)
    return mul_rrule(L, x)
end

function ChainRulesCore.rrule(::typeof(*), H::HermTri, x::Real)
    return mul_rrule(H, x)
end

function ChainRulesCore.rrule(::typeof(*), H::SymTri, x::Real)
    return mul_rrule(H, x)
end
