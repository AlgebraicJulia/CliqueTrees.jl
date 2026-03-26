# Kernel functions for x \ A where x is a scalar

# ===== frule =====

function ldiv_frule_impl(x::Number, A::MaybeHermOrSymTri{UPLO}, dx::Number, dA::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, dA)
    y = x \ A
    return y, (dA - dx * parent(y)) / x
end

function ldiv_frule_impl(x::Number, A::MaybeHermOrSymTri{UPLO}, dx::ZeroTangent, dA::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(A, dA)
    y = x \ A
    return y, dA / x
end

function ldiv_frule_impl(x::Number, A::MaybeHermOrSymTri, dx::Number, dA::ZeroTangent)
    y = x \ A
    return y, -dx * parent(y) / x
end

function ldiv_frule_impl(x::Number, A::MaybeHermOrSymTri, dx::ZeroTangent, dA::ZeroTangent)
    return x \ A, ZeroTangent()
end

function ChainRulesCore.frule((_, dx, dL)::Tuple, ::typeof(\), x::Number, L::ChordalTriangular{:N})
    return ldiv_frule_impl(x, L, dx, dL)
end

function ChainRulesCore.frule((_, dx, dH)::Tuple, ::typeof(\), x::Real, H::HermOrSymTri)
    return ldiv_frule_impl(x, H, dx, dH)
end

# ===== rrule =====

function ldiv_rrule_impl(x::Number, L::ChordalTriangular{:N, UPLO}, y::ChordalTriangular{:N, UPLO}, Δy::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(L, y, Δy)
    Δx = @thunk -dot(y, Δy) / conj(x)
    ΔL = @thunk conj(x) \ Δy
    return Δx, ΔL
end

function ldiv_rrule_impl(x::Real, H::HermTri{UPLO}, y::HermTri{UPLO}, Δy::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(H, y, Δy)
    Δx = @thunk -dot(y, Hermitian(Δy, UPLO)) / x
    ΔH = @thunk x \ Δy
    return Δx, ΔH
end

function ldiv_rrule_impl(x::Real, H::SymTri{UPLO}, y::SymTri{UPLO}, Δy::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(H, y, Δy)
    Δx = @thunk -dot(y, Symmetric(Δy, UPLO)) / x
    ΔH = @thunk x \ Δy
    return Δx, ΔH
end

function ldiv_rrule(x::Number, A::MaybeHermOrSymTri)
    y = x \ A

    function pullback(Δy)
        if Δy isa ZeroTangent
            return NoTangent(), ZeroTangent(), ZeroTangent()
        else
            Δx, ΔA = ldiv_rrule_impl(x, A, y, Δy)
            return NoTangent(), Δx, ΔA
        end
    end

    return y, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(\), x::Number, L::ChordalTriangular{:N})
    return ldiv_rrule(x, L)
end

# type ambiguity with ChainRules
function ChainRulesCore.rrule(::typeof(\), x::RealOrComplex, L::ChordalTriangular{:N, <:Any, <:RealOrComplex})
    return ldiv_rrule(x, L)
end

function ChainRulesCore.rrule(::typeof(\), x::Real, H::HermTri)
    return ldiv_rrule(x, H)
end

function ChainRulesCore.rrule(::typeof(\), x::Real, H::SymTri)
    return ldiv_rrule(x, H)
end
