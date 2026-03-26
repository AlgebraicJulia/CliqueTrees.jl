function unflattri(uplo::Val{UPLO}, S::ChordalSymbolic, x::AbstractVector) where {UPLO}
    nd = ndz(S)
    nl = nlz(S)
    Dval = similar(x, nd)
    Lval = similar(x, nl)
    copyto!(Dval, 1, x,      1, nd)
    copyto!(Lval, 1, x, nd + 1, nl)
    return ChordalTriangular{:N, UPLO}(S, Dval, Lval)
end

function unflatsym(uplo::Val{UPLO}, S::ChordalSymbolic, x::AbstractVector) where {UPLO}
    L = unflattri(uplo, S, x)
    return Hermitian(unscale!(L), UPLO)
end

# Kernel functions for unflattri
function unflattri_frule_impl(uplo::Val{UPLO}, S::ChordalSymbolic, x::AbstractVector, dx::AbstractVector) where {UPLO}
    L = unflattri(uplo, S, x)
    dL = unflattri(uplo, S, dx)
    return L, dL
end

function unflattri_rrule_impl(uplo::Val{UPLO}, S::ChordalSymbolic, x::AbstractVector, L::ChordalTriangular{:N, UPLO}, ΔL::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(L, ΔL)
    _, _, Δx = flat(ΔL)
    return Δx
end

# Kernel functions for unflatsym
function unflatsym_frule_impl(uplo::Val{UPLO}, S::ChordalSymbolic, x::AbstractVector, dx::AbstractVector) where {UPLO}
    H = unflatsym(uplo, S, x)
    dH = unscale!(unflattri(uplo, S, dx))
    return H, dH
end

function unflatsym_rrule_impl(uplo::Val{UPLO}, S::ChordalSymbolic, x::AbstractVector, H::HermOrSymTri{UPLO}, ΔH::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(H, ΔH)
    return fflat(clean!, ΔH)
end

function ChainRulesCore.frule((_, _, _, dx)::Tuple, ::typeof(unflattri), uplo::Val{UPLO}, S::ChordalSymbolic, x::AbstractVector) where {UPLO}
    return unflattri_frule_impl(uplo, S, x, dx)
end

function ChainRulesCore.frule((_, _, _, dx)::Tuple, ::typeof(unflatsym), uplo::Val{UPLO}, S::ChordalSymbolic, x::AbstractVector) where {UPLO}
    return unflatsym_frule_impl(uplo, S, x, dx)
end

function ChainRulesCore.rrule(::typeof(unflattri), uplo::Val{UPLO}, S::ChordalSymbolic, x::AbstractVector) where {UPLO}
    L = unflattri(uplo, S, x)

    function pullback(ΔL)
        if ΔL isa ZeroTangent
            return NoTangent(), NoTangent(), NoTangent(), ZeroTangent()
        else
            return NoTangent(), NoTangent(), NoTangent(), unflattri_rrule_impl(uplo, S, x, L, ΔL)
        end
    end

    return L, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(unflatsym), uplo::Val{UPLO}, S::ChordalSymbolic, x::AbstractVector) where {UPLO}
    H = unflatsym(uplo, S, x)

    function pullback(ΔH)
        if ΔH isa ZeroTangent
            return NoTangent(), NoTangent(), NoTangent(), ZeroTangent()
        else
            return NoTangent(), NoTangent(), NoTangent(), unflatsym_rrule_impl(uplo, S, x, H, ΔH)
        end
    end

    return H, pullback ∘ unthunk
end
