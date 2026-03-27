function unflattri(x::AbstractVector, S::ChordalSymbolic, uplo::Val{UPLO}=Val(DEFAULT_UPLO)) where {UPLO}
    nd = ndz(S)
    nl = nlz(S)
    Dval = similar(x, nd)
    Lval = similar(x, nl)
    copyto!(Dval, 1, x,      1, nd)
    copyto!(Lval, 1, x, nd + 1, nl)
    return ChordalTriangular{:N, UPLO}(S, Dval, Lval)
end

function unflatsym(x::AbstractVector, S::ChordalSymbolic, uplo::Val{UPLO}=Val(DEFAULT_UPLO)) where {UPLO}
    L = unflattri(x, S, uplo)
    return Hermitian(unscale!(L), UPLO)
end

# Kernel functions for unflattri
function unflattri_frule_impl(x::AbstractVector, S::ChordalSymbolic, uplo::Val{UPLO}, dx::AbstractVector) where {UPLO}
    L = unflattri(x, S, uplo)
    dL = unflattri(dx, S, uplo)
    return L, dL
end

function unflattri_rrule_impl(x::AbstractVector, S::ChordalSymbolic, uplo::Val{UPLO}, L::ChordalTriangular{:N, UPLO}, ΔL::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(L, ΔL)
    return flat(ΔL)
end

# Kernel functions for unflatsym
function unflatsym_frule_impl(x::AbstractVector, S::ChordalSymbolic, uplo::Val{UPLO}, dx::AbstractVector) where {UPLO}
    H = unflatsym(x, S, uplo)
    dH = unscale!(unflattri(dx, S, uplo))
    return H, dH
end

function unflatsym_rrule_impl(x::AbstractVector, S::ChordalSymbolic, uplo::Val{UPLO}, H::HermOrSymTri{UPLO}, ΔH::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(H, ΔH)
    return fflat(clean!, ΔH)
end

function ChainRulesCore.frule((_, dx, _, _)::Tuple, ::typeof(unflattri), x::AbstractVector, S::ChordalSymbolic, uplo::Val{UPLO}) where {UPLO}
    return unflattri_frule_impl(x, S, uplo, dx)
end

function ChainRulesCore.frule((_, dx, _, _)::Tuple, ::typeof(unflatsym), x::AbstractVector, S::ChordalSymbolic, uplo::Val{UPLO}) where {UPLO}
    return unflatsym_frule_impl(x, S, uplo, dx)
end

function ChainRulesCore.rrule(::typeof(unflattri), x::AbstractVector, S::ChordalSymbolic, uplo::Val{UPLO}) where {UPLO}
    L = unflattri(x, S, uplo)

    function pullback(ΔL)
        if ΔL isa ZeroTangent
            return NoTangent(), ZeroTangent(), NoTangent(), NoTangent()
        else
            return NoTangent(), unflattri_rrule_impl(x, S, uplo, L, ΔL), NoTangent(), NoTangent()
        end
    end

    return L, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(unflatsym), x::AbstractVector, S::ChordalSymbolic, uplo::Val{UPLO}) where {UPLO}
    H = unflatsym(x, S, uplo)

    function pullback(ΔH)
        if ΔH isa ZeroTangent
            return NoTangent(), ZeroTangent(), NoTangent(), NoTangent()
        else
            return NoTangent(), unflatsym_rrule_impl(x, S, uplo, H, ΔH), NoTangent(), NoTangent()
        end
    end

    return H, pullback ∘ unthunk
end
