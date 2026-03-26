# unflattri(uplo::Val, S::ChordalSymbolic, x::AbstractVector) -> ChordalTriangular
# unflatsym(uplo::Val, S::ChordalSymbolic, x::AbstractVector) -> Hermitian{ChordalTriangular}

using CliqueTrees.Multifrontal.Differential: unflattri, unflatsym,
    unflattri_rrule_impl, unflatsym_rrule_impl, flat

@is_primitive MinimalCtx Tuple{typeof(unflattri), Val, ChordalSymbolic, AbstractVector}
@is_primitive MinimalCtx Tuple{typeof(unflatsym), Val, ChordalSymbolic, AbstractVector}

function Mooncake.rrule!!(
    ::CoDual{typeof(unflattri)},
    uplo::CoDual{Val{UPLO}},
    S::CoDual{<:ChordalSymbolic},
    x::CoDual{<:AbstractVector}
) where {UPLO}
    uplo = primal(uplo)
    S = primal(S)
    px = primal(x)
    tx = tangent(x)

    pL = unflattri(uplo, S, px)
    tL = zero(pL)

    function pullback!!(::NoRData)
        Δx = unflattri_rrule_impl(uplo, S, px, pL, tL)
        increment_and_get_rdata!(tx, NoRData(), Δx)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end

    return CoDual(pL, tL), pullback!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(unflatsym)},
    uplo::CoDual{Val{UPLO}},
    S::CoDual{<:ChordalSymbolic},
    x::CoDual{<:AbstractVector}
) where {UPLO}
    uplo = primal(uplo)
    S = primal(S)
    px = primal(x)
    tx = tangent(x)

    pH = unflatsym(uplo, S, px)
    tH = zero(parent(pH))

    function pullback!!(::NoRData)
        Δx = unflatsym_rrule_impl(uplo, S, px, pH, tH)
        increment_and_get_rdata!(tx, NoRData(), Δx)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end

    return CoDual(pH, tH), pullback!!
end
