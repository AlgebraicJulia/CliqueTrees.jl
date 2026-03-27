# unflattri(x::AbstractVector, S::ChordalSymbolic, uplo::Val) -> ChordalTriangular
# unflatsym(x::AbstractVector, S::ChordalSymbolic, uplo::Val) -> Hermitian{ChordalTriangular}

using CliqueTrees.Multifrontal.Differential: unflattri, unflatsym,
    unflattri_rrule_impl, unflatsym_rrule_impl, flat

@is_primitive MinimalCtx Tuple{typeof(unflattri), AbstractVector, ChordalSymbolic, Val}
@is_primitive MinimalCtx Tuple{typeof(unflatsym), AbstractVector, ChordalSymbolic, Val}

function Mooncake.rrule!!(
    ::CoDual{typeof(unflattri)},
    x::CoDual{<:AbstractVector},
    S::CoDual{<:ChordalSymbolic},
    uplo::CoDual{Val{UPLO}}
) where {UPLO}
    px = primal(x)
    tx = tangent(x)
    S = primal(S)
    uplo = primal(uplo)

    pL = unflattri(px, S, uplo)
    tL = zero(pL)

    function pullback!!(::NoRData)
        Δx = unflattri_rrule_impl(px, S, uplo, pL, tL)
        increment_and_get_rdata!(tx, NoRData(), Δx)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end

    return CoDual(pL, tL), pullback!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(unflatsym)},
    x::CoDual{<:AbstractVector},
    S::CoDual{<:ChordalSymbolic},
    uplo::CoDual{Val{UPLO}}
) where {UPLO}
    px = primal(x)
    tx = tangent(x)
    S = primal(S)
    uplo = primal(uplo)

    pH = unflatsym(px, S, uplo)
    tH = zero(parent(pH))

    function pullback!!(::NoRData)
        Δx = unflatsym_rrule_impl(px, S, uplo, pH, tH)
        increment_and_get_rdata!(tx, NoRData(), Δx)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end

    return CoDual(pH, tH), pullback!!
end
