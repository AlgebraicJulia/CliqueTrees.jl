function uncholesky(L::ChordalTriangular{:N, UPLO}) where {UPLO}
    H = copy(L)
    uncholesky!(H)
    return Hermitian(H, UPLO)
end

function ChainRulesCore.frule((_, dL)::Tuple, ::typeof(uncholesky), L::ChordalTriangular{:N})
    return uncholesky_frule_impl(L, dL)
end

function ChainRulesCore.rrule(::typeof(uncholesky), L::ChordalTriangular{:N})
    H = uncholesky(L)
    pullback(ΔH) = (NoTangent(), uncholesky_rrule_impl(L, H, ΔH))
    return H, pullback ∘ unthunk
end

function uncholesky_rrule_impl(L::ChordalTriangular{:N}, H::HermOrSymTri, ΔH)
    if ΔH isa ZeroTangent
        ΔL = ZeroTangent()
    else
        ΔL = copylike(L, ΔH)
        dfcholesky!(ΔL, L; adj=true, inv=true)
    end

    return ΔL
end

function uncholesky_frule_impl(L::ChordalTriangular{:N, UPLO}, dL) where {UPLO}
    H = uncholesky(L)

    if dL isa ZeroTangent
        dH = ZeroTangent()
    else
        dH = Hermitian(copylike(L, dL), UPLO)
        dfcholesky!(parent(dH), L; adj=false, inv=true)
    end

    return H, dH
end
