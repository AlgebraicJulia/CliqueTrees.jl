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

    function pullback(ΔH)
        if ΔH isa ZeroTangent
            ΔL = ZeroTangent()
        else
            ΔL = uncholesky_rrule_impl(L, H, ΔH)
        end

        return NoTangent(), ΔL
    end

    return H, pullback ∘ unthunk
end

function uncholesky_rrule_impl(L::ChordalTriangular{:N, UPLO}, H::HermOrSymTri{UPLO}, dH::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(L, H, dH)
    ΔL = copy(dH)
    dfcholesky!(ΔL, L; adj=true, inv=true)
    return ΔL
end

function uncholesky_frule_impl(L::ChordalTriangular{:N, UPLO}, dL::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(L, dL)
    H = uncholesky(L)
    dH = copy(dL)
    dfcholesky!(dH, L; adj=false, inv=true)
    return H, dH
end
