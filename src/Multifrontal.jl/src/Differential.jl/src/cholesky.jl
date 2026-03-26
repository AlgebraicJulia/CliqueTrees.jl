function LinearAlgebra.cholesky(H::HermOrSymTri)
    L = copy(parent(H))
    cholesky!(L)
    return L
end

function ChainRulesCore.frule((_, dH)::Tuple, ::typeof(cholesky), H::HermOrSymTri)
    return cholesky_frule_impl(H, dH)
end

function ChainRulesCore.rrule(::typeof(cholesky), H::HermOrSymTri)
    L = cholesky(H)

    function pullback(ΔL)
        if ΔL isa ZeroTangent
            ΔH = ZeroTangent()
        else
            ΔH = cholesky_rrule_impl(H, L, ΔL)
        end

        return NoTangent(), ΔH
    end

    return L, pullback ∘ unthunk
end

function cholesky_rrule_impl(H::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, dL::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(H, L, dL)
    ΔH = copy(dL)
    dfcholesky!(ΔH, L; adj=true, inv=false)
    return ΔH
end

function cholesky_frule_impl(H::HermOrSymTri{UPLO}, dH::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(H, dH)
    L = cholesky(H)
    dL = copy(dH)
    dfcholesky!(dL, L; adj=false, inv=false)
    return L, dL
end
