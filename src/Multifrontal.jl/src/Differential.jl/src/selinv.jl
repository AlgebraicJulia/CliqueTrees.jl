function selinv(L::ChordalTriangular{:N, UPLO}) where {UPLO}
    Y = copy(L)
    selinv!(Y)
    return Hermitian(Y, UPLO)
end

function ChainRulesCore.frule((_, dL)::Tuple, ::typeof(selinv), L::ChordalTriangular{:N})
    return selinv_frule_impl(L, dL)
end

function ChainRulesCore.rrule(::typeof(selinv), L::ChordalTriangular{:N})
    H = selinv(L)

    function pullback(ΔH)
        if ΔH isa ZeroTangent
            ΔL = ZeroTangent()
        else
            ΔL = selinv_rrule_impl(L, H, ΔH)
        end

        return NoTangent(), ΔL
    end

    return H, pullback ∘ unthunk
end

function selinv_rrule_impl(L::ChordalTriangular{:N, UPLO}, H::HermOrSymTri{UPLO}, dH::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(L, H, dH)
    ΔL = copy(dH)
    fisher!(ΔL, L, parent(H); inv=false)
    rmul!(ΔL, -1)
    dfcholesky!(ΔL, L; adj=true, inv=true)
    return ΔL
end

function selinv_frule_impl(L::ChordalTriangular{:N, UPLO}, dL::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(L, dL)
    H = selinv(L)
    dH = copy(dL)
    dfcholesky!(dH, L; adj=false, inv=true)
    fisher!(dH, L, parent(H); inv=false)
    rmul!(dH, -1)
    return H, dH
end
