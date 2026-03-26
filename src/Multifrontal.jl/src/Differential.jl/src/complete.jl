function complete(H::HermOrSymTri)
    L = copy(parent(H))
    complete!(L)
    return L
end

function ChainRulesCore.frule((_, dH)::Tuple, ::typeof(complete), H::HermOrSymTri)
    return complete_frule_impl(H, dH)
end

function ChainRulesCore.rrule(::typeof(complete), H::HermOrSymTri)
    L = complete(H)

    function pullback(ΔL)
        if ΔL isa ZeroTangent
            ΔH = ZeroTangent()
        else
            ΔH = complete_rrule_impl(H, L, ΔL)
        end

        return NoTangent(), ΔH
    end

    return L, pullback ∘ unthunk
end

function complete_rrule_impl(H::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, dL::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(H, L, dL)
    ΔH = copy(dL)
    dfcholesky!(ΔH, L; adj=true, inv=false)
    fisher!(ΔH, L, parent(H); inv=true)
    rmul!(ΔH, -1)
    return ΔH
end

function complete_frule_impl(H::HermOrSymTri{UPLO}, dH::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(H, dH)
    L = complete(H)
    dL = copy(dH)
    fisher!(dL, L, parent(H); inv=true)
    rmul!(dL, -1)
    dfcholesky!(dL, L; adj=false, inv=false)
    return L, dL
end
