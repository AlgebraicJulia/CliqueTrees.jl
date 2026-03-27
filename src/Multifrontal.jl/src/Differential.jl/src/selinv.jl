# Kernel functions for selinv

function selinv(L::ChordalTriangular{:N, UPLO}) where {UPLO}
    Y = copy(L)
    selinv!(Y)
    return Hermitian(Y, UPLO)
end

function selinv(A::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}) where {UPLO}
    return selinv(L)
end

# ===== frule =====

function selinv_frule_impl(L::ChordalTriangular{:N, UPLO}, dL::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(L, dL)
    H = selinv(L)
    dH = copy(dL)
    dfcholesky!(dH, L; adj=false, inv=true)
    fisher!(dH, L, parent(H); inv=false)
    rmul!(dH, -1)
    return H, dH
end

function selinv_frule_impl(L::ChordalTriangular{:N}, dL::ZeroTangent)
    return selinv(L), ZeroTangent()
end

function selinv_frule_impl(A::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, dA::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(L, A, dA)
    H = selinv(L)
    dH = copy(dA)
    fisher!(dH, L, parent(H); inv=false)
    rmul!(dH, -1)
    return H, dH
end

function selinv_frule_impl(A::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, dA::ZeroTangent) where {UPLO}
    return selinv(A, L), ZeroTangent()
end

function ChainRulesCore.frule((_, dL)::Tuple, ::typeof(selinv), L::ChordalTriangular{:N})
    return selinv_frule_impl(L, dL)
end

function ChainRulesCore.frule((_, dA, _)::Tuple, ::typeof(selinv), A::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}) where {UPLO}
    return selinv_frule_impl(A, L, dA)
end

# ===== rrule =====

function selinv_rrule_impl(L::ChordalTriangular{:N, UPLO}, H::HermOrSymTri{UPLO}, ΔH::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(L, H, ΔH)
    ΔL = copy(ΔH)
    fisher!(ΔL, L, parent(H); inv=false)
    rmul!(ΔL, -1)
    dfcholesky!(ΔL, L; adj=true, inv=true)
    return ΔL
end

function selinv_rrule_impl(A::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, H::HermOrSymTri{UPLO}, ΔH::ChordalTriangular{:N, UPLO}) where {UPLO}
    @assert checksymbolic(L, A, H, ΔH)
    ΔA = copy(ΔH)
    fisher!(ΔA, L, parent(H); inv=false)
    rmul!(ΔA, -1)
    return ΔA
end

function ChainRulesCore.rrule(::typeof(selinv), L::ChordalTriangular{:N})
    H = selinv(L)

    function pullback(ΔH)
        if ΔH isa ZeroTangent
            return NoTangent(), ZeroTangent()
        else
            return NoTangent(), selinv_rrule_impl(L, H, ΔH)
        end
    end

    return H, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(selinv), A::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}) where {UPLO}
    H = selinv(L)

    function pullback(ΔH)
        if ΔH isa ZeroTangent
            return NoTangent(), ZeroTangent(), NoTangent()
        else
            return NoTangent(), selinv_rrule_impl(A, L, H, ΔH), NoTangent()
        end
    end

    return H, pullback ∘ unthunk
end
