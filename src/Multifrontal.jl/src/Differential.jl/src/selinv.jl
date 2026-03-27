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

function selinv_frule_impl(L::ChordalTriangular{:N}, dL)
    H = selinv(L)

    if dL isa ZeroTangent
        dH = ZeroTangent()
    else
        dH = copy(dL)
        dfcholesky!(dH, L; adj=false, inv=true)
        fisher!(dH, L, parent(H); inv=false)
        rmul!(dH, -1)
    end

    return H, dH
end

function selinv_frule_impl(A::HermOrSymTri, L::ChordalTriangular{:N}, dA)
    H = selinv(L)

    if dA isa ZeroTangent
        dH = ZeroTangent()
    else
        dH = copy(dA)
        fisher!(dH, L, parent(H); inv=false)
        rmul!(dH, -1)
    end

    return H, dH
end

function ChainRulesCore.frule((_, dL)::Tuple, ::typeof(selinv), L::ChordalTriangular{:N})
    return selinv_frule_impl(L, dL)
end

function ChainRulesCore.frule((_, dA, _)::Tuple, ::typeof(selinv), A::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}) where {UPLO}
    return selinv_frule_impl(A, L, dA)
end

# ===== rrule =====

function selinv_rrule_impl(L::ChordalTriangular{:N, UPLO}, H::HermOrSymTri{UPLO}, ΔH) where {UPLO}
    if ΔH isa ZeroTangent
        ΔL = ZeroTangent()
    else
        ΔL = copy(ΔH)
        fisher!(ΔL, L, parent(H); inv=false)
        rmul!(ΔL, -1)
        dfcholesky!(ΔL, L; adj=true, inv=true)
    end

    return ΔL
end

function selinv_rrule_impl(A::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}, H::HermOrSymTri{UPLO}, ΔH) where {UPLO}
    if ΔH isa ZeroTangent
        ΔA = ZeroTangent()
    else
        ΔA = copy(ΔH)
        fisher!(ΔA, L, parent(H); inv=false)
        rmul!(ΔA, -1)
    end

    return ΔA
end

function ChainRulesCore.rrule(::typeof(selinv), L::ChordalTriangular{:N})
    H = selinv(L)
    pullback(ΔH) = (NoTangent(), selinv_rrule_impl(L, H, ΔH))
    return H, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(selinv), A::HermOrSymTri{UPLO}, L::ChordalTriangular{:N, UPLO}) where {UPLO}
    H = selinv(L)
    pullback(ΔH) = (NoTangent(), selinv_rrule_impl(A, L, H, ΔH), NoTangent())
    return H, pullback ∘ unthunk
end
