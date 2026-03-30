# Kernel functions for selinv

function selinv(L::ChordalTriangular{:N, UPLO}) where {UPLO}
    Y = copy(L)
    selinv!(Y)
    return Hermitian(Y, UPLO)
end

function selinv(A::MaybeHermOrSymSparse, L::ChordalTriangular{:N}, P::Permutation)
    H = selinv(L)
    return project(A, H, P)
end

# ===== frule =====

function selinv_frule_impl(L::ChordalTriangular{:N, UPLO}, dL) where {UPLO}
    H = selinv(L)

    if dL isa ZeroTangent
        dH = ZeroTangent()
    else
        dH = Hermitian(copylike(L, dL), UPLO)
        dfcholesky!(parent(dH), L; adj=false, inv=true)
        fisher!(parent(dH), L, parent(H); inv=false)
        rmul!(dH, -1)
    end

    return H, dH
end

function selinv_frule_impl(A::MaybeHermOrSymSparse, L::ChordalTriangular{:N}, P::Permutation, dA)
    H = selinv(L)

    if dA isa ZeroTangent
        dH = ZeroTangent()
    else
        dH = chordal(dA, P, L.S, L.uplo)
        fisher!(parent(dH), L, parent(H); inv=false)
        rmul!(dH, -1)
    end

    return project(A, H, P), project(A, dH, P)
end

function ChainRulesCore.frule((_, dL)::Tuple, ::typeof(selinv), L::ChordalTriangular{:N})
    return selinv_frule_impl(L, dL)
end

function ChainRulesCore.frule((_, dA, _, _)::Tuple, ::typeof(selinv), A::MaybeHermOrSymSparse, L::ChordalTriangular{:N}, P::Permutation)
    return selinv_frule_impl(A, L, P, dA)
end

# ===== rrule =====

function selinv_rrule_impl(L::ChordalTriangular{:N}, H::HermOrSymTri, ΔH)
    if ΔH isa ZeroTangent
        ΔL = ZeroTangent()
    else
        ΔL = copylike(L, ΔH)
        fisher!(ΔL, L, parent(H); inv=false)
        rmul!(ΔL, -1)
        dfcholesky!(ΔL, L; adj=true, inv=true)
    end

    return ΔL
end

function selinv_rrule_impl(A::MaybeHermOrSymSparse, L::ChordalTriangular{:N}, P::Permutation, H::HermOrSymTri, ΔH)
    if ΔH isa ZeroTangent
        ΔA = ZeroTangent()
    else
        ΔA = chordal(ΔH, P, L.S, L.uplo)
        fisher!(parent(ΔA), L, parent(H); inv=false)
        rmul!(ΔA, -1)
    end

    return project(A, ΔA, P)
end

function ChainRulesCore.rrule(::typeof(selinv), L::ChordalTriangular{:N})
    H = selinv(L)
    pullback(ΔH) = (NoTangent(), selinv_rrule_impl(L, H, ΔH))
    return H, pullback ∘ unthunk
end

function ChainRulesCore.rrule(::typeof(selinv), A::MaybeHermOrSymSparse, L::ChordalTriangular{:N}, P::Permutation)
    H = selinv(L)
    pullback(ΔH) = (NoTangent(), selinv_rrule_impl(A, L, P, H, ΔH), NoTangent(), NoTangent())
    return project(A, H, P), pullback ∘ unthunk
end
