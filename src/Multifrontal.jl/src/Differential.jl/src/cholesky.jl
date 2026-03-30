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
    pullback(ΔL) = (NoTangent(), cholesky_rrule_impl(H, L, ΔL))
    return L, pullback ∘ unthunk
end

function cholesky_rrule_impl(H::HermOrSymTri, L::ChordalTriangular{:N}, ΔL)
    if ΔL isa ZeroTangent
        ΔH = ZeroTangent()
    else
        ΔH = copylike(H, ΔL)
        dfcholesky!(parent(ΔH), L; adj=true, inv=false)
    end

    return ΔH
end

function cholesky_frule_impl(H::HermOrSymTri, dH)
    L = cholesky(H)

    if dH isa ZeroTangent
        dL = ZeroTangent()
    else
        dL = copylike(L, dH)
        dfcholesky!(dL, L; adj=false, inv=false)
    end

    return L, dL
end
