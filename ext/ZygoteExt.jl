module ZygoteExt

using ChainRulesCore: unthunk
using CliqueTrees.Multifrontal: AdjTri, TransTri, ChordalTriangular, HermTri, SymTri
using CliqueTrees.Multifrontal.Differential: ldiv_rrule_impl, tr_rrule_impl
using LinearAlgebra: tr
using Zygote

Zygote.@adjoint function \(L::ChordalTriangular{:N}, x::AbstractVecOrMat)
    y = L \ x

    function pullback(Δy)
        if isnothing(Δy)
            return (nothing, nothing)
        else
            ΔL, Δx = ldiv_rrule_impl(L, x, y, Δy)
            return (unthunk(ΔL), Δx)
        end
    end

    return y, pullback
end

Zygote.@adjoint function \(A::AdjTri{:N}, x::AbstractVecOrMat)
    y = A \ x

    function pullback(Δy)
        if isnothing(Δy)
            return (nothing, nothing)
        else
            ΔL, Δx = ldiv_rrule_impl(A, x, y, Δy)
            return (unthunk(ΔL), Δx)
        end
    end

    return y, pullback
end

Zygote.@adjoint function \(A::TransTri{:N}, x::AbstractVecOrMat)
    y = A \ x

    function pullback(Δy)
        if isnothing(Δy)
            return (nothing, nothing)
        else
            ΔL, Δx = ldiv_rrule_impl(A, x, y, Δy)
            return (unthunk(ΔL), Δx)
        end
    end

    return y, pullback
end

Zygote.@adjoint function tr(L::ChordalTriangular{:N})
    y = tr(L)

    function pullback(Δy)
        if isnothing(Δy)
            return (nothing,)
        else
            return (tr_rrule_impl(L, y, Δy),)
        end
    end

    return y, pullback
end

Zygote.@adjoint function tr(H::HermTri)
    y = tr(H)

    function pullback(Δy)
        if isnothing(Δy)
            return (nothing,)
        else
            return (tr_rrule_impl(H, y, Δy),)
        end
    end

    return y, pullback
end

Zygote.@adjoint function tr(H::SymTri)
    y = tr(H)

    function pullback(Δy)
        if isnothing(Δy)
            return (nothing,)
        else
            return (tr_rrule_impl(H, y, Δy),)
        end
    end

    return y, pullback
end

end
