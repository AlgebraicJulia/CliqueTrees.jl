module ZygoteExt

using ChainRulesCore: unthunk
using CliqueTrees.Multifrontal: AdjTri, TransTri, ChordalTriangular, HermTri, SymTri, Permutation
using CliqueTrees.Multifrontal.Differential: ldiv_rrule_impl, rdiv_rrule_impl, mul_rrule_impl, tr_rrule_impl
using LinearAlgebra: tr, Diagonal, UniformScaling
using Zygote

# A \ x

Zygote.@adjoint function \(A::ChordalTriangular{:N}, x::AbstractVecOrMat)
    y = A \ x

    function pullback(Δy)
        isnothing(Δy) && return (nothing, nothing)
        ΔA, Δx = ldiv_rrule_impl(A, x, y, Δy)
        return (unthunk(ΔA), Δx)
    end

    return y, pullback
end

Zygote.@adjoint function \(A::AdjTri{:N}, x::AbstractVecOrMat)
    y = A \ x

    function pullback(Δy)
        isnothing(Δy) && return (nothing, nothing)
        ΔA, Δx = ldiv_rrule_impl(A, x, y, Δy)
        return (unthunk(ΔA), Δx)
    end

    return y, pullback
end

Zygote.@adjoint function \(A::TransTri{:N}, x::AbstractVecOrMat)
    y = A \ x

    function pullback(Δy)
        isnothing(Δy) && return (nothing, nothing)
        ΔA, Δx = ldiv_rrule_impl(A, x, y, Δy)
        return (unthunk(ΔA), Δx)
    end

    return y, pullback
end

# tr(A)

Zygote.@adjoint function tr(A::ChordalTriangular{:N})
    y = tr(A)

    function pullback(Δy)
        isnothing(Δy) && return (nothing,)
        return (tr_rrule_impl(A, y, Δy),)
    end

    return y, pullback
end

Zygote.@adjoint function tr(A::HermTri)
    y = tr(A)

    function pullback(Δy)
        isnothing(Δy) && return (nothing,)
        return (tr_rrule_impl(A, y, Δy),)
    end

    return y, pullback
end

Zygote.@adjoint function tr(A::SymTri)
    y = tr(A)

    function pullback(Δy)
        isnothing(Δy) && return (nothing,)
        return (tr_rrule_impl(A, y, Δy),)
    end

    return y, pullback
end

# Permutation operations

Zygote.@adjoint function \(P::Permutation, x::AbstractVecOrMat)
    y = P \ x

    function pullback(Δy)
        isnothing(Δy) && return (nothing, nothing)
        _, Δx = ldiv_rrule_impl(P, x, y, Δy)
        return (nothing, Δx)
    end

    return y, pullback
end

Zygote.@adjoint function /(x::AbstractVecOrMat, P::Permutation)
    y = x / P

    function pullback(Δy)
        isnothing(Δy) && return (nothing, nothing)
        Δx, _ = rdiv_rrule_impl(x, P, y, Δy)
        return (Δx, nothing)
    end

    return y, pullback
end

Zygote.@adjoint function *(P::Permutation, x::AbstractVecOrMat)
    y = P * x

    function pullback(Δy)
        isnothing(Δy) && return (nothing, nothing)
        _, Δx = mul_rrule_impl(P, x, y, Δy)
        return (nothing, Δx)
    end

    return y, pullback
end

Zygote.@adjoint function *(x::AbstractVecOrMat, P::Permutation)
    y = x * P

    function pullback(Δy)
        isnothing(Δy) && return (nothing, nothing)
        Δx, _ = mul_rrule_impl(x, P, y, Δy)
        return (Δx, nothing)
    end

    return y, pullback
end

# Accumulation for chordal types - use + to preserve structure
# (Default accum uses broadcast which falls back to dense Matrix)

for T in (HermTri, SymTri, ChordalTriangular)
    @eval Zygote.accum(x::$T, y::$T) = x + y
    @eval Zygote.accum(x::$T, y::Diagonal) = x + y
    @eval Zygote.accum(x::Diagonal, y::$T) = x + y
    @eval Zygote.accum(x::$T, y::UniformScaling) = x + y
    @eval Zygote.accum(x::UniformScaling, y::$T) = x + y
end

end
