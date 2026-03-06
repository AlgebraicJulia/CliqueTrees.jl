abstract type AbstractFactorization{DIAG, UPLO, T, I, Prm <: AbstractVector{I}} <: Factorization{T} end

const IFactorization{DIAG, UPLO, T, I} = AbstractFactorization{DIAG, UPLO, T, I, OneTo{I}}

# ===== Base methods =====

function Base.getproperty(F::AbstractFactorization{DIAG, UPLO}, s::Symbol) where {DIAG, UPLO}
    if s === :P
        return Permutation(getfield(F, :perm))
    elseif s === :D
        return Diagonal(getfield(F, :d))
    elseif s === :L || s === :U
        if s === UPLO
            return triangular(F)
        else
            return triangular(F)'
        end
    elseif s === :uplo
        return Val(UPLO)
    elseif s === :diag
        return Val(DIAG)
    else
        return getfield(F, s)
    end
end

function Base.copyto!(F::AbstractFactorization{DIAG, UPLO}, A::AbstractMatrix; check::Bool=true) where {DIAG, UPLO}
    if !check || ishermitian(A)
        return copyto!(F, Hermitian(A, UPLO))
    elseif istril(A)
        return copyto!(F, Hermitian(A, :L))
    elseif istriu(A)
        return copyto!(F, Hermitian(A, :U))
    end

    error()
end

function Base.copyto!(F::AbstractFactorization, A::HermOrSym; check::Bool=true)
    sympermute!(triangular(F), parent(A), F.invp, A.uplo, char(F.uplo))
    return F
end

function Base.copy!(F::AbstractFactorization, A::AbstractMatrix; check::Bool=true)
    return copyto!(F, A; check)
end

function Base.size(F::AbstractFactorization)
    return size(triangular(F))
end

function Base.size(F::AbstractFactorization, args...)
    return size(triangular(F), args...)
end

function Base.axes(F::AbstractFactorization, args...)
    return axes(triangular(F), args...)
end

function Base.fill!(F::AbstractFactorization, x)
    fill!(triangular(F), x)
    return F
end

function Base.adjoint(F::AbstractFactorization)
    return F
end

# ===== LinearAlgebra =====

function LinearAlgebra.issuccess(F::AbstractFactorization)
    return iszero(F.info[])
end

function LinearAlgebra.rank(F::AbstractFactorization{DIAG}; kw...) where {DIAG}
    if DIAG === :N
        return rank(F.L; kw...)
    else
        return rank(F.D; kw...)
    end
end

function LinearAlgebra.isposdef(F::AbstractFactorization{DIAG}) where {DIAG}
    if DIAG === :N
        return iszero(F.info[])
    else
        return isposdef(F.D)
    end
end

function LinearAlgebra.det(F::AbstractFactorization{DIAG}) where {DIAG}
    if DIAG === :N
        return det(F.L)^2
    else
        return det(F.L)^2 * det(F.D)
    end
end

function LinearAlgebra.logdet(F::AbstractFactorization{DIAG}) where {DIAG}
    if DIAG === :N
        return 2logdet(F.L)
    else
        d, s = logabsdet(F)
        return d + log(s)
    end
end

function LinearAlgebra.logabsdet(F::AbstractFactorization{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    if DIAG === :N
        return (2logdet(F.L), one(T))
    else
        return logabsdet(F.D)
    end
end

function LinearAlgebra.diag(F::AbstractFactorization{DIAG}) where {DIAG}
    if DIAG === :N
        return diag(F.L).^2
    else
        return diag(F.D)
    end
end

function LinearAlgebra.cond(F::AbstractFactorization, p::Real=2)
    if p == 1 || p == Inf
        condest1(F)
    elseif p == 2
        condest2(F)
    else
        error()
    end
end

function LinearAlgebra.opnorm(F::AbstractFactorization, p::Real=2)
    if p == 1 || p == Inf
        opnormest1(F)
    elseif p == 2
        opnormest2(F)
    else
        error()
    end
end

function Base.Matrix{T}(F::AbstractFactorization) where {T}
    B = Matrix{T}(I, size(F))
    return lmul!(F, B)
end

function Base.Matrix(F::AbstractFactorization{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    return Matrix{T}(F)
end

function ncl(F::AbstractFactorization)
    return size(F, 1)
end
