abstract type AbstractFactorization{DIAG, UPLO, T, I, Prm <: AbstractVector{I}, Ivp <: AbstractVector{I}} <: Factorization{T} end

const AbstractCholesky = AbstractFactorization{:N}
const AbstractLDLt = AbstractFactorization{:U}
const NaturalFactorization{DIAG, UPLO, T, I} = AbstractFactorization{DIAG, UPLO, T, I, OneTo{I}, OneTo{I}}

# ===== Base methods =====

function Base.:(==)(F::AbstractFactorization, G::AbstractFactorization)
    return false
end

function Base.:(==)(F::AbstractFactorization{DIAG, UPLO}, G::AbstractFactorization{DIAG, UPLO}) where {DIAG, UPLO}
    out = F.P === G.P && triangular(F) == triangular(G)

    if DIAG === :U
        out = out && F.D == G.D
    end

    return out
end

function Base.getproperty(F::AbstractFactorization{DIAG, UPLO}, s::Symbol) where {DIAG, UPLO}
    if s === :P
        return Permutation(getfield(F, :perm), getfield(F, :invp))
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

function Base.copyto!(F::AbstractFactorization{DIAG}, G::AbstractFactorization{DIAG}) where {DIAG}
    if DIAG === :U
        copyto!(F.d, G.d)
    end

    copyto!(triangular(F, Val(:N)), triangular(G, Val(:N)))
    return F
end

function Base.copy(F::AbstractFactorization)
    return copyto!(similar(F), F)
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

function Base.copy!(F::AbstractFactorization, A; kw...)
    return copyto!(F, A; kw...)
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

function inertiadiag(D::Diagonal{T}; atol::Real=zero(real(T)), rtol::Real=zero(real(T))) where {T}
    R = real(T)

    if R <: AbstractFloat && !ispositive(rtol) && !ispositive(atol)
        tol = size(D, 1) * eps(R)
    else
        tol = convert(R, max(atol, rtol * maximum(abs, D.diag)))
    end

    np = nn = nz = 0

    for i in axes(D, 1)
        Dii = D.diag[i]

        if real(Dii) > tol
            np += 1
        elseif real(Dii) < -tol
            nn += 1
        else
            nz += 1
        end
    end

    return (np, nn, nz)
end

@static if VERSION >= v"1.11"
    function LinearAlgebra.inertia(F::AbstractFactorization{DIAG}; kw...) where {DIAG}
        if DIAG === :N
            return (ncl(F), 0, 0)
        else
            return inertiadiag(F.D; kw...)
        end
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
        return F.P' * diag(F.L).^2
    else
        return F.P' * diag(F.D)
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

function ncl(F::AbstractFactorization)
    return size(F, 1)
end

function Base.length(F::AbstractFactorization)
    return ncl(F)^2
end

function uncopy(F::NaturalFactorization{DIAG, UPLO}) where {DIAG, UPLO}
    return Hermitian(triangular(F, Val(:N)), UPLO)
end

function uncopy(F::AbstractFactorization{DIAG, UPLO}) where {DIAG, UPLO}
    return cong(uncopy(NaturalFactorization(F)), F.P)
end

function unfactorize(F::AbstractFactorization)
    G = unfactorize!(copy(F))
    return uncopy(G)
end

function (::Type{Mat})(F::AbstractFactorization) where {Mat <: Matrix}
    return Mat(unfactorize(F))
end

function SparseArrays.sparse(F::AbstractFactorization)
    return sparse(unfactorize(F))
end
