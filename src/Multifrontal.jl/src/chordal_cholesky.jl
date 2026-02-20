"""
    ChordalCholesky{UPLO, T, I, Val} <: Factorization{T}

A Cholesky decomposition of a sparse positive-definite matrix ``A``:

```math
    P A P^\\mathsf{T} = L L^\\mathsf{T}.
```

where ``P`` is a permutation matrix and ``L`` is lower triangular.
The type parameter `UPLO` specifies which triangular factor is
stored.

   - `UPLO`: `:L` or `:U` (lower / upper triangular)

### Fields

   - `F.P`: permutation matrix
   - `F.L`: lower triangular factor
   - `F.U`: upper triangular factor
   - `F.S`: symbolic factorization

"""
struct ChordalCholesky{UPLO, T, I, Val <: AbstractVector{T}, Prm <: AbstractVector{I}} <: Factorization{T}
    S::ChordalSymbolic{I}
    Dval::Val
    Lval::Val
    perm::Prm
    invp::Prm
    info::FScalar{I}
end

const FChordalCholesky{UPLO, T, I} = ChordalCholesky{UPLO, T, I, FVector{T}, FVector{I}}

function ChordalCholesky{UPLO}(S::ChordalSymbolic{I}, Dval::Val, Lval::Val, perm::Prm, invp::Prm, info::FScalar{I}) where {UPLO, I <: Integer, T, Val <: AbstractVector{T}, Prm <: AbstractVector{I}}
    return ChordalCholesky{UPLO, T, I, Val, Prm}(S, Dval, Lval, perm, invp, info)
end

function ChordalCholesky{UPLO, T}(perm::Prm, S::ChordalSymbolic{I}) where {UPLO, T, I <: Integer, Prm <: AbstractVector{I}}
    n = nv(S.res)
    Dval = FVector{T}(undef, S.Dptr[n + one(I)] - one(I))
    Lval = FVector{T}(undef, S.Lptr[n + one(I)] - one(I))
    info = FScalar{I}(undef); info[] = zero(I)
    invp = similar(perm)

    @inbounds for i in eachindex(perm)
        invp[perm[i]] = i
    end

    return ChordalCholesky{UPLO}(S, Dval, Lval, perm, invp, info)
end

"""
    ChordalCholesky(A::AbstractMatrix; check=true)

Construct a Cholesky factorization object from a positive-definite matrix `A`.
Perform the factorization using the function [`cholesky!`](@ref).

### Arguments

- `check`: check if `A` is symmetric

"""
function ChordalCholesky(A::AbstractMatrix; kw...)
    return ChordalCholesky{:L}(A; kw...)
end

function ChordalCholesky(A::AbstractMatrix, perm::AbstractVector, S::ChordalSymbolic)
    return ChordalCholesky{:L}(A, perm, S)
end

function ChordalCholesky{UPLO}(A::AbstractMatrix; kw...) where {UPLO}
    return ChordalCholesky{UPLO}(sparse(A); kw...)
end

function ChordalCholesky{UPLO}(A::AbstractMatrix, perm::AbstractVector, S::ChordalSymbolic) where {UPLO}
    return ChordalCholesky{UPLO}(sparse(A), perm, S)
end

function ChordalCholesky{UPLO}(A::SparseMatrixCSC; check::Bool=true, kw...) where {UPLO}
    if !check || ishermitian(A)
        perm, S = symbolic(A; check=false, kw...)
        return ChordalCholesky{UPLO}(Hermitian(A, UPLO), perm, S)
    elseif istril(A)
        return ChordalCholesky{UPLO}(Hermitian(A, :L); kw...)
    elseif istriu(A)
        return ChordalCholesky{UPLO}(Hermitian(A, :U); kw...)
    end

    error()
end

function ChordalCholesky{UPLO}(A::HermOrSym; kw...) where {UPLO}
    perm, S = symbolic(A; kw...)
    return ChordalCholesky{UPLO}(A, perm, S)
end

"""
    ChordalCholesky(A::AbstractMatrix, perm::AbstractVector, S::ChordalSymbolic)

Construct a Cholesky factorization object from a positive-definite matrix `A`.
Perform the factorization using the function [`cholesky!`](@ref).
"""
function ChordalCholesky{UPLO}(A::AbstractMatrix{T}, perm::Prm, S::ChordalSymbolic{I}) where {UPLO, T, I <: Integer, Prm <: AbstractVector{I}}
    return copy!(ChordalCholesky{UPLO, T}(perm, S), A)
end

function ChordalCholesky{UPLO}(A::AbstractMatrix{T}, perm::Prm, S::ChordalSymbolic{I}) where {UPLO, T <: Integer, I <: Integer, Prm <: AbstractVector{I}}
    R = float(T)
    return ChordalCholesky{UPLO}(convert(AbstractMatrix{R}, A), perm, S)
end

function Base.propertynames(::ChordalCholesky)
    return (:L, :U, :P, :S, :Dval, :Lval, :perm, :invp, :info)
end

function Base.getproperty(F::ChordalCholesky{UPLO}, d::Symbol) where {UPLO}
    if d === :P
        return Permutation(getfield(F, :perm))
    elseif d === :L || d === :U
        A = ChordalTriangular(F)

        if d === UPLO
            return A
        else
            return A'
        end
    else
        return getfield(F, d)
    end
end

function Base.show(io::IO, F::T) where {T <: ChordalCholesky}
    n = size(F, 1)
    print(io, "$n×$n $T with $(nnz(F)) stored entries")
    return
end

function Base.show(io::IO, ::MIME"text/plain", F::T) where {UPLO, T <: ChordalCholesky{UPLO}}
    n = size(F, 1)
    println(io, "$n×$n $T with $(nnz(F)) stored entries:")

    if n < 16
        print_matrix(io, ChordalTriangular(F))
    else
        showsymbolic(io, F.S, Val(UPLO))
    end

    return
end

function Base.copy(F::ChordalCholesky{UPLO, T, I, Val, Prm}) where {UPLO, T, I, Val, Prm}
    return ChordalCholesky{UPLO, T, I, Val, Prm}(F.S, copy(F.Dval), copy(F.Lval), copy(F.perm), copy(F.invp), copy(F.info))
end

function Base.copy!(F::ChordalCholesky{UPLO}, A::SparseMatrixCSC; check::Bool=true) where {UPLO}
    if !check || ishermitian(A)
        return copy!(F, Hermitian(A, UPLO))
    elseif istril(A)
        return copy!(F, Hermitian(A, :L))
    elseif istriu(A)
        return copy!(F, Hermitian(A, :U))
    end

    error()
end

function Base.copy!(F::ChordalCholesky{UPLO}, A::HermOrSym; check::Bool=true) where {UPLO}
    if UPLO === :L
        B = sympermute(parent(A), F.invp, A.uplo, 'U')
    else
        B = sympermute(parent(A), F.invp, A.uplo, 'L')
    end

    copy!(ChordalTriangular(F), copy(adjoint(B)))
    return F
end

function Base.fill!(F::ChordalCholesky, x)
    fill!(F.L, x)
    return F
end

function flatindices(F::ChordalCholesky{UPLO}, A::SparseMatrixCSC; check::Bool=true) where {UPLO}
    if !check || ishermitian(A)
        return flatindices(F, Hermitian(A, UPLO))
    elseif istril(A)
        return flatindices(F, Hermitian(A, :L))
    elseif istriu(A)
        return flatindices(F, Hermitian(A, :U))
    end

    error()
end

function flatindices(F::ChordalCholesky{UPLO}, A::HermOrSym; check::Bool=true) where {UPLO}
    colptr = parent(A).colptr
    rowval = parent(A).rowval
    nzval = collect(oneto(nnz(parent(A))))
    B = SparseMatrixCSC{Int, Int}(size(A)..., colptr, rowval, nzval)

    if UPLO === :L
        B = sympermute(B, F.invp, A.uplo, 'U')
    else
        B = sympermute(B, F.invp, A.uplo, 'L')
    end

    C = copy(adjoint(B))
    P = flatindices(ChordalTriangular(F), C)
    return invpermute!(P, C.nzval)
end

function getflatindex(F::ChordalCholesky, p::Integer)
    return getflatindex(F.L, p)
end

function setflatindex!(F::ChordalCholesky, x, p::Integer)
    setflatindex!(F.L, x, p)
    return F
end

# ===== Abstract Matrix Interface =====

function SparseArrays.nnz(F::ChordalCholesky)
    return nnz(F.S)
end

function Base.size(F::ChordalCholesky)
    return size(F.S)
end

function Base.size(F::ChordalCholesky, args...)
    return size(F.S, args...)
end

function Base.axes(F::ChordalCholesky, args...)
    return axes(F.S, args...)
end

function LinearAlgebra.isposdef(F::ChordalCholesky)
    return iszero(F.info[])
end

function LinearAlgebra.issuccess(F::ChordalCholesky)
    return iszero(F.info[])
end

function LinearAlgebra.rank(F::ChordalCholesky; kw...)
    return rank(F.L; kw...)
end

function Base.adjoint(F::ChordalCholesky)
    return F
end

function LinearAlgebra.det(F::ChordalCholesky)
    return det(F.L)^2
end

function LinearAlgebra.logdet(F::ChordalCholesky)
    return 2logdet(F.L)
end

function LinearAlgebra.logabsdet(F::ChordalCholesky{UPLO, T}) where {UPLO, T}
    return (2logdet(F.L), one(T))
end
