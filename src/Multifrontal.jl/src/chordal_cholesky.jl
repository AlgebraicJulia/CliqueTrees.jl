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
    info::FScalar{I}
    rank::FScalar{I}
end

const FChordalCholesky{UPLO, T, I} = ChordalCholesky{UPLO, T, I, FVector{T}, FVector{I}}

function ChordalCholesky{UPLO}(S::ChordalSymbolic{I}, Dval::Val, Lval::Val, perm::Prm, info::FScalar{I}, rank::FScalar{I}) where {UPLO, I <: Integer, T, Val <: AbstractVector{T}, Prm <: AbstractVector{I}}
    return ChordalCholesky{UPLO, T, I, Val, Prm}(S, Dval, Lval, perm, info, rank)
end

function ChordalCholesky{UPLO, T}(perm::Prm, S::ChordalSymbolic{I}) where {UPLO, T, I <: Integer, Prm <: AbstractVector{I}}
    n = nv(S.res)
    Dval = FVector{T}(undef, S.Dptr[n + one(I)] - one(I))
    Lval = FVector{T}(undef, S.Lptr[n + one(I)] - one(I))
    info = FScalar{I}(undef); info[] = zero(I)
    rank = FScalar{I}(undef); rank[] = nov(S.res)
    return ChordalCholesky{UPLO}(S, Dval, Lval, perm, info, rank)
end

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

function ChordalCholesky{UPLO}(A::SparseMatrixCSC; kw...) where {UPLO}
    perm, S = symbolic(A; kw...)
    return ChordalCholesky{UPLO}(A, perm, S)
end

function ChordalCholesky{UPLO}(A::SparseMatrixCSC{T}, perm::Prm, S::ChordalSymbolic{I}) where {UPLO, T, I <: Integer, Prm <: AbstractVector{I}}
    return copy!(ChordalCholesky{UPLO, T}(perm, S), A)
end

function ChordalCholesky{UPLO}(A::SparseMatrixCSC{T}, perm::Prm, S::ChordalSymbolic{I}) where {UPLO, T <: Integer, I <: Integer, Prm <: AbstractVector{I}}
    return ChordalCholesky{UPLO}(SparseMatrixCSC{float(T)}(A), perm, S)
end

function Base.propertynames(::ChordalCholesky)
    return (:L, :U, :P, :S, :Dval, :Lval, :perm, :info, :rank)
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
    return ChordalCholesky{UPLO, T, I, Val, Prm}(F.S, copy(F.Dval), copy(F.Lval), copy(F.perm), copy(F.info), copy(F.rank))
end

function Base.copy!(F::ChordalCholesky{UPLO, T, I}, A::SparseMatrixCSC{T, I}) where {UPLO, T, I <: Integer}
    A = permute(A, F.perm, F.perm)
    copy_D!(F.S.Dptr, F.Dval, F.S.res, A)
    copy_L!(F.S.Lptr, F.Lval, F.S.res, F.S.sep, A, Val{UPLO}())
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

function LinearAlgebra.rank(F::ChordalCholesky)
    return convert(Int, F.rank[])
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
