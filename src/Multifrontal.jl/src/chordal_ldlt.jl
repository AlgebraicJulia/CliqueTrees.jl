"""
    ChordalLDLt{UPLO, T, I, Val} <: Factorization{T}

An LDLᵀ decomposition of a sparse symmetric matrix ``A``:

```math
    P A P^\\mathsf{T} = L D L^\\mathsf{T}.
```

where ``P`` is a permutation matrix, ``L`` is unit lower triangular, and ``D`` is diagonal.
The type parameter `UPLO` specifies which triangular factor is
stored.

   - `UPLO`: `:L` or `:U` (lower / upper triangular)

### Fields

   - `F.P`: permutation matrix
   - `F.L`: unit lower triangular factor
   - `F.U`: unit upper triangular factor
   - `F.D`: diagonal factor
   - `F.S`: symbolic factorization

"""
struct ChordalLDLt{UPLO, T, I, Val <: AbstractVector{T}, Prm <: AbstractVector{I}} <: Factorization{T}
    S::ChordalSymbolic{I}
    d::Val
    Dval::Val
    Lval::Val
    perm::Prm
    info::FScalar{I}
end

const FChordalLDLt{UPLO, T, I} = ChordalLDLt{UPLO, T, I, FVector{T}, FVector{I}}

function ChordalLDLt{UPLO}(S::ChordalSymbolic{I}, d::Val, Dval::Val, Lval::Val, perm::Prm, info::FScalar{I}) where {UPLO, I <: Integer, T, Val <: AbstractVector{T}, Prm <: AbstractVector{I}}
    return ChordalLDLt{UPLO, T, I, Val, Prm}(S, d, Dval, Lval, perm, info)
end

function ChordalLDLt{UPLO, T}(perm::Prm, S::ChordalSymbolic{I}) where {UPLO, T, I <: Integer, Prm <: AbstractVector{I}}
    n = nv(S.res)
    d = FVector{T}(undef, nov(S.res))
    Dval = FVector{T}(undef, S.Dptr[n + one(I)] - one(I))
    Lval = FVector{T}(undef, S.Lptr[n + one(I)] - one(I))
    info = FScalar{I}(undef); info[] = zero(I)
    return ChordalLDLt{UPLO}(S, d, Dval, Lval, perm, info)
end

function ChordalLDLt(A::AbstractMatrix; kw...)
    return ChordalLDLt{:L}(A; kw...)
end

function ChordalLDLt{UPLO}(A::AbstractMatrix; kw...) where {UPLO}
    return ChordalLDLt{UPLO}(sparse(A); kw...)
end

function ChordalLDLt{UPLO}(A::SparseMatrixCSC; kw...) where {UPLO}
    perm, S = symbolic(A; kw...)
    return ChordalLDLt{UPLO}(A, perm, S)
end

function ChordalLDLt{UPLO}(A::SparseMatrixCSC{T}, perm::Prm, S::ChordalSymbolic{I}) where {UPLO, T, I <: Integer, Prm <: AbstractVector{I}}
    return copy!(ChordalLDLt{UPLO, T}(perm, S), A)
end

function ChordalLDLt{UPLO}(A::SparseMatrixCSC{T}, perm::Prm, S::ChordalSymbolic{I}) where {UPLO, T <: Integer, I <: Integer, Prm <: AbstractVector{I}}
    return ChordalLDLt{UPLO}(SparseMatrixCSC{float(T)}(A), perm, S)
end

function Base.propertynames(::ChordalLDLt)
    return (:L, :U, :D, :P, :S, :d, :Dval, :Lval, :perm, :info)
end

function Base.getproperty(F::ChordalLDLt{UPLO}, d::Symbol) where {UPLO}
    if d === :P
        return Permutation(getfield(F, :perm))
    elseif d === :D
        return Diagonal(getfield(F, :d))
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

function Base.show(io::IO, F::T) where {T <: ChordalLDLt}
    n = size(F, 1)
    print(io, "$n×$n $T with $(nnz(F)) stored entries")
    return
end

function Base.show(io::IO, ::MIME"text/plain", F::T) where {UPLO, T <: ChordalLDLt{UPLO}}
    n = size(F, 1)
    println(io, "$n×$n $T with $(nnz(F)) stored entries:")

    if n < 16
        print_matrix(io, ChordalTriangular(F))
        println(io)
        println(io)
        print_matrix(io, F.D)
    else
        showsymbolic(io, F.S, Val(UPLO))
    end

    return
end

function Base.copy(F::ChordalLDLt{UPLO, T, I, Val, Prm}) where {UPLO, T, I, Val, Prm}
    return ChordalLDLt{UPLO, T, I, Val, Prm}(F.S, copy(F.d), copy(F.Dval), copy(F.Lval), copy(F.perm), copy(F.info))
end

function Base.copy!(F::ChordalLDLt{UPLO, T, I}, A::SparseMatrixCSC) where {UPLO, T, I <: Integer}
    A = permute(A, F.perm, F.perm)
    copy_D!(F.S.Dptr, F.Dval, F.S.res, A)
    copy_L!(F.S.Lptr, F.Lval, F.S.res, F.S.sep, A, Val{UPLO}())
    return F
end

# ===== Abstract Matrix Interface =====

function SparseArrays.nnz(F::ChordalLDLt)
    return nnz(F.S)
end

function Base.size(F::ChordalLDLt)
    return size(F.S)
end

function Base.size(F::ChordalLDLt, args...)
    return size(F.S, args...)
end

function Base.axes(F::ChordalLDLt, args...)
    return axes(F.S, args...)
end

function LinearAlgebra.isposdef(F::ChordalLDLt)
    return isposdef(F.D)
end

function LinearAlgebra.issuccess(F::ChordalLDLt)
    return iszero(F.info[])
end

function Base.adjoint(F::ChordalLDLt)
    return F
end

function LinearAlgebra.det(F::ChordalLDLt)
    return det(F.L)^2 * det(F.D)
end

function LinearAlgebra.logabsdet(F::ChordalLDLt)
    return logabsdet(F.D)
end

function LinearAlgebra.logdet(F::ChordalLDLt)
    out, sgn = logabsdet(F)
    ispositive(real(sgn)) || error()
    return out
end
