struct ChordalFactorization{DIAG, UPLO, T, I, Dia <: AbstractVector{T}, Val <: AbstractVector{T}, Prm <: AbstractVector{I}} <: AbstractFactorization{DIAG, UPLO, T, I, Prm}
    S::ChordalSymbolic{I}
    d::Dia
    Dval::Val
    Lval::Val
    perm::Prm
    invp::Prm
    info::FScalar{I}
end

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
const ChordalCholesky = ChordalFactorization{:N}
const FChordalCholesky{UPLO, T, I} = ChordalCholesky{UPLO, T, I, Ones{T, 1, Tuple{OneTo{Int}}}, FVector{T}, FVector{I}}

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
const ChordalLDLt = ChordalFactorization{:U}
const FChordalLDLt{UPLO, T, I} = ChordalLDLt{UPLO, T, I, FVector{T}, FVector{T}, FVector{I}}

# ===== Constructors =====

function ChordalFactorization{DIAG}(A::AbstractMatrix; kw...) where {DIAG}
    return ChordalFactorization{DIAG, DEFAULT_UPLO}(A; kw...)
end

function ChordalFactorization{DIAG, UPLO}(S::ChordalSymbolic{I}, d::Dia, Dval::Val, Lval::Val, perm::Prm, invp::Prm, info::FScalar{I}) where {DIAG, UPLO, I <: Integer, T, Dia <: AbstractVector{T}, Val <: AbstractVector{T}, Prm <: AbstractVector{I}}
    return ChordalFactorization{DIAG, UPLO, T, I, Dia, Val, Prm}(S, d, Dval, Lval, perm, invp, info)
end

function ChordalFactorization{DIAG, UPLO, T}(perm::Prm, S::ChordalSymbolic{I}) where {DIAG, UPLO, T, I <: Integer, Prm <: AbstractVector{I}}
    n = nfr(S)
    Dval = FVector{T}(undef, convert(Int, S.Dptr[n + 1]) - 1)
    Lval = FVector{T}(undef, convert(Int, S.Lptr[n + 1]) - 1)
    info = FScalar{I}(undef); info[] = zero(I)
    invp = similar(perm)

    @inbounds for i in eachindex(perm)
        invp[perm[i]] = i
    end

    if DIAG === :N
        d = Ones{T}(ncl(S))
    else
        d = FVector{T}(undef, ncl(S))
    end

    return ChordalFactorization{DIAG, UPLO}(S, d, Dval, Lval, perm, invp, info)
end

function ChordalFactorization{DIAG, UPLO}(A::AbstractMatrix; kw...) where {DIAG, UPLO}
    return ChordalFactorization{DIAG, UPLO}(sparse(A); kw...)
end

function ChordalFactorization{DIAG, UPLO}(A::AbstractMatrix, perm::AbstractVector, S::ChordalSymbolic) where {DIAG, UPLO}
    return ChordalFactorization{DIAG, UPLO}(sparse(A), perm, S)
end

function ChordalFactorization{DIAG, UPLO}(A::SparseMatrixCSC; check::Bool=true, kw...) where {DIAG, UPLO}
    if !check || ishermitian(A)
        perm, S = symbolic(A; check=false, kw...)
        return ChordalFactorization{DIAG, UPLO}(Hermitian(A, UPLO), perm, S)
    elseif istril(A)
        return ChordalFactorization{DIAG, UPLO}(Hermitian(A, :L); kw...)
    elseif istriu(A)
        return ChordalFactorization{DIAG, UPLO}(Hermitian(A, :U); kw...)
    end

    error()
end

function ChordalFactorization{DIAG, UPLO}(A::HermOrSym; kw...) where {DIAG, UPLO}
    perm, S = symbolic(A; kw...)
    return ChordalFactorization{DIAG, UPLO}(A, perm, S)
end

function ChordalFactorization{DIAG, UPLO}(A::AbstractMatrix{T}, perm::Prm, S::ChordalSymbolic{I}) where {DIAG, UPLO, T, I <: Integer, Prm <: AbstractVector{I}}
    return copy!(ChordalFactorization{DIAG, UPLO, T}(perm, S), A)
end

function ChordalFactorization{DIAG, UPLO}(A::AbstractMatrix{T}, perm::Prm, S::ChordalSymbolic{I}) where {DIAG, UPLO, T <: Integer, I <: Integer, Prm <: AbstractVector{I}}
    R = float(T)
    return ChordalFactorization{DIAG, UPLO}(convert(AbstractMatrix{R}, A), perm, S)
end

function IFactorization(F::ChordalFactorization{DIAG, UPLO, T, I}) where {DIAG, UPLO, T, I}
    perm = invp = OneTo{I}(ncl(F))
    return ChordalFactorization{DIAG, UPLO}(F.S, F.d, F.Dval, F.Lval, perm, invp, F.info)
end

function triangular(F::ChordalFactorization)
    return ChordalTriangular(F)
end

# ===== Base methods =====

function Base.show(io::IO, ::Type{FChordalCholesky{UPLO, T, I}}) where {UPLO, T, I}
    if !isdefined(get(io, :module, Main), :FChordalCholesky)
        print(io, "Multifrontal.")
    end

    print(io, "FChordalCholesky{", repr(UPLO), ", ", T, ", ", I, "}")
end

function Base.show(io::IO, ::Type{FChordalLDLt{UPLO, T, I}}) where {UPLO, T, I}
    if !isdefined(get(io, :module, Main), :FChordalLDLt)
        print(io, "Multifrontal.")
    end

    print(io, "FChordalLDLt{", repr(UPLO), ", ", T, ", ", I, "}")
end

function Base.show(io::IO, F::T) where {T <: ChordalFactorization}
    n = ncl(F)
    print(io, "$n×$n $T with $(nnz(F)) stored entries")
    return
end

function Base.show(io::IO, ::MIME"text/plain", F::T) where {DIAG, UPLO, T <: ChordalFactorization{DIAG, UPLO}}
    n = ncl(F)
    println(io, "$n×$n $T with $(nnz(F)) stored entries:")

    if n < 16
        print_matrix(io, ChordalTriangular(F))

        if DIAG === :U
            println(io)
            println(io)
            print_matrix(io, F.D)
        end
    else
        showsymbolic(io, F.S, F.uplo)
    end

    return
end

function Base.propertynames(::ChordalFactorization)
    return (:L, :U, :D, :P, :S, :d, :Dval, :Lval, :perm, :invp, :info)
end

function Base.copy(F::ChordalFactorization{DIAG, UPLO}) where {DIAG, UPLO}
    return ChordalFactorization{DIAG, UPLO}(F.S, copy(F.d), copy(F.Dval), copy(F.Lval), copy(F.perm), copy(F.invp), copy(F.info))
end

# ===== flatindices =====

function flatindices(F::ChordalFactorization{DIAG, UPLO}, A::SparseMatrixCSC; check::Bool=true) where {DIAG, UPLO}
    if !check || ishermitian(A)
        return flatindices(F, Hermitian(A, UPLO))
    elseif istril(A)
        return flatindices(F, Hermitian(A, :L))
    elseif istriu(A)
        return flatindices(F, Hermitian(A, :U))
    end

    error()
end

function flatindices(F::ChordalFactorization{DIAG, UPLO, T, I}, A::HermOrSym; check::Bool=true) where {DIAG, UPLO, T, I}
    m = convert(I, nnz(parent(A)))
    colptr = parent(A).colptr
    rowval = parent(A).rowval
    nzval = collect(oneto(m))
    B = SparseMatrixCSC(size(A)..., colptr, rowval, nzval)

    if UPLO === :L
        C = sympermute(B, F.invp, A.uplo, 'L')
    else
        C = sympermute(B, F.invp, A.uplo, 'U')
    end

    P = flatindices(ChordalTriangular(F), C)
    fill!(nzval, zero(I))

    for (i, j) in zip(nonzeros(C), P)
        nzval[i] = j
    end

    return nzval
end

function getflatindex(F::ChordalFactorization, p::Integer)
    return getflatindex(ChordalTriangular(F), p)
end

function setflatindex!(F::ChordalFactorization, x, p::Integer)
    setflatindex!(ChordalTriangular(F), x, p)
    return F
end

# ===== Abstract Matrix Interface =====

function SparseArrays.nnz(F::ChordalFactorization)
    return nnz(ChordalTriangular(F))
end

function nfr(F::ChordalFactorization)
    return nfr(ChordalTriangular(F))
end

function fronts(F::ChordalFactorization)
    return fronts(ChordalTriangular(F))
end

function diagblock(F::ChordalFactorization, j::Integer)
    return diagblock(ChordalTriangular(F), j)
end

function offdblock(F::ChordalFactorization, j::Integer)
    return offdblock(ChordalTriangular(F), j)
end

