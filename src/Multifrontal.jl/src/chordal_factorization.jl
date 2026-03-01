struct ChordalFactorization{DIAG, UPLO, T, I, Dia <: AbstractVector{T}, Val <: AbstractVector{T}, Prm <: AbstractVector{I}} <: Factorization{T}
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

const FChordalCholesky{UPLO, T, I} = ChordalCholesky{UPLO, T, I, Ones{T, 1, Tuple{OneTo{Int}}}, FVector{T}, FVector{I}}
const FChordalLDLt{UPLO, T, I} = ChordalLDLt{UPLO, T, I, FVector{T}, FVector{T}, FVector{I}}

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

# ===== Constructors =====

function ChordalFactorization{DIAG}(A::AbstractMatrix; kw...) where {DIAG}
    return ChordalFactorization{DIAG, DEFAULT_UPLO}(A; kw...)
end

function ChordalFactorization{DIAG, UPLO}(S::ChordalSymbolic{I}, d::Dia, Dval::Val, Lval::Val, perm::Prm, invp::Prm, info::FScalar{I}) where {DIAG, UPLO, I <: Integer, T, Dia <: AbstractVector{T}, Val <: AbstractVector{T}, Prm <: AbstractVector{I}}
    return ChordalFactorization{DIAG, UPLO, T, I, Dia, Val, Prm}(S, d, Dval, Lval, perm, invp, info)
end

function ChordalFactorization{DIAG, UPLO, T}(perm::Prm, S::ChordalSymbolic{I}) where {DIAG, UPLO, T, I <: Integer, Prm <: AbstractVector{I}}
    n = nv(S.res)
    Dval = FVector{T}(undef, S.Dptr[n + one(I)] - one(I))
    Lval = FVector{T}(undef, S.Lptr[n + one(I)] - one(I))
    info = FScalar{I}(undef); info[] = zero(I)
    invp = similar(perm)

    @inbounds for i in eachindex(perm)
        invp[perm[i]] = i
    end

    if DIAG === :N
        d = Ones{T}(nov(S.res))
    else
        d = FVector{T}(undef, nov(S.res))
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

# ===== Base methods =====

function Base.copy!(F::ChordalFactorization{DIAG, UPLO}, A::SparseMatrixCSC; check::Bool=true) where {DIAG, UPLO}
    if !check || ishermitian(A)
        return copy!(F, Hermitian(A, UPLO))
    elseif istril(A)
        return copy!(F, Hermitian(A, :L))
    elseif istriu(A)
        return copy!(F, Hermitian(A, :U))
    end

    error()
end

function Base.copy!(F::ChordalFactorization{DIAG, UPLO}, A::HermOrSym; check::Bool=true) where {DIAG, UPLO}
    if UPLO === :L
        B = sympermute(parent(A), F.invp, A.uplo, 'U')
    else
        B = sympermute(parent(A), F.invp, A.uplo, 'L')
    end

    copy!(ChordalTriangular(F), copy(adjoint(B)))
    return F
end

function Base.fill!(F::ChordalFactorization, x)
    fill!(ChordalTriangular(F), x)
    return F
end

function Base.adjoint(F::ChordalFactorization)
    return F
end

function Base.show(io::IO, F::T) where {T <: ChordalFactorization}
    n = size(F, 1)
    print(io, "$n×$n $T with $(nnz(F)) stored entries")
    return
end

function Base.show(io::IO, ::MIME"text/plain", F::T) where {DIAG, UPLO, T <: ChordalFactorization{DIAG, UPLO}}
    n = size(F, 1)
    println(io, "$n×$n $T with $(nnz(F)) stored entries:")

    if n < 16
        print_matrix(io, ChordalTriangular(F))
        if DIAG === :U
            println(io)
            println(io)
            print_matrix(io, F.D)
        end
    else
        showsymbolic(io, F.S, Val(UPLO))
    end

    return
end

function Base.propertynames(::ChordalFactorization)
    return (:L, :U, :D, :P, :S, :d, :Dval, :Lval, :perm, :invp, :info)
end

function Base.getproperty(F::ChordalFactorization{DIAG, UPLO}, d::Symbol) where {DIAG, UPLO}
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

function Base.copy(F::ChordalFactorization{DIAG, UPLO}) where {DIAG, UPLO}
    return ChordalFactorization{DIAG, UPLO}(F.S, copy(F.d), copy(F.Dval), copy(F.Lval), copy(F.perm), copy(F.invp), copy(F.info))
end

# ===== LinearAlgebra =====

function LinearAlgebra.isposdef(F::ChordalFactorization{DIAG}) where {DIAG}
    if DIAG === :N
        return iszero(F.info[])
    else
        return isposdef(F.D)
    end
end

function LinearAlgebra.rank(F::ChordalFactorization{DIAG}; kw...) where {DIAG}
    if DIAG === :N
        return rank(F.L; kw...)
    else
        return rank(F.D; kw...)
    end
end

function LinearAlgebra.det(F::ChordalFactorization{DIAG}) where {DIAG}
    if DIAG === :N
        return det(F.L)^2
    else
        return det(F.L)^2 * det(F.D)
    end
end

function LinearAlgebra.logdet(F::ChordalFactorization{DIAG}) where {DIAG}
    if DIAG === :N
        return 2logdet(F.L)
    else
        d, s = logabsdet(F)
        return d + log(s)
    end
end

function LinearAlgebra.logabsdet(F::ChordalFactorization{DIAG}) where {DIAG}
    if DIAG === :N
        return (2logdet(F.L), one(eltype(F)))
    else
        return logabsdet(F.D)
    end
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
        B = sympermute(B, F.invp, A.uplo, 'U')
    else
        B = sympermute(B, F.invp, A.uplo, 'L')
    end

    C = copy(adjoint(B)); P = flatindices(ChordalTriangular(F), C)
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

function Base.Matrix{T}(F::ChordalFactorization{DIAG}) where {DIAG, T}
    B = Matrix{T}(I, size(F))
    return lmul!(F, B)
end

function Base.Matrix(F::ChordalFactorization{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    return Matrix{T}(F)
end

function LinearAlgebra.diag(F::ChordalFactorization{DIAG}) where {DIAG}
    if DIAG === :N
        return diag(F.L).^2
    else
        return diag(F.D)
    end
end

function SparseArrays.nnz(F::ChordalFactorization)
    return nnz(F.S)
end

function Base.size(F::ChordalFactorization)
    return size(F.S)
end

function Base.size(F::ChordalFactorization, args...)
    return size(F.S, args...)
end

function Base.axes(F::ChordalFactorization, args...)
    return axes(F.S, args...)
end

function LinearAlgebra.issuccess(F::ChordalFactorization)
    return iszero(F.info[])
end

function LinearAlgebra.cond(F::ChordalFactorization, p::Real)
    if p == 1 || p == Inf
        condest1(F)
    elseif p == 2
        condest2(F)
    else
        error()
    end
end

function LinearAlgebra.opnorm(F::ChordalFactorization, p::Real)
    if p == 1 || p == Inf
        opnormest1(F)
    elseif p == 2
        opnormest2(F)
    else
        error()
    end
end
