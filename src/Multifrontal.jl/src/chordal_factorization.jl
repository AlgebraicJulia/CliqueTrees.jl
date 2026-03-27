struct ChordalFactorization{
        DIAG,
        UPLO,
        T,
        I,
        Dia <: AbstractVector{T},
        Dvl <: AbstractVector{T},
        Lvl <: AbstractVector{T},
        Prm <: AbstractVector{I},
        Ivp <: AbstractVector{I},
        Ifo,
    } <: AbstractFactorization{DIAG, UPLO, T, I, Prm, Ivp}
    S::ChordalSymbolic{I}
    d::Dia
    Dval::Dvl
    Lval::Lvl
    perm::Prm
    invp::Ivp
    info::Ifo
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

const FChordalCholesky{UPLO, T, I} = ChordalCholesky{
    UPLO,
    T,
    I,
    IOnes{T},
    FVector{T},
    FVector{T},
    FVector{I},
    FVector{I},
    FScalar{I},
}

const DChordalCholesky{UPLO, T, I} = ChordalCholesky{
    UPLO,
    T,
    I,
    IOnes{T},
    Vector{T},
    Vector{T},
    Vector{I},
    Vector{I},
    Scalar{I},
}

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

const FChordalLDLt{UPLO, T, I} = ChordalLDLt{
    UPLO,
    T,
    I,
    FVector{T},
    FVector{T},
    FVector{T},
    FVector{I},
    FVector{I},
    FScalar{I},
}

const DChordalLDLt{UPLO, T, I} = ChordalLDLt{
    UPLO,
    T,
    I,
    Vector{T},
    Vector{T},
    Vector{T},
    Vector{I},
    Vector{I},
    Scalar{I},
}

# ===== Constructors =====

function ChordalFactorization{DIAG, UPLO}(
        S::ChordalSymbolic{I},
        d::Dia,
        Dval::Dvl,
        Lval::Lvl,
        perm::Prm,
        invp::Ivp,
        info::Ifo,
    ) where {
        DIAG,
        UPLO,
        I <: Integer,
        T,
        Dia <: AbstractVector{T},
        Dvl <: AbstractVector{T},
        Lvl <: AbstractVector{T},
        Prm <: AbstractVector{I},
        Ivp <: AbstractVector{I},
        Ifo,
    }
    return ChordalFactorization{DIAG, UPLO, T, I, Dia, Dvl, Lvl, Prm, Ivp, Ifo}(S, d, Dval, Lval, perm, invp, info)
end

function ChordalFactorization{DIAG, UPLO, T, I, Dia, Dvl, Lvl, Prm, Ivp, Ifo}(
        P::Permutation{I},
        S::ChordalSymbolic{I},
    ) where {
        DIAG,
        UPLO,
        T,
        I <: Integer,
        Dia <: AbstractVector{T},
        Dvl <: AbstractVector{T},
        Lvl <: AbstractVector{T},
        Prm <: AbstractVector{I},
        Ivp <: AbstractVector{I},
        Ifo,
    }
    d = allocate(Dia, ncl(S))
    Dval = allocate(Dvl, ndz(S))
    Lval = allocate(Lvl, nlz(S))
    info = allocate(Ifo)

    return ChordalFactorization{DIAG, UPLO, T, I, Dia, Dvl, Lvl, Prm, Ivp, Ifo}(S, d, Dval, Lval, P.perm, P.invp, info)
end

for Fac in (:FChordalCholesky, :DChordalCholesky, :FChordalLDLt, :DChordalLDLt)
    @eval function $Fac{UPLO, T}(
            P::Permutation{I},
            S::ChordalSymbolic{I},
        ) where {
            UPLO,
            T,
            I <: Integer,
        }
        return $Fac{UPLO, T, I}(P, S)
    end
end

function ChordalFactorization{DIAG, UPLO, T}(P::Permutation{I, Prm, Ivp}, S::ChordalSymbolic{I}) where {DIAG, UPLO, T, I <: Integer, Prm <: AbstractVector{I}, Ivp <: AbstractVector{I}}
    if DIAG === :N
        Dia = IOnes{T}
    else
        Dia = FVector{T}
    end

    return ChordalFactorization{DIAG, UPLO, T, I, Dia, FVector{T}, FVector{T}, Prm, Ivp, FScalar{I}}(P, S)
end

function (::Type{Fac})(A::AbstractMatrix; kw...) where {DIAG, Fac <: ChordalFactorization{DIAG}}
    return Fac{DEFAULT_UPLO}(A; kw...)
end

function (::Type{Fac})(A::AbstractMatrix; kw...) where {DIAG, UPLO, Fac <: ChordalFactorization{DIAG, UPLO}}
    return Fac(sparse(A); kw...)
end

function (::Type{Fac})(A::SparseMatrixCSC; check::Bool=true, kw...) where {DIAG, UPLO, Fac <: ChordalFactorization{DIAG, UPLO}}
    if !check || ishermitian(A)
        return Fac(Hermitian(A, UPLO), symbolic(A; check=false, kw...)...)
    elseif istril(A)
        return Fac(Hermitian(A, :L); kw...)
    elseif istriu(A)
        return Fac(Hermitian(A, :U); kw...)
    end

    error()
end

function (::Type{Fac})(A::HermOrSym; kw...) where {DIAG, UPLO, Fac <: ChordalFactorization{DIAG, UPLO}}
    return Fac(A, symbolic(A; kw...)...)
end

function (::Type{Fac})(A::AbstractMatrix, P::Permutation, S::ChordalSymbolic; check::Bool=true) where {DIAG, Fac <: ChordalFactorization{DIAG}}
    return Fac{DEFAULT_UPLO}(A, P, S; check)
end

function (::Type{Fac})(A::AbstractMatrix{T}, P::Permutation, S::ChordalSymbolic; check::Bool=true) where {DIAG, UPLO, T, Fac <: ChordalFactorization{DIAG, UPLO}}
    return Fac{T}(A, P, S; check)
end

function (::Type{Fac})(A::AbstractMatrix{T}, P::Permutation, S::ChordalSymbolic; check::Bool=true) where {DIAG, UPLO, T <: Integer, Fac <: ChordalFactorization{DIAG, UPLO}}
    return Fac{float(T)}(A, P, S; check)
end

function (::Type{Fac})(A::AbstractMatrix{T}, P::Permutation, S::ChordalSymbolic; check::Bool=true) where {DIAG, UPLO, T <: Integer, R, Fac <: ChordalFactorization{DIAG, UPLO, R}}
    return copyto!(Fac(P, S), A; check)
end

function (::Type{Fac})(A::AbstractMatrix, P::Permutation, S::ChordalSymbolic; check::Bool=true) where {DIAG, UPLO, T, Fac <: ChordalFactorization{DIAG, UPLO, T}}
    return copyto!(Fac(P, S), A; check)
end

function (::Type{Fac})(A::AbstractMatrix, S::ChordalSymbolic{I}; check::Bool=true) where {DIAG, I, Fac <: ChordalFactorization{DIAG}}
    P = NaturalPermutation{I}(ncl(S))
    return Fac(A, P, S; check)
end

function NaturalFactorization(F::ChordalFactorization{DIAG, UPLO, T, I}) where {DIAG, UPLO, T, I}
    perm = invp = OneTo{I}(ncl(F))
    return ChordalFactorization{DIAG, UPLO}(F.S, F.d, F.Dval, F.Lval, perm, invp, F.info)
end

function triangular(F::ChordalFactorization, ::Val{DIAG}=F.diag) where {DIAG}
    return ChordalTriangular{DIAG}(F)
end

# ===== Base methods =====

for Fac in (:FChordalCholesky, :FChordalLDLt)
    @eval function Base.show(io::IO, ::Type{$Fac{UPLO, T, I}}) where {UPLO, T, I}
        if !isdefined(get(io, :module, Main), $(QuoteNode(Fac)))
            print(io, "Multifrontal.")
        end

        print(io, $("$Fac{"), repr(UPLO), ", ", T, ", ", I, "}")
    end
end

for Fac in (:DChordalCholesky, :DChordalLDLt)
    @eval function Base.show(io::IO, ::Type{$Fac{UPLO, T, I}}) where {UPLO, T, I}
        print(io, $("$Fac{"), repr(UPLO), ", ", T, ", ", I, "}")
    end
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

function Base.similar(F::ChordalFactorization{DIAG, UPLO}, ::Type{T}=eltype(F)) where {DIAG, UPLO, T}
    if DIAG === :N
        d = F.d
    else
        d = similar(F.d, T)
    end

    return ChordalFactorization{DIAG, UPLO}(F.S, d, similar(F.Dval, T), similar(F.Lval, T), F.perm, F.invp, similar(F.info))
end

# ===== flatindices =====

function flatindices(F::ChordalFactorization{DIAG, UPLO}, A::SparseMatrixCSC; check::Bool=true) where {DIAG, UPLO}
    return flatindices(F.invp, F.S, A, Val(UPLO); check)
end

function flatindices(F::ChordalFactorization{DIAG, UPLO}, A::HermOrSym) where {DIAG, UPLO}
    return flatindices(F.invp, F.S, A, Val(UPLO))
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

function ndz(F::ChordalFactorization)
    return ndz(ChordalTriangular(F))
end

function nlz(F::ChordalFactorization)
    return nlz(ChordalTriangular(F))
end

function nfr(F::ChordalFactorization)
    return nfr(F.S)
end

function fronts(F::ChordalFactorization)
    return fronts(F.S)
end

function diagblock(F::ChordalFactorization, j::Integer)
    return diagblock(ChordalTriangular(F), j)
end

function offdblock(F::ChordalFactorization, j::Integer)
    return offdblock(ChordalTriangular(F), j)
end

function symbolic(F::ChordalFactorization)
    return symbolic(triangular(F))
end

