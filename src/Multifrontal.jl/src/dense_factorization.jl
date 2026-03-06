struct DenseFactorization{DIAG, UPLO, T, Mat <: AbstractMatrix{T}, Dia <: AbstractVector{T}, Prm <: AbstractVector{Int}} <: AbstractFactorization{DIAG, UPLO, T, Int, Prm}
    M::Mat
    d::Dia
    perm::Prm
    invp::Prm
    info::FScalar{Int}
end

const IDenseFactorization{DIAG, UPLO, T, Mat, Dia} = DenseFactorization{DIAG, UPLO, T, Mat, Dia, OneTo{Int}}
const DenseCholeskyPivoted = DenseFactorization{:N}
const DenseLDLtPivoted = DenseFactorization{:U}
const DenseCholesky = IDenseFactorization{:N}
const DenseLDLt = IDenseFactorization{:U}

# ===== IFactorization =====

function DenseFactorization{DIAG, UPLO}(
        M::Mat,
        d::Dia,
        perm::Prm,
        invp::Prm,
        info::FScalar{Int},
    ) where {DIAG, UPLO, T, Mat <: AbstractMatrix{T}, Dia <: AbstractVector{T}, Prm <: AbstractVector{Int}}
    return DenseFactorization{DIAG, UPLO, T, Mat, Dia, Prm}(M, d, perm, invp, info)
end

function DenseFactorization{DIAG}(A::AbstractMatrix) where {DIAG}
    return DenseFactorization{DIAG, DEFAULT_UPLO}(A)
end

function IDenseFactorization{DIAG}(A::AbstractMatrix) where {DIAG}
    return IDenseFactorization{DIAG, DEFAULT_UPLO}(A)
end

function DenseFactorization{DIAG, UPLO}(A::HermOrSym) where {DIAG, UPLO}
    @assert A.uplo === char(Val(UPLO))
    return DenseFactorization{DIAG, UPLO}(parent(A))
end

function IDenseFactorization{DIAG, UPLO}(A::HermOrSym) where {DIAG, UPLO}
    @assert A.uplo === char(Val(UPLO))
    return IDenseFactorization{DIAG, UPLO}(parent(A))
end

function DenseFactorization{DIAG, UPLO}(A::AbstractMatrix{T}) where {DIAG, UPLO, T}
    n = size(A, 1)

    if DIAG === :N
        d = Ones{T}(n)
    else
        d = FVector{T}(undef, size(A, 1))
    end

    perm = FVector{Int}(undef, n)
    invp = FVector{Int}(undef, n)
    info = FScalar{Int}(undef); info[] = 0
    return DenseFactorization{DIAG, UPLO}(A, d, perm, invp, info)
end

function IDenseFactorization{DIAG, UPLO}(A::AbstractMatrix{T}) where {DIAG, UPLO, T}
    n = size(A, 1)

    if DIAG === :N
        d = Ones{T}(n)
    else
        d = FVector{T}(undef, size(A, 1))
    end

    perm = axes(A, 1)
    invp = axes(A, 1)
    info = FScalar{Int}(undef); info[] = 0
    return DenseFactorization{DIAG, UPLO}(A, d, perm, invp, info)
end

function IFactorization(F::DenseFactorization{DIAG, UPLO}) where {DIAG, UPLO}
    perm = invp = OneTo{Int}(size(F, 1))
    return DenseFactorization{DIAG, UPLO}(F.M, F.d, perm, invp, F.info)
end

# ===== Base methods =====

function Base.propertynames(::DenseFactorization)
    return (:L, :U, :D, :P, :d, :perm, :invp, :info)
end

function Base.copy(F::DenseFactorization{DIAG, UPLO}) where {DIAG, UPLO}
    return DenseFactorization{DIAG, UPLO}(copy(F.M), copy(F.d), copy(F.perm), copy(F.invp), F.info)
end

function triangular(F::DenseFactorization)
    return tri(F.uplo, F.diag, getfield(F, :M))
end

# ===== show =====

function Base.show(io::IO, F::DenseFactorization{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    n = size(F, 1)
    print(io, "$n×$n DenseFactorization{", repr(DIAG), ", ", repr(UPLO), ", ", T, "}")
end

function Base.show(io::IO, ::MIME"text/plain", F::DenseFactorization{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    n = size(F, 1)
    println(io, "$n×$n DenseFactorization{", repr(DIAG), ", ", repr(UPLO), ", ", T, "}:")
    print_matrix(io, triangular(F))

    if DIAG === :U
        println(io)
        println(io)
        print_matrix(io, F.D)
    end
end

