struct DenseFactorization{
        DIAG,
        UPLO,
        T,
        Mat <: AbstractMatrix{T},
        Dia <: AbstractVector{T},
        Prm <: AbstractVector{Int},
        Ivp <: AbstractVector{Int},
        Ifo <: AbstractScalar{Int},
    } <: AbstractFactorization{DIAG, UPLO, T, Int, Prm, Ivp}
    M::Mat
    d::Dia
    perm::Prm
    invp::Ivp
    info::Ifo
end

const NaturalDenseFactorization{DIAG, UPLO, T, Mat, Dia, Ifo} = DenseFactorization{DIAG, UPLO, T, Mat, Dia, OneTo{Int}, OneTo{Int}, Ifo}
const DenseCholeskyPivoted = DenseFactorization{:N}
const DenseLDLtPivoted = DenseFactorization{:U}
const DenseCholesky = NaturalDenseFactorization{:N}
const DenseLDLt = NaturalDenseFactorization{:U}

const FDenseCholesky{UPLO, T} = DenseCholesky{
    UPLO,
    T,
    FMatrix{T},
    IOnes{T},
    FScalar{Int},
}

const FDenseLDLt{UPLO, T} = DenseLDLt{
    UPLO,
    T,
    FMatrix{T},
    FVector{T},
    FScalar{Int},
}

const FDenseCholeskyPivoted{UPLO, T} = DenseCholeskyPivoted{
    UPLO,
    T,
    FMatrix{T},
    IOnes{T},
    FVector{Int},
    FVector{Int},
    FScalar{Int},
}

const FDenseLDLtPivoted{UPLO, T} = DenseLDLtPivoted{
    UPLO,
    T,
    FMatrix{T},
    FVector{T},
    FVector{Int},
    FVector{Int},
    FScalar{Int},
}

const DDenseCholesky{UPLO, T} = DenseCholesky{
    UPLO,
    T,
    Matrix{T},
    IOnes{T},
    Scalar{Int},
}

const DDenseLDLt{UPLO, T} = DenseLDLt{
    UPLO,
    T,
    Matrix{T},
    Vector{T},
    Scalar{Int},
}

const DDenseCholeskyPivoted{UPLO, T} = DenseCholeskyPivoted{
    UPLO,
    T,
    Matrix{T},
    IOnes{T},
    Vector{Int},
    Vector{Int},
    Scalar{Int},
}

const DDenseLDLtPivoted{UPLO, T} = DenseLDLtPivoted{
    UPLO,
    T,
    Matrix{T},
    Vector{T},
    Vector{Int},
    Vector{Int},
    Scalar{Int},
}

# ===== NaturalFactorization =====

function DenseFactorization{DIAG, UPLO, T, Mat, Dia, Prm, Ivp, Ifo}(A::AbstractMatrix) where {
        DIAG,
        UPLO,
        T,
        Mat <: AbstractMatrix{T},
        Dia <: AbstractVector{T},
        Prm <: AbstractVector{Int},
        Ivp <: AbstractVector{Int},
        Ifo <: AbstractScalar{Int},
    }

    if A isa HermOrSym
        B = parent(A)
    else
        B = A
    end

    n = size(A, 1)
    d = allocate(Dia, n)
    perm = allocate(Prm, n)
    invp = allocate(Ivp, n)
    info = allocate(Ifo)

    if !(Prm <: AbstractRange)
        perm .= 1:n
    end

    if !(Ivp <: AbstractRange)
        invp .= 1:n
    end

    return DenseFactorization{DIAG, UPLO, T, Mat, Dia, Prm, Ivp, Ifo}(B, d, perm, invp, info)
end

function DenseFactorization{DIAG, UPLO}(
        M::Mat,
        d::Dia,
        perm::Prm,
        invp::Ivp,
        info::Ifo,
    ) where {
        DIAG,
        UPLO,
        T,
        Mat <: AbstractMatrix{T},
        Dia <: AbstractVector{T},
        Prm <: AbstractVector{Int},
        Ivp <: AbstractVector{Int},
        Ifo <: AbstractScalar{Int},
    }
    return DenseFactorization{DIAG, UPLO, T, Mat, Dia, Prm, Ivp, Ifo}(M, d, perm, invp, info)
end

function (::Type{Fac})(n::Integer) where {DIAG, UPLO, T, Fac <: Union{DenseFactorization{DIAG, UPLO, T}, NaturalDenseFactorization{DIAG, UPLO, T}}}
    M = FMatrix{T}(undef, n, n)
    return Fac(M)
end

function (::Type{Fac})(n::Integer) where {DIAG, UPLO, T, Mat <: AbstractMatrix{T}, Fac <: Union{DenseFactorization{DIAG, UPLO, T, Mat}, NaturalDenseFactorization{DIAG, UPLO, T, Mat}}}
    M = allocate(Mat, n, n)
    return Fac(M)
end

function DenseFactorization{DIAG}(A::AbstractMatrix) where {DIAG}
    return DenseFactorization{DIAG, DEFAULT_UPLO}(A)
end

function NaturalDenseFactorization{DIAG}(A::AbstractMatrix) where {DIAG}
    return NaturalDenseFactorization{DIAG, DEFAULT_UPLO}(A)
end

function DenseFactorization{DIAG, UPLO}(A::AbstractMatrix{T}) where {DIAG, UPLO, T}
    return DenseFactorization{DIAG, UPLO, T}(A)
end

function NaturalDenseFactorization{DIAG, UPLO}(A::AbstractMatrix{T}) where {DIAG, UPLO, T}
    return NaturalDenseFactorization{DIAG, UPLO, T}(A)
end

function DenseFactorization{DIAG, UPLO, T}(A::Mat) where {DIAG, UPLO, T, Mat <: AbstractMatrix{T}}
    if DIAG === :N
        Dia = IOnes{T}
    else
        Dia = FVector{T}
    end

    if A isa HermOrSym
        Pnt = typeof(parent(A))
    else
        Pnt = Mat
    end

    Prm = FVector{Int}
    Ivp = FVector{Int}
    Ifo = FScalar{Int}
    return DenseFactorization{DIAG, UPLO, T, Pnt, Dia, Prm, Ivp, Ifo}(A)
end 

function NaturalDenseFactorization{DIAG, UPLO, T}(A::Mat) where {DIAG, UPLO, T, Mat <: AbstractMatrix{T}}
    if DIAG === :N
        Dia = IOnes{T}
    else
        Dia = FVector{T}
    end 
 
    if A isa HermOrSym
        Pnt = typeof(parent(A))
    else
        Pnt = Mat
    end
   
    Ifo = FScalar{Int}
    return NaturalDenseFactorization{DIAG, UPLO, T, Pnt, Dia, Ifo}(A)
end 

function NaturalFactorization(F::DenseFactorization{DIAG, UPLO}) where {DIAG, UPLO}
    perm = invp = OneTo{Int}(size(F, 1))
    return DenseFactorization{DIAG, UPLO}(F.M, F.d, perm, invp, F.info)
end

# ===== Base methods =====

function Base.propertynames(::DenseFactorization)
    return (:L, :U, :D, :P, :d, :perm, :invp, :info)
end

function Base.similar(F::DenseFactorization{DIAG, UPLO}, ::Type{T}=eltype(F)) where {DIAG, UPLO, T}
    if DIAG === :N
        d = F.d
    else
        d = similar(F.d, T)
    end

    return DenseFactorization{DIAG, UPLO}(similar(F.M, T), d, F.perm, F.invp, similar(F.info))
end

function triangular(F::DenseFactorization, diag::Val=F.diag)
    return tri(F.uplo, diag, getfield(F, :M))
end

# ===== show =====

for Fac in (
    :FDenseCholesky,
    :FDenseLDLt,
    :FDenseCholeskyPivoted,
    :FDenseLDLtPivoted,
    :DDenseCholesky,
    :DDenseLDLt,
    :DDenseCholeskyPivoted,
    :DDenseLDLtPivoted,
)
    @eval function Base.show(io::IO, ::Type{$Fac{UPLO, T}}) where {UPLO, T}
        print(io, $("$Fac{"), repr(UPLO), ", ", T, "}")
    end
end

function Base.show(io::IO, F::Fac) where {Fac <: DenseFactorization}
    n = size(F, 1)
    print(io, "$n×$n $Fac")
end

function Base.show(io::IO, ::MIME"text/plain", F::Fac) where {DIAG, Fac <: DenseFactorization{DIAG}}
    n = size(F, 1)
    println(io, "$n×$n $Fac:")
    print_matrix(io, triangular(F))

    if DIAG === :U
        println(io)
        println(io)
        print_matrix(io, F.D)
    end
end

