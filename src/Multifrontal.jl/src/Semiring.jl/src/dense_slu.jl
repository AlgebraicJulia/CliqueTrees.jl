struct DenseSLU{Sem <: AbstractSemiring, T, Mat <: AbstractMatrix{T}} <: AbstractSLU{T}
    s::Sem
    A::Mat
end

const FDenseSLU{Sem, T} = DenseSLU{Sem, T, FMatrix{T}}
const DDenseSLU{Sem, T} = DenseSLU{Sem, T, Matrix{T}}

function DenseSLU{Sem}(F::DenseSLU) where {Sem}
    A = F.A
    return DenseSLU(Sem(), A)
end

function Base.size(F::DenseSLU)
    return size(F.A)
end

function Base.size(F::DenseSLU, d::Integer)
    return size(F.A, d)
end

function Base.copyto!(F::DenseSLU, A::AbstractMatrix)
    copyto!(F.A, A)
    return F
end

# ===== sgetrf! =====

function sgetrf!(F::DenseSLU; nt::Integer = nthreads())
    sgetrf!(F.s, F.A; nt)
    return F
end

# ===== sgetrs! =====

function sgetrs!(F::DenseSLU, side::Val, trans::Val, B::AbstractVecOrMat; nt::Integer = nthreads())
    return sgetrs!(F.s, side, trans, F.A, B; nt)
end

# ===== sgetri! =====

function sgetri!(F::DenseSLU, C::AbstractMatrix; nt::Integer = nthreads())
    return sgetri!(F.s, C, F.A; nt)
end

# ===== stpqxt! =====
#
# Given the factorization of A*, compute the factorization of
#
#     (A ⊕ X Y)*
#
# where X is n × m (or a vector: m = 1) and Y is m × n (or a vector: the row y).
#
function stpqxt!(F::DenseSLU{Sem, T}, X::AbstractVecOrMat, Y::AbstractVecOrMat; nt::Integer = nthreads()) where {Sem, T}
    @assert size(X, 1) == size(F, 1)

    if Y isa AbstractVector
        @assert size(X, 2) == 1
        @assert length(Y) == size(F, 1)
    else
        @assert size(Y, 1) == size(X, 2)
        @assert size(Y, 2) == size(F, 1)
    end

    n = size(F, 1)
    m = size(X, 2)

    X₁ = FMatrix{T}(undef, n, m)
    Y₁ = FMatrix{T}(undef, m, n)
    copyto!(X₁, X)
    copyto!(Y₁, Y)

    stpqxt!(F.s, F.A, X₁, Y₁; nt)
    return F
end
