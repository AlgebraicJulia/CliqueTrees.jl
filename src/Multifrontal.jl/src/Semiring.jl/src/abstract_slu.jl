abstract type AbstractSLU{T} <: Factorization{T} end

const TransSLU{T} = TransposeFactorization{T, <:AbstractSLU{T}}

const MaybeTransSLU{T} = Union{
    AbstractSLU{T},
     TransSLU{T},
}

function Base.Matrix(F::AbstractSLU{T}) where {T}
    n = size(F, 1)
    C = Matrix{T}(undef, n, n)
    return sgetri!(F, C)
end

function Base.parent(F::AbstractSLU)
    return F
end

function Base.adjoint(F::AbstractSLU)
    return TransposeFactorization(F)
end

function Base.transpose(F::AbstractSLU)
    return TransposeFactorization(F)
end

# ===== mlu =====

function mlu(s::AbstractSemiring, A::SparseMatrixCSC)
    F = ChordalSLU(s, A)
    copyto!(F, A)
    return lu!(F)
end

#
# Factorize the closure of a matrix A:
#
#   A* = U* L*
#
function mlu(s::AbstractSemiring, A::AbstractMatrix)
    F = DenseSLU(s, A)
    return lu!(F)
end

# ===== mstar =====

#
# Compute the closure off a matrix A:
#
#   A*
#
function mstar(s::AbstractSemiring, A::AbstractMatrix)
    return Matrix(mlu(s, A))
end

# ===== pstar =====

#
# Project the closure of a matrix A
#
#   A*
#
# onto the sparsity pattern of A.
#
function pstar(s::AbstractSemiring, A::SparseMatrixCSC)
    F = mlu(s, A); sgetrp!(F)
    L = sparse(F.L); tril!(L, -1)
    U = sparse(F.U)
    return permute(L + U, F.rinvp, F.cinvp)
end

# ===== lu! =====

#
# Factorize the closure of a matrix A:
#
#   A* = U* L*
#
function LinearAlgebra.lu!(F::AbstractSLU)
    return sgetrf!(F)
end

# ===== lmul! / rmul! =====

#
# Find the least solution to the
# fixpoint equation
#
#   AX + B = X.
#
function LinearAlgebra.lmul!(F::MaybeTransSLU, B::AbstractVecOrMat)
    P, trans = unwrap(F)
    return sgetrs!(P, Val(:L), trans, B)
end

#
# Find the least solution to the
# fixpoint equation
#
#   XA + B = X.
#
function LinearAlgebra.rmul!(B::AbstractMatrix, F::MaybeTransSLU)
    P, trans = unwrap(F)
    return sgetrs!(P, Val(:R), trans, B)
end

# ===== ldiv! / rdiv! =====

#
# Find the greatest solution to the
# fixpoint equation
#
#   A \ X ∧ B = X.
#
function LinearAlgebra.ldiv!(F::AbstractSLU, B::AbstractVecOrMat)
    return sgetrs!(F, Val(:L), Val(:C), B)
end

function LinearAlgebra.ldiv!(F::TransSLU, B::AbstractVecOrMat)
    return sgetrs!(parent(F), Val(:L), Val(:R), B)
end

#
# Find the greatest solution to the
# fixpoint equation
#
#   X / A ∧ B = X.
#
function LinearAlgebra.rdiv!(B::AbstractMatrix, F::AbstractSLU)
    return sgetrs!(F, Val(:R), Val(:C), B)
end

function LinearAlgebra.rdiv!(B::AbstractMatrix, F::TransSLU)
    return sgetrs!(parent(F), Val(:R), Val(:R), B)
end

# ===== * =====

function Base.:*(F::MaybeTransSLU, B::AbstractVecOrMat)
    return lmul!(F, copy(B))
end

function Base.:*(B::AbstractMatrix, F::MaybeTransSLU)
    return rmul!(copy(B), F)
end

# ===== \ / / =====

function Base.:\(F::MaybeTransSLU, B::AbstractVecOrMat)
    return ldiv!(F, copy(B))
end

function Base.:/(B::AbstractMatrix, F::MaybeTransSLU)
    return rdiv!(copy(B), F)
end
