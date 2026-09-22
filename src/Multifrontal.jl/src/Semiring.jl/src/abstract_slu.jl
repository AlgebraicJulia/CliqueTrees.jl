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

function mlu(s::AbstractSemiring, A::AbstractMatrix)
    F = DenseSLU(s, A)
    return lu!(F)
end

# ===== mstar =====

function mstar(s::AbstractSemiring, A::AbstractMatrix)
    return Matrix(mlu(s, A))
end

# ===== lu! =====

function LinearAlgebra.lu!(F::AbstractSLU)
    return sgetrf!(F)
end

# ===== lmul! / rmul! =====

function LinearAlgebra.lmul!(F::MaybeTransSLU, B::AbstractVecOrMat)
    P, trans = unwrap(F)
    return sgetrs!(P, Val(:L), trans, B)
end

function LinearAlgebra.rmul!(B::AbstractMatrix, F::MaybeTransSLU)
    P, trans = unwrap(F)
    return sgetrs!(P, Val(:R), trans, B)
end

# ===== ldiv! / rdiv! =====

function LinearAlgebra.ldiv!(F::AbstractSLU, B::AbstractVecOrMat)
    return sgetrs!(F, Val(:L), Val(:C), B)
end

function LinearAlgebra.ldiv!(F::TransSLU, B::AbstractVecOrMat)
    return sgetrs!(parent(F), Val(:L), Val(:R), B)
end

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
