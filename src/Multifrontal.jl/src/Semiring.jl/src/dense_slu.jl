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
