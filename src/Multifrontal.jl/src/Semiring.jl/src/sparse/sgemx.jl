function sgemx!(s::AbstractSemiring, tA::Val, tB::Val, C::AbstractMatrix, A::SparseMatrixCSC, B::AbstractMatrix)
    return sgemx_sparse!(s, tA, tB, C, A, B)
end

function sgemx!(s::AbstractSemiring, tA::N_OR_R, tB::Val, c::AbstractVector, A::SparseMatrixCSC, b::AbstractVector)
    return sgemx_sparse!(s, tA, tB, c, A, b)
end

function sgemx!(s::AbstractSemiring, tA::T_OR_C, tB::Val, c::AbstractVector, A::SparseMatrixCSC, b::AbstractVector)
    return sgemx_sparse!(s, tA, tB, c, A, b)
end

function sgemx_sparse!(s::AbstractSemiring, tA::N_OR_R, tB::Val, C::AbstractVecOrMat, A::SparseMatrixCSC, B::AbstractVecOrMat)
    @inbounds for k in axes(C, 2)
        for j in axes(A, 2)
            if C isa AbstractVector || tB isa N_OR_R
                u = B[j, k]
            else
                u = B[k, j]
            end

            for p in nzrange(A, j)
                i = rowvals(A)[p]
                C[i, k] = smuladd(s, nonzeros(A)[p], u, C[i, k], tA, tB)
            end
        end
    end

    return C
end

function sgemx_sparse!(s::AbstractSemiring, tA::T_OR_C, tB::N_OR_R, C::AbstractVecOrMat, A::SparseMatrixCSC, B::AbstractVecOrMat)
    @inbounds for k in axes(C, 2)
        for j in axes(A, 2)
            u = C[j, k]

            @simd for p in nzrange(A, j)
                i = rowvals(A)[p]
                u = smuladd(s, nonzeros(A)[p], B[i, k], u, tA, tB)
            end

            C[j, k] = u
        end
    end

    return C
end

function sgemx_sparse!(s::AbstractSemiring, tA::T_OR_C, tB::T_OR_C, C::AbstractVecOrMat, A::SparseMatrixCSC, B::AbstractVecOrMat)
    @inbounds for j in axes(A, 2)
        for p in nzrange(A, j)
            i = rowvals(A)[p]
            v = nonzeros(A)[p]

            for k in axes(C, 2)
                if C isa AbstractVector
                    u = B[i]
                else
                    u = B[k, i]
                end

                C[j, k] = smuladd(s, v, u, C[j, k], tA, tB)
            end
        end
    end

    return C
end
