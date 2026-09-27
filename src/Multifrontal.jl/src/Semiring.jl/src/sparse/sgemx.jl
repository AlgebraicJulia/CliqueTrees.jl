function sgemx!(s::AbstractSemiring, tA::Val, tB::Val, C::AbstractMatrix, A::SparseMatrixCSC, B::AbstractMatrix; nt::Integer = nthreads())
    return sgemx_sparse!(s, tA, tB, C, A, B, nt)
end

function sgemx!(s::AbstractSemiring, tA::N_OR_R, tB::Val, c::AbstractVector, A::SparseMatrixCSC, b::AbstractVector; nt::Integer = nthreads())
    return sgemx_sparse!(s, tA, tB, c, A, b, nt)
end

function sgemx!(s::AbstractSemiring, tA::T_OR_C, tB::Val, c::AbstractVector, A::SparseMatrixCSC, b::AbstractVector; nt::Integer = nthreads())
    return sgemx_sparse!(s, tA, tB, c, A, b, nt)
end

function sgemx!(s::AbstractSemiring, tA::Val, tB::Val, C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::SparseMatrixCSC{T}; nt::Integer = nthreads()) where {T}
    return sgemx_sparse!(s, tA, tB, C, A, B, nt)
end

function sgemx!(s::AbstractSemiring, tA::N_OR_R, tB::N_OR_R, c::AbstractVector{T}, a::AbstractVector{T}, B::SparseMatrixCSC{T}; nt::Integer = nthreads()) where {T}
    return sgemx_sparse!(s, tA, tB, c, a, B, nt)
end

function sgemx!(s::AbstractSemiring, tA::N_OR_R, tB::T_OR_C, c::AbstractVector{T}, a::AbstractVector{T}, B::SparseMatrixCSC{T}; nt::Integer = nthreads()) where {T}
    return sgemx_sparse!(s, tA, tB, c, a, B, nt)
end

function sgemx_sparse!(s::AbstractSemiring, tA::Val, tB::Val, C::AbstractVecOrMat, A::AbstractVecOrMat, B::SparseMatrixCSC, nt::Integer)
    m = size(C, 1)

    if C isa AbstractVector || nt <= 1 || m < 2nt
        sgemx_sparse_st!(s, tA, tB, C, A, B)
    else
        tsize = fld(m, nt)

        @threads for t in 1:nt
            tstrt = (t - 1) * tsize + 1

            if t < nt
                tstop = t * tsize
            else
                tstop = m
            end

            Ct = view(C, tstrt:tstop, :)

            if tA isa N_OR_R
                At = view(A, tstrt:tstop, :)
            else
                At = view(A, :, tstrt:tstop)
            end

            sgemx_sparse_st!(s, tA, tB, Ct, At, B)
        end
    end

    return C
end

function sgemx_sparse!(s::AbstractSemiring, tA::Val, tB::Val, C::AbstractVecOrMat, A::SparseMatrixCSC, B::AbstractVecOrMat, nt::Integer)
    m = size(C, 2)

    if C isa AbstractVector || nt <= 1 || m < 2nt
        sgemx_sparse_st!(s, tA, tB, C, A, B)
    else
        tsize = fld(m, nt)

        @threads for t in 1:nt
            tstrt = (t - 1) * tsize + 1

            if t < nt
                tstop = t * tsize
            else
                tstop = m
            end

            Ct = view(C, :, tstrt:tstop)

            if tB isa N_OR_R
                Bt = view(B, :, tstrt:tstop)
            else
                Bt = view(B, tstrt:tstop, :)
            end

            sgemx_sparse_st!(s, tA, tB, Ct, A, Bt)
        end
    end

    return C
end

function sgemx_sparse_st!(s::AbstractSemiring, tA::N_OR_R, tB::N_OR_R, C::AbstractMatrix, A::AbstractMatrix, B::SparseMatrixCSC)
    @inbounds for j in axes(B, 2)
        for p in nzrange(B, j)
            k = rowvals(B)[p]
            v = nonzeros(B)[p]
            saxpy!(s, tA, tB, Val(:R), v, view(A, :, k), view(C, :, j))
        end
    end

    return C
end

function sgemx_sparse_st!(s::AbstractSemiring, tA::T_OR_C, tB::N_OR_R, C::AbstractMatrix, A::AbstractMatrix, B::SparseMatrixCSC)
    @inbounds for j in axes(B, 2)
        for p in nzrange(B, j)
            k = rowvals(B)[p]
            v = nonzeros(B)[p]

            for i in axes(C, 1)
                C[i, j] = smuladd(s, A[k, i], v, C[i, j], tA, tB)
            end
        end
    end

    return C
end

function sgemx_sparse_st!(s::AbstractSemiring, tA::Val, tB::N_OR_R, c::AbstractVector, a::AbstractVector, B::SparseMatrixCSC)
    @inbounds for j in axes(B, 2)
        for p in nzrange(B, j)
            k = rowvals(B)[p]
            v = nonzeros(B)[p]
            c[j] = smuladd(s, a[k], v, c[j], tA, tB)
        end
    end

    return c
end

function sgemx_sparse_st!(s::AbstractSemiring, tA::N_OR_R, tB::T_OR_C, C::AbstractMatrix, A::AbstractMatrix, B::SparseMatrixCSC)
    @inbounds for k in axes(B, 2)
        for p in nzrange(B, k)
            j = rowvals(B)[p]
            v = nonzeros(B)[p]
            saxpy!(s, tA, tB, Val(:R), v, view(A, :, k), view(C, :, j))
        end
    end

    return C
end

function sgemx_sparse_st!(s::AbstractSemiring, tA::T_OR_C, tB::T_OR_C, C::AbstractMatrix, A::AbstractMatrix, B::SparseMatrixCSC)
    @inbounds for k in axes(B, 2)
        for p in nzrange(B, k)
            j = rowvals(B)[p]
            v = nonzeros(B)[p]

            for i in axes(C, 1)
                C[i, j] = smuladd(s, A[k, i], v, C[i, j], tA, tB)
            end
        end
    end

    return C
end

function sgemx_sparse_st!(s::AbstractSemiring, tA::Val, tB::T_OR_C, c::AbstractVector, a::AbstractVector, B::SparseMatrixCSC)
    @inbounds for k in axes(B, 2)
        for p in nzrange(B, k)
            j = rowvals(B)[p]
            v = nonzeros(B)[p]
            c[j] = smuladd(s, a[k], v, c[j], tA, tB)
        end
    end

    return c
end

function sgemx_sparse_st!(s::AbstractSemiring, tA::N_OR_R, tB::Val, C::AbstractVecOrMat, A::SparseMatrixCSC, B::AbstractVecOrMat)
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

function sgemx_sparse_st!(s::AbstractSemiring, tA::T_OR_C, tB::N_OR_R, C::AbstractVecOrMat, A::SparseMatrixCSC, B::AbstractVecOrMat)
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

function sgemx_sparse_st!(s::AbstractSemiring, tA::T_OR_C, tB::T_OR_C, C::AbstractVecOrMat, A::SparseMatrixCSC, B::AbstractVecOrMat)
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
