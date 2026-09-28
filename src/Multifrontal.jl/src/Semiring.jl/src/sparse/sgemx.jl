function sgemx!(s::AbstractSemiring, tA::Val, tB::Val, C::AbstractMatrix, A::SparseMatrixCSC{<:Any, I}, B::AbstractMatrix; nt::Integer = nthreads()) where {I}
    jstrt = one(I)
    jstop = convert(I, size(A, 2))
    return sgemx_sparse!(s, tA, tB, C, getcolptr(A), rowvals(A), nonzeros(A), jstrt, jstop, B, nt)
end

function sgemx!(s::AbstractSemiring, tA::N_OR_R, tB::Val, c::AbstractVector, A::SparseMatrixCSC{<:Any, I}, b::AbstractVector; nt::Integer = nthreads()) where {I}
    jstrt = one(I)
    jstop = convert(I, size(A, 2))
    return sgemx_sparse!(s, tA, tB, c, getcolptr(A), rowvals(A), nonzeros(A), jstrt, jstop, b, nt)
end

function sgemx!(s::AbstractSemiring, tA::T_OR_C, tB::Val, c::AbstractVector, A::SparseMatrixCSC{<:Any, I}, b::AbstractVector; nt::Integer = nthreads()) where {I}
    jstrt = one(I)
    jstop = convert(I, size(A, 2))
    return sgemx_sparse!(s, tA, tB, c, getcolptr(A), rowvals(A), nonzeros(A), jstrt, jstop, b, nt)
end

function sgemx!(s::AbstractSemiring, tA::Val, tB::Val, C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::SparseMatrixCSC{T, I}; nt::Integer = nthreads()) where {T, I}
    jstrt = one(I)
    jstop = convert(I, size(B, 2))
    return sgemx_sparse!(s, tA, tB, C, A, getcolptr(B), rowvals(B), nonzeros(B), jstrt, jstop, nt)
end

function sgemx!(s::AbstractSemiring, tA::N_OR_R, tB::N_OR_R, c::AbstractVector{T}, a::AbstractVector{T}, B::SparseMatrixCSC{T, I}; nt::Integer = nthreads()) where {T, I}
    jstrt = one(I)
    jstop = convert(I, size(B, 2))
    return sgemx_sparse!(s, tA, tB, c, a, getcolptr(B), rowvals(B), nonzeros(B), jstrt, jstop, nt)
end

function sgemx!(s::AbstractSemiring, tA::N_OR_R, tB::T_OR_C, c::AbstractVector{T}, a::AbstractVector{T}, B::SparseMatrixCSC{T, I}; nt::Integer = nthreads()) where {T, I}
    jstrt = one(I)
    jstop = convert(I, size(B, 2))
    return sgemx_sparse!(s, tA, tB, c, a, getcolptr(B), rowvals(B), nonzeros(B), jstrt, jstop, nt)
end

function sgemx_sparse!(s::AbstractSemiring, tA::Val, tB::Val, C::AbstractVecOrMat, A::AbstractVecOrMat, Bptr::AbstractVector{I}, Btgt::AbstractVector{I}, Bval::AbstractVector, jstrt::I, jstop::I, nt::Integer) where {I}
    m = size(C, 1)

    if C isa AbstractVector || nt <= 1 || m < 2nt
        sgemx_sparse_st!(s, tA, tB, C, A, Bptr, Btgt, Bval, jstrt, jstop)
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

            sgemx_sparse_st!(s, tA, tB, Ct, At, Bptr, Btgt, Bval, jstrt, jstop)
        end
    end

    return C
end

function sgemx_sparse!(s::AbstractSemiring, tA::Val, tB::Val, C::AbstractVecOrMat, Aptr::AbstractVector{I}, Atgt::AbstractVector{I}, Aval::AbstractVector, jstrt::I, jstop::I, B::AbstractVecOrMat, nt::Integer) where {I}
    m = size(C, 2)

    if C isa AbstractVector || nt <= 1 || m < 2nt
        sgemx_sparse_st!(s, tA, tB, C, Aptr, Atgt, Aval, jstrt, jstop, B)
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

            sgemx_sparse_st!(s, tA, tB, Ct, Aptr, Atgt, Aval, jstrt, jstop, Bt)
        end
    end

    return C
end

function sgemx_sparse_st!(s::AbstractSemiring, tA::N_OR_R, tB::N_OR_R, C::AbstractMatrix, A::AbstractMatrix, Bptr::AbstractVector{I}, Btgt::AbstractVector{I}, Bval::AbstractVector, jstrt::I, jstop::I) where {I}
    @inbounds for j in jstrt:jstop
        pstrt = Bptr[j]
        pstop = Bptr[j + one(I)] - one(I)

        for p in pstrt:pstop
            k = Btgt[p]
            saxpy!(s, tA, tB, Val(:R), Bval[p], view(A, :, k), view(C, :, j))
        end
    end

    return C
end

function sgemx_sparse_st!(s::AbstractSemiring, tA::T_OR_C, tB::N_OR_R, C::AbstractMatrix, A::AbstractMatrix, Bptr::AbstractVector{I}, Btgt::AbstractVector{I}, Bval::AbstractVector, jstrt::I, jstop::I) where {I}
    @inbounds for j in jstrt:jstop
        pstrt = Bptr[j]
        pstop = Bptr[j + one(I)] - one(I)

        for p in pstrt:pstop
            k = Btgt[p]
            v = Bval[p]

            for i in axes(C, 1)
                C[i, j] = smuladd(s, A[k, i], v, C[i, j], tA, tB)
            end
        end
    end

    return C
end

function sgemx_sparse_st!(s::AbstractSemiring, tA::Val, tB::N_OR_R, c::AbstractVector, a::AbstractVector, Bptr::AbstractVector{I}, Btgt::AbstractVector{I}, Bval::AbstractVector, jstrt::I, jstop::I) where {I}
    @inbounds for j in jstrt:jstop
        pstrt = Bptr[j]
        pstop = Bptr[j + one(I)] - one(I)

        for p in pstrt:pstop
            k = Btgt[p]
            c[j] = smuladd(s, a[k], Bval[p], c[j], tA, tB)
        end
    end

    return c
end

function sgemx_sparse_st!(s::AbstractSemiring, tA::N_OR_R, tB::T_OR_C, C::AbstractMatrix, A::AbstractMatrix, Bptr::AbstractVector{I}, Btgt::AbstractVector{I}, Bval::AbstractVector, kstrt::I, kstop::I) where {I}
    @inbounds for k in kstrt:kstop
        pstrt = Bptr[k]
        pstop = Bptr[k + one(I)] - one(I)

        for p in pstrt:pstop
            j = Btgt[p]
            saxpy!(s, tA, tB, Val(:R), Bval[p], view(A, :, k), view(C, :, j))
        end
    end

    return C
end

function sgemx_sparse_st!(s::AbstractSemiring, tA::T_OR_C, tB::T_OR_C, C::AbstractMatrix, A::AbstractMatrix, Bptr::AbstractVector{I}, Btgt::AbstractVector{I}, Bval::AbstractVector, kstrt::I, kstop::I) where {I}
    @inbounds for k in kstrt:kstop
        pstrt = Bptr[k]
        pstop = Bptr[k + one(I)] - one(I)

        for p in pstrt:pstop
            j = Btgt[p]
            v = Bval[p]

            for i in axes(C, 1)
                C[i, j] = smuladd(s, A[k, i], v, C[i, j], tA, tB)
            end
        end
    end

    return C
end

function sgemx_sparse_st!(s::AbstractSemiring, tA::Val, tB::T_OR_C, c::AbstractVector, a::AbstractVector, Bptr::AbstractVector{I}, Btgt::AbstractVector{I}, Bval::AbstractVector, kstrt::I, kstop::I) where {I}
    @inbounds for k in kstrt:kstop
        pstrt = Bptr[k]
        pstop = Bptr[k + one(I)] - one(I)

        for p in pstrt:pstop
            j = Btgt[p]
            c[j] = smuladd(s, a[k], Bval[p], c[j], tA, tB)
        end
    end

    return c
end

function sgemx_sparse_st!(s::AbstractSemiring, tA::N_OR_R, tB::Val, C::AbstractVecOrMat, Aptr::AbstractVector{I}, Atgt::AbstractVector{I}, Aval::AbstractVector, jstrt::I, jstop::I, B::AbstractVecOrMat) where {I}
    @inbounds for k in axes(C, 2)
        for j in jstrt:jstop
            if C isa AbstractVector || tB isa N_OR_R
                u = B[j, k]
            else
                u = B[k, j]
            end

            pstrt = Aptr[j]
            pstop = Aptr[j + one(I)] - one(I)

            for p in pstrt:pstop
                i = Atgt[p]
                C[i, k] = smuladd(s, Aval[p], u, C[i, k], tA, tB)
            end
        end
    end

    return C
end

function sgemx_sparse_st!(s::AbstractSemiring, tA::T_OR_C, tB::N_OR_R, C::AbstractVecOrMat, Aptr::AbstractVector{I}, Atgt::AbstractVector{I}, Aval::AbstractVector, jstrt::I, jstop::I, B::AbstractVecOrMat) where {I}
    @inbounds for k in axes(C, 2)
        for j in jstrt:jstop
            u = C[j, k]
            pstrt = Aptr[j]
            pstop = Aptr[j + one(I)] - one(I)

            @simd for p in pstrt:pstop
                u = smuladd(s, Aval[p], B[Atgt[p], k], u, tA, tB)
            end

            C[j, k] = u
        end
    end

    return C
end

function sgemx_sparse_st!(s::AbstractSemiring, tA::T_OR_C, tB::T_OR_C, C::AbstractVecOrMat, Aptr::AbstractVector{I}, Atgt::AbstractVector{I}, Aval::AbstractVector, jstrt::I, jstop::I, B::AbstractVecOrMat) where {I}
    @inbounds for j in jstrt:jstop
        pstrt = Aptr[j]
        pstop = Aptr[j + one(I)] - one(I)

        for p in pstrt:pstop
            i = Atgt[p]
            v = Aval[p]

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
