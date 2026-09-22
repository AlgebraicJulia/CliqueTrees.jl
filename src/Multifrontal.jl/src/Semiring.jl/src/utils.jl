function szerorec!(s::AbstractSemiring, A::AbstractVecOrMat{T}, trans::Val) where {T}
    fill!(A, szero(s, T, trans))
    return A
end

function sscatteradd!(s::AbstractSemiring, C::AbstractMatrix, M::AbstractMatrix, ind::AbstractVector, ::Val{:L})
    @inbounds for j in axes(M, 2)
        for i in axes(M, 1)
            C[ind[i], j] = splus(s, C[ind[i], j], M[i, j], Val(:N))
        end
    end

    return C
end

function sscatteradd!(s::AbstractSemiring, C::AbstractMatrix, M::AbstractMatrix, ind::AbstractVector, ::Val{:R})
    @inbounds for j in axes(M, 2)
        indj = ind[j]

        for i in axes(M, 1)
            C[i, indj] = splus(s, C[i, indj], M[i, j], Val(:N))
        end
    end

    return C
end

function sscatteradd!(s::AbstractSemiring, C::AbstractVector, M::AbstractVector, ind::AbstractVector)
    @inbounds for i in axes(M, 1)
        C[ind[i]] = splus(s, C[ind[i]], M[i], Val(:N))
    end

    return C
end

function sscatteradd!(s::AbstractSemiring, trans::Val, C::AbstractMatrix, M::AbstractMatrix, ind::AbstractVector, ::Val{:L})
    @inbounds for j in axes(M, 2)
        for i in axes(M, 1)
            C[ind[i], j] = splus(s, C[ind[i], j], M[i, j], trans)
        end
    end

    return C
end

function sscatteradd!(s::AbstractSemiring, trans::Val, C::AbstractMatrix, M::AbstractMatrix, ind::AbstractVector, ::Val{:R})
    @inbounds for j in axes(M, 2)
        indj = ind[j]

        for i in axes(M, 1)
            C[i, indj] = splus(s, C[i, indj], M[i, j], trans)
        end
    end

    return C
end

function sscatteradd!(s::AbstractSemiring, trans::Val, C::AbstractVector, M::AbstractVector, ind::AbstractVector)
    @inbounds for i in axes(M, 1)
        C[ind[i]] = splus(s, C[ind[i]], M[i], trans)
    end

    return C
end

function permuterows!(A::AbstractVecOrMat, work::AbstractVector, perm::AbstractVector)
    m = size(A, 1)
    n = size(A, 2)
    k = min(8, n)

    B = reshape(view(work, oneto(m * k)), m, k)

    @inbounds for jstrt in 1:k:n
        jsize = min(jstrt + k - 1, n) - jstrt + 1

        for j in 1:jsize
            for i in 1:m
                B[i, j] = A[i, jstrt + j - 1]
            end
        end

        for j in 1:jsize
            for i in 1:m
                A[perm[i], jstrt + j - 1] = B[i, j]
            end
        end
    end

    return A
end

function permutecols!(A::AbstractVecOrMat, work::AbstractVector, perm::AbstractVector)
    m = size(A, 1)
    n = size(A, 2)
    k = min(8, m)

    B = reshape(view(work, oneto(k * n)), k, n)

    @inbounds for istrt in 1:k:m
        isize = min(istrt + k - 1, m) - istrt + 1

        for j in 1:n
            for i in 1:isize
                B[i, j] = A[istrt + i - 1, j]
            end
        end

        for j in 1:n
            for i in 1:isize
                A[istrt + i - 1, perm[j]] = B[i, j]
            end
        end
    end

    return A
end
