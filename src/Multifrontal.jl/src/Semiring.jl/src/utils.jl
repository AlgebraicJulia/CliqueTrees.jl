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
