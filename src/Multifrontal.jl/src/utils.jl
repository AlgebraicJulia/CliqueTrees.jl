function char(::Val{:N})
    return 'N'
end

function char(::Val{:C})
    return 'C'
end

function char(::Val{:T})
    return 'T'
end

function char(::Val{:L})
    return 'L'
end

function char(::Val{:U})
    return 'U'
end

function char(::Val{:R})
    return 'R'
end

function tri(uplo::Val{:L}, diag::Val{:N}, A::AbstractMatrix)
    return LowerTriangular(A)
end

function tri(uplo::Val{:L}, diag::Val{:U}, A::AbstractMatrix)
    return UnitLowerTriangular(A)
end

function tri(uplo::Val{:U}, diag::Val{:N}, A::AbstractMatrix)
    return UpperTriangular(A)
end

function tri(uplo::Val{:U}, diag::Val{:U}, A::AbstractMatrix)
    return UnitUpperTriangular(A)
end

function sym(::Val{UPLO}, A::AbstractMatrix) where {UPLO}
    return Hermitian(A, UPLO)
end

function adj(tA::Val{:N}, A::AbstractMatrix)
    return A
end

function adj(tA::Val{:T}, A::AbstractMatrix)
    return transpose(A)
end

function adj(tA::Val{:C}, A::AbstractMatrix)
    return adjoint(A)
end

function zerotri!(A::AbstractMatrix{T}, uplo::Val{Q}) where {Q, T}
    @assert size(A, 1) == size(A, 2)

    @inbounds for w in axes(A, 1)
        if Q === :L
            for v in w:size(A, 1)
                A[v, w] = zero(T)
            end
        else
            for v in oneto(w)
                A[v, w] = zero(T)
            end
        end
    end

    return A
end

function zerorec!(A::AbstractArray{T}) where {T}
    fill!(A, zero(T))
    return A
end

function copytri!(
        A::AbstractMatrix{T},
        B::AbstractMatrix{T},
        uplo::Val{Q},
        ind::AbstractVector{I}=axes(A, 1),
    ) where {Q, T, I <: Integer}
    @assert size(A, 1) == size(A, 2) == length(ind)
    @assert size(B, 1) == size(B, 2)

    @inbounds for j in eachindex(ind)
        if Q === :L
            for i in j:length(ind)
                A[i, j] = B[ind[i], ind[j]]
            end
        else
            for i in oneto(j)
                A[i, j] = B[ind[i], ind[j]]
            end
        end
    end

    return A
end

function addtri!(
        A::AbstractMatrix{T},
        B::AbstractMatrix{T},
        uplo::Val{Q},
        ind::AbstractVector{I}=axes(B, 1),
    ) where {Q, T, I <: Integer}
    @assert size(A, 1) == size(A, 2)
    @assert size(B, 1) == size(B, 2) == length(ind)

    @inbounds for j in eachindex(ind)
        if Q === :L
            for i in j:length(ind)
                A[ind[i], ind[j]] += B[i, j]
            end
        else
            for i in oneto(j)
                A[ind[i], ind[j]] += B[i, j]
            end
        end
    end

    return A
end

function addrec!(
        A::AbstractArray{T},
        B::AbstractVector{T},
        row::AbstractVector{I}=axes(B, 1),
    ) where {T, I <: Integer}
    @assert length(row) == length(B)

    @inbounds for i in eachindex(row, B)
        A[row[i]] += B[i]
    end

    return A
end

function addrec!(
        A::AbstractArray{T},
        B::AbstractMatrix{T},
        row::AbstractVector{I}=axes(B, 1),
        col::AbstractVector{I}=axes(B, 2),
    ) where {T, I <: Integer}
    @assert length(row) == size(B, 1)
    @assert length(col) == size(B, 2)

    @inbounds for j in eachindex(col)
        cj = col[j]

        for i in eachindex(row)
            A[row[i], cj] += B[i, j]
        end
    end

    return A
end

function copyrec!(
        A::AbstractVector{T},
        B::AbstractVector{T},
        row::AbstractVector{I}=axes(A, 1),
    ) where {T, I <: Integer}
    @assert length(row) == length(A)

    @inbounds for i in eachindex(row)
        A[i] = B[row[i]]
    end

    return A
end

function copyrec!(
        A::AbstractMatrix{T},
        B::AbstractMatrix{T},
        row::AbstractVector{I}=axes(A, 1),
        col::AbstractVector{I}=axes(A, 2),
    ) where {T, I <: Integer}
    @assert length(row) == size(A, 1)
    @assert length(col) == size(A, 2)

    @inbounds for j in eachindex(col)
        cj = col[j]

        for i in eachindex(row)
            A[i, j] = B[row[i], cj]
        end
    end

    return A
end

function unwrap(A::AbstractVecOrMat)
    return (A, Val(:N))
end

function unwrap(A::Adjoint)
    return (parent(A), Val(:C))
end

function unwrap(A::Transpose)
    return (parent(A), Val(:T))
end

"""
    isforward(UPLO, TRANS, SIDE) -> Bool

Determine whether to traverse the elimination tree forward (leaves → root) or backward (root → leaves).

Forward traversal: L from left, Lᵀ from right (for lower); Uᵀ from left, U from right (for upper).
Backward traversal: Lᵀ from left, L from right (for lower); U from left, Uᵀ from right (for upper).
"""
function isforward(UPLO, TRANS, SIDE)
    return UPLO === :L && (TRANS === :N && SIDE === :L || TRANS !== :N && SIDE === :R) ||
           UPLO === :U && (TRANS !== :N && SIDE === :L || TRANS === :N && SIDE === :R)
end

function copy_D!(
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        res::AbstractGraph{I},
        A::AbstractMatrix{T},
    ) where {T, I <: Integer}

    nwr = one(I)

    for j in vertices(res)
        pj = Dptr[j] - one(I)

        swr = nwr
        nwr = pointers(res)[j + one(I)]

        for vr in swr:nwr - one(I)
            wr = swr

            for pa in nzrange(A, vr)
                wa = rowvals(A)[pa]
                wa < swr && continue
                wa < nwr || break

                while wr < wa
                    pj += one(I); Dval[pj] = zero(T); wr += one(I)
                end

                pj += one(I); Dval[pj] = nonzeros(A)[pa]; wr += one(I)
            end

            while wr < nwr
                pj += one(I); Dval[pj] = zero(T); wr += one(I)
            end
        end
    end

    return
end

function copy_L!(
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        A::AbstractMatrix{T},
        ::Val{:L},
    ) where {T, I <: Integer}

    npr = one(I)

    for j in vertices(res)
        pj = Lptr[j] - one(I)

        spr = npr
        npr = pointers(sep)[j + one(I)]
        spr >= npr && continue

        swr = targets(sep)[spr]
        nwr = targets(sep)[npr - one(I)] + one(I)

        for vr in neighbors(res, j)
            pr = spr

            for pa in nzrange(A, vr)
                wr = targets(sep)[pr]
                wa = rowvals(A)[pa]
                wa < swr && continue
                wa < nwr || break

                while wr < wa
                    pj += one(I); Lval[pj] = zero(T); pr += one(I); wr = targets(sep)[pr]
                end

                pj += one(I); Lval[pj] = nonzeros(A)[pa]; pr += one(I)
            end

            while pr < npr
                pj += one(I); Lval[pj] = zero(T); pr += one(I)
            end
        end
    end

    return
end

function copy_L!(
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        A::AbstractMatrix{T},
        ::Val{:U},
    ) where {T, I <: Integer}

    nwr = one(I)

    for j in vertices(res)
        pj = Lptr[j] - one(I)

        swr = nwr
        nwr = pointers(res)[j + one(I)]

        for vr in neighbors(sep, j)
            wr = swr

            for pa in nzrange(A, vr)
                wa = rowvals(A)[pa]
                wa < swr && continue
                wa < nwr || break

                while wr < wa
                    pj += one(I); Lval[pj] = zero(T); wr += one(I)
                end

                pj += one(I); Lval[pj] = nonzeros(A)[pa]; wr += one(I)
            end

            while wr < nwr
                pj += one(I); Lval[pj] = zero(T); wr += one(I)
            end
        end
    end

    return
end
