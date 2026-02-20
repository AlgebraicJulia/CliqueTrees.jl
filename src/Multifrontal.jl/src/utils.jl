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

function symmetric(graph, uplo::Char)
    return symmetric(BipartiteGraph(graph), uplo)
end

function symmetric(graph::AbstractGraph{V}, uplo::Char) where {V}
    E = etype(graph)

    fwd = graph; bwd = reverse(fwd)

    n = nv(fwd)
    m = ne(fwd) + ne(bwd)

    mrk = FVector{V}(undef, n)
    ptr = FVector{E}(undef, n + one(V))
    tgt = FVector{V}(undef, m)

    for v in vertices(fwd)
        mrk[v] = zero(V)
    end

    p = zero(E)

    for v in vertices(fwd)
        mrk[v] = v
        ptr[v] = p + one(E)

        for w in neighbors(fwd, v)
            uplo === 'L' && w < v && continue
            uplo === 'U' && w > v && continue

            if mrk[w] < v
                mrk[w] = v
                p += one(E); tgt[p] = w
            end
        end

        for w in neighbors(bwd, v)
            if mrk[w] < v
                mrk[w] = v
                p += one(E); tgt[p] = w
            end
        end
    end

    ptr[n + one(V)] = p + one(E)
    return BipartiteGraph{V, E}(n, n, m, ptr, tgt)
end

function sympermute(A::SparseMatrixCSC{T, I}, invp::AbstractVector, src::Char, tgt::Char) where {T, I}
    n = size(A, 1)
    m = zero(I)
    colptr = zeros(I, n + 1)

    @inbounds for j in axes(A, 2)
        pj = invp[j]

        for p in nzrange(A, j)
            i = rowvals(A)[p]

            src === 'L' && i < j && continue
            src === 'U' && i > j && continue

            pi = invp[i]

            if tgt === 'L'
                hi = min(pi, pj)
            else
                hi = max(pi, pj)
            end

            if hi < n
                colptr[hi + two(I)] += one(I)
            end

            m += one(I)
        end
    end

    @inbounds colptr[1] = p = one(I)

    @inbounds for i in axes(A, 1)
        colptr[i + 1] = p += colptr[i + 1]
    end

    rowval = Vector{I}(undef, m)
    nzval = Vector{T}(undef, m)

    @inbounds for j in axes(A, 2)
        pj = invp[j]

        for p in nzrange(A, j)
            i = rowvals(A)[p]

            src === 'L' && i < j && continue
            src === 'U' && i > j && continue

            pi = invp[i]

            if tgt === 'L'
                hi, lo = minmax(pi, pj)
            else
                lo, hi = minmax(pi, pj)
            end

            q = colptr[hi + one(I)]
            rowval[q] = lo
            v = nonzeros(A)[p]

            if (i > j) == (pi > pj)
                nzval[q] = conj(v)
            else
                nzval[q] = v
            end

            colptr[hi + one(I)] += one(I)
        end
    end

    return SparseMatrixCSC(n, n, colptr, rowval, nzval)
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

function zerotri!(A::AbstractMatrix{T}, uplo::Val{Q}, col::AbstractVector{I}=axes(A, 1)) where {Q, T, I <: Integer}
    @assert size(A, 1) == size(A, 2)

    @inbounds for j in eachindex(col)
        cj = col[j]

        if Q === :L
            for i in j:length(col)
                A[col[i], cj] = zero(T)
            end
        else
            for i in oneto(j)
                A[col[i], cj] = zero(T)
            end
        end
    end

    return A
end

function zerorec!(A::AbstractVector{T}, row::AbstractVector{I}=axes(A, 1)) where {T, I <: Integer}
    @inbounds for i in eachindex(row)
        A[row[i]] = zero(T)
    end

    return A
end

function zerorec!(A::AbstractMatrix{T}, row::AbstractVector{I}=axes(A, 1), col::AbstractVector{J}=axes(A, 2)) where {T, I <: Integer, J <: Integer}
    @inbounds for j in eachindex(col)
        cj = col[j]

        for i in eachindex(row)
            A[row[i], cj] = zero(T)
        end
    end

    return A
end

function copytri!(
        A::AbstractMatrix{T},
        B::AbstractMatrix{T},
        uplo::Val{Q},
        col::AbstractVector{I}=axes(A, 1),
    ) where {Q, T, I <: Integer}
    @assert size(A, 1) == size(A, 2) == length(col)
    @assert size(B, 1) == size(B, 2)

    @inbounds for j in eachindex(col)
        cj = col[j]

        if Q === :L
            for i in j:length(col)
                A[i, j] = B[col[i], cj]
            end
        else
            for i in oneto(j)
                A[i, j] = B[col[i], cj]
            end
        end
    end

    return A
end

function addtri!(
        A::AbstractMatrix{T},
        B::AbstractMatrix{T},
        uplo::Val{Q},
        col::AbstractVector{I}=axes(B, 1),
    ) where {Q, T, I <: Integer}
    @assert size(A, 1) == size(A, 2)
    @assert size(B, 1) == size(B, 2) == length(col)

    @inbounds for j in eachindex(col)
        cj = col[j]

        if Q === :L
            for i in j:length(col)
                A[col[i], cj] += B[i, j]
            end
        else
            for i in oneto(j)
                A[col[i], cj] += B[i, j]
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

function isforward(UPLO, TRANS, SIDE)
    return UPLO === :L && (TRANS === :N && SIDE === :L || TRANS !== :N && SIDE === :R) ||
           UPLO === :U && (TRANS !== :N && SIDE === :L || TRANS === :N && SIDE === :R)
end

function copy_D!(
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        res::AbstractGraph{I},
        A::SparseMatrixCSC,
    ) where {T, I <: Integer}

    nwr = one(I)

    @inbounds for j in vertices(res)
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
        A::SparseMatrixCSC,
        ::Val{:L},
    ) where {T, I <: Integer}

    npr = one(I)

    @inbounds for j in vertices(res)
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
        A::SparseMatrixCSC,
        ::Val{:U},
    ) where {T, I <: Integer}

    nwr = one(I)

    @inbounds for j in vertices(res)
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

function flat_D!(
        Dptr::AbstractVector{I},
        Aval::AbstractVector,
        res::AbstractGraph{I},
        A::SparseMatrixCSC,
    ) where {I <: Integer}

    nwr = one(I)

    @inbounds for j in vertices(res)
        pj = Dptr[j] - one(I)

        swr = nwr
        nwr = pointers(res)[j + one(I)]

        for vr in swr:nwr - one(I)
            wr = swr

            for pa in nzrange(A, vr)
                wa = rowvals(A)[pa]
                wa < swr && continue
                wa < nwr || break

                pj += wa - wr; wr = wa
                pj += one(I); Aval[pa] = pj; wr += one(I)
            end

            pj += nwr - wr
        end
    end

    return
end

function flat_L!(
        Lptr::AbstractVector{I},
        Aval::AbstractVector,
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        A::SparseMatrixCSC,
        uplo::Val{:L},
    ) where {I <: Integer}

    npr = one(I)

    @inbounds for j in vertices(res)
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
                    pj += one(I); pr += one(I); wr = targets(sep)[pr]
                end

                pj += one(I); Aval[pa] = -pj; pr += one(I)
            end

            pj += npr - pr
        end
    end

    return
end

function flat_L!(
        Lptr::AbstractVector{I},
        Aval::AbstractVector,
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        A::SparseMatrixCSC,
        uplo::Val{:U},
    ) where {I <: Integer}

    nwr = one(I)

    @inbounds for j in vertices(res)
        pj = Lptr[j] - one(I)

        swr = nwr
        nwr = pointers(res)[j + one(I)]

        for vr in neighbors(sep, j)
            wr = swr

            for pa in nzrange(A, vr)
                wa = rowvals(A)[pa]
                wa < swr && continue
                wa < nwr || break

                pj += wa - wr; wr = wa
                pj += one(I); Aval[pa] = -pj; wr += one(I)
            end

            pj += nwr - wr
        end
    end

    return
end

function swaprec!(v::AbstractVector, j::Integer, k::Integer)
    @inbounds v[j], v[k] = v[k], v[j]
    return
end

# Hermitian swap: exchange rows/cols j and k for Hermitian matrix stored in lower triangle
function swaptri!(A::AbstractMatrix, j::Integer, k::Integer, ::Val{:L})
    n = size(A, 1)

    @inbounds begin
        A[j, j], A[k, k] = A[k, k], A[j, j]

        for i in k+1:n
            A[i, j], A[i, k] = A[i, k], A[i, j]
        end

        for i in j+1:k-1
            A[i, j], A[k, i] = conj(A[k, i]), conj(A[i, j])
        end

        for i in 1:j-1
            A[j, i], A[k, i] = A[k, i], A[j, i]
        end

        A[k, j] = conj(A[k, j])
    end

    return
end

# Hermitian swap: exchange rows/cols j and k for Hermitian matrix stored in upper triangle
function swaptri!(A::AbstractMatrix, j::Integer, k::Integer, ::Val{:U})
    n = size(A, 1)

    @inbounds begin
        A[j, j], A[k, k] = A[k, k], A[j, j]

        for i in 1:j-1
            A[i, j], A[i, k] = A[i, k], A[i, j]
        end

        for i in j+1:k-1
            A[j, i], A[i, k] = conj(A[i, k]), conj(A[j, i])
        end

        for i in k+1:n
            A[j, i], A[k, i] = A[k, i], A[j, i]
        end

        A[j, k] = conj(A[j, k])
    end

    return
end
