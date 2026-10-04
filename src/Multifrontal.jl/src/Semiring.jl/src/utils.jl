# ===== sccs =====

function sccs(A::SparseMatrixCSC)
    return sccs(BipartiteGraph(A))
end

function sccs(graph::BipartiteGraph{I}) where {I}
    n = nv(graph); np1 = n + one(I)

    low  = FVector{I}(undef, n)
    arc  = FVector{I}(undef, n)
    ptr  = FVector{I}(undef, np1)
    tgt  = FVector{I}(undef, n)

    return sccs!(low, arc, ptr, tgt, graph)
end

function sccs!(low::AbstractVector{I}, arc::AbstractVector{I}, ptr::AbstractVector{I}, tgt::AbstractVector{I}, graph::BipartiteGraph{I}) where {I}
    n = nv(graph); np1 = n + one(I)

    fill!(low, zero(I))

    r = np1; c = zero(I); i = one(I)

    @inbounds for u in vertices(graph)
        if iszero(low[u])
            r -= one(I); tgt[r] = u; l = low[u] = np1 - r
            d  = one(I); ptr[np1 - d] = r
            v  = u; p = pointers(graph)[v]; pstop = pointers(graph)[v + one(I)]

            while true
                if p < pstop
                    w = targets(graph)[p]; p += one(I); m = low[w]

                    if iszero(m)
                        low[v] = l; arc[d] = p
                        r -= one(I); tgt[r] = w; l = low[w] = np1 - r
                        d += one(I); ptr[np1 - d] = r
                        v  = w; p = pointers(graph)[v]; pstop = pointers(graph)[v + one(I)]
                    else
                        l = min(l, m)
                    end
                else
                    rstop = ptr[np1 - d]

                    if l + rstop == np1
                        c += one(I); ptr[c] = i

                        for j in r:rstop
                            w = tgt[i] = tgt[j]; low[w] = np1; i += one(I)
                        end

                        r = rstop + one(I)
                    else
                        low[v] = l
                    end

                    d -= one(I); iszero(d) && break

                    v  = tgt[ptr[np1 - d]]
                    l  = min(low[v], l)
                    p  = arc[d]; pstop = pointers(graph)[v + one(I)]
                end
            end
        end
    end

    ptr[c + one(I)] = i
    return BipartiteGraph{I, I}(n, c, n, ptr, tgt)
end

#
# A[p, q] for a sparse matrix whose columns are sorted and duplicate-free: the same arrays as
# SparseArrays.permute(A, p, q), built in parallel. Column j of the result is column q[j] of A with its
# rows renamed by p⁻¹ and sorted; the columns are independent, so threads fill disjoint ranges.
# (SparseArrays' version is serial and goes through two transposes: 19 ms for 2M entries.)
#
function permute_csc(A::SparseMatrixCSC{T, I}, p::AbstractVector, q::AbstractVector) where {T, I}
    m, n = size(A)
    Aptr = getcolptr(A); Arow = rowvals(A); Aval = nonzeros(A)
    ip = Vector{I}(undef, m)

    @inbounds for i in oneto(m)
        ip[p[i]] = i
    end

    ptr = Vector{I}(undef, n + 1)
    @inbounds ptr[1] = one(I)

    @inbounds for j in oneto(n)
        c = q[j]
        ptr[j + 1] = ptr[j] + (Aptr[c + 1] - Aptr[c])
    end

    nz = Int(ptr[n + 1]) - 1
    row = Vector{I}(undef, nz)
    val = Vector{T}(undef, nz)
    nchunk = nz < 2^16 ? 1 : 8 * nthreads()

    if isone(nchunk)                               # (no threads to wake for a small matrix)
        permute_csc_columns!(row, val, ptr, Aptr, Arow, Aval, ip, q, 1, n)
    else
        @threads for k in 1:nchunk
            permute_csc_columns!(row, val, ptr, Aptr, Arow, Aval, ip, q, cld((k - 1) * n, nchunk) + 1, cld(k * n, nchunk))
        end
    end

    return SparseMatrixCSC(m, n, ptr, row, val)
end

# the graph of the pattern of A[p, q] (columns sorted), without the values
function permute_pattern(A::SparseMatrixCSC{<:Any, I}, p::AbstractVector, q::AbstractVector) where {I}
    m, n = size(A)
    Aptr = getcolptr(A); Arow = rowvals(A)
    ip = Vector{I}(undef, m)

    @inbounds for i in oneto(m)
        ip[p[i]] = i
    end

    ptr = FVector{I}(undef, n + 1)
    @inbounds ptr[1] = one(I)

    @inbounds for j in oneto(n)
        c = q[j]
        ptr[j + 1] = ptr[j] + (Aptr[c + 1] - Aptr[c])
    end

    nz = ptr[n + 1] - one(I)
    row = FVector{I}(undef, nz)
    nchunk = nz < 2^16 ? 1 : 8 * nthreads()

    if isone(nchunk)                               # (no threads to wake for a small matrix)
        permute_pattern_columns!(row, ptr, Aptr, Arow, ip, q, 1, n)
    else
        @threads for k in 1:nchunk
            permute_pattern_columns!(row, ptr, Aptr, Arow, ip, q, cld((k - 1) * n, nchunk) + 1, cld(k * n, nchunk))
        end
    end

    return BipartiteGraph{I, I}(m, n, nz, ptr, row)
end

function permute_pattern_columns!(row::AbstractVector{I}, ptr, Aptr, Arow, ip, q, j0::Integer, j1::Integer) where {I}
    @inbounds for j in j0:j1
        c = q[j]; a = Aptr[c]; o = ptr[j]; d = ptr[j + 1] - o

        for t in 0:(d - 1)
            row[o + t] = ip[Arow[a + t]]
        end

        if d > 24                                  # rows are distinct: an unstable in-place sort will do
            r = view(row, o:(o + d - 1))
            issorted(r) || sort!(r; alg = QuickSort)  # (sorted already when the relabelling keeps the order)
            continue
        end

        for t in 1:(d - 1)
            r = row[o + t]; u = t - 1

            while u >= 0 && row[o + u] > r
                row[o + u + 1] = row[o + u]
                u -= 1
            end

            row[o + u + 1] = r
        end
    end

    return
end

function permute_csc_columns!(row::Vector{I}, val::Vector{T}, ptr, Aptr, Arow, Aval, ip, q, j0::Integer, j1::Integer) where {I, T}
    @inbounds for j in j0:j1
        c = q[j]; a = Aptr[c]; o = ptr[j]; d = ptr[j + 1] - o

        for t in 0:(d - 1)
            row[o + t] = ip[Arow[a + t]]
            val[o + t] = Aval[a + t]
        end

        if d > 32                                  # a long column: sort it through a permutation
            rs = view(row, o:(o + d - 1)); vs = view(val, o:(o + d - 1))
            σ = sortperm(rs)
            vs .= vs[σ]; rs .= rs[σ]
            continue
        end

        # insertion sort of a short column by row (rows are distinct)
        for t in 1:(d - 1)
            r = row[o + t]; v = val[o + t]; u = t - 1

            while u >= 0 && row[o + u] > r
                row[o + u + 1] = row[o + u]; val[o + u + 1] = val[o + u]
                u -= 1
            end

            row[o + u + 1] = r; val[o + u + 1] = v
        end
    end

    return
end

function subgraph(graph::BipartiteGraph{I}, strt::I, stop::I) where {I}
    @assert one(I) <= strt <= stop <= nv(graph)

    n = stop - strt + one(I)

    ptr = FVector{I}(undef, n + one(I))

    p = one(I)

    @inbounds for j in oneto(n)
        ptr[j] = p

        for i in neighbors(graph, strt + j - one(I))
            if strt <= i <= stop
                p += one(I)
            end
        end
    end

    ptr[n + one(I)] = p; m = p - one(I)

    tgt = FVector{I}(undef, m)

    @inbounds for j in oneto(n)
        p = ptr[j]

        for i in neighbors(graph, strt + j - one(I))
            if strt <= i <= stop
                tgt[p] = i - strt + one(I); p += one(I)
            end
        end
    end

    return BipartiteGraph{I, I}(n, n, m, ptr, tgt)
end

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

function intriangle(::Val{:L}, i, j)
    return i >= j
end

function intriangle(::Val{:U}, i, j)
    return i <= j
end
