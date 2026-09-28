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
