function dijkstra!(s::AbstractSemiring, A::SparseMatrixCSC{T, I}, b::AbstractVector{T}) where {T, I}
    n = size(A, 2)
    fwd = FVector{I}(undef, n)
    bwd = FVector{I}(undef, n)
    return dijkstra_st!(s, b, fwd, bwd, A)
end

function dijkstra!(s::AbstractSemiring, A::SparseMatrixCSC{T, I}, B::AbstractMatrix{T}; nt::Integer = nthreads()) where {T, I}
    n = size(A, 2)

    @threads for t in 1:nt
        fwd = FVector{I}(undef, n)
        bwd = FVector{I}(undef, n)

        @inbounds for k in t:nt:size(B, 2)
            Bk = view(B, :, k)
            dijkstra_st!(s, Bk, fwd, bwd, A)
        end
    end

    return B
end

function dijkstra_st!(s::AbstractSemiring, val::AbstractVector{T}, fwd::AbstractVector{I}, bwd::AbstractVector{I}, A::SparseMatrixCSC{T, I}) where {T, I}
    n = convert(I, size(A, 2))
    m = zero(I)

    fill!(bwd, zero(I))

    @inbounds for u in oneto(n)
        if val[u] != szero(s, T, Val(:N))
            m += one(I)
            fwd[m] = u
            bwd[u] = m
        end
    end

    @inbounds for i in (m + two(I)) >> 2:-one(I):one(I)
        hfall!(s, fwd, bwd, val, i, m)
    end

    @inbounds while m > zero(I)
        u = fwd[1]
        a = val[u]
        bwd[u] = zero(I)

        w = fwd[m]
        m -= one(I)

        if m > zero(I)
            fwd[1] = w
            bwd[w] = one(I)
            hfall!(s, fwd, bwd, val, one(I), m)
        end

        pstrt = A.colptr[u]
        pstop = A.colptr[u + one(I)] - one(I)

        for p in pstrt:pstop
            v = A.rowval[p]
            b = sprod(s, A.nzval[p], a, Val(:N), Val(:N))

            if !sgte(s, val[v], b)
                val[v] = b

                if iszero(bwd[v])
                    m += one(I)
                    fwd[m] = v
                    bwd[v] = m
                    hrise!(s, fwd, bwd, val, m)
                else
                    hrise!(s, fwd, bwd, val, bwd[v])
                end
            end
        end
    end

    return val
end

function hswap!(fwd::AbstractVector{I}, bwd::AbstractVector{I}, i::I, j::I) where {I}
    @inbounds u = fwd[i]
    @inbounds v = fwd[j]
    @inbounds fwd[i] = v
    @inbounds fwd[j] = u
    @inbounds bwd[v] = i
    @inbounds bwd[u] = j
    return
end

function hrise!(s::AbstractSemiring, fwd::AbstractVector{I}, bwd::AbstractVector{I}, val::AbstractVector, i::I) where {I}
    @inbounds while i > one(I)
        j = (i + two(I)) >> 2

        if slte(s, val[fwd[i]], val[fwd[j]])
            break
        end

        hswap!(fwd, bwd, i, j)
        i = j
    end

    return
end

@inline function hfall!(s::AbstractSemiring, fwd::AbstractVector{I}, bwd::AbstractVector{I}, val::AbstractVector, i::I, m::I) where {I}
    @inbounds while true
        jstrt = four(I) * i - two(I)

        if jstrt > m
            break
        end

        jstop = min(four(I) * i + one(I), m)

        k = jstrt
        a = val[fwd[k]]

        for j in jstrt + one(I):jstop
            b = val[fwd[j]]

            if slte(s, a, b)
                a = b
                k = j
            end
        end

        if slte(s, a, val[fwd[i]])
            break
        end

        hswap!(fwd, bwd, i, k)
        i = k
    end

    return
end
