const SLU_NB = 128

# ===== sgetrf! =====

function sgetrf!(s::AbstractSemiring, A::AbstractMatrix{V}; nt::Integer = nthreads()) where {V}
    @assert size(A, 2) == size(A, 1)

    n = size(A, 1)

    if n <= SLU_NB
        sgetrf2!(s, A)
    else
        sgetrf_mt!(s, A, spool_mt(V, nt, n, n, n), nt)
    end

    return A
end

# ===== sgetrf_mt! =====

function sgetrf_mt!(s::AbstractSemiring, A::AbstractMatrix, pool::AbstractVector, nt::Integer)
    @assert size(A, 2) == size(A, 1)

    n = size(A, 1)

    if n <= SLU_NB
        sgetrf2!(s, A)
    else
        for k in 1:SLU_NB:n
            #
            #   A = [ Akk Akn ]
            #       [ Ank Ann ]
            #
            b = min(SLU_NB, n - k + 1)
            Akk = view(A, k:k + b - 1, k:k + b - 1)
            #
            # factorize
            #
            #   Akk* = Ukk* Lkk*
            #
            # and write
            #
            #   Akk ← Lkk + Ukk
            #
            sgetrf2!(s, Akk)

            if k + b <= n
                Akn = view(A, k:k + b - 1, k + b:n)
                Ank = view(A, k + b:n, k:k + b - 1)
                Ann = view(A, k + b:n, k + b:n)
                #
                #   Akn ← Lkk* Akn
                #
                strsx_mt!(s, Val(:L), Val(:N), Val(:L), Val(:U), Akk, Akn, pool, nt)
                #
                #   Ank ← Ank Ukk*
                #
                strsx_mt!(s, Val(:R), Val(:N), Val(:U), Val(:N), Akk, Ank, pool, nt)
                #
                #   Ann ← Ank Akn + Ann
                #
                sgemx_mt!(s, Val(:N), Val(:N), Ann, Ank, Akn, pool, nt)
            end
        end
    end

    return A
end

# ===== sgetrf2! =====

function sgetrf2!(s::AbstractSemiring, A::AbstractMatrix{T}) where {T}
    @assert size(A, 2) == size(A, 1)

    n = size(A, 1)

    Z = sizeof(T)
    sA = stride(A, 2)

    @preserve A begin
        pA = pointer(A)
        i = 1

        #
        #   pivots four at a time
        #
        @inbounds while i + 3 <= n
            #
            #   the columns i:i + 3, one pivot at a time
            #
            for p in i:i + 3
                if !isintegral(s)
                    sApp = sstar(s, A[p, p])

                    for k in p + 1:n
                        A[k, p] = sprod(s, A[k, p], sApp, Val(:N), Val(:N))
                    end
                end

                for j in p + 1:i + 3
                    saxpy_kern!(s, Val(:N), Val(:N), Val(:R), pA + ((j - 1) * sA + p) * Z, pA + ((p - 1) * sA + p) * Z, A[p, j], n - p)
                end
            end
            #
            #   each column j to the right: rows i:i + 3 receive the four
            #   pivots with scalars, then rows i + 4:n with one 4-column axpy
            #
            for j in i + 4:n
                for p in i:i + 2
                    Apj = A[p, j]

                    for q in p + 1:i + 3
                        A[q, j] = smul(s, Val(:N), Val(:N), Val(:R), A[q, p], Apj, A[q, j])
                    end
                end

                saxpy_kern!(s, Val(:N), Val(:N), Val(:R), pA + ((j - 1) * sA + i + 3) * Z, pA + ((i - 1) * sA + i + 3) * Z, sA, n - i - 3, A[i, j], A[i + 1, j], A[i + 2, j], A[i + 3, j])
            end

            i += 4
        end
        #
        #   the remaining pivots one at a time
        #
        @inbounds while i <= n
            #
            #   A = [ Aii Ain ]
            #       [ Ani Ann ]
            #
            if !isintegral(s)
                #
                #   Ani ← Ani Aii*
                #
                sAii = sstar(s, A[i, i])

                for k in i + 1:n
                    A[k, i] = sprod(s, A[k, i], sAii, Val(:N), Val(:N))
                end
            end
            #
            #   Ann ← Ani Ain + Ann
            #
            for j in i + 1:n
                saxpy_kern!(s, Val(:N), Val(:N), Val(:R), pA + ((j - 1) * sA + i) * Z, pA + ((i - 1) * sA + i) * Z, A[i, j], n - i)
            end

            i += 1
        end
    end

    return A
end
