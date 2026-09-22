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

function sgetrf_mt!(s::AbstractSemiring, A::AbstractMatrix, pool::Channel, nt::Integer)
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
                strsx_mt!(s, Val(:L), Val(:N), Val(:L), Akk, Akn, pool, nt)
                #
                #   Ank ← Ank Ukk*
                #
                strsx_mt!(s, Val(:R), Val(:N), Val(:U), Akk, Ank, pool, nt)
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

function sgetrf2!(s::AbstractSemiring, A::AbstractMatrix)
    @assert size(A, 2) == size(A, 1)

    n = size(A, 1)

    @inbounds for i in 1:n
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
            Aij = A[i, j]

            for k in i + 1:n
                A[k, j] = smuladd(s, A[k, i], Aij, A[k, j], Val(:N), Val(:N))
            end
        end
    end

    return A
end
