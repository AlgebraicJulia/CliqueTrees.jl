const SLU_NB = 128

# ===== slu! =====

function slu!(s::AbstractSemiring, A::AbstractMatrix{V}) where {V}
    n = size(A, 1)
    @assert size(A, 2) == n

    if n <= SLU_NB
        slu2!(s, A)
    else
        nt = nthreads()
        slu_mt!(s, A, spool(V, nt), nt)
    end

    return A
end

# ===== slu_mt! =====

function slu_mt!(s::AbstractSemiring, A::AbstractMatrix, pool::Channel, nt::Integer)
    @assert size(A, 2) == size(A, 1)

    n = size(A, 1)

    if n <= SLU_NB
        slu2!(s, A)
    else
        depth = ceil(Int, log2(nt)) + 1

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
            slu2!(s, Akk)

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
                sgemx_mt!(s, Ann, Ank, Akn, pool, depth)
            end
        end
    end

    return A
end

# ===== slu2! =====

function slu2!(s::AbstractSemiring, A::AbstractMatrix)
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
