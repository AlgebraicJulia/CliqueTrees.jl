const STRTRI_LEAF = 128

# ===== strtri! =====

function strtri!(s::AbstractSemiring, uplo::Val, diag::Val, A::AbstractMatrix{T}; nt::Integer = nthreads()) where {T}
    n = size(A, 1)

    if nt <= 1
        AP, BP, CP = spool_st(s, T, n, n, n)
        strtri_st!(s, uplo, diag, A, AP, BP, CP)
    else
        pool = spool_mt(s, T, nt, n, n, n)
        strtri_mt!(s, uplo, diag, A, pool, nt)
    end

    return A
end

# ===== strtri_st! =====

function strtri_st!(s::AbstractSemiring, uplo::Val{UPLO}, diag::Val, A::AbstractMatrix, AP::AbstractVector, BP::AbstractVector, CP::AbstractVector) where {UPLO}
    n = size(A, 1)

    if n <= STRTRI_LEAF
        strtri2!(s, uplo, diag, A)
    else
        m = n >> 1
        A₁₁ = view(A, 1:m,     1:m)
        A₂₂ = view(A, m + 1:n, m + 1:n)

        if UPLO === :L
            A₂₁ = view(A, m + 1:n, 1:m)
            #
            #   A₂₁ ← A₂₁ A₁₁*
            #   A₂₁ ← A₂₂* A₂₁
            #
            strsx_st!(s, Val(:R), Val(:N), uplo, diag, A₁₁, A₂₁, AP, BP, CP)
            strsx_st!(s, Val(:L), Val(:N), uplo, diag, A₂₂, A₂₁, AP, BP, CP)
        else
            A₁₂ = view(A, 1:m, m + 1:n)
            #
            #   A₁₂ ← A₁₁* A₁₂
            #   A₁₂ ← A₁₂ A₂₂*
            #
            strsx_st!(s, Val(:L), Val(:N), uplo, diag, A₁₁, A₁₂, AP, BP, CP)
            strsx_st!(s, Val(:R), Val(:N), uplo, diag, A₂₂, A₁₂, AP, BP, CP)
        end

        strtri_st!(s, uplo, diag, A₁₁, AP, BP, CP)
        strtri_st!(s, uplo, diag, A₂₂, AP, BP, CP)
    end

    return A
end

# ===== strtri_mt! =====

function strtri_mt!(s::AbstractSemiring, uplo::Val{UPLO}, diag::Val, A::AbstractMatrix, pool::AbstractVector, nt::Integer) where {UPLO}
    n = size(A, 1)

    if nt <= 1 || n <= STRTRI_LEAF
        AP, BP, CP = pool[1]
        strtri_st!(s, uplo, diag, A, AP, BP, CP)
    else
        m = n >> 1
        A₁₁ = view(A, 1:m,     1:m)
        A₂₂ = view(A, m + 1:n, m + 1:n)

        if UPLO === :L
            A₂₁ = view(A, m + 1:n, 1:m)
            #
            #   A₂₁ ← A₂₁ A₁₁*
            #   A₂₁ ← A₂₂* A₂₁
            #
            strsx_mt!(s, Val(:R), Val(:N), uplo, diag, A₁₁, A₂₁, pool, nt)
            strsx_mt!(s, Val(:L), Val(:N), uplo, diag, A₂₂, A₂₁, pool, nt)
        else
            A₁₂ = view(A, 1:m, m + 1:n)
            #
            #   A₁₂ ← A₁₁* A₁₂
            #   A₁₂ ← A₁₂ A₂₂*
            #
            strsx_mt!(s, Val(:L), Val(:N), uplo, diag, A₁₁, A₁₂, pool, nt)
            strsx_mt!(s, Val(:R), Val(:N), uplo, diag, A₂₂, A₁₂, pool, nt)
        end

        nt₁ = nt >> 1
        pool₁ = view(pool, 1:nt₁)
        pool₂ = view(pool, nt₁ + 1:nt)
        task = @spawn strtri_mt!(s, uplo, diag, $A₁₁, $pool₁, $nt₁)
        strtri_mt!(s, uplo, diag, A₂₂, pool₂, nt - nt₁)
        wait(task)
    end

    return A
end

# ===== strtri2! =====

function strtri2!(s::AbstractSemiring, ::Val{:L}, ::Val{DIAG}, A::AbstractMatrix{T}) where {T, DIAG}
    @assert size(A, 1) == size(A, 2)

    n = size(A, 1)

    Z = sizeof(T)
    sA = stride(A, 2)

    if DIAG === :N
        @inbounds for k in 1:n
            A[k, k] = sstar(s, A[k, k])
        end
    end

    @preserve A begin
        pA = pointer(A)

        @inbounds for j in 1:n
            if DIAG === :N && !isintegral(s)
                for i in j + 1:n
                    A[i, j] = sprod(s, A[i, j], A[j, j], Val(:N), Val(:N))
                end
            end

            #
            #   rows j + 1:n of column j, four at a time: the 4 × 4
            #   diagonal block with scalars, then the rows below it
            #   with one 4-column axpy
            #
            k = j + 1

            while k + 3 <= n
                for p in k:k + 3
                    if DIAG === :N && !isintegral(s)
                        A[p, j] = sprod(s, A[p, p], A[p, j], Val(:N), Val(:N))
                    end

                    Apj = A[p, j]

                    for q in p + 1:k + 3
                        A[q, j] = smul(s, Val(:N), Val(:N), Val(:R), A[q, p], Apj, A[q, j])
                    end
                end

                saxpy_kern!(s, Val(:N), Val(:N), Val(:R), pA + ((j - 1) * sA + k + 3) * Z, pA + ((k - 1) * sA + k + 3) * Z, sA, n - k - 3, A[k, j], A[k + 1, j], A[k + 2, j], A[k + 3, j])
                k += 4
            end

            while k <= n
                if DIAG === :N && !isintegral(s)
                    Akj = A[k, j] = sprod(s, A[k, k], A[k, j], Val(:N), Val(:N))
                else
                    Akj = A[k, j]
                end

                saxpy_kern!(s, Val(:N), Val(:N), Val(:R), pA + ((j - 1) * sA + k) * Z, pA + ((k - 1) * sA + k) * Z, Akj, n - k)
                k += 1
            end
        end
    end

    return A
end

function strtri2!(s::AbstractSemiring, ::Val{:U}, ::Val{DIAG}, A::AbstractMatrix{T}) where {T, DIAG}
    @assert size(A, 1) == size(A, 2)

    n = size(A, 1)

    Z = sizeof(T)
    sA = stride(A, 2)

    if DIAG === :N
        @inbounds for k in 1:n
            A[k, k] = sstar(s, A[k, k])
        end
    end

    @preserve A begin
        pA = pointer(A)

        @inbounds for j in n:-1:1
            if DIAG === :N && !isintegral(s)
                for i in 1:j - 1
                    A[i, j] = sprod(s, A[i, j], A[j, j], Val(:N), Val(:N))
                end
            end

            for k in j - 1:-1:1
                if DIAG === :N && !isintegral(s)
                    Akj = A[k, j] = sprod(s, A[k, k], A[k, j], Val(:N), Val(:N))
                else
                    Akj = A[k, j]
                end

                saxpy_kern!(s, Val(:N), Val(:N), Val(:R), pA + (j - 1) * sA * Z, pA + (k - 1) * sA * Z, Akj, k - 1)
            end
        end
    end

    return A
end
