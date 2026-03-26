# ===== trtri! =====

function trtri!(uplo::Val, diag::Val, A::AbstractMatrix{T}) where {T <: BlasFloat}
    LAPACK.trtri!(char(uplo), char(diag), A)
    return
end

function trtri!(uplo::Val{UPLO}, diag::Val{DIAG}, A::AbstractMatrix{T}) where {T, UPLO, DIAG}
    n = size(A, 1)

    if n <= THRESHOLD
        trtri2!(uplo, diag, A)
        return
    end

    m = prevpow(2, n) >> 1

    A₁₁ = view(A,     1:m,     1:m)
    A₂₂ = view(A, m + 1:n, m + 1:n)

    trtri!(uplo, diag, A₁₁)
    trtri!(uplo, diag, A₂₂)

    if UPLO === :L
        A₂₁ = view(A, m + 1:n, 1:m)
        trmm!(Val(:R), uplo, Val(:N), diag,  one(T), A₁₁, A₂₁)
        trmm!(Val(:L), uplo, Val(:N), diag, -one(T), A₂₂, A₂₁)
    else
        A₁₂ = view(A, 1:m, m + 1:n)
        trmm!(Val(:L), uplo, Val(:N), diag,  one(T), A₁₁, A₁₂)
        trmm!(Val(:R), uplo, Val(:N), diag, -one(T), A₂₂, A₁₂)
    end

    return
end

# ===== trtri2! =====

function trtri2!(::Val{:L}, ::Val{DIAG}, A::AbstractMatrix{T}) where {T, DIAG}
    n = size(A, 1)

    @inbounds @fastmath for j in 1:n
        if DIAG === :N
            A[j, j] = inv(A[j, j])
            Ajj = -A[j, j]
        else
            Ajj = -one(T)
        end

        for i in j + 1:n
            Aij = A[i, j] * Ajj

            for k in j + 1:i - 1
                Aij -= A[i, k] * A[k, j]
            end

            if DIAG === :N
                A[i, j] = Aij / A[i, i]
            else
                A[i, j] = Aij
            end
        end
    end
    return
end

function trtri2!(::Val{:U}, ::Val{DIAG}, A::AbstractMatrix{T}) where {T, DIAG}
    n = size(A, 1)

    @inbounds @fastmath for j in n:-1:1
        if DIAG === :N
            A[j, j] = inv(A[j, j])
            Ajj = -A[j, j]
        else
            Ajj = -one(T)
        end

        for i in j - 1:-1:1
            Aij = A[i, j] * Ajj

            for k in i + 1:j - 1
                Aij -= A[i, k] * A[k, j]
            end

            if DIAG === :N
                A[i, j] = Aij / A[i, i]
            else
                A[i, j] = Aij
            end
        end
    end
    return
end
