# ===== potrf2! =====

function potrf2!(::Val{:L}, A::AbstractMatrix{T}, d::AbstractVector{T}, S::AbstractVector{T}, R::AbstractRegularization, ::Val{DIAG}) where {T, DIAG}
    @inbounds @fastmath for j in axes(A, 1)
        Ajj = regularize(R, S, real(A[j, j]), j)

        iszero(Ajj) && return j

        if DIAG === :N
            A[j, j] = Ajj = sqrt(Ajj)
        else
            d[j] = Ajj
        end

        iDjj = inv(Ajj)

        for i in j + 1:size(A, 1)
            A[i, j] *= iDjj
        end

        for k in j + 1:size(A, 1)
            Akj = A[k, j]; cAkj = conj(Akj)

            if DIAG === :N
                A[k, k] -= abs2(Akj)
            else
                A[k, k] -= Ajj * abs2(Akj)
            end

            for i in k + 1:size(A, 1)
                if DIAG === :N
                    A[i, k] -= A[i, j] * cAkj
                else
                    A[i, k] -= A[i, j] * Ajj * cAkj
                end
            end
        end
    end

    return 0
end

function potrf2!(::Val{:U}, A::AbstractMatrix{T}, d::AbstractVector{T}, S::AbstractVector{T}, R::AbstractRegularization, ::Val{DIAG}) where {T, DIAG}
    @inbounds @fastmath for j in axes(A, 1)
        Ajj = real(A[j, j])

        for k in 1:j - 1
            if DIAG === :N
                Ajj -= abs2(A[k, j])
            else
                Ajj -= abs2(A[k, j]) * real(d[k])
            end
        end

        Ajj = regularize(R, S, Ajj, j)

        iszero(Ajj) && return j

        if DIAG === :N
            A[j, j] = Ajj = sqrt(Ajj)
        else
            d[j] = Ajj
        end

        iDjj = inv(Ajj)

        for i in j + 1:size(A, 1)
            for k in 1:j - 1
                cAkj = conj(A[k, j])

                if DIAG === :N
                    A[j, i] -= A[k, i] * cAkj
                else
                    A[j, i] -= A[k, i] * d[k] * cAkj
                end
            end

            A[j, i] *= iDjj
        end
    end

    return 0
end

# ===== potrf! =====

function potrf!(uplo::Val{UPLO}, A::AbstractMatrix{T}) where {T <: BlasFloat, UPLO}
    _, info = LAPACK.potrf!(char(uplo), A)
    return info
end

function potrf!(uplo::Val{UPLO}, A::AbstractMatrix{T}) where {T, UPLO}
    n = size(A, 1)
    W = Ones{T}(n * n)
    d = Ones{T}(n)
    S = Ones{T}(n)
    R = NoRegularization()
    diag = Val(:N)
    return potrf!(uplo, W, A, d, S, R, diag)
end

function potrf!(uplo::Val{UPLO}, ::AbstractVector{T}, A::AbstractMatrix{T}, ::AbstractVector{T}, ::AbstractVector{T}, ::NoRegularization, ::Val{:N}) where {T <: BlasFloat, UPLO}
    return potrf!(uplo, A)
end

function potrf!(uplo::Val{UPLO}, W::AbstractVector{T}, D::AbstractMatrix{T}, d::AbstractVector{T}, S::AbstractVector{T}, R::GMW81, diag::Val{DIAG}) where {T, UPLO, DIAG}
    n = size(D, 1)

    if UPLO === :L
        L = Ones{T}(0, n)
    else
        L = Ones{T}(n, 0)
    end

    return chol_factor!(D, L, W, d, S, R, uplo, diag)
end

function potrf!(uplo::Val{UPLO}, W::AbstractVector{T}, L₁::AbstractMatrix{T}, d₁::AbstractVector{T}, S₁::AbstractVector{T}, R::SE99, diag::Val{DIAG}) where {T, UPLO, DIAG}
    n = size(L₁, 1)

    if UPLO === :L
        L₂ = Ones{T}(0, n)
    else
        L₂ = Ones{T}(n, 0)
    end

    M = Ones{T}(0, 0)
    d₂ = Ones{T}(0)
    S₂ = Ones{T}(0)

    chol_se99_kernel!(L₁, L₂, M, W, d₁, d₂, S₁, S₂, R, true, zero(real(T)), zero(real(T)), uplo, diag)

    return 0
end

function potrf!(uplo::Val{UPLO}, W::AbstractVector{T}, A::AbstractMatrix{T}, d::AbstractVector{T}, S::AbstractVector{T}, R::AbstractRegularization, diag::Val{DIAG}) where {T, UPLO, DIAG}
    n = size(A, 1)
    n <= THRESHOLD && return potrf2!(uplo, A, d, S, R, diag)

    n₁  = 2^floor(Int, log2(n)) ÷ 2
    A₁₁ = view(A, 1:n₁, 1:n₁)
    A₂₂ = view(A, n₁+1:n, n₁+1:n)

    d₁₁ = view(d, 1:n₁)
    d₂₂ = view(d, n₁+1:n)
    S₁₁ = view(S, 1:n₁)
    S₂₂ = view(S, n₁+1:n)
    #
    # factorize A₁₁
    #
    info = potrf!(uplo, W, A₁₁, d₁₁, S₁₁, R, diag)
    !iszero(info) && return info

    if UPLO === :L
        B = view(A, n₁+1:n, 1:n₁)
        side = Val(:R)
        trans = Val(:N)
    else
        B = view(A, 1:n₁, n₁+1:n)
        side = Val(:L)
        trans = Val(:C)
    end
    #
    # B ← B A₁₁⁻ᴴ D₁₁⁻¹
    #
    trsm!(side, uplo, Val(:C), diag, one(T), A₁₁, B)
    cdiv!(side, diag, B, d₁₁)
    #
    # A₂₂ ← A₂₂ - B Bᴴ       (Cholesky)
    # A₂₂ ← A₂₂ - B D₁₁ Bᴴ   (LDLt)
    #
    syrk!(uplo, trans, -one(real(T)), W, B, d₁₁, one(real(T)), A₂₂, diag)
    #
    # factorize A₂₂
    #
    info = potrf!(uplo, W, A₂₂, d₂₂, S₂₂, R, diag)
    !iszero(info) && return n₁ + info
    return 0
end
