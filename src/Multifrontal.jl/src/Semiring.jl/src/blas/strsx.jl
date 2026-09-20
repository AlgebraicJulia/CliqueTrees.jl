# ===== strsx2! =====

function strsx2!(s::AbstractSemiring, ::Val{:L}, ::Val{:N}, ::Val{:L}, A::AbstractMatrix, B::AbstractVecOrMat)
    n = size(A, 1)
    m = size(B, 2)

    @inbounds for j in 1:m
        for i in 1:n
            Bij = B[i, j]

            for k in i + 1:n
                B[k, j] = smuladd(s, A[k, i], Bij, B[k, j], Val(:N), Val(:N))
            end
        end
    end

    return B
end

function strsx2!(s::AbstractSemiring, ::Val{:L}, ::Val{:N}, ::Val{:U}, A::AbstractMatrix, B::AbstractVecOrMat)
    n = size(A, 1)
    m = size(B, 2)

    @inbounds for i in 1:m
        for j in n:-1:1
            if isintegral(s)
                Bji = B[j, i]
            else
                Bji = sprod(s, sstar(s, A[j, j]), B[j, i], Val(:N), Val(:N))
            end

            B[j, i] = Bji

            for k in 1:j - 1
                B[k, i] = smuladd(s, A[k, j], Bji, B[k, i], Val(:N), Val(:N))
            end
        end
    end

    return B
end

function strsx2!(s::AbstractSemiring, ::Val{:L}, trans::Union{Val{:T}, Val{:C}}, ::Val{:L}, A::AbstractMatrix{T}, B::AbstractVecOrMat) where {T}
    n = size(A, 1)
    m = size(B, 2)

    @inbounds for j in 1:m
        for k in n:-1:1
            Δ = szero(s, T, trans)

            @simd for i in k + 1:n
                Δ = smuladd(s, A[i, k], B[i, j], Δ, trans, Val(:N))
            end

            B[k, j] = splus(s, B[k, j], Δ, trans)
        end
    end

    return B
end

function strsx2!(s::AbstractSemiring, ::Val{:L}, trans::Union{Val{:T}, Val{:C}}, ::Val{:U}, A::AbstractMatrix{T}, B::AbstractVecOrMat) where {T}
    n = size(A, 1)
    m = size(B, 2)

    @inbounds for j in 1:m
        for k in 1:n
            Δ = szero(s, T, trans)

            @simd for i in 1:k - 1
                Δ = smuladd(s, A[i, k], B[i, j], Δ, trans, Val(:N))
            end

            Bk = splus(s, B[k, j], Δ, trans)

            if isintegral(s)
                B[k, j] = Bk
            else
                B[k, j] = sprod(s, sstar(s, A[k, k]), Bk, trans, Val(:N))
            end
        end
    end

    return B
end

function strsx2!(s::AbstractSemiring, ::Val{:R}, ::Val{:N}, ::Val{:U}, A::AbstractMatrix, B::AbstractMatrix)
    n = size(A, 1)
    m = size(B, 1)

    @inbounds for j in 1:n
        for k in 1:j - 1
            Akj = A[k, j]

            for i in 1:m
                B[i, j] = smuladd(s, B[i, k], Akj, B[i, j], Val(:N), Val(:N))
            end
        end

        if !isintegral(s)
            sAjj = sstar(s, A[j, j])

            for i in 1:m
                B[i, j] = sprod(s, B[i, j], sAjj, Val(:N), Val(:N))
            end
        end
    end

    return B
end

function strsx2!(s::AbstractSemiring, ::Val{:R}, ::Val{:N}, ::Val{:L}, A::AbstractMatrix, B::AbstractMatrix)
    n = size(A, 1)
    m = size(B, 1)

    @inbounds for j in n:-1:1
        for k in j + 1:n
            Akj = A[k, j]

            for i in 1:m
                B[i, j] = smuladd(s, B[i, k], Akj, B[i, j], Val(:N), Val(:N))
            end
        end
    end

    return B
end

function strsx2!(s::AbstractSemiring, ::Val{:R}, trans::Union{Val{:T}, Val{:C}}, ::Val{:L}, A::AbstractMatrix, B::AbstractMatrix)
    n = size(A, 1)
    m = size(B, 1)

    @inbounds for k in 1:n
        for j in k + 1:n
            Ajk = A[j, k]

            for i in 1:m
                B[i, j] = smuladd(s, B[i, k], Ajk, B[i, j], Val(:N), trans)
            end
        end
    end

    return B
end

function strsx2!(s::AbstractSemiring, ::Val{:R}, trans::Union{Val{:T}, Val{:C}}, ::Val{:U}, A::AbstractMatrix, B::AbstractMatrix)
    n = size(A, 1)
    m = size(B, 1)

    @inbounds for k in n:-1:1
        if !isintegral(s)
            sAkk = sstar(s, A[k, k])

            for i in 1:m
                B[i, k] = sprod(s, B[i, k], sAkk, Val(:N), trans)
            end
        end

        for j in 1:k - 1
            Ajk = A[j, k]

            for i in 1:m
                B[i, j] = smuladd(s, B[i, k], Ajk, B[i, j], Val(:N), trans)
            end
        end
    end

    return B
end

function strsx2!(s::AbstractSemiring, ::Val{:R}, ::Val{:N}, ::Val{:U}, A::AbstractMatrix, b::AbstractVector)
    n = size(A, 1)

    @inbounds for j in 1:n
        bj = b[j]

        @simd for k in 1:j - 1
            bj = smuladd(s, b[k], A[k, j], bj, Val(:N), Val(:N))
        end

        if isintegral(s)
            b[j] = bj
        else
            b[j] = sprod(s, bj, sstar(s, A[j, j]), Val(:N), Val(:N))
        end
    end

    return b
end

function strsx2!(s::AbstractSemiring, ::Val{:R}, ::Val{:N}, ::Val{:L}, A::AbstractMatrix, b::AbstractVector)
    n = size(A, 1)

    @inbounds for j in n:-1:1
        bj = b[j]

        @simd for k in j + 1:n
            bj = smuladd(s, b[k], A[k, j], bj, Val(:N), Val(:N))
        end

        b[j] = bj
    end

    return b
end

function strsx2!(s::AbstractSemiring, ::Val{:R}, trans::Union{Val{:T}, Val{:C}}, ::Val{:L}, A::AbstractMatrix, b::AbstractVector)
    n = size(A, 1)

    @inbounds for k in 1:n
        bk = b[k]

        @simd for j in k + 1:n
            b[j] = smuladd(s, bk, A[j, k], b[j], Val(:N), trans)
        end
    end

    return b
end

function strsx2!(s::AbstractSemiring, ::Val{:R}, trans::Union{Val{:T}, Val{:C}}, ::Val{:U}, A::AbstractMatrix, b::AbstractVector)
    n = size(A, 1)

    @inbounds for k in n:-1:1
        if isintegral(s)
            bk = b[k]
        else
            bk = sprod(s, b[k], sstar(s, A[k, k]), Val(:N), trans)
        end

        b[k] = bk

        @simd for j in 1:k - 1
            b[j] = smuladd(s, bk, A[j, k], b[j], Val(:N), trans)
        end
    end

    return b
end

# ===== strsx! =====

function strsx!(s::AbstractSemiring, side::Val, trans::Val, uplo::Val, A::AbstractMatrix, b::AbstractVector; nt::Integer = nthreads())
    strsx2!(s, side, trans, uplo, A, b)
    return b
end

function strsx!(s::AbstractSemiring, side::Val{SIDE}, trans::Val, uplo::Val, A::AbstractMatrix, B::AbstractMatrix{T}; nt::Integer = nthreads()) where {SIDE, T}
    m = size(B, 1)
    n = size(B, 2)

    if SIDE === :L
        c = n
    else
        c = m
    end

    if nt <= 1 || c <= THRESHOLD
        mr = sgemx_width(T)
        AP = FVector{T}(undef, cld(SGEMX_LEAF, mr) * mr * SGEMX_LEAF)
        BP = FVector{T}(undef, cld(SGEMX_LEAF, SGEMX_NR) * SGEMX_NR * SGEMX_LEAF)
        CP = FVector{T}(undef, mr * SGEMX_NR)
        strsx_st!(s, side, trans, uplo, A, B, AP, BP, CP)
    else
        pool = spool(T, nt)
        strsx_mt!(s, side, trans, uplo, A, B, pool, nt)
    end

    return B
end

# ===== strsx_mt! =====

function strsx_mt!(s::AbstractSemiring, side::Val{SIDE}, trans::Val, uplo::Val, A::AbstractMatrix, B::AbstractMatrix, pool::Channel, w::Integer) where {SIDE}
    m = size(B, 1)
    n = size(B, 2)

    if SIDE === :L
        c = n
    else
        c = m
    end

    if w <= 1 || c <= THRESHOLD
        AP, BP, CP = take!(pool)

        try
            strsx_st!(s, side, trans, uplo, A, B, AP, BP, CP)
        finally
            put!(pool, (AP, BP, CP))
        end
    else
        h = c >> 1

        if SIDE === :L
            B₁ = view(B, 1:m,     1:h)
            B₂ = view(B, 1:m, h + 1:n)
        else
            B₁ = view(B,     1:h, 1:n)
            B₂ = view(B, h + 1:m, 1:n)
        end

        w₁ = w >> 1
        task = @spawn strsx_mt!(s, side, trans, uplo, A, B₁, pool, w₁)
        strsx_mt!(s, side, trans, uplo, A, B₂, pool, w - w₁)
        wait(task)
    end

    return B
end

# ===== strsx_st! =====

function strsx_st!(s::AbstractSemiring, side::Val{SIDE}, trans::Val{TRANS}, uplo::Val{UPLO}, A::AbstractMatrix, B::AbstractMatrix, AP::AbstractVector, BP::AbstractVector, CP::AbstractVector) where {SIDE, TRANS, UPLO}
    n = size(A, 1)

    if n <= THRESHOLD
        strsx2!(s, side, trans, uplo, A, B)
    else
        m = prevpow(2, n) >> 1

        A₁₁ = view(A,     1:m,     1:m)
        A₂₂ = view(A, m + 1:n, m + 1:n)

        if UPLO === :L
            A₂₁ = view(A, m + 1:n, 1:m)
        else
            A₂₁ = view(A, 1:m, m + 1:n)
        end

        if SIDE === :L
            q = size(B, 2)
            B₁ = view(B,     1:m, 1:q)
            B₂ = view(B, m + 1:n, 1:q)
        else
            q = size(B, 1)
            B₁ = view(B, 1:q,     1:m)
            B₂ = view(B, 1:q, m + 1:n)
        end

        if isforward(UPLO, TRANS, SIDE)
            strsx_st!(s, side, trans, uplo, A₁₁, B₁, AP, BP, CP)

            if SIDE === :L
                sgemx_st!(s, trans, Val(:N), B₂, A₂₁, B₁, AP, BP, CP)
            else
                sgemx_st!(s, Val(:N), trans, B₂, B₁, A₂₁, AP, BP, CP)
            end

            strsx_st!(s, side, trans, uplo, A₂₂, B₂, AP, BP, CP)
        else
            strsx_st!(s, side, trans, uplo, A₂₂, B₂, AP, BP, CP)

            if SIDE === :L
                sgemx_st!(s, trans, Val(:N), B₁, A₂₁, B₂, AP, BP, CP)
            else
                sgemx_st!(s, Val(:N), trans, B₁, B₂, A₂₁, AP, BP, CP)
            end

            strsx_st!(s, side, trans, uplo, A₁₁, B₁, AP, BP, CP)
        end
    end

    return B
end
