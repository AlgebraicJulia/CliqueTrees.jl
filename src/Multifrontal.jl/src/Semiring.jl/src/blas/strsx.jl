# ===== strsx2! =====

function strsx2!(s::AbstractSemiring, ::Val{:L}, ::Val{:L}, A::AbstractMatrix, B::AbstractVecOrMat)
    n = size(A, 1)
    m = size(B, 2)

    @inbounds for j in 1:m
        for i in 1:n
            Bij = B[i, j]

            for k in i + 1:n
                B[k, j] = smuladd(s, A[k, i], Bij, B[k, j])
            end
        end
    end

    return B
end

function strsx2!(s::AbstractSemiring, ::Val{:R}, ::Val{:U}, A::AbstractMatrix, B::AbstractMatrix)
    n = size(A, 1)
    m = size(B, 1)

    @inbounds for j in 1:n
        for k in 1:j - 1
            Akj = A[k, j]

            for i in 1:m
                B[i, j] = smuladd(s, B[i, k], Akj, B[i, j])
            end
        end

        if !isintegral(s)
            sAjj = sstar(s, A[j, j])

            for i in 1:m
                B[i, j] = sprod(s, B[i, j], sAjj)
            end
        end
    end

    return B
end

function strsx2!(s::AbstractSemiring, ::Val{:L}, ::Val{:U}, A::AbstractMatrix, B::AbstractVecOrMat)
    n = size(A, 1)
    m = size(B, 2)

    @inbounds for i in 1:m
        for j in n:-1:1
            if isintegral(s)
                Bji = B[j, i]
            else
                Bji = sprod(s, sstar(s, A[j, j]), B[j, i])
            end

            B[j, i] = Bji

            for k in 1:j - 1
                B[k, i] = smuladd(s, A[k, j], Bji, B[k, i])
            end
        end
    end

    return B
end

function strsx2!(s::AbstractSemiring, ::Val{:R}, ::Val{:L}, A::AbstractMatrix, B::AbstractMatrix)
    n = size(A, 1)
    m = size(B, 1)

    @inbounds for j in n:-1:1
        for k in j + 1:n
            Akj = A[k, j]

            for i in 1:m
                B[i, j] = smuladd(s, B[i, k], Akj, B[i, j])
            end
        end
    end

    return B
end

function strsx2!(s::AbstractSemiring, ::Val{:R}, ::Val{:U}, A::AbstractMatrix, b::AbstractVector)
    n = size(A, 1)

    @inbounds for j in 1:n
        bj = b[j]

        for k in 1:j - 1
            bj = smuladd(s, b[k], A[k, j], bj)
        end

        if isintegral(s)
            b[j] = bj
        else
            b[j] = sprod(s, sstar(s, A[j, j]), bj)
        end
    end

    return b
end

function strsx2!(s::AbstractSemiring, ::Val{:R}, ::Val{:L}, A::AbstractMatrix, b::AbstractVector)
    n = size(A, 1)

    @inbounds for j in n:-1:1
        bj = b[j]

        for k in j + 1:n
            bj = smuladd(s, b[k], A[k, j], bj)
        end

        b[j] = bj
    end

    return b
end

# ===== strsx! =====

function strsx!(s::AbstractSemiring, side::Val, uplo::Val, A::AbstractMatrix, b::AbstractVector; nt::Integer = nthreads())
    strsx2!(s, side, uplo, A, b)
    return b
end

function strsx!(s::AbstractSemiring, side::Val{SIDE}, uplo::Val, A::AbstractMatrix, B::AbstractMatrix{T}; nt::Integer = nthreads()) where {SIDE, T}
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
        strsx_st!(s, side, uplo, A, B, AP, BP, CP)
    else
        pool = spool(T, nt)
        strsx_mt!(s, side, uplo, A, B, pool, nt)
    end

    return B
end

# ===== strsx_mt! =====

function strsx_mt!(s::AbstractSemiring, side::Val{SIDE}, uplo::Val, A::AbstractMatrix, B::AbstractMatrix, pool::Channel, w::Integer) where {SIDE}
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
            strsx_st!(s, side, uplo, A, B, AP, BP, CP)
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
        task = @spawn strsx_mt!(s, side, uplo, A, B₁, pool, w₁)
        strsx_mt!(s, side, uplo, A, B₂, pool, w - w₁)
        wait(task)
    end

    return B
end

# ===== strsx_st! =====

function strsx_st!(s::AbstractSemiring, side::Val{SIDE}, uplo::Val{UPLO}, A::AbstractMatrix, B::AbstractMatrix, AP::AbstractVector, BP::AbstractVector, CP::AbstractVector) where {SIDE, UPLO}
    n = size(A, 1)

    if n <= THRESHOLD
        strsx2!(s, side, uplo, A, B)
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

        if isforward(UPLO, :N, SIDE)
            strsx_st!(s, side, uplo, A₁₁, B₁, AP, BP, CP)

            if SIDE === :L
                sgemx_st!(s, B₂, A₂₁, B₁, AP, BP, CP)
            else
                sgemx_st!(s, B₂, B₁, A₂₁, AP, BP, CP)
            end

            strsx_st!(s, side, uplo, A₂₂, B₂, AP, BP, CP)
        else
            strsx_st!(s, side, uplo, A₂₂, B₂, AP, BP, CP)

            if SIDE === :L
                sgemx_st!(s, B₁, A₂₁, B₂, AP, BP, CP)
            else
                sgemx_st!(s, B₁, B₂, A₂₁, AP, BP, CP)
            end

            strsx_st!(s, side, uplo, A₁₁, B₁, AP, BP, CP)
        end
    end

    return B
end
