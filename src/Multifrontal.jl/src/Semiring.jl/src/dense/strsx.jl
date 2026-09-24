const STRSX_WORK = 8192
const STRSX_CH = 16

# ===== strsx! =====

function strsx!(s::AbstractSemiring, side::Val, trans::Val, uplo::Val, diag::Val, A::AbstractMatrix, b::AbstractVector; nt::Integer = nthreads())
    strsx2!(s, side, trans, uplo, diag, A, b)
    return b
end

function strsx!(s::AbstractSemiring, side::Val{SIDE}, trans::Val, uplo::Val, diag::Val, A::AbstractMatrix, B::AbstractMatrix{T}; nt::Integer = nthreads()) where {SIDE, T}
    m = size(B, 1)
    n = size(B, 2)

    if SIDE === :L
        c = n
        d = m
    else
        c = m
        d = n
    end

    if nt <= 1 || c <= THRESHOLD || size(A, 1) * c < STRSX_WORK
        AP, BP, CP = spool_st(T, m, d, n)
        strsx_st!(s, side, trans, uplo, diag, A, B, AP, BP, CP)
    else
        pool = spool_mt(T, nt, m, d, n)
        strsx_mt!(s, side, trans, uplo, diag, A, B, pool, nt)
    end

    return B
end

# ===== strsx_mt! =====

function strsx_mt!(s::AbstractSemiring, side::Val{SIDE}, trans::Val, uplo::Val, diag::Val, A::AbstractMatrix, B::AbstractMatrix, pool::Channel, nt::Integer) where {SIDE}
    m = size(B, 1)
    n = size(B, 2)

    if SIDE === :L
        c = n
    else
        c = m
    end

    if nt <= 1 || c <= THRESHOLD || size(A, 1) * c < STRSX_WORK
        AP, BP, CP = take!(pool)

        try
            strsx_st!(s, side, trans, uplo, diag, A, B, AP, BP, CP)
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

        nt₁ = nt >> 1
        task = @spawn strsx_mt!(s, side, trans, uplo, diag, A, B₁, pool, nt₁)
        strsx_mt!(s, side, trans, uplo, diag, A, B₂, pool, nt - nt₁)
        wait(task)
    end

    return B
end

# ===== strsx_st! =====

function strsx_st!(s::AbstractSemiring, side::Val{SIDE}, trans::Val{TRANS}, uplo::Val{UPLO}, diag::Val, A::AbstractMatrix, B::AbstractMatrix, AP::AbstractVector, BP::AbstractVector, CP::AbstractVector) where {SIDE, TRANS, UPLO}
    n = size(A, 1)

    if n <= THRESHOLD
        strsx2!(s, side, trans, uplo, diag, A, B)
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
            strsx_st!(s, side, trans, uplo, diag, A₁₁, B₁, AP, BP, CP)

            if SIDE === :L
                sgemx_st!(s, trans, Val(:N), B₂, A₂₁, B₁, AP, BP, CP)
            else
                sgemx_st!(s, Val(:N), trans, B₂, B₁, A₂₁, AP, BP, CP)
            end

            strsx_st!(s, side, trans, uplo, diag, A₂₂, B₂, AP, BP, CP)
        else
            strsx_st!(s, side, trans, uplo, diag, A₂₂, B₂, AP, BP, CP)

            if SIDE === :L
                sgemx_st!(s, trans, Val(:N), B₁, A₂₁, B₂, AP, BP, CP)
            else
                sgemx_st!(s, Val(:N), trans, B₁, B₂, A₂₁, AP, BP, CP)
            end

            strsx_st!(s, side, trans, uplo, diag, A₁₁, B₁, AP, BP, CP)
        end
    end

    return B
end

# ===== strsx2! =====

function strsx2!(s::AbstractSemiring, ::Val{:L}, trans::N_OR_R, ::Val{:L}, ::Val{DIAG}, A::AbstractMatrix{T}, B::AbstractVecOrMat) where {T, DIAG}
    n = size(A, 1)
    m = size(B, 2)

    Z = sizeof(T)
    sA = stride(A, 2)
    sB = stride(B, 2)

    @preserve A B begin
        pA = pointer(A)
        pB = pointer(B)

        if m == 1 || !(DIAG === :N && !isintegral(s))
            @inbounds for j in 1:m
                pBj = pB + (j - 1) * sB * Z

                for i in 1:n
                    if DIAG === :N && !isintegral(s)
                        B[i, j] = sprod(s, sstar(s, A[i, i]), B[i, j], trans, Val(:N))
                    end

                    Bij = B[i, j]
                    saxpy_kern!(s, trans, Val(:N), Val(:R), pBj + i * Z, pA + ((i - 1) * sA + i) * Z, Bij, n - i)
                end
            end

            return B
        end

        z = szero(s, T, Val(:N))

        @inbounds for cstrt in 1:STRSX_CH:n
            cstop = min(cstrt + STRSX_CH - 1, n)
            csize = cstop - cstrt + 1

            stars = ntuple(Val(STRSX_CH)) do t
                if t <= csize
                    sstar(s, A[cstrt + t - 1, cstrt + t - 1])
                else
                    z
                end
            end

            for j in 1:m
                pBj = pB + (j - 1) * sB * Z

                for i in cstrt:cstop
                    Bij = B[i, j] = sprod(s, stars[i - cstrt + 1], B[i, j], trans, Val(:N))
                    saxpy_kern!(s, trans, Val(:N), Val(:R), pBj + i * Z, pA + ((i - 1) * sA + i) * Z, Bij, n - i)
                end
            end
        end
    end

    return B
end

function strsx2!(s::AbstractSemiring, ::Val{:L}, trans::N_OR_R, ::Val{:U}, ::Val{DIAG}, A::AbstractMatrix{T}, B::AbstractVecOrMat) where {T, DIAG}
    n = size(A, 1)
    m = size(B, 2)

    Z = sizeof(T)
    sA = stride(A, 2)
    sB = stride(B, 2)

    @preserve A B begin
        pA = pointer(A)
        pB = pointer(B)

        if m == 1 || !(DIAG === :N && !isintegral(s))
            @inbounds for i in 1:m
                pBi = pB + (i - 1) * sB * Z

                for j in n:-1:1
                    if DIAG === :N && !isintegral(s)
                        Bji = B[j, i] = sprod(s, sstar(s, A[j, j]), B[j, i], trans, Val(:N))
                    else
                        Bji = B[j, i]
                    end

                    saxpy_kern!(s, trans, Val(:N), Val(:R), pBi, pA + (j - 1) * sA * Z, Bji, j - 1)
                end
            end

            return B
        end

        z = szero(s, T, Val(:N))

        @inbounds for cstop in n:-STRSX_CH:1
            csize = min(STRSX_CH, cstop)
            cstrt = cstop - csize + 1

            stars = ntuple(Val(STRSX_CH)) do t
                if t <= csize
                    sstar(s, A[cstrt + t - 1, cstrt + t - 1])
                else
                    z
                end
            end

            for i in 1:m
                pBi = pB + (i - 1) * sB * Z

                for j in cstop:-1:cstrt
                    Bji = B[j, i] = sprod(s, stars[j - cstrt + 1], B[j, i], trans, Val(:N))
                    saxpy_kern!(s, trans, Val(:N), Val(:R), pBi, pA + (j - 1) * sA * Z, Bji, j - 1)
                end
            end
        end
    end

    return B
end

function strsx2!(s::AbstractSemiring, ::Val{:L}, trans::T_OR_C, ::Val{:L}, ::Val{DIAG}, A::AbstractMatrix{T}, B::AbstractVecOrMat) where {T, DIAG}
    n = size(A, 1)
    m = size(B, 2)

    Z = sizeof(T)
    sA = stride(A, 2)
    sB = stride(B, 2)

    op = compose(trans, Val(:N))

    @preserve A B begin
        pA = pointer(A)
        pB = pointer(B)

        if m == 1 || !(DIAG === :N && !isintegral(s))
            @inbounds for j in 1:m
                pBj = pB + (j - 1) * sB * Z

                for k in n:-1:1
                    Bkj = splus(s, B[k, j], sdot_kern!(s, trans, Val(:N), op, pA + ((k - 1) * sA + k) * Z, pBj + k * Z, n - k), trans)

                    if DIAG === :N && !isintegral(s)
                        B[k, j] = sprod(s, sstar(s, A[k, k]), Bkj, trans, Val(:N))
                    else
                        B[k, j] = Bkj
                    end
                end
            end

            return B
        end

        z = szero(s, T, Val(:N))

        @inbounds for cstop in n:-STRSX_CH:1
            csize = min(STRSX_CH, cstop)
            cstrt = cstop - csize + 1

            stars = ntuple(Val(STRSX_CH)) do t
                if t <= csize
                    sstar(s, A[cstrt + t - 1, cstrt + t - 1])
                else
                    z
                end
            end

            for j in 1:m
                pBj = pB + (j - 1) * sB * Z

                for k in cstop:-1:cstrt
                    Bkj = splus(s, B[k, j], sdot_kern!(s, trans, Val(:N), op, pA + ((k - 1) * sA + k) * Z, pBj + k * Z, n - k), trans)
                    B[k, j] = sprod(s, stars[k - cstrt + 1], Bkj, trans, Val(:N))
                end
            end
        end
    end

    return B
end

function strsx2!(s::AbstractSemiring, ::Val{:L}, trans::T_OR_C, ::Val{:U}, ::Val{DIAG}, A::AbstractMatrix{T}, B::AbstractVecOrMat) where {T, DIAG}
    n = size(A, 1)
    m = size(B, 2)

    Z = sizeof(T)
    sA = stride(A, 2)
    sB = stride(B, 2)

    op = compose(trans, Val(:N))

    @preserve A B begin
        pA = pointer(A)
        pB = pointer(B)

        if m == 1 || !(DIAG === :N && !isintegral(s))
            @inbounds for j in 1:m
                pBj = pB + (j - 1) * sB * Z

                for k in 1:n
                    Bkj = splus(s, B[k, j], sdot_kern!(s, trans, Val(:N), op, pA + (k - 1) * sA * Z, pBj, k - 1), trans)

                    if DIAG === :N && !isintegral(s)
                        B[k, j] = sprod(s, sstar(s, A[k, k]), Bkj, trans, Val(:N))
                    else
                        B[k, j] = Bkj
                    end
                end
            end

            return B
        end

        z = szero(s, T, Val(:N))

        @inbounds for cstrt in 1:STRSX_CH:n
            cstop = min(cstrt + STRSX_CH - 1, n)
            csize = cstop - cstrt + 1

            stars = ntuple(Val(STRSX_CH)) do t
                if t <= csize
                    sstar(s, A[cstrt + t - 1, cstrt + t - 1])
                else
                    z
                end
            end

            for j in 1:m
                pBj = pB + (j - 1) * sB * Z

                for k in cstrt:cstop
                    Bkj = splus(s, B[k, j], sdot_kern!(s, trans, Val(:N), op, pA + (k - 1) * sA * Z, pBj, k - 1), trans)
                    B[k, j] = sprod(s, stars[k - cstrt + 1], Bkj, trans, Val(:N))
                end
            end
        end
    end

    return B
end

function strsx2!(s::AbstractSemiring, ::Val{:R}, trans::N_OR_R, ::Val{:U}, ::Val{DIAG}, A::AbstractMatrix{T}, B::AbstractMatrix) where {T, DIAG}
    n = size(A, 1)
    m = size(B, 1)

    Z = sizeof(T)
    sB = stride(B, 2)

    @preserve B begin
        pB = pointer(B)

        @inbounds for j in 1:n
            pBj = pB + (j - 1) * sB * Z

            for k in 1:j - 1
                saxpy_kern!(s, Val(:N), trans, Val(:R), pBj, pB + (k - 1) * sB * Z, A[k, j], m)
            end

            if DIAG === :N && !isintegral(s)
                sAjj = sstar(s, A[j, j])

                for i in 1:m
                    B[i, j] = sprod(s, B[i, j], sAjj, Val(:N), trans)
                end
            end
        end
    end

    return B
end

function strsx2!(s::AbstractSemiring, ::Val{:R}, trans::N_OR_R, ::Val{:L}, ::Val{DIAG}, A::AbstractMatrix{T}, B::AbstractMatrix) where {T, DIAG}
    n = size(A, 1)
    m = size(B, 1)

    Z = sizeof(T)
    sB = stride(B, 2)

    @preserve B begin
        pB = pointer(B)

        @inbounds for j in n:-1:1
            pBj = pB + (j - 1) * sB * Z

            for k in j + 1:n
                saxpy_kern!(s, Val(:N), trans, Val(:R), pBj, pB + (k - 1) * sB * Z, A[k, j], m)
            end

            if DIAG === :N && !isintegral(s)
                sAjj = sstar(s, A[j, j])

                for i in 1:m
                    B[i, j] = sprod(s, B[i, j], sAjj, Val(:N), trans)
                end
            end
        end
    end

    return B
end

function strsx2!(s::AbstractSemiring, ::Val{:R}, trans::T_OR_C, ::Val{:L}, ::Val{DIAG}, A::AbstractMatrix{T}, B::AbstractMatrix) where {T, DIAG}
    n = size(A, 1)
    m = size(B, 1)

    Z = sizeof(T)
    sB = stride(B, 2)

    @preserve B begin
        pB = pointer(B)

        @inbounds for k in 1:n
            pBk = pB + (k - 1) * sB * Z

            if DIAG === :N && !isintegral(s)
                sAkk = sstar(s, A[k, k])

                for i in 1:m
                    B[i, k] = sprod(s, B[i, k], sAkk, Val(:N), trans)
                end
            end

            for j in k + 1:n
                saxpy_kern!(s, Val(:N), trans, Val(:R), pB + (j - 1) * sB * Z, pBk, A[j, k], m)
            end
        end
    end

    return B
end

function strsx2!(s::AbstractSemiring, ::Val{:R}, trans::T_OR_C, ::Val{:U}, ::Val{DIAG}, A::AbstractMatrix{T}, B::AbstractMatrix) where {T, DIAG}
    n = size(A, 1)
    m = size(B, 1)

    Z = sizeof(T)
    sB = stride(B, 2)

    @preserve B begin
        pB = pointer(B)

        @inbounds for k in n:-1:1
            pBk = pB + (k - 1) * sB * Z

            if DIAG === :N && !isintegral(s)
                sAkk = sstar(s, A[k, k])

                for i in 1:m
                    B[i, k] = sprod(s, B[i, k], sAkk, Val(:N), trans)
                end
            end

            for j in 1:k - 1
                saxpy_kern!(s, Val(:N), trans, Val(:R), pB + (j - 1) * sB * Z, pBk, A[j, k], m)
            end
        end
    end

    return B
end

function strsx2!(s::AbstractSemiring, ::Val{:R}, trans::N_OR_R, ::Val{:U}, ::Val{DIAG}, A::AbstractMatrix{T}, b::AbstractVector) where {T, DIAG}
    n = size(A, 1)

    Z = sizeof(T)
    sA = stride(A, 2)

    op = compose(Val(:N), trans)

    @preserve A b begin
        pA = pointer(A)
        pb = pointer(b)

        @inbounds for j in 1:n
            bj = splus(s, b[j], sdot_kern!(s, Val(:N), trans, op, pb, pA + (j - 1) * sA * Z, j - 1), op)

            if DIAG === :N && !isintegral(s)
                b[j] = sprod(s, bj, sstar(s, A[j, j]), Val(:N), trans)
            else
                b[j] = bj
            end
        end
    end

    return b
end

function strsx2!(s::AbstractSemiring, ::Val{:R}, trans::N_OR_R, ::Val{:L}, ::Val{DIAG}, A::AbstractMatrix{T}, b::AbstractVector) where {T, DIAG}
    n = size(A, 1)

    Z = sizeof(T)
    sA = stride(A, 2)

    op = compose(Val(:N), trans)

    @preserve A b begin
        pA = pointer(A)
        pb = pointer(b)

        @inbounds for j in n:-1:1
            bj = splus(s, b[j], sdot_kern!(s, Val(:N), trans, op, pb + j * Z, pA + ((j - 1) * sA + j) * Z, n - j), op)

            if DIAG === :N && !isintegral(s)
                b[j] = sprod(s, bj, sstar(s, A[j, j]), Val(:N), trans)
            else
                b[j] = bj
            end
        end
    end

    return b
end

function strsx2!(s::AbstractSemiring, ::Val{:R}, trans::T_OR_C, ::Val{:L}, ::Val{DIAG}, A::AbstractMatrix{T}, b::AbstractVector) where {T, DIAG}
    n = size(A, 1)

    Z = sizeof(T)
    sA = stride(A, 2)

    @preserve A b begin
        pA = pointer(A)
        pb = pointer(b)

        @inbounds for k in 1:n
            if DIAG === :N && !isintegral(s)
                bk = b[k] = sprod(s, b[k], sstar(s, A[k, k]), Val(:N), trans)
            else
                bk = b[k]
            end

            saxpy_kern!(s, Val(:N), trans, Val(:L), pb + k * Z, pA + ((k - 1) * sA + k) * Z, bk, n - k)
        end
    end

    return b
end

function strsx2!(s::AbstractSemiring, ::Val{:R}, trans::T_OR_C, ::Val{:U}, ::Val{DIAG}, A::AbstractMatrix{T}, b::AbstractVector) where {T, DIAG}
    n = size(A, 1)

    Z = sizeof(T)
    sA = stride(A, 2)

    @preserve A b begin
        pA = pointer(A)
        pb = pointer(b)

        @inbounds for k in n:-1:1
            if DIAG === :N && !isintegral(s)
                bk = b[k] = sprod(s, b[k], sstar(s, A[k, k]), Val(:N), trans)
            else
                bk = b[k]
            end

            saxpy_kern!(s, Val(:N), trans, Val(:L), pb, pA + (k - 1) * sA * Z, bk, k - 1)
        end
    end

    return b
end
