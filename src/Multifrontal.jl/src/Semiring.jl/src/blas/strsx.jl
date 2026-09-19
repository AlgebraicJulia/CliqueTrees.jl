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

function strsx2!(s::S, ::Val{:R}, ::Val{:U}, A::AbstractMatrix, B::AbstractMatrix) where {S <: AbstractSemiring}
    n = size(A, 1)
    m = size(B, 1)

    @inbounds for j in 1:n
        for k in 1:j - 1
            Akj = A[k, j]

            for i in 1:m
                B[i, j] = smuladd(s, B[i, k], Akj, B[i, j])
            end
        end

        if !(S <: IntegralQuantale)
            sAjj = sstar(s, A[j, j])

            for i in 1:m
                B[i, j] = sprod(s, B[i, j], sAjj)
            end
        end
    end

    return B
end

function strsx2!(s::S, ::Val{:L}, ::Val{:U}, A::AbstractMatrix, B::AbstractVecOrMat) where {S <: AbstractSemiring}
    n = size(A, 1)
    m = size(B, 2)

    @inbounds for i in 1:m
        for j in n:-1:1
            if S <: IntegralQuantale
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

function strsx2!(s::S, ::Val{:R}, ::Val{:U}, A::AbstractMatrix, b::AbstractVector) where {S <: AbstractSemiring}
    n = size(A, 1)

    @inbounds for j in 1:n
        bj = b[j]

        for k in 1:j - 1
            bj = smuladd(s, b[k], A[k, j], bj)
        end

        if S <: IntegralQuantale
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

function strsx!(s::AbstractSemiring, side::Val{S}, uplo::Val, A::AbstractMatrix, B::AbstractMatrix; nt::Integer = nthreads()) where {S}
    m = size(B, 1)
    n = size(B, 2)

    if S === :L
        c = n
    else
        c = m
    end

    if nt <= 1 || c <= THRESHOLD
        strsx_impl!(s, side, uplo, A, B; nt)
    else
        h = c >> 1

        if S === :L
            B₁ = view(B, 1:m,     1:h)
            B₂ = view(B, 1:m, h + 1:n)
        else
            B₁ = view(B,     1:h, 1:n)
            B₂ = view(B, h + 1:m, 1:n)
        end

        nt₁ = nt >> 1

        task = @spawn strsx!(s, side, uplo, A, B₁; nt = nt₁)
        strsx!(s, side, uplo, A, B₂; nt = nt - nt₁)
        wait(task)
    end

    return B
end

# ===== strsx_impl! =====

function strsx_impl!(s::AbstractSemiring, side::Val{SIDE}, uplo::Val{UPLO}, A::AbstractMatrix, B::AbstractMatrix; nt::Integer = nthreads()) where {SIDE, UPLO}
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
            strsx_impl!(s, side, uplo, A₁₁, B₁; nt)

            if SIDE === :L
                sgemx!(s, B₂, A₂₁, B₁; nt)
            else
                sgemx!(s, B₂, B₁, A₂₁; nt)
            end

            strsx_impl!(s, side, uplo, A₂₂, B₂; nt)
        else
            strsx_impl!(s, side, uplo, A₂₂, B₂; nt)

            if SIDE === :L
                sgemx!(s, B₁, A₂₁, B₂; nt)
            else
                sgemx!(s, B₁, B₂, A₂₁; nt)
            end

            strsx_impl!(s, side, uplo, A₁₁, B₁; nt)
        end
    end

    return B
end
