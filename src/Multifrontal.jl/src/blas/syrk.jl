# ===== syrk! =====

function syrk!(uplo::Val, trans::Val, α, A::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T}
    n = size(C, 1)
    W = Ones{T}(n * n)
    D = Ones{T}(n)
    syrk!(uplo, trans, α, W, A, D, β, C, Val(:N))
    return
end

# ===== syrk2! =====

function syrk2!(::Val{UPLO}, ::Val{:N}, α, A::AbstractMatrix{T}, D::AbstractVector{T}, β, C::AbstractMatrix{T}, ::Val{DIAG}) where {T, UPLO, DIAG}
    m = size(C, 1)
    n = size(A, 2)

    @inbounds @fastmath for j in 1:m
        if UPLO === :L
            rng = j:m
        else
            rng = 1:j
        end

        if iszero(β)
            for i in rng
                C[i, j] = β
            end
        else
            for i in rng
                C[i, j] *= β
            end
        end
    end

    @inbounds @fastmath for k in 1:n
        for j in 1:m
            if UPLO === :L
                rng = j:m
            else
                rng = 1:j
            end

            if DIAG === :N
                Ajk = α * conj(A[j, k])
            else
                Ajk = α * D[k] * conj(A[j, k])
            end

            for i in rng
                C[i, j] += A[i, k] * Ajk
            end
        end
    end

    return
end

function syrk2!(::Val{UPLO}, ::Val, α, A::AbstractMatrix{T}, D::AbstractVector{T}, β, C::AbstractMatrix{T}, ::Val{DIAG}) where {T, UPLO, DIAG}
    m = size(C, 1)
    n = size(A, 1)

    @inbounds @fastmath for j in 1:m
        if UPLO === :L
            rng = j:m
        else
            rng = 1:j
        end

        for i in rng
            Δij = zero(T)

            for k in 1:n
                if DIAG === :N
                    Δij += α * conj(A[k, i]) * A[k, j]
                else
                    Δij += α * D[k] * conj(A[k, i]) * A[k, j]
                end
            end

            if iszero(β)
                C[i, j] = Δij
            else
                C[i, j] = Δij + β * C[i, j]
            end
        end
    end

    return
end

# ===== 9-arg syrk! recursive fallback =====

function syrk!(uplo::Val{UPLO}, trans::Val{TRANS}, α, W::AbstractVector{T}, A::AbstractMatrix{T}, D::AbstractVector{T}, β, C::AbstractMatrix{T}, diag::Val{DIAG}) where {T, TRANS, UPLO, DIAG}
    n = size(C, 1)

    if n <= THRESHOLD
        syrk2!(uplo, trans, α, A, D, β, C, diag)
    else
        m = prevpow(2, n) >> 1

        if TRANS === :N
            A₁ = view(A,     1:m, :)
            A₂ = view(A, m + 1:n, :)
        else
            A₁ = view(A, :,     1:m)
            A₂ = view(A, :, m + 1:n)
        end

        C₁₁ = view(C,     1:m,     1:m)
        C₂₂ = view(C, m + 1:n, m + 1:n)

        if UPLO === :L
            C₂₁ = view(C, m + 1:n, 1:m)

            if TRANS === :N
                gemm!(Val(:N), Val(:T), α, W, A₂, A₁, D, β, C₂₁, diag)
            else
                gemm!(Val(:T), Val(:N), α, W, A₂, A₁, D, β, C₂₁, diag)
            end
        else
            C₁₂ = view(C, 1:m, m + 1:n)

            if TRANS === :N
                gemm!(Val(:N), Val(:T), α, W, A₁, A₂, D, β, C₁₂, diag)
            else
                gemm!(Val(:T), Val(:N), α, W, A₁, A₂, D, β, C₁₂, diag)
            end
        end

        syrk!(uplo, trans, α, W, A₁, D, β, C₁₁, diag)
        syrk!(uplo, trans, α, W, A₂, D, β, C₂₂, diag)
    end

    return
end

function syrk!(uplo::Val, trans::Val{TRANS}, α, ::AbstractVector{T}, A::AbstractMatrix{T}, ::AbstractVector{T}, β, C::AbstractMatrix{T}, ::Val{:N}) where {T <: BlasFloat, TRANS}
    if T <: Complex
        BLAS.herk!(char(uplo), char(trans), convert(real(T), α), A, convert(real(T), β), C)
    else
        BLAS.syrk!(char(uplo), char(trans), convert(T, α), A, convert(T, β), C)
    end

    return
end

function syrk!(uplo::Val, trans::Val{TRANS}, α, W::AbstractVector{T}, A::AbstractMatrix{T}, D::AbstractVector{T}, β, C::AbstractMatrix{T}, ::Val{:U}) where {T <: BlasFloat, TRANS}
    B = reshape(view(W, 1:length(A)), size(A))
    copyrec!(B, A)

    if TRANS === :N
        cmul!(Val(:R), Val(:U), B, D)
    else
        cmul!(Val(:L), Val(:U), B, D)
    end

    trrk!(uplo, trans, α, B, A, β, C)
    return
end
