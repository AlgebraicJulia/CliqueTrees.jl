function qdtrf2!(::Val{:L}, A::AbstractMatrix{T}, D::AbstractVector, R::Union{DynamicRegularization, Nothing}) where {T}
    @inbounds @fastmath for j in axes(A, 1)
        Ajj = real(A[j, j])

        if !isnothing(R)
            if R.signs[j] * Ajj < R.epsilon
                Ajj = R.delta * R.signs[j]
            end
        end

        if iszero(Ajj)
            return j
        else
            D[j] = Ajj; iDjj = inv(Ajj)

            for i in j + 1:size(A, 1)
                A[i, j] *= iDjj
            end

            for k in j + 1:size(A, 1)
                Akj = A[k, j]; cAkj = Ajj * conj(Akj)

                A[k, k] -= Ajj * abs2(Akj)

                for i in k + 1:size(A, 1)
                    A[i, k] -= A[i, j] * cAkj
                end
            end
        end
    end

    return 0
end

function qdtrf2!(::Val{:U}, A::AbstractMatrix{T}, D::AbstractVector, R::Union{DynamicRegularization, Nothing}) where {T}
    @inbounds @fastmath for j in axes(A, 1)
        Ajj = real(A[j, j])

        for k in 1:j - 1
            Ajj -= abs2(A[k, j]) * D[k]
        end

        if !isnothing(R)
            if R.signs[j] * Ajj < R.epsilon
                Ajj = R.delta * R.signs[j]
            end
        end

        if iszero(Ajj)
            return j
        else
            D[j] = Ajj; iDjj = inv(Ajj)

            for i in j + 1:size(A, 1)
                for k in 1:j - 1
                    A[j, i] -= A[k, i] * D[k] * conj(A[k, j])
                end

                A[j, i] *= iDjj
            end
        end
    end

    return 0
end

function qdtrf!(uplo::Val{UPLO}, W::AbstractMatrix{T}, A::AbstractMatrix{T}, D::AbstractVector, R::Union{DynamicRegularization, Nothing}) where {T, UPLO}
    n = size(A, 1)
    n <= THRESHOLD && return qdtrf2!(uplo, A, D, R)

    n₁  = 2^floor(Int, log2(n)) ÷ 2
    A₁₁ = view(A, 1:n₁, 1:n₁)
    D₁₁ = view(D, 1:n₁)
    A₂₂ = view(A, n₁+1:n, n₁+1:n)
    D₂₂ = view(D, n₁+1:n)

    if isnothing(R)
        R₁ = nothing
        R₂ = nothing
    else
        R₁ = view(R, 1:n₁)
        R₂ = view(R, n₁+1:n)
    end
    #
    # factorize A₁₁
    #
    info = qdtrf!(uplo, W, A₁₁, D₁₁, R₁)
    !iszero(info) && return info

    if UPLO === :L
        A₂₁ = view(A, n₁+1:n, 1:n₁)
        W₂₁ = view(W, 1:n-n₁, 1:n₁)
        #
        # A₂₁ ← A₂₁ L₁₁⁻ᴴ
        #
        trsm!(Val(:R), Val(:L), Val(:C), Val(:U), one(T), A₁₁, A₂₁)
        #
        # W₂₁ ← A₂₁
        #
        copyrec!(W₂₁, A₂₁)
        #
        # A₂₁ ← A₂₁ D₁₁⁻¹
        #
        @inbounds for j in axes(A₂₁, 2)
            iD₁₁ = inv(D₁₁[j])

            for i in axes(A₂₁, 1)
                A₂₁[i, j] *= iD₁₁
            end
        end
        #
        # A₂₂ ← A₂₂ - W₂₁ A₂₁ᴴ
        #
        trrk!(uplo, Val(:N), W₂₁, A₂₁, A₂₂)
    else
        A₁₂ = view(A, 1:n₁, n₁+1:n)
        W₁₂ = view(W, 1:n₁, 1:n-n₁)
        #
        # A₁₂ ← U₁₁⁻ᴴ A₁₂
        #
        trsm!(Val(:L), Val(:U), Val(:C), Val(:U), one(T), A₁₁, A₁₂)
        #
        # W₁₂ ← A₁₂
        #
        copyrec!(W₁₂, A₁₂)
        #
        # A₁₂ ← D₁₁⁻¹ A₁₂
        #
        @inbounds for j in axes(A₁₂, 2)
            for i in axes(A₁₂, 1)
                A₁₂[i, j] *= inv(D₁₁[i])
            end
        end
        #
        # A₂₂ ← A₂₂ - W₁₂ᴴ A₁₂
        #
        trrk!(uplo, Val(:C), W₁₂, A₁₂, A₂₂)
    end

    #
    # factorize A₂₂
    #
    info = qdtrf!(uplo, W, A₂₂, D₂₂, R₂)
    !iszero(info) && return n₁ + info
    return 0
end

function trrk!(uplo::Val, ::Val{:N}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}) where {T <: BlasFloat}
    if T <: Complex
        BLAS.her2k!(char(uplo), 'N', convert(T, -1/2), A, B, one(real(T)), C)
    else
        BLAS.syr2k!(char(uplo), 'N', convert(T, -1/2), A, B, one(T), C)
    end

    return
end

function trrk!(uplo::Val, ::Val{:C}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}) where {T <: BlasFloat}
    if T <: Complex
        BLAS.her2k!(char(uplo), 'C', convert(T, -1/2), A, B, one(real(T)), C)
    else
        BLAS.syr2k!(char(uplo), 'C', convert(T, -1/2), A, B, one(T), C)
    end
    return
end

function trrk!(uplo::Val{UPLO}, tA::Val{TRANS}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}) where {T, UPLO, TRANS}
    @inbounds for j in axes(C, 2)
        if UPLO === :L
            rows = j:size(C, 1)
        else
            rows = 1:j
        end

        if TRANS === :N
            for k in axes(A, 2)
                Bjk = conj(B[j, k])

                for i in rows
                    C[i, j] -= A[i, k] * Bjk
                end
            end
        else
            for k in axes(A, 1)
                Bkj = B[k, j]

                for i in rows
                    C[i, j] -= conj(A[k, i]) * Bkj
                end
            end
        end
    end

    return
end

function gemm!(tA::Val, tB::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.gemm!(char(tA), char(tB), α, A, B, β, C)
    return
end

function gemm!(tA::Val, tB::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) where {T}
    mul!(C, adj(tA, A), adj(tB, B), α, β)
    return
end

function gemv!(tA::Val, α::T, A::AbstractMatrix{T}, b::AbstractVector{T}, β::T, c::AbstractVector{T}) where {T <: BlasFloat}
    BLAS.gemv!(char(tA), α, A, b, β, c)
    return
end

function gemv!(tA::Val, α::T, A::AbstractMatrix{T}, b::AbstractVector{T}, β::T, c::AbstractVector{T}) where {T}
    mul!(c, adj(tA, A), b, α, β)
    return
end

function trsm!(::Val{:L}, uplo::Val, tA::Val, diag::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.trsm!('L', char(uplo), char(tA), char(diag), α, A, B)
    return
end

function trsm!(::Val{:R}, uplo::Val, tA::Val, diag::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.trsm!('R', char(uplo), char(tA), char(diag), α, A, B)
    return
end

function trsm!(::Val{:L}, uplo::Val, tA::Val, diag::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    ldiv!(adj(tA, tri(uplo, diag, A)), B)
    lmul!(α, B)
    return
end

function trsm!(::Val{:R}, uplo::Val, tA::Val, diag::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    rdiv!(B, adj(tA, tri(uplo, diag, A)))
    lmul!(α, B)
    return
end

function trsv!(uplo::Val, tA::Val, diag::Val, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T <: BlasFloat}
    BLAS.trsv!(char(uplo), char(tA), char(diag), A, b)
    return
end

function trsv!(uplo::Val, tA::Val, diag::Val, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    ldiv!(adj(tA, tri(uplo, diag, A)), b)
    return
end

function trmm!(::Val{:L}, uplo::Val, tA::Val, diag::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.trmm!('L', char(uplo), char(tA), char(diag), α, A, B)
    return
end

function trmm!(::Val{:R}, uplo::Val, tA::Val, diag::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.trmm!('R', char(uplo), char(tA), char(diag), α, A, B)
    return
end

function trmm!(::Val{:L}, uplo::Val, tA::Val, diag::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    lmul!(adj(tA, tri(uplo, diag, A)), B)
    lmul!(α, B)
    return
end

function trmm!(::Val{:R}, uplo::Val, tA::Val, diag::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    rmul!(B, adj(tA, tri(uplo, diag, A)))
    lmul!(α, B)
    return
end

function trmv!(uplo::Val, tA::Val, diag::Val, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T <: BlasFloat}
    BLAS.trmv!(char(uplo), char(tA), char(diag), A, b)
    return
end

function trmv!(uplo::Val, tA::Val, diag::Val, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    lmul!(adj(tA, tri(uplo, diag, A)), b)
    return
end

function syrk!(uplo::Val, ::Val{:N}, α, A::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T <: BlasFloat}
    if T <: Complex
        BLAS.herk!(char(uplo), 'N', real(α), A, real(β), C)
    else
        BLAS.syrk!(char(uplo), 'N', α, A, β, C)
    end
    return
end

function syrk!(uplo::Val, ::Val{:C}, α, A::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T <: BlasFloat}
    if T <: Complex
        BLAS.herk!(char(uplo), 'C', real(α), A, real(β), C)
    else
        BLAS.syrk!(char(uplo), 'C', α, A, β, C)
    end
    return
end

function syrk!(uplo::Val{UPLO}, ::Val{:N}, α, A::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T, UPLO}
    m = size(C, 1)

    @inbounds for j in axes(C, 2)
        if UPLO === :L
            rows = j:m
        else
            rows = 1:j
        end

        for i in rows
            C[i, j] *= β
        end

        for k in axes(A, 2)
            Ajk = α * conj(A[j, k])

            for i in rows
                C[i, j] += A[i, k] * Ajk
            end
        end
    end

    return
end

function syrk!(uplo::Val{UPLO}, ::Val{:C}, α, A::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T, UPLO}
    m = size(C, 1)

    @inbounds for j in axes(C, 2)
        if UPLO === :L
            rows = j:m
        else
            rows = 1:j
        end

        for i in rows
            C[i, j] *= β
        end

        for k in axes(A, 1)
            Akj = α * conj(A[k, j])

            for i in rows
                C[i, j] += conj(A[k, i]) * Akj
            end
        end
    end

    return
end

function symm!(::Val{:L}, uplo::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) where {T <: BlasFloat}
    if T <: Complex
        BLAS.hemm!('L', char(uplo), α, A, B, β, C)
    else
        BLAS.symm!('L', char(uplo), α, A, B, β, C)
    end
    return
end

function symm!(::Val{:R}, uplo::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) where {T <: BlasFloat}
    if T <: Complex
        BLAS.hemm!('R', char(uplo), α, A, B, β, C)
    else
        BLAS.symm!('R', char(uplo), α, A, B, β, C)
    end
    return
end

function symm!(::Val{:L}, uplo::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) where {T}
    mul!(C, sym(uplo, A), B, α, β)
    return
end

function symm!(::Val{:R}, uplo::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) where {T}
    mul!(C, B, sym(uplo, A), α, β)
    return
end

function potrf!(uplo::Val, A::AbstractMatrix{T}) where {T <: BlasFloat}
    _, info = LAPACK.potrf!(char(uplo), A)
    return info
end

function potrf!(uplo::Val, A::AbstractMatrix{T}) where {T}
    F = cholesky!(sym(uplo, A))
    return F.info
end

for (pstrf, T, R) in
    ((:dpstrf_, :Float64,    :Float64),
     (:spstrf_, :Float32,    :Float32),
     (:zpstrf_, :ComplexF64, :Float64),
     (:cpstrf_, :ComplexF32, :Float32))
    @eval begin
        function pstrf!(uplo::AbstractChar, A::AbstractMatrix{$T}, piv::AbstractVector{BlasInt}, work::AbstractVector{$R}, tol::Real)
            n = checksquare(A)
            @assert length(piv) >= n
            @assert length(work) >= 2n
            require_one_based_indexing(A, piv, work)
            LAPACK.chkuplo(uplo)
            chkstride1(A)

            lda = max(1, stride(A, 2))
            rank = Ref{BlasInt}()
            info = Ref{BlasInt}()

            ccall((BLAS.@blasfunc($pstrf), BLAS.libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{BlasInt},
                   Ptr{BlasInt}, Ref{$R}, Ptr{$R}, Ref{BlasInt}, Clong),
                  uplo, n, A, lda, piv, rank, tol, work, info, 1)

            LAPACK.chkargsok(info[])
            return info[], rank[]
        end
    end
end

function pstrf!(uplo::Val, A::AbstractMatrix{T}, piv::AbstractVector{BlasInt}, work::AbstractVector{<:Real}, tol::Real) where {T <: BlasFloat}
    return pstrf!(char(uplo), A, piv, work, tol)
end

function pstrf!(uplo::Val, A::AbstractMatrix{T}, piv::AbstractVector{BlasInt}, work::AbstractVector{<:Real}, tol::Real) where {T}
    F = cholesky!(sym(uplo, A), RowMaximum(); tol=tol, check=false)
    copyto!(piv, F.piv)
    return F.info, F.rank
end

function trtri!(uplo::Val, diag::Val, A::AbstractMatrix{T}) where {T <: BlasFloat}
    LAPACK.trtri!(char(uplo), char(diag), A)
    return
end

function trtri!(uplo::Val, diag::Val, A::AbstractMatrix{T}) where {T}
    copyto!(A, inv(tri(uplo, diag, A)))
    return
end

function ger!(α, x::AbstractVector{T}, y::AbstractVector{T}, A::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.ger!(convert(T, α), x, y, A)
    return
end

function ger!(α, x::AbstractVector{T}, y::AbstractVector{T}, A::AbstractMatrix{T}) where {T}
    @inbounds for j in axes(A, 2)
        αyj = α * conj(y[j])

        for i in axes(A, 1)
            A[i, j] += x[i] * αyj
        end
    end

    return
end

function syr!(uplo::Val, α, x::AbstractVector{T}, A::AbstractMatrix{T}) where {T <: BlasFloat}
    if T <: Complex
        BLAS.her!(char(uplo), real(α), x, A)
    else
        BLAS.syr!(char(uplo), α, x, A)
    end

    return
end

function syr!(::Val{UPLO}, α, x::AbstractVector{T}, A::AbstractMatrix{T}) where {UPLO, T}
    @inbounds if UPLO === :L
        for k in axes(A, 1)
            αxk = α * conj(x[k])

            A[k, k] += x[k] * αxk

            for i in k + 1:size(A, 1)
                A[i, k] += x[i] * αxk
            end
        end
    else
        for j in axes(A, 2)
            αxj = α * conj(x[j])

            for i in 1:j
                A[i, j] += x[i] * αxj
            end
        end
    end

    return
end

