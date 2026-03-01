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

# ===== pstrf2! =====

function pstrf2!(::Val{:L}, A::AbstractMatrix{T}, D::AbstractVector, P::AbstractVector, bstrt::Int, bstop::Int, S::AbstractVector, R::AbstractRegularization, tol::Real, ::Val{DIAG}) where {T, DIAG}
    @inbounds for j in bstrt:bstop
        Ajj = real(A[j, j]) - D[j]

        if R isa NoRegularization && iszero(S[j])
            maxval = abs(Ajj)
        else
            maxval = real(S[j]) * Ajj
        end

        maxind = j

        for i in j + 1:size(A, 1)
            Aii = real(A[i, i]) - D[i]

            if R isa NoRegularization && iszero(S[i])
                sAii = abs(Aii)
            else
                sAii = real(S[i]) * Aii
            end

            if sAii > maxval
                maxval = sAii
                maxind = i
            end
        end

        if R isa NoRegularization && maxval < tol
            for i in j:size(A, 1)
                D[i] = zero(real(T))
            end

            return 0, j - 1
        end

        if maxind != j
            swaptri!(A, j, maxind, Val(:L))
            swaprec!(P, j, maxind)
            swaprec!(D, j, maxind)
            swaprec!(S, j, maxind)
        end

        for k in bstrt:j - 1
            cLjk = conj(A[j, k])

            for i in j + 1:size(A, 1)
                if DIAG === :N
                    A[i, j] -= A[i, k] * cLjk
                else
                    A[i, j] -= A[i, k] * real(D[k]) * cLjk
                end
            end
        end

        Djj = regularize(R, S, real(A[j, j]) - real(D[j]), j)

        iszero(Djj) && return 0, j - 1

        if DIAG === :N
            A[j, j] = Djj = sqrt(Djj)
        else
            D[j] = Djj
        end

        iDjj = inv(Djj)

        for i in j + 1:size(A, 1)
            A[i, j] *= iDjj
        end

        for i in j + 1:size(A, 1)
            if DIAG === :N
                D[i] += abs2(A[i, j])
            else
                D[i] += Djj * abs2(A[i, j])
            end
        end
    end

    return 0, bstop
end

function pstrf2!(::Val{:U}, A::AbstractMatrix{T}, D::AbstractVector, P::AbstractVector, bstrt::Int, bstop::Int, S::AbstractVector, R::AbstractRegularization, tol::Real, ::Val{DIAG}) where {T, DIAG}
    @inbounds for j in bstrt:bstop
        if R isa NoRegularization && iszero(S[j])
            maxval = abs(D[j])
        else
            maxval = real(S[j]) * D[j]
        end

        maxind = j

        for i in j + 1:size(A, 1)
            if R isa NoRegularization && iszero(S[i])
                sAii = abs(D[i])
            else
                sAii = real(S[i]) * D[i]
            end

            if sAii > maxval
                maxval = sAii
                maxind = i
            end
        end

        if R isa NoRegularization && maxval < tol
            for i in j:size(A, 1)
                D[i] = zero(real(T))
            end

            return 0, j - 1
        end

        if maxind != j
            swaptri!(A, j, maxind, Val(:U))
            swaprec!(P, j, maxind)
            swaprec!(D, j, maxind)
            swaprec!(S, j, maxind)
        end

        Djj = regularize(R, S, real(D[j]), j)

        iszero(Djj) && return 0, j - 1

        if DIAG === :N
            A[j, j] = Djj = sqrt(Djj)
        else
            D[j] = Djj
        end

        iDjj = inv(Djj)

        for i in j + 1:size(A, 1)
            for k in bstrt:j - 1
                cAkj = conj(A[k, j])

                if DIAG === :N
                    A[j, i] -= A[k, i] * cAkj
                else
                    A[j, i] -= A[k, i] * real(D[k]) * cAkj
                end
            end

            A[j, i] *= iDjj

            if DIAG === :N
                D[i] -= abs2(A[j, i])
            else
                D[i] -= Djj * abs2(A[j, i])
            end
        end
    end

    return 0, bstop
end

# ===== pstrf! =====

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

function pstrf!(uplo::Val, W::AbstractVector{T}, A::AbstractMatrix{T}, D::AbstractVector{T}, P::AbstractVector{BlasInt}, ::AbstractVector{T}, ::NoRegularization, tol::Real, ::Val{:N}) where {T <: BlasFloat}
    return pstrf!(char(uplo), A, P, reinterpret(real(T), W), tol)
end

function pstrf!(uplo::Val{UPLO}, W::AbstractVector{T}, A::AbstractMatrix{T}, D::AbstractVector{T}, P::AbstractVector, S::AbstractVector{T}, R::AbstractRegularization, tol::Real, diag::Val{DIAG}) where {T, UPLO, DIAG}
    if isnegative(tol)
        maxdiag = zero(real(T))

        @inbounds for i in axes(A, 1)
            maxdiag = max(maxdiag, abs(real(A[i, i])))
        end

        tol = size(A, 1) * eps(real(T)) * maxdiag
    end

    @inbounds for j in axes(A, 1)
        P[j] = j

        if UPLO === :L
            D[j] = zero(real(T))
        else
            D[j] = real(A[j, j])
        end
    end

    rank = size(A, 1)

    @inbounds for bstrt in 1:THRESHOLD:size(A, 1)
        bstop = min(bstrt + THRESHOLD - 1, size(A, 1))
        bsize = bstop - bstrt + 1

        info, rank = pstrf2!(uplo, A, D, P, bstrt, bstop, S, R, tol, diag)

        if rank < bstop
            return info, rank
        end

        if bstop < size(A, 1)
            brest = size(A, 1) - bstop

            Arr = view(A, bstop + 1:size(A, 1), bstop + 1:size(A, 1))
            Dbb = view(D, bstrt:bstop)

            if UPLO === :L
                Arb = view(A, bstop + 1:size(A, 1), bstrt:bstop)

                syrk!(Val(:L), Val(:N), -one(real(T)), W, Arb, Dbb, one(real(T)), Arr, diag)

                @inbounds for j in bstop + 1:size(A, 1)
                    D[j] = zero(real(T))
                end
            else
                Arb = view(A, bstrt:bstop, bstop + 1:size(A, 1))

                syrk!(Val(:U), Val(:C), -one(real(T)), W, Arb, Dbb, one(real(T)), Arr, diag)
            end
        end
    end

    return 0, rank
end

# ===== trrk! =====

function trrk!(uplo::Val, trans::Val{TRANS}, alpha::Real, A::AbstractMatrix{T}, B::AbstractMatrix{T}, beta::Real, C::AbstractMatrix{T}) where {T <: BlasFloat, TRANS}
    if T <: Complex
        BLAS.her2k!(char(uplo), char(trans), convert(T, alpha / 2), A, B, beta, C)
    else
        BLAS.syr2k!(char(uplo), char(trans), convert(T, alpha / 2), A, B, beta, C)
    end

    return
end

function trrk!(uplo::Val{UPLO}, trans::Val{TRANS}, alpha::Real, A::AbstractMatrix{T}, B::AbstractMatrix{T}, beta::Real, C::AbstractMatrix{T}) where {T, TRANS, UPLO}
    @inbounds for j in axes(C, 2)
        if UPLO === :L
            for i in 1:j-1
                C[i, j] = zero(T)
            end
        else
            for i in j + 1:size(C, 1)
                C[i, j] = zero(T)
            end
        end
    end

    if TRANS === :N
        mul!(C, A, B', alpha, beta)
    else
        mul!(C, A', B, alpha, beta)
    end

    return
end

# ===== syrk! =====

function syrk!(uplo::Val, trans::Val, α, A::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T <: BlasFloat}
    if T <: Complex
        BLAS.herk!(char(uplo), char(trans), real(α), A, real(β), C)
    else
        BLAS.syrk!(char(uplo), char(trans), α, A, β, C)
    end

    return
end

function syrk!(uplo::Val{UPLO}, trans::Val{TRANS}, α, A::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T, TRANS, UPLO}
    @inbounds for j in axes(C, 2)
        if UPLO === :L
            for i in 1:j-1
                C[i, j] = zero(T)
            end
        else
            for i in j + 1:size(C, 1)
                C[i, j] = zero(T)
            end
        end
    end

    if TRANS === :N
        mul!(C, A, A', α, β)
    else
        mul!(C, A', A, α, β)
    end

    return
end

function syrk!(uplo::Val, trans::Val, α, ::AbstractVector{T}, A::AbstractMatrix{T}, ::AbstractVector{T}, β, C::AbstractMatrix{T}, ::Val{:N}) where {T}
    return syrk!(uplo, trans, α, A, β, C)
end

function syrk!(uplo::Val, trans::Val{TRANS}, alpha::Real, W::AbstractVector{T}, A::AbstractMatrix{T}, d::AbstractVector{T}, beta::Real, C::AbstractMatrix{T}, ::Val{:U}) where {T, TRANS}
    B = reshape(view(W, 1:length(A)), size(A))
    copyrec!(B, A)

    if TRANS === :N
        cmul!(Val(:R), B, d)
    else
        cmul!(Val(:L), B, d)
    end

    trrk!(uplo, trans, alpha, B, A, beta, C)
    return
end

# ===== gemm! =====

function gemm!(tA::Val, tB::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.gemm!(char(tA), char(tB), α, A, B, β, C)
    return
end

function gemm!(tA::Val, tB::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) where {T}
    mul!(C, adj(tA, A), adj(tB, B), α, β)
    return
end

function gemm!(tA::Val, tB::Val, α::T, ::AbstractVector{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, ::AbstractVector{T}, β::T, C::AbstractMatrix{T}, ::Val{:N}) where {T}
    return gemm!(tA, tB, α, A, B, β, C)
end

function gemm!(tA::Val{TA}, tB::Val{TB}, α::T, W::AbstractVector{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, d::AbstractVector{T}, β::T, C::AbstractMatrix{T}, ::Val{:U}) where {T, TA, TB}
    D = reshape(view(W, 1:length(A)), size(A))
    copyrec!(D, A)

    if TA === :N
        cmul!(Val(:R), D, d)
    else
        cmul!(Val(:L), D, d)
    end

    gemm!(tA, tB, α, D, B, β, C)
    return
end

# ===== gemv! =====

function gemv!(tA::Val, α::T, A::AbstractMatrix{T}, b::AbstractVector{T}, β::T, c::AbstractVector{T}) where {T <: BlasFloat}
    BLAS.gemv!(char(tA), α, A, b, β, c)
    return
end

function gemv!(tA::Val, α::T, A::AbstractMatrix{T}, b::AbstractVector{T}, β::T, c::AbstractVector{T}) where {T}
    mul!(c, adj(tA, A), b, α, β)
    return
end

# ===== trsm! =====

function trsm!(side::Val, uplo::Val, tA::Val, diag::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.trsm!(char(side), char(uplo), char(tA), char(diag), α, A, B)
    return
end

function trsm!(side::Val{SIDE}, uplo::Val, tA::Val, diag::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T, SIDE}
    if SIDE === :L
        ldiv!(adj(tA, tri(uplo, diag, A)), B)
    else
        rdiv!(B, adj(tA, tri(uplo, diag, A)))
    end

    lmul!(α, B)
    return
end

# ===== trsv! =====

function trsv!(uplo::Val, tA::Val, diag::Val, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T <: BlasFloat}
    BLAS.trsv!(char(uplo), char(tA), char(diag), A, b)
    return
end

function trsv!(uplo::Val, tA::Val, diag::Val, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    ldiv!(adj(tA, tri(uplo, diag, A)), b)
    return
end

# ===== trmm! =====

function trmm!(side::Val, uplo::Val, tA::Val, diag::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.trmm!(char(side), char(uplo), char(tA), char(diag), α, A, B)
    return
end

function trmm!(side::Val{SIDE}, uplo::Val, tA::Val, diag::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T, SIDE}
    if SIDE === :L
        lmul!(adj(tA, tri(uplo, diag, A)), B)
    else
        rmul!(B, adj(tA, tri(uplo, diag, A)))
    end

    lmul!(α, B)
    return
end

# ===== trmv! =====

function trmv!(uplo::Val, tA::Val, diag::Val, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T <: BlasFloat}
    BLAS.trmv!(char(uplo), char(tA), char(diag), A, b)
    return
end

function trmv!(uplo::Val, tA::Val, diag::Val, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    lmul!(adj(tA, tri(uplo, diag, A)), b)
    return
end

# ===== symm! =====

function symm!(side::Val, uplo::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) where {T <: BlasFloat}
    if T <: Complex
        BLAS.hemm!(char(side), char(uplo), α, A, B, β, C)
    else
        BLAS.symm!(char(side), char(uplo), α, A, B, β, C)
    end

    return
end

function symm!(side::Val{SIDE}, uplo::Val, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β::T, C::AbstractMatrix{T}) where {T, SIDE}
    if SIDE === :L
        mul!(C, sym(uplo, A), B, α, β)
    else
        mul!(C, B, sym(uplo, A), α, β)
    end

    return
end

# ===== trtri! =====

function trtri!(uplo::Val, diag::Val, A::AbstractMatrix{T}) where {T <: BlasFloat}
    LAPACK.trtri!(char(uplo), char(diag), A)
    return
end

function trtri!(uplo::Val, diag::Val, A::AbstractMatrix{T}) where {T}
    copyto!(A, inv(tri(uplo, diag, A)))
    return
end

# ===== ger! =====

function ger!(α, x::AbstractVector{T}, y::AbstractVector{T}, A::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.ger!(convert(T, α), x, y, A)
    return
end

function ger!(α, x::AbstractVector{T}, y::AbstractVector{T}, A::AbstractMatrix{T}) where {T}
    mul!(A, x, y', α, true)
    return
end

# ===== syr! =====

function syr!(uplo::Val, α, x::AbstractVector{T}, A::AbstractMatrix{T}) where {T <: BlasFloat}
    if T <: Complex
        BLAS.her!(char(uplo), real(α), x, A)
    else
        BLAS.syr!(char(uplo), α, x, A)
    end

    return
end

function syr!(uplo::Val, α, x::AbstractVector{T}, A::AbstractMatrix{T}) where {T}
    ger!(α, x, x, A)
    return
end
