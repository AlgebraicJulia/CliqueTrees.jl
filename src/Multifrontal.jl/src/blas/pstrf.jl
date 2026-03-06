# ===== pstrf_pivot! =====

function pstrf_pivot!(uplo::Val, A::AbstractMatrix{T}, D::AbstractVector, P::AbstractVector, S::AbstractVector, R::AbstractRegularization, tol::Real, j::Int) where {T}
    n = size(A, 1)

    if R isa NoRegularization && iszero(S[j])
        maxval = abs(D[j])
    else
        maxval = real(S[j]) * D[j]
    end

    maxind = j

    for i in j + 1:n
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
        for i in j:n
            D[i] = zero(real(T))
        end

        return false
    end

    if maxind != j
        swaptri!(A, j, maxind, uplo)
        swaprec!(P, j, maxind)
        swaprec!(D, j, maxind)
        swaprec!(S, j, maxind)
    end

    return true
end

# ===== pstrf2! =====

function pstrf2!(uplo::Val{:L}, A::AbstractMatrix{T}, D::AbstractVector, P::AbstractVector, bstrt::Int, bstop::Int, S::AbstractVector, R::AbstractRegularization, tol::Real, ::Val{DIAG}) where {T, DIAG}
    n = size(A, 1)

    @inbounds for j in bstrt:bstop
        pstrf_pivot!(uplo, A, D, P, S, R, tol, j) || return 0, j - 1

        for k in bstrt:j - 1
            cLjk = conj(A[j, k])

            for i in j + 1:n
                if DIAG === :N
                    A[i, j] -= A[i, k] * cLjk
                else
                    A[i, j] -= A[i, k] * real(D[k]) * cLjk
                end
            end
        end

        Djj = regularize(R, S, real(D[j]), j)

        iszero(Djj) && return 0, j - 1

        if DIAG === :N
            A[j, j] = Djj = sqrt(Djj)
        else
            D[j] = Djj
        end

        iDjj = inv(Djj)

        for i in j + 1:n
            A[i, j] *= iDjj
        end

        for i in j + 1:n
            if DIAG === :N
                D[i] -= abs2(A[i, j])
            else
                D[i] -= Djj * abs2(A[i, j])
            end
        end
    end

    return 0, bstop
end

function pstrf2!(uplo::Val{:U}, A::AbstractMatrix{T}, D::AbstractVector, P::AbstractVector, bstrt::Int, bstop::Int, S::AbstractVector, R::AbstractRegularization, tol::Real, ::Val{DIAG}) where {T, DIAG}
    n = size(A, 1)

    @inbounds for j in bstrt:bstop
        pstrf_pivot!(uplo, A, D, P, S, R, tol, j) || return 0, j - 1

        Djj = regularize(R, S, real(D[j]), j)

        iszero(Djj) && return 0, j - 1

        if DIAG === :N
            A[j, j] = Djj = sqrt(Djj)
        else
            D[j] = Djj
        end

        iDjj = inv(Djj)

        for i in j + 1:n
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

function pstrf!(uplo::Val{UPLO}, W::AbstractVector{T}, A::AbstractMatrix{T}, D::AbstractVector{T}, P::AbstractVector, S::AbstractVector{T}, R::GMW81, tol::Real, diag::Val{DIAG}) where {T, UPLO, DIAG}
    n = size(A, 1)

    if UPLO === :L
        L = Ones{T}(0, n)
    else
        L = Ones{T}(n, 0)
    end

    return chol_piv_factor!(A, L, W, D, P, S, R, uplo, diag)
end

function pstrf!(uplo::Val{UPLO}, W::AbstractVector{T}, L₁::AbstractMatrix{T}, d₁::AbstractVector{T}, P::AbstractVector, S₁::AbstractVector{T}, R::SE99, tol::Real, diag::Val{DIAG}) where {T, UPLO, DIAG}
    n = size(L₁, 1)

    if UPLO === :L
        L₂ = Ones{T}(0, n)
    else
        L₂ = Ones{T}(n, 0)
    end

    M = Ones{T}(0, 0)
    d₂ = Ones{T}(0)
    g = FVector{T}(undef, n)
    S₂ = Ones{T}(0)

    chol_se99_piv_kernel!(L₁, L₂, M, W, d₁, d₂, g, P, S₁, S₂, R, true, zero(real(T)), zero(real(T)), uplo, diag)

    return 0, n
end

function pstrf!(uplo::Val{UPLO}, W::AbstractVector{T}, A::AbstractMatrix{T}, D::AbstractVector{T}, P::AbstractVector, S::AbstractVector{T}, R::AbstractRegularization, tol::Real, diag::Val{DIAG}) where {T, UPLO, DIAG}
    n = size(A, 1)

    if isnegative(tol)
        maxdiag = zero(real(T))

        @inbounds for i in axes(A, 1)
            maxdiag = max(maxdiag, abs(real(A[i, i])))
        end

        tol = n * eps(real(T)) * maxdiag
    end

    @inbounds for j in axes(A, 1)
        P[j] = j
        D[j] = real(A[j, j])
    end

    rank = n

    @inbounds for bstrt in 1:THRESHOLD:n
        bstop = min(bstrt + THRESHOLD - 1, n)

        info, rank = pstrf2!(uplo, A, D, P, bstrt, bstop, S, R, tol, diag)

        if rank < bstop
            return info, rank
        elseif bstop < n
            Arr = view(A, bstop + 1:n, bstop + 1:n)
            Dbb = view(D, bstrt:bstop)

            if UPLO === :L
                Arb = view(A, bstop + 1:n, bstrt:bstop)
                trans = Val(:N)
            else
                Arb = view(A, bstrt:bstop, bstop + 1:n)
                trans = Val(:C)
            end

            syrk!(uplo, trans, -one(real(T)), W, Arb, Dbb, one(real(T)), Arr, diag)
        end
    end

    return 0, rank
end
