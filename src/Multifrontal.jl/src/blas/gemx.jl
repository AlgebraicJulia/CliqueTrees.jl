# ===== gemm! =====

function gemm!(tA::Val, tB::Val, α, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β, C::AbstractMatrix{T}) where {T <: BlasFloat}
    BLAS.gemm!(char(tA), char(tB), convert(T, α), A, B, convert(T, β), C)
    return
end

function gemm!(tA::Val, tB::Val, α, A::AbstractMatrix, B::AbstractMatrix, β, C::AbstractMatrix)
    gemx!(tA, tB, α, A, B, β, C)
    return
end

function gemm!(tA::Val, tB::Val, α, ::AbstractVector, A::AbstractMatrix, B::AbstractMatrix, ::AbstractVector, β, C::AbstractMatrix, ::Val{:N})
    return gemm!(tA, tB, α, A, B, β, C)
end

function gemm!(tA::Val{TA}, tB::Val{TB}, α, W::AbstractVector, A::AbstractMatrix, B::AbstractMatrix, d::AbstractVector, β, C::AbstractMatrix, ::Val{:U}) where {TA, TB}
    D = reshape(view(W, 1:length(A)), size(A))
    copyrec!(D, A)

    if TA === :N
        cmul!(Val(:R), Val(:U), D, d)
    else
        cmul!(Val(:L), Val(:U), D, d)
    end

    gemm!(tA, tB, α, D, B, β, C)
    return
end

# ===== gemv! =====

function gemv!(tA::Val, α, A::AbstractMatrix{T}, b::AbstractVector{T}, β, c::AbstractVector{T}) where {T <: BlasFloat}
    BLAS.gemv!(char(tA), convert(T, α), A, b, convert(T, β), c)
    return
end

function gemv!(tA::Val, α, A::AbstractMatrix, b::AbstractVector, β, c::AbstractVector)
    gemx!(tA, Val(:N), α, A, b, β, c)
    return
end

# ===== gemx! =====

function gemx!(tA::Val{TA}, tB::Val{TB}, α, A::AbstractMatrix, B::AbstractVector, β, C::AbstractVector) where {TA, TB}
    gemx2!(tA, tB, α, A, B, β, C)
    return
end

function gemx!(tA::Val{TA}, tB::Val{TB}, α, A::AbstractMatrix, B::AbstractMatrix, β, C::AbstractMatrix) where {TA, TB}
    m = size(C, 1)
    n = size(C, 2)

    if TA === :N
        k = size(A, 2)
    else
        k = size(A, 1)
    end

    maxdim = max(m, n, k)

    if maxdim <= THRESHOLD
        gemx2!(tA, tB, α, A, B, β, C)
    else
        l = prevpow(2, maxdim) >> 1

        if m == maxdim
            C₁ = view(C,     1:l, :)
            C₂ = view(C, l + 1:m,  :)

            if TA === :N
                A₁ = view(A,     1:l, :)
                A₂ = view(A, l + 1:m,  :)
            else
                A₁ = view(A, :,     1:l)
                A₂ = view(A, :, l + 1:m)
            end

            gemx!(tA, tB, α, A₁, B, β, C₁)
            gemx!(tA, tB, α, A₂, B, β, C₂)

        elseif n == maxdim
            C₁ = view(C, :,     1:l)
            C₂ = view(C, :, l + 1:n)

            if TB === :N
                B₁ = view(B, :,     1:l)
                B₂ = view(B, :, l + 1:n)
            else
                B₁ = view(B,     1:l, :)
                B₂ = view(B, l + 1:n,  :)
            end

            gemx!(tA, tB, α, A, B₁, β, C₁)
            gemx!(tA, tB, α, A, B₂, β, C₂)

        else
            if TA === :N
                A₁ = view(A, :,     1:l)
                A₂ = view(A, :, l + 1:k)
            else
                A₁ = view(A,     1:l, :)
                A₂ = view(A, l + 1:k,  :)
            end

            if TB === :N
                B₁ = view(B,     1:l, :)
                B₂ = view(B, l + 1:k,  :)
            else
                B₁ = view(B, :,     1:l)
                B₂ = view(B, :, l + 1:k)
            end

            gemx!(tA, tB, α, A₁, B₁, β, C)
            gemx!(tA, tB, α, A₂, B₂, 1, C)
        end
    end

    return
end

@generated function gemx_tile!(c::AbstractVector, A::AbstractMatrix, b::AbstractVector, α, istrt, ::Val{TILE}) where {TILE}
    accm = Vector{Symbol}(undef, TILE)
    init = Vector{Expr}(undef,   TILE)
    updt = Vector{Expr}(undef,   TILE)
    stor = Vector{Expr}(undef,   TILE)

    for t in 1:TILE
        accm[t] = Symbol(:a, t)
        init[t] = :($(accm[t]) = z)
        updt[t] = :($(accm[t]) = muladd(A[istrt + $(t - 1), k], bk, $(accm[t])))
        stor[t] = :(@inbounds c[istrt + $(t - 1)] = muladd(α, $(accm[t]), c[istrt + $(t - 1)]))
    end

    return quote
        $(Expr(:meta, :inline))
        z = zero(promote_eltype(A, b))

        $(init...)

        @inbounds @simd for k in axes(A, 2)
            bk = b[k]
            $(updt...)
        end

        $(stor...)

        return
    end
end

function gemx2!(::Val{:N}, ::Val{:N}, α, A::AbstractMatrix, b::AbstractVector, β, c::AbstractVector)
    m = size(A, 1)
    #
    #     c ← β c
    #
    if iszero(β)
        @inbounds for i in 1:m
            c[i] = β
        end
    elseif !isone(β)
        @inbounds for i in 1:m
            c[i] *= β
        end
    end
    #
    #     c ← c + α A b
    #
    i = 1

    @inbounds while i + 15 <= m; gemx_tile!(c, A, b, α, i, Val(16)); i += 16; end
    @inbounds while i +  7 <= m; gemx_tile!(c, A, b, α, i, Val( 8)); i +=  8; end
    @inbounds while i +  3 <= m; gemx_tile!(c, A, b, α, i, Val( 4)); i +=  4; end
    @inbounds while i +  1 <= m; gemx_tile!(c, A, b, α, i, Val( 2)); i +=  2; end
    @inbounds while i      <= m; gemx_tile!(c, A, b, α, i, Val( 1)); i +=  1; end

    return
end

function gemx2!(::Val{:N}, ::Val{:N}, α, A::AbstractMatrix, B::AbstractMatrix, β, C::AbstractMatrix)
    if iszero(β)
        @inbounds @fastmath for j in axes(C, 2)
            for i in axes(C, 1)
                C[i, j] = β
            end
        end
    else
        @inbounds @fastmath for j in axes(C, 2)
            for i in axes(C, 1)
                C[i, j] *= β
            end
        end
    end

    @inbounds @fastmath for k in axes(A, 2)
        for j in axes(C, 2)
            Bkj = α * B[k, j]

            for i in axes(C, 1)
                C[i, j] += A[i, k] * Bkj
            end
        end
    end

    return
end

function gemx2!(::Val{:N}, ::Val{TB}, α, A::AbstractMatrix, B::AbstractVecOrMat, β, C::AbstractVecOrMat) where {TB}
    if iszero(β)
        @inbounds @fastmath for j in axes(C, 2)
            for i in axes(C, 1)
                C[i, j] = β
            end
        end
    else
        @inbounds @fastmath for j in axes(C, 2)
            for i in axes(C, 1)
                C[i, j] *= β
            end
        end
    end

    @inbounds @fastmath for k in axes(A, 2)
        for j in axes(C, 2)
            if TB === :C
                Bjk = α * conj(B[j, k])
            else
                Bjk = α * B[j, k]
            end

            for i in axes(C, 1)
                C[i, j] += A[i, k] * Bjk
            end
        end
    end

    return
end

@generated function gemx_dot_tile!(c::AbstractVector, A::AbstractMatrix, b::AbstractVector, α, ::Val{TA}, jstrt, ::Val{JAM}) where {TA, JAM}
    accm = Vector{Symbol}(undef, JAM)
    init = Vector{Expr}(undef,   JAM)
    updt = Vector{Expr}(undef,   JAM)
    stor = Vector{Expr}(undef,   JAM)

    for t in 1:JAM
        accm[t] = Symbol(:a, t)
        init[t] = :($(accm[t]) = z)

        if TA === :C
            updt[t] = :($(accm[t]) = muladd(conj(A[k, jstrt + $(t - 1)]), bk, $(accm[t])))
        else
            updt[t] = :($(accm[t]) = muladd(     A[k, jstrt + $(t - 1)],  bk, $(accm[t])))
        end

        stor[t] = :(@inbounds c[jstrt + $(t - 1)] = muladd(α, $(accm[t]), c[jstrt + $(t - 1)]))
    end

    return quote
        $(Expr(:meta, :inline))
        z = zero(promote_eltype(A, b))

        $(init...)

        @inbounds @simd for k in axes(A, 1)
            bk = b[k]
            $(updt...)
        end

        $(stor...)

        return
    end
end

function gemx2!(tA::Val{TA}, ::Val{:N}, α, A::AbstractMatrix, b::AbstractVector, β, c::AbstractVector) where {TA}
    n = size(A, 2)
    #
    #     c ← β c
    #
    if iszero(β)
        @inbounds for j in 1:n
            c[j] = β
        end
    elseif !isone(β)
        @inbounds for j in 1:n
            c[j] *= β
        end
    end
    #
    #     c ← c + α op(A) b   (column-jammed dot; outputs cascade 8→4→2→1)
    #
    j = 1

    @inbounds while j + 7 <= n; gemx_dot_tile!(c, A, b, α, tA, j, Val(8)); j += 8; end
    @inbounds while j + 3 <= n; gemx_dot_tile!(c, A, b, α, tA, j, Val(4)); j += 4; end
    @inbounds while j + 1 <= n; gemx_dot_tile!(c, A, b, α, tA, j, Val(2)); j += 2; end
    @inbounds while j     <= n; gemx_dot_tile!(c, A, b, α, tA, j, Val(1)); j += 1; end

    return
end

function gemx2!(::Val{TA}, ::Val{:N}, α, A::AbstractMatrix, B::AbstractMatrix, β, C::AbstractMatrix) where {TA}
    @inbounds @fastmath for j in axes(C, 2)
        for i in axes(C, 1)
            Δ = zero(promote_eltype(A, B))

            for k in axes(A, 1)
                if TA === :C
                    Δ += conj(A[k, i]) * B[k, j]
                else
                    Δ += A[k, i] * B[k, j]
                end
            end

            if iszero(β)
                C[i, j] = α * Δ
            else
                C[i, j] = α * Δ + β * C[i, j]
            end
        end
    end

    return
end

function gemx2!(::Val{TA}, ::Val{TB}, α, A::AbstractMatrix, B::AbstractVecOrMat, β, C::AbstractVecOrMat) where {TA, TB}
    @inbounds @fastmath for j in axes(C, 2)
        for i in axes(C, 1)
            Δ = zero(promote_eltype(A, B))

            for k in axes(A, 1)
                if TA === :C
                    Aki = conj(A[k, i])
                else
                    Aki = A[k, i]
                end

                if TB === :C
                    Δ += Aki * conj(B[j, k])
                else
                    Δ += Aki * B[j, k]
                end
            end

            if iszero(β)
                C[i, j] = α * Δ
            else
                C[i, j] = α * Δ + β * C[i, j]
            end
        end
    end

    return
end
