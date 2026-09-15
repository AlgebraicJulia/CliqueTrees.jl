const GEMX_MR   = 8
const GEMX_NR   = 4
const GEMX_LEAF = 256

# ===== gemm! =====

function gemm!(tA::Val, tB::Val, α, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β, C::AbstractMatrix{T}; nt::Integer = nthreads()) where {T <: BlasFloat}
    BLAS.gemm!(char(tA), char(tB), convert(T, α), A, B, convert(T, β), C)
    return
end

function gemm!(tA::Val, tB::Val, α, A::AbstractMatrix, B::AbstractMatrix, β, C::AbstractMatrix; nt::Integer = nthreads())
    gemx!(tA, tB, α, A, B, β, C; nt)
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

function gemx!(tA::Val{TA}, tB::Val{TB}, α, A::AbstractMatrix{T}, B::AbstractMatrix{T}, β, C::AbstractMatrix{T}; nt::Integer = nthreads()) where {TA, TB, T}
    #
    #   C ← β C
    #
    if iszero(β)
        fill!(C, zero(T))
    elseif !isone(β)
        C .*= β
    end

    m = size(C, 1)
    n = size(C, 2)

    if TA === :N
        k = size(A, 2)
    else
        k = size(A, 1)
    end

    mc = min(m, GEMX_LEAF)
    nc = min(n, GEMX_LEAF)
    kc = min(k, GEMX_LEAF)
    #
    #   C ← α A B + C
    #
    if nt <= 1 || max(m, n, k) <= GEMX_LEAF
        AP = FVector{T}(undef, cld(mc, GEMX_MR) * GEMX_MR * kc)
        BP = FVector{T}(undef, cld(nc, GEMX_NR) * GEMX_NR * kc)
        gemx_st!(tA, tB, C, A, B, α, AP, BP)
    else
        depth = ceil(Int, log2(nt)) + 1
        work = Channel{Tuple{FVector{T}, FVector{T}}}(nt)

        for _ in 1:nt
            AP = FVector{T}(undef, cld(mc, GEMX_MR) * GEMX_MR * kc)
            BP = FVector{T}(undef, cld(nc, GEMX_NR) * GEMX_NR * kc)
            put!(work, (AP, BP))
        end

        gemx_mt!(tA, tB, C, A, B, α, work, depth)
    end

    return
end

function gemx_mt!(tA::Val{TA}, tB::Val{TB}, C::AbstractMatrix{T}, A::AbstractMatrix, B::AbstractMatrix, α, work::Channel, depth::Int) where {TA, TB, T}
    m = size(C, 1)
    n = size(C, 2)

    if TA === :N
        k = size(A, 2)
    else
        k = size(A, 1)
    end

    if depth <= 0 || (m <= GEMX_LEAF && n <= GEMX_LEAF && k <= GEMX_LEAF)
        AP, BP = take!(work)

        try
            gemx_st!(tA, tB, C, A, B, α, AP, BP)
        finally
            put!(work, (AP, BP))
        end
    else
        mx = max(m, n, k)

        if m == mx
            h = (m >> 1); h -= h % GEMX_MR; h = max(h, GEMX_MR)

            if TA === :N
                A₁ = view(A, 1:h, :); A₂ = view(A, h + 1:m, :)
            else
                A₁ = view(A, :, 1:h); A₂ = view(A, :, h + 1:m)
            end

            task = @spawn gemx_mt!(tA, tB, view(C, 1:h, :), A₁, B, α, work, depth - 1)
            gemx_mt!(tA, tB, view(C, h + 1:m, :), A₂, B, α, work, depth - 1)
            wait(task)
        elseif n == mx
            h = (n >> 1); h -= h % GEMX_NR; h = max(h, GEMX_NR)

            if TB === :N
                B₁ = view(B, :, 1:h); B₂ = view(B, :, h + 1:n)
            else
                B₁ = view(B, 1:h, :); B₂ = view(B, h + 1:n, :)
            end

            task = @spawn gemx_mt!(tA, tB, view(C, :, 1:h), A, B₁, α, work, depth - 1)
            gemx_mt!(tA, tB, view(C, :, h + 1:n), A, B₂, α, work, depth - 1)
            wait(task)
        else
            h = k >> 1

            if TA === :N
                A₁ = view(A, :, 1:h); A₂ = view(A, :, h + 1:k)
            else
                A₁ = view(A, 1:h, :); A₂ = view(A, h + 1:k, :)
            end

            if TB === :N
                B₁ = view(B, 1:h, :); B₂ = view(B, h + 1:k, :)
            else
                B₁ = view(B, :, 1:h); B₂ = view(B, :, h + 1:k)
            end

            gemx_mt!(tA, tB, C, A₁, B₁, α, work, depth)
            gemx_mt!(tA, tB, C, A₂, B₂, α, work, depth)
        end
    end

    return C
end

function gemx_st!(tA::Val{TA}, tB::Val{TB}, C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α, AP::AbstractVector, BP::AbstractVector) where {TA, TB}
    m = size(C, 1)
    n = size(C, 2)

    if TA === :N
        k = size(A, 2)
    else
        k = size(A, 1)
    end

    if m <= GEMX_LEAF && n <= GEMX_LEAF && k <= GEMX_LEAF
        gemx_col!(tA, tB, C, A, B, α, AP, BP)
    else
        mx = max(m, n, k)

        if m == mx
            h = (m >> 1); h -= h % GEMX_MR; h = max(h, GEMX_MR)

            if TA === :N
                A₁ = view(A, 1:h, :); A₂ = view(A, h + 1:m, :)
            else
                A₁ = view(A, :, 1:h); A₂ = view(A, :, h + 1:m)
            end
            #
            #   [ C₁ ]     [ A₁ ]
            #   [ C₂ ]  ←  [ A₂ ] B + C
            #
            gemx_st!(tA, tB, view(C, 1:h, :),     A₁, B, α, AP, BP)
            gemx_st!(tA, tB, view(C, h + 1:m, :), A₂, B, α, AP, BP)
        elseif n == mx
            h = (n >> 1); h -= h % GEMX_NR; h = max(h, GEMX_NR)

            if TB === :N
                B₁ = view(B, :, 1:h); B₂ = view(B, :, h + 1:n)
            else
                B₁ = view(B, 1:h, :); B₂ = view(B, h + 1:n, :)
            end
            #
            #   [ C₁ C₂ ]  ←  A [ B₁ B₂ ] + C
            #
            gemx_st!(tA, tB, view(C, :, 1:h),     A, B₁, α, AP, BP)
            gemx_st!(tA, tB, view(C, :, h + 1:n), A, B₂, α, AP, BP)
        else
            h = k >> 1

            if TA === :N
                A₁ = view(A, :, 1:h); A₂ = view(A, :, h + 1:k)
            else
                A₁ = view(A, 1:h, :); A₂ = view(A, h + 1:k, :)
            end

            if TB === :N
                B₁ = view(B, 1:h, :); B₂ = view(B, h + 1:k, :)
            else
                B₁ = view(B, :, 1:h); B₂ = view(B, :, h + 1:k)
            end
            #
            #   C  ←  [ A₁ A₂ ] [ B₁ ] + C
            #                   [ B₂ ]
            #
            gemx_st!(tA, tB, C, A₁, B₁, α, AP, BP)
            gemx_st!(tA, tB, C, A₂, B₂, α, AP, BP)
        end
    end

    return C
end

function gemx_col!(tA::Val{TA}, tB::Val{TB}, C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, α, AP::AbstractVector, BP::AbstractVector) where {TA, TB}
    m = size(C, 1)
    n = size(C, 2)

    if TA === :N
        k = size(A, 2)
    else
        k = size(A, 1)
    end
    #
    #   BP ← α B
    #
    @inbounds for j0 in 1:GEMX_NR:n
        nt = min(GEMX_NR, n - j0 + 1)
        off = (j0 - 1) * k

        for p in 1:k
            for j in 1:nt
                if TB === :N
                    Bpj = B[p, j0 + j - 1]
                else
                    Bpj = conj(B[j0 + j - 1, p])
                end

                BP[off + (p - 1) * GEMX_NR + j] = α * Bpj
            end
        end
    end
    #
    #   AP ← A
    #
    @inbounds for i0 in 1:GEMX_MR:m
        mt = min(GEMX_MR, m - i0 + 1)
        off = (i0 - 1) * k

        for p in 1:k
            for i in 1:mt
                if TA === :N
                    Aip = A[i0 + i - 1, p]
                else
                    Aip = conj(A[p, i0 + i - 1])
                end

                AP[off + (p - 1) * GEMX_MR + i] = Aip
            end
        end
    end
    #
    #   C ← A B + C
    #
    @inbounds for j0 in 1:GEMX_NR:n
        for i0 in 1:GEMX_MR:m
            mt = min(GEMX_MR, m - i0 + 1)
            nt = min(GEMX_NR, n - j0 + 1)

            if mt == GEMX_MR && nt == GEMX_NR
                gemx_col_kernel!(C, AP, (i0 - 1) * k + 1, BP, (j0 - 1) * k + 1, i0, j0, k)
            else
                for j in 1:nt
                    for p in 1:k
                        b = BP[(j0 - 1) * k + (p - 1) * GEMX_NR + j]

                        for i in 1:mt
                            C[i0 + i - 1, j0 + j - 1] = muladd(AP[(i0 - 1) * k + (p - 1) * GEMX_MR + i], b, C[i0 + i - 1, j0 + j - 1])
                        end
                    end
                end
            end
        end
    end

    return C
end

@generated function gemx_col_kernel!(C::AbstractMatrix, AP::AbstractVector, ap0::Int, BP::AbstractVector, bp0::Int, i0::Int, j0::Int, k::Int)
    load  = Expr[]   # Δij ← C[i, j]
    la    = Expr[]   # ai  ← AP[…]
    lb    = Expr[]   # bj  ← BP[…]
    fma   = Expr[]   # Δij ← ai bj + Δij
    store = Expr[]   # C[i, j] ← Δij

    for i in 1:GEMX_MR
        ai = Symbol(:a, i)
        push!(la, :($ai = AP[aoff + $(i - 1)]))
    end

    for j in 1:GEMX_NR
        bj = Symbol(:b, j)
        push!(lb, :($bj = BP[boff + $(j - 1)]))

        for i in 1:GEMX_MR
            ai  = Symbol(:a, i)
            Δij = Symbol(:Δ, i, j)
            push!(load,  :($Δij = C[i0 + $(i - 1), j0 + $(j - 1)]))
            push!(fma,   :($Δij = muladd($ai, $bj, $Δij)))
            push!(store, :(C[i0 + $(i - 1), j0 + $(j - 1)] = $Δij))
        end
    end

    return quote
        $(Expr(:meta, :inline))
        @inbounds @fastmath begin
            $(load...)

            for p in 1:k
                aoff = ap0 + (p - 1) * GEMX_MR
                boff = bp0 + (p - 1) * GEMX_NR

                $(la...)
                $(lb...)
                $(fma...)
            end

            $(store...)
        end

        return
    end
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
    #     c ← c + α A b
    #
    j = 1

    @inbounds while j + 7 <= n; gemx_dot_tile!(c, A, b, α, tA, j, Val(8)); j += 8; end
    @inbounds while j + 3 <= n; gemx_dot_tile!(c, A, b, α, tA, j, Val(4)); j += 4; end
    @inbounds while j + 1 <= n; gemx_dot_tile!(c, A, b, α, tA, j, Val(2)); j += 2; end
    @inbounds while j     <= n; gemx_dot_tile!(c, A, b, α, tA, j, Val(1)); j += 1; end

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
