const SGEMX_MR   = 8
const SGEMX_NR   = 4
const SGEMX_LEAF = 256

# ===== sgemx! =====

function sgemx!(s::AbstractSemiring, C::AbstractMatrix{V}, A::AbstractMatrix{V}, B::AbstractMatrix{V}; nt::Integer = nthreads()) where {V}
    @assert size(C, 1) == size(A, 1)
    @assert size(C, 2) == size(B, 2)
    @assert size(A, 2) == size(B, 1)

    m = size(C, 1)
    n = size(C, 2)
    k = size(A, 2)

    mc = min(m, SGEMX_LEAF)
    nc = min(n, SGEMX_LEAF)
    kc = min(k, SGEMX_LEAF)

    if nt <= 1 || max(m, n, k) <= SGEMX_LEAF
        AP = FVector{V}(undef, cld(mc, SGEMX_MR) * SGEMX_MR * kc)
        BP = FVector{V}(undef, cld(nc, SGEMX_NR) * SGEMX_NR * kc)
        sgemx_st!(s, C, A, B, AP, BP)
    else
        depth = ceil(Int, log2(nt)) + 1
        work = Channel{Tuple{FVector{V}, FVector{V}}}(nt)

        for _ in 1:nt
            AP = FVector{V}(undef, cld(mc, SGEMX_MR) * SGEMX_MR * kc)
            BP = FVector{V}(undef, cld(nc, SGEMX_NR) * SGEMX_NR * kc)
            put!(work, (AP, BP))
        end

        sgemx_mt!(s, C, A, B, work, depth)
    end

    return C
end

function sgemx_mt!(s::AbstractSemiring, C::AbstractMatrix{V}, A::AbstractMatrix, B::AbstractMatrix, work::Channel, depth::Int) where {V}
    m = size(C, 1)
    n = size(C, 2)
    k = size(A, 2)

    if depth <= 0 || (m <= SGEMX_LEAF && n <= SGEMX_LEAF && k <= SGEMX_LEAF)
        AP, BP = take!(work)

        try
            sgemx_st!(s, C, A, B, AP, BP)
        finally
            put!(work, (AP, BP))
        end
    else
        mx = max(m, n, k)

        if m == mx
            h = (m >> 1); h -= h % SGEMX_MR; h = max(h, SGEMX_MR)
            task = @spawn sgemx_mt!(s, view(C, 1:h, :), view(A, 1:h, :), B, work, depth - 1)
            sgemx_mt!(s, view(C, h + 1:m, :), view(A, h + 1:m, :), B, work, depth - 1)
            wait(task)
        elseif n == mx
            h = (n >> 1); h -= h % SGEMX_NR; h = max(h, SGEMX_NR)
            task = @spawn sgemx_mt!(s, view(C, :, 1:h), A, view(B, :, 1:h), work, depth - 1)
            sgemx_mt!(s, view(C, :, h + 1:n), A, view(B, :, h + 1:n), work, depth - 1)
            wait(task)
        else
            h = k >> 1
            sgemx_mt!(s, C, view(A, :, 1:h), view(B, 1:h, :), work, depth)
            sgemx_mt!(s, C, view(A, :, h + 1:k), view(B, h + 1:k, :), work, depth)
        end
    end

    return C
end

function sgemx_st!(s::AbstractSemiring, C::AbstractMatrix{V}, A::AbstractMatrix, B::AbstractMatrix, AP::AbstractVector, BP::AbstractVector) where {V}
    m = size(C, 1)
    n = size(C, 2)
    k = size(A, 2)

    if m <= SGEMX_LEAF && n <= SGEMX_LEAF && k <= SGEMX_LEAF
        sgemx_leaf!(s, C, A, B, AP, BP)
    else
        mx = max(m, n, k)

        if m == mx
            h = (m >> 1); h -= h % SGEMX_MR; h = max(h, SGEMX_MR)
            sgemx_st!(s, view(C, 1:h, :), view(A, 1:h, :), B, AP, BP)
            sgemx_st!(s, view(C, h + 1:m, :), view(A, h + 1:m, :), B, AP, BP)
        elseif n == mx
            h = (n >> 1); h -= h % SGEMX_NR; h = max(h, SGEMX_NR)
            sgemx_st!(s, view(C, :, 1:h), A, view(B, :, 1:h), AP, BP)
            sgemx_st!(s, view(C, :, h + 1:n), A, view(B, :, h + 1:n), AP, BP)
        else
            h = k >> 1
            sgemx_st!(s, C, view(A, :, 1:h), view(B, 1:h, :), AP, BP)
            sgemx_st!(s, C, view(A, :, h + 1:k), view(B, h + 1:k, :), AP, BP)
        end
    end

    return C
end

function sgemx_leaf!(s::AbstractSemiring, C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, AP::AbstractVector, BP::AbstractVector)
    if s isa PlusProd
        sgemx_col!(s, C, A, B, AP, BP)
    else
        sgemx_row!(s, C, A, B, AP, BP)
    end

    return C
end

# ===== sgemx_row! =====

function sgemx_row!(s::AbstractSemiring, C::AbstractMatrix{V}, A::AbstractMatrix, B::AbstractMatrix, AP::AbstractVector, BP::AbstractVector) where {V}
    m = size(C, 1)
    n = size(C, 2)
    k = size(A, 2)

    @inbounds for i in 1:m
        off = (i - 1) * k

        for p in 1:k
            AP[off + p] = A[i, p]
        end
    end

    @inbounds for j in 1:n
        off = (j - 1) * k

        for p in 1:k
            BP[off + p] = B[p, j]
        end
    end

    j0 = 1

    @inbounds while j0 + SGEMX_NR - 1 <= n
        sgemx_row_kernel!(s, C, AP, BP, j0, m, k)
        j0 += SGEMX_NR
    end

    @inbounds while j0 <= n
        bp0 = (j0 - 1) * k

        for i in 1:m
            ap0 = (i - 1) * k
            Δ = C[i, j0]

            @simd for p in 1:k
                Δ = smuladd(s, AP[ap0 + p], BP[bp0 + p], Δ)
            end

            C[i, j0] = Δ
        end

        j0 += 1
    end

    return C
end

@inline function sgemx_row_kernel!(s::AbstractSemiring, C::AbstractMatrix, AP::AbstractVector, BP::AbstractVector, j0::Int, m::Int, k::Int)
    b0 = (j0 - 1) * k
    b1 = b0 + k
    b2 = b1 + k
    b3 = b2 + k

    @inbounds for i in 1:m
        ap0 = (i - 1) * k

        Δ0 = C[i, j0]
        Δ1 = C[i, j0 + 1]
        Δ2 = C[i, j0 + 2]
        Δ3 = C[i, j0 + 3]

        @simd for p in 1:k
            a = AP[ap0 + p]
            Δ0 = smuladd(s, a, BP[b0 + p], Δ0)
            Δ1 = smuladd(s, a, BP[b1 + p], Δ1)
            Δ2 = smuladd(s, a, BP[b2 + p], Δ2)
            Δ3 = smuladd(s, a, BP[b3 + p], Δ3)
        end

        C[i, j0]     = Δ0
        C[i, j0 + 1] = Δ1
        C[i, j0 + 2] = Δ2
        C[i, j0 + 3] = Δ3
    end

    return
end

# ===== sgemx_col! =====

function sgemx_col!(s::AbstractSemiring, C::AbstractMatrix{V}, A::AbstractMatrix, B::AbstractMatrix, AP::AbstractVector, BP::AbstractVector) where {V}
    m = size(C, 1)
    n = size(C, 2)
    k = size(A, 2)

    @inbounds for j0 in 1:SGEMX_NR:n
        nt = min(SGEMX_NR, n - j0 + 1)
        off = (j0 - 1) * k

        for p in 1:k
            for j in 1:nt
                BP[off + (p - 1) * SGEMX_NR + j] = B[p, j0 + j - 1]
            end
        end
    end

    @inbounds for i0 in 1:SGEMX_MR:m
        mt = min(SGEMX_MR, m - i0 + 1)
        off = (i0 - 1) * k

        for p in 1:k
            for i in 1:mt
                AP[off + (p - 1) * SGEMX_MR + i] = A[i0 + i - 1, p]
            end
        end
    end

    @inbounds for j0 in 1:SGEMX_NR:n
        for i0 in 1:SGEMX_MR:m
            mt = min(SGEMX_MR, m - i0 + 1)
            nt = min(SGEMX_NR, n - j0 + 1)

            if mt == SGEMX_MR && nt == SGEMX_NR
                sgemx_col_kernel!(s, C, AP, (i0 - 1) * k + 1, BP, (j0 - 1) * k + 1, i0, j0, k)
            else
                for j in 1:nt
                    for p in 1:k
                        b = BP[(j0 - 1) * k + (p - 1) * SGEMX_NR + j]

                        for i in 1:mt
                            C[i0 + i - 1, j0 + j - 1] = smuladd(s, AP[(i0 - 1) * k + (p - 1) * SGEMX_MR + i], b, C[i0 + i - 1, j0 + j - 1])
                        end
                    end
                end
            end
        end
    end

    return C
end

@generated function sgemx_col_kernel!(s::AbstractSemiring, C::AbstractMatrix, AP::AbstractVector, ap0::Int, BP::AbstractVector, bp0::Int, i0::Int, j0::Int, k::Int)
    load  = Expr[]
    la    = Expr[]
    lb    = Expr[]
    fma   = Expr[]
    store = Expr[]

    for i in 1:SGEMX_MR
        ai = Symbol(:a, i)
        push!(la, :($ai = AP[aoff + $(i - 1)]))
    end

    for j in 1:SGEMX_NR
        bj = Symbol(:b, j)
        push!(lb, :($bj = BP[boff + $(j - 1)]))

        for i in 1:SGEMX_MR
            ai  = Symbol(:a, i)
            Δij = Symbol(:Δ, i, j)
            push!(load,  :($Δij = C[i0 + $(i - 1), j0 + $(j - 1)]))
            push!(fma,   :($Δij = smuladd(s, $ai, $bj, $Δij)))
            push!(store, :(C[i0 + $(i - 1), j0 + $(j - 1)] = $Δij))
        end
    end

    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $(load...)

            for p in 1:k
                aoff = ap0 + (p - 1) * SGEMX_MR
                boff = bp0 + (p - 1) * SGEMX_NR

                $(la...)
                $(lb...)
                $(fma...)
            end

            $(store...)
        end

        return
    end
end

# ===== sgemv! =====

function sgemv!(s::AbstractSemiring, c::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    m = size(A, 1)
    k = size(A, 2)

    @inbounds for j in 1:k
        bj = b[j]

        for i in 1:m
            c[i] = smuladd(s, A[i, j], bj, c[i])
        end
    end

    return c
end

function sgemx!(s::AbstractSemiring, C::AbstractVector, A::AbstractMatrix, b::AbstractVector; nt::Integer = nthreads())
    sgemv!(s, C, A, b)
    return C
end

function sgemx!(s::AbstractSemiring, C::AbstractVector, a::AbstractVector, B::AbstractMatrix; nt::Integer = nthreads())
    sgemv!(s, C, transpose(B), a)
    return C
end
