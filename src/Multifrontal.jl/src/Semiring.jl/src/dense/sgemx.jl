const SGEMX_LEAF = 256

# ===== register tile =====

if test_cpu_feature(JL_X86_avx512f)
    const SGEMX_ISA = :avx512
elseif test_cpu_feature(JL_X86_avx2) && test_cpu_feature(JL_X86_fma)
    const SGEMX_ISA = :avx2
else
    const SGEMX_ISA = :other
end

@static if SGEMX_ISA === :avx512
    const SGEMX_MV = 2
    const SGEMX_NR = 8
elseif SGEMX_ISA === :avx2
    const SGEMX_MV = 1
    const SGEMX_NR = 6
else
    const SGEMX_MV = 1
    const SGEMX_NR = 4
end

# ===== sgemx! =====

function sgemx!(s::AbstractSemiring, tA::Val{TA}, tB::Val{TB}, C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}; nt::Integer = nthreads()) where {T, TA, TB}
    @assert stride(C, 1) == 1

    ni = size(C, 1)
    nk = size(C, 2)

    if TA === :N || TA === :R
        nj = size(A, 2)
    else
        nj = size(A, 1)
    end

    if nt <= 1 || max(ni, nk) <= SGEMX_LEAF || ni * nj * nk < SGEMX_LEAF * max(ni, nj, nk)
        AP, BP, CP = spool_st(T, ni, nj, nk)
        sgemx_st!(s, tA, tB, C, A, B, AP, BP, CP)
    else
        pool = spool_mt(T, nt, ni, nj, nk)
        sgemx_mt!(s, tA, tB, C, A, B, pool, nt)
    end

    return C
end

function sgemx!(s::AbstractSemiring, tA::R_OR_C, tB::R_OR_C, C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}; nt::Integer = nthreads()) where {T}
    return error("not supported")
end

function sgemx!(s::AbstractSemiring, tA::N_OR_R, tB::Val, c::AbstractVector{T}, A::AbstractMatrix{T}, b::AbstractVector; nt::Integer = nthreads()) where {T}
    ni = size(A, 1)
    nj = size(A, 2)
    sj = stride(A, 2)

    Z = sizeof(T)

    @preserve c A begin
        pc = pointer(c)

        @inbounds for j in 1:nj
            pa = pointer(A) + (j - 1) * sj * Z
            saxpy_kern!(s, tA, tB, Val(:R), pc, pa, b[j], ni)
        end
    end

    return c
end

function sgemx!(s::AbstractSemiring, tA::T_OR_C, tB::Val, c::AbstractVector, A::AbstractMatrix{T}, b::AbstractVector; nt::Integer = nthreads()) where {T}
    ni = size(A, 2)
    nj = size(A, 1)
    sj = stride(A, 2)

    Z = sizeof(T)

    op = compose(tA, tB)

    @preserve A b begin
        pb = pointer(b)

        @inbounds for i in 1:ni
            pa = pointer(A) + (i - 1) * sj * Z
            c[i] = splus(s, c[i], sdot_kern!(s, tA, tB, op, pa, pb, nj), op)
        end
    end

    return c
end

function sgemx!(s::AbstractSemiring, tA::N_OR_R, tB::N_OR_R, c::AbstractVector, a::AbstractVector, B::AbstractMatrix{T}; nt::Integer = nthreads()) where {T}
    ni = size(B, 2)
    nj = size(B, 1)
    sj = stride(B, 2)

    Z = sizeof(T)

    op = compose(tA, tB)

    @preserve a B begin
        pa = pointer(a)

        @inbounds for i in 1:ni
            pb = pointer(B) + (i - 1) * sj * Z
            c[i] = splus(s, c[i], sdot_kern!(s, tA, tB, op, pa, pb, nj), op)
        end
    end

    return c
end

function sgemx!(s::AbstractSemiring, tA::N_OR_R, tB::T_OR_C, c::AbstractVector{T}, a::AbstractVector, B::AbstractMatrix{T}; nt::Integer = nthreads()) where {T}
    ni = size(B, 1)
    nj = size(B, 2)
    sj = stride(B, 2)

    Z = sizeof(T)

    @preserve c B begin
        pc = pointer(c)

        @inbounds for j in 1:nj
            pb = pointer(B) + (j - 1) * sj * Z
            saxpy_kern!(s, tA, tB, Val(:L), pc, pb, a[j], ni)
        end
    end

    return c
end

# ===== sgemx_mt! =====

function sgemx_mt!(s::AbstractSemiring, tA::Val{TA}, tB::Val{TB}, C::AbstractMatrix{T}, A::AbstractMatrix, B::AbstractMatrix, pool::AbstractVector, nt::Integer) where {T, TA, TB}
    ni = size(C, 1)
    nk = size(C, 2)

    if TA === :N || TA === :R
        nj = size(A, 2)
    else
        nj = size(A, 1)
    end

    if nt <= 1 || max(ni, nk) <= SGEMX_LEAF || ni * nj * nk < SGEMX_LEAF * max(ni, nj, nk)
        AP, BP, CP = pool[1]
        sgemx_st!(s, tA, tB, C, A, B, AP, BP, CP)
    else
        mx = max(ni, nj, nk)

        if ni == mx
            #
            #   [ C₁ ] = [ A₁ ] B
            #   [ C₂ ]   [ A₂ ]
            #
            mr = SGEMX_MV * vecwidth(T)

            hi = ni >> 1
            hi -= hi % mr
            hi = max(hi, mr)

            C₁ = view(C,      1:hi, 1:nk)
            C₂ = view(C, hi + 1:ni, 1:nk)

            if TA === :N || TA === :R
                A₁ = view(A,      1:hi, 1:nj)
                A₂ = view(A, hi + 1:ni, 1:nj)
            else
                A₁ = view(A, 1:nj,      1:hi)
                A₂ = view(A, 1:nj, hi + 1:ni)
            end

            nt₁ = nt >> 1
            pool₁ = view(pool, 1:nt₁)
            pool₂ = view(pool, nt₁ + 1:nt)
            task = @spawn sgemx_mt!(s, tA, tB, $C₁, $A₁, B, $pool₁, $nt₁)
            sgemx_mt!(s, tA, tB, C₂, A₂, B, pool₂, nt - nt₁)
            wait(task)
        elseif nk == mx
            #
            #   [ C₁ C₂ ] = A [ B₁ B₂ ]
            #
            hk = nk >> 1
            hk -= hk % SGEMX_NR
            hk = max(hk, SGEMX_NR)

            C₁ = view(C, 1:ni,      1:hk)
            C₂ = view(C, 1:ni, hk + 1:nk)

            if TB === :N || TB === :R
                B₁ = view(B, 1:nj,      1:hk)
                B₂ = view(B, 1:nj, hk + 1:nk)
            else
                B₁ = view(B,      1:hk, 1:nj)
                B₂ = view(B, hk + 1:nk, 1:nj)
            end

            nt₁ = nt >> 1
            pool₁ = view(pool, 1:nt₁)
            pool₂ = view(pool, nt₁ + 1:nt)
            task = @spawn sgemx_mt!(s, tA, tB, $C₁, A, $B₁, $pool₁, $nt₁)
            sgemx_mt!(s, tA, tB, C₂, A, B₂, pool₂, nt - nt₁)
            wait(task)
        else
            #
            #   C = [ A₁ A₂ ] [ B₁ ]
            #                 [ B₂ ]
            #
            hj = nj >> 1

            if TA === :N || TA === :R
                A₁ = view(A, 1:ni,      1:hj)
                A₂ = view(A, 1:ni, hj + 1:nj)
            else
                A₁ = view(A,      1:hj, 1:ni)
                A₂ = view(A, hj + 1:nj, 1:ni)
            end

            if TB === :N || TB === :R
                B₁ = view(B,      1:hj, 1:nk)
                B₂ = view(B, hj + 1:nj, 1:nk)
            else
                B₁ = view(B, 1:nk,      1:hj)
                B₂ = view(B, 1:nk, hj + 1:nj)
            end

            sgemx_mt!(s, tA, tB, C, A₁, B₁, pool, nt)
            sgemx_mt!(s, tA, tB, C, A₂, B₂, pool, nt)
        end
    end

    return C
end

# ===== sgemx_st! =====

function sgemx_st!(s::AbstractSemiring, C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, AP::AbstractVector, BP::AbstractVector, CP::AbstractVector)
    return sgemx_st!(s, Val(:N), Val(:N), C, A, B, AP, BP, CP)
end

function sgemx_st!(s::AbstractSemiring, tA::Val{TA}, tB::Val{TB}, C::AbstractMatrix{T}, A::AbstractMatrix, B::AbstractMatrix, AP::AbstractVector, BP::AbstractVector, CP::AbstractVector) where {T, TA, TB}
    ni = size(C, 1)
    nk = size(C, 2)

    if TA === :N || TA === :R
        nj = size(A, 2)
    else
        nj = size(A, 1)
    end

    if ni <= SGEMX_LEAF && nj <= SGEMX_LEAF && nk <= SGEMX_LEAF
        sgemx2!(s, tA, tB, C, A, B, AP, BP, CP)
    else
        mx = max(ni, nj, nk)

        if ni == mx
            #
            #   [ C₁ ] = [ A₁ ] B
            #   [ C₂ ]   [ A₂ ]
            #
            mr = SGEMX_MV * vecwidth(T)

            hi = ni >> 1
            hi -= hi % mr
            hi = max(hi, mr)

            C₁ = view(C,      1:hi, 1:nk)
            C₂ = view(C, hi + 1:ni, 1:nk)

            if TA === :N || TA === :R
                A₁ = view(A,      1:hi, 1:nj)
                A₂ = view(A, hi + 1:ni, 1:nj)
            else
                A₁ = view(A, 1:nj,      1:hi)
                A₂ = view(A, 1:nj, hi + 1:ni)
            end

            sgemx_st!(s, tA, tB, C₁, A₁, B, AP, BP, CP)
            sgemx_st!(s, tA, tB, C₂, A₂, B, AP, BP, CP)
        elseif nk == mx
            #
            #   [ C₁ C₂ ] = A [ B₁ B₂ ]
            #
            hk = nk >> 1
            hk -= hk % SGEMX_NR
            hk = max(hk, SGEMX_NR)

            C₁ = view(C, 1:ni,      1:hk)
            C₂ = view(C, 1:ni, hk + 1:nk)

            if TB === :N || TB === :R
                B₁ = view(B, 1:nj,      1:hk)
                B₂ = view(B, 1:nj, hk + 1:nk)
            else
                B₁ = view(B,      1:hk, 1:nj)
                B₂ = view(B, hk + 1:nk, 1:nj)
            end

            sgemx_st!(s, tA, tB, C₁, A, B₁, AP, BP, CP)
            sgemx_st!(s, tA, tB, C₂, A, B₂, AP, BP, CP)
        else
            #
            #   C = [ A₁ A₂ ] [ B₁ ]
            #                 [ B₂ ]
            #
            hj = nj >> 1

            if TA === :N || TA === :R
                A₁ = view(A, 1:ni,      1:hj)
                A₂ = view(A, 1:ni, hj + 1:nj)
            else
                A₁ = view(A,      1:hj, 1:ni)
                A₂ = view(A, hj + 1:nj, 1:ni)
            end

            if TB === :N || TB === :R
                B₁ = view(B,      1:hj, 1:nk)
                B₂ = view(B, hj + 1:nj, 1:nk)
            else
                B₁ = view(B, 1:nk,      1:hj)
                B₂ = view(B, 1:nk, hj + 1:nj)
            end

            sgemx_st!(s, tA, tB, C, A₁, B₁, AP, BP, CP)
            sgemx_st!(s, tA, tB, C, A₂, B₂, AP, BP, CP)
        end
    end

    return C
end

# ===== sgemx2! =====

function sgemx2!(s::AbstractSemiring, tA::Val{TA}, tB::Val{TB}, C::AbstractMatrix{T}, A::AbstractMatrix, B::AbstractMatrix, AP::AbstractVector, BP::AbstractVector, CP::AbstractVector, mr::Val{MR} = Val(SGEMX_MV * vecwidth(T))) where {T, MR, TA, TB}
    ni = size(C, 1)
    nk = size(C, 2)

    if TA === :N || TA === :R
        nj = size(A, 2)
    else
        nj = size(A, 1)
    end

    z = szero(s, T, Val(:N))

    sgemx_pack_A!(s, tA, tB, AP, A, ni, nj, z, mr)
    sgemx_pack_B!(s, tA, tB, BP, B, nk, nj, z)

    @inbounds for k0 in 0:SGEMX_NR:nk - 1
        kt = min(SGEMX_NR, nk - k0); kp0 = k0 * nj

        for i0 in 0:MR:ni - 1
            it = min(MR, ni - i0); ip0 = i0 * nj

            if it == MR && kt == SGEMX_NR
                sgemx_kern!(s, tA, tB, C, i0, k0, AP, ip0 + 1, BP, kp0 + 1, nj, mr)
            else
                for kp in 1:kt
                    for ip in 1:it
                        CP[(kp - 1) * MR + ip] = C[i0 + ip, k0 + kp]
                    end

                    for ip in it + 1:MR
                        CP[(kp - 1) * MR + ip] = z
                    end
                end

                for kp in kt + 1:SGEMX_NR
                    for ip in 1:MR
                        CP[(kp - 1) * MR + ip] = z
                    end
                end

                @preserve CP begin
                    sgemx_kern!(s, tA, tB, pointer(CP), MR, AP, ip0 + 1, BP, kp0 + 1, nj, mr)
                end

                for kp in 1:kt
                    for ip in 1:it
                        C[i0 + ip, k0 + kp] = CP[(kp - 1) * MR + ip]
                    end
                end
            end
        end
    end

    return C
end

# ===== sgemx_pack_A! =====

function sgemx_pack_A!(s::AbstractSemiring, tA::Val{TA}, tB::Val{TB}, AP::AbstractVector, A::AbstractMatrix, ni::Int, nj::Int, z, ::Val{MR}) where {TA, TB, MR}
    @inbounds for i0 in 0:MR:ni - 1
        it = min(MR, ni - i0); ip0 = i0 * nj

        for j in 1:nj
            for ip in 1:it
                if TA === :N || TA === :R
                    AP[ip0 + (j - 1) * MR + ip] = A[i0 + ip, j]
                else
                    AP[ip0 + (j - 1) * MR + ip] = A[j, i0 + ip]
                end
            end

            for ip in it + 1:MR
                AP[ip0 + (j - 1) * MR + ip] = z
            end
        end
    end

    return AP
end

# ===== sgemx_pack_B! =====

function sgemx_pack_B!(s::AbstractSemiring, tA::Val{TA}, tB::Val{TB}, BP::AbstractVector, B::AbstractMatrix, nk::Int, nj::Int, z) where {TA, TB}
    @inbounds for k0 in 0:SGEMX_NR:nk - 1
        kt = min(SGEMX_NR, nk - k0); kp0 = k0 * nj

        for j in 1:nj
            for kp in 1:kt
                if TB === :N || TB === :R
                    BP[kp0 + (j - 1) * SGEMX_NR + kp] = B[j, k0 + kp]
                else
                    BP[kp0 + (j - 1) * SGEMX_NR + kp] = B[k0 + kp, j]
                end
            end

            for kp in kt + 1:SGEMX_NR
                BP[kp0 + (j - 1) * SGEMX_NR + kp] = z
            end
        end
    end

    return BP
end

# ===== sgemx_kern! =====

@generated function sgemx_kern!(s::AbstractSemiring, tA::Val, tB::Val, pC::Ptr{T}, ldC::Int, AP::AbstractVector, ip0::Int, BP::AbstractVector, kp0::Int, nj::Int, ::Val{MR}) where {T, MR}
    W = vecwidth(T)
    MV = MR ÷ W
    NR = SGEMX_NR
    Z = sizeof(T)

    @assert MR == MV * W

    c(v, k) = Symbol(:c_, v, :_, k)
    a(v) = Symbol(:a_, v)
    b(k) = Symbol(:b_, k)

    init = Expr(:block)
    body = Expr(:block)
    term = Expr(:block)

    for k in 1:NR, v in 1:MV
        off = :(($(k - 1) * ldC + $((v - 1) * W)) * $Z)
        push!(init.args, :($(c(v, k)) = vload(Vec{$W, $T}, pC + $off)))
        push!(term.args, :(vstore($(c(v, k)), pC + $off)))
    end

    for v in 1:MV
        push!(body.args, :($(a(v)) = vload(Vec{$W, $T}, pA + $((v - 1) * W * Z))))
    end

    for k in 1:NR
        push!(body.args, :($(b(k)) = unsafe_load(pB, $k)))

        for v in 1:MV
            push!(body.args, :($(c(v, k)) = @inline smuladd(s, $(a(v)), $(b(k)), $(c(v, k)), tA, tB)))
        end
    end

    return quote
        Base.GC.@preserve AP BP begin
            pA = pointer(AP, ip0)
            pB = pointer(BP, kp0)
            $init

            for _ in 1:nj
                $body
                pA += $(MR * Z)
                pB += $(NR * Z)
            end

            $term
        end

        return
    end
end

function sgemx_kern!(s::AbstractSemiring, tA::Val, tB::Val, C::AbstractMatrix{T}, i0::Int, k0::Int, AP::AbstractVector, ip0::Int, BP::AbstractVector, kp0::Int, nj::Int, mr::Val{MR}) where {T, MR}
    @preserve C begin
        pC = unsafe_convert(Ptr{T}, C) + (k0 * stride(C, 2) + i0) * sizeof(T)
        sgemx_kern!(s, tA, tB, pC, stride(C, 2), AP, ip0, BP, kp0, nj, mr)
    end

    return
end
