const SGEMX_NR   = 4
const SGEMX_LEAF = 256

function sgemx_width(::Type{V}) where {V}
    return 64 ÷ sizeof(V)
end

# ===== sgemx! =====

function sgemx!(s::AbstractSemiring, C::AbstractMatrix{V}, A::AbstractMatrix{V}, B::AbstractMatrix{V}; nt::Integer = nthreads()) where {V}
    @assert size(C, 1) == size(A, 1)
    @assert size(C, 2) == size(B, 2)
    @assert size(A, 2) == size(B, 1)

    ni = size(C, 1)
    nj = size(A, 2)
    nk = size(C, 2)

    mr  = sgemx_width(V)
    nic = min(ni, SGEMX_LEAF)
    njc = min(nj, SGEMX_LEAF)
    nkc = min(nk, SGEMX_LEAF)

    apn = cld(nic, mr) * mr * njc
    bpn = cld(nkc, SGEMX_NR) * SGEMX_NR * njc
    cpn = mr * SGEMX_NR

    if nt <= 1 || max(ni, nj, nk) <= SGEMX_LEAF
        AP = Vector{V}(undef, apn)
        BP = Vector{V}(undef, bpn)
        CP = Vector{V}(undef, cpn)
        sgemx_st!(s, C, A, B, AP, BP, CP)
    else
        depth = ceil(Int, log2(nt)) + 1
        work = Channel{Tuple{Vector{V}, Vector{V}, Vector{V}}}(nt)

        for _ in 1:nt
            AP = Vector{V}(undef, apn)
            BP = Vector{V}(undef, bpn)
            CP = Vector{V}(undef, cpn)
            put!(work, (AP, BP, CP))
        end

        sgemx_mt!(s, C, A, B, work, depth)
    end

    return C
end

function sgemx_mt!(s::AbstractSemiring, C::AbstractMatrix{V}, A::AbstractMatrix, B::AbstractMatrix, work::Channel, depth::Int) where {V}
    ni = size(C, 1)
    nj = size(A, 2)
    nk = size(C, 2)

    if depth <= 0 || (ni <= SGEMX_LEAF && nj <= SGEMX_LEAF && nk <= SGEMX_LEAF)
        AP, BP, CP = take!(work)

        try
            sgemx_st!(s, C, A, B, AP, BP, CP)
        finally
            put!(work, (AP, BP, CP))
        end
    else
        mx = max(ni, nj, nk)

        if ni == mx
            #
            #   [ C₁ ] = [ A₁ ] B
            #   [ C₂ ]   [ A₂ ]
            #
            mr = sgemx_width(V)

            hi = ni >> 1
            hi -= hi % mr
            hi = max(hi, mr)

            C₁ = view(C,      1:hi, 1:nk)
            C₂ = view(C, hi + 1:ni, 1:nk)
            A₁ = view(A,      1:hi, 1:nj)
            A₂ = view(A, hi + 1:ni, 1:nj)

            task = @spawn sgemx_mt!(s, C₁, A₁, B, work, depth - 1)
            sgemx_mt!(s, C₂, A₂, B, work, depth - 1)
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
            B₁ = view(B, 1:nj,      1:hk)
            B₂ = view(B, 1:nj, hk + 1:nk)

            task = @spawn sgemx_mt!(s, C₁, A, B₁, work, depth - 1)
            sgemx_mt!(s, C₂, A, B₂, work, depth - 1)
            wait(task)
        else
            #
            #   C = [ A₁ A₂ ] [ B₁ ]
            #                 [ B₂ ]
            #
            hj = nj >> 1

            A₁ = view(A,      1:ni,      1:hj)
            A₂ = view(A,      1:ni, hj + 1:nj)
            B₁ = view(B,      1:hj,      1:nk)
            B₂ = view(B, hj + 1:nj,      1:nk)

            sgemx_mt!(s, C, A₁, B₁, work, depth)
            sgemx_mt!(s, C, A₂, B₂, work, depth)
        end
    end

    return C
end

function sgemx_st!(s::AbstractSemiring, C::AbstractMatrix{V}, A::AbstractMatrix, B::AbstractMatrix, AP::AbstractVector, BP::AbstractVector, CP::AbstractVector) where {V}
    ni = size(C, 1)
    nj = size(A, 2)
    nk = size(C, 2)

    if ni <= SGEMX_LEAF && nj <= SGEMX_LEAF && nk <= SGEMX_LEAF
        sgemx_leaf!(s, C, A, B, AP, BP, CP)
    else
        mx = max(ni, nj, nk)

        if ni == mx
            #
            #   [ C₁ ] = [ A₁ ] B
            #   [ C₂ ]   [ A₂ ]
            #
            mr = sgemx_width(V)

            hi = ni >> 1
            hi -= hi % mr
            hi = max(hi, mr)

            C₁ = view(C,      1:hi, 1:nk)
            C₂ = view(C, hi + 1:ni, 1:nk)
            A₁ = view(A,      1:hi, 1:nj)
            A₂ = view(A, hi + 1:ni, 1:nj)

            sgemx_st!(s, C₁, A₁, B, AP, BP, CP)
            sgemx_st!(s, C₂, A₂, B, AP, BP, CP)
        elseif nk == mx
            #
            #   [ C₁ C₂ ] = A [ B₁ B₂ ]
            #
            hk = nk >> 1
            hk -= hk % SGEMX_NR
            hk = max(hk, SGEMX_NR)

            C₁ = view(C, 1:ni,      1:hk)
            C₂ = view(C, 1:ni, hk + 1:nk)
            B₁ = view(B, 1:nj,      1:hk)
            B₂ = view(B, 1:nj, hk + 1:nk)

            sgemx_st!(s, C₁, A, B₁, AP, BP, CP)
            sgemx_st!(s, C₂, A, B₂, AP, BP, CP)
        else
            #
            #   C = [ A₁ A₂ ] [ B₁ ]
            #                 [ B₂ ]
            #
            hj = nj >> 1

            A₁ = view(A,      1:ni,      1:hj)
            A₂ = view(A,      1:ni, hj + 1:nj)
            B₁ = view(B,      1:hj,      1:nk)
            B₂ = view(B, hj + 1:nj,      1:nk)

            sgemx_st!(s, C, A₁, B₁, AP, BP, CP)
            sgemx_st!(s, C, A₂, B₂, AP, BP, CP)
        end
    end

    return C
end

# ===== sgemx_leaf! =====

function sgemx_leaf!(s::AbstractSemiring, C::AbstractMatrix{T}, A::AbstractMatrix, B::AbstractMatrix, AP::AbstractVector, BP::AbstractVector, CP::AbstractVector, mr::Val{MR} = Val(sgemx_width(T))) where {T, MR}
    ni = size(C, 1)
    nj = size(A, 2)
    nk = size(C, 2)
    z = szero(s, T)

    @inbounds for i0 in 0:MR:ni - 1
        it = min(MR, ni - i0); ip0 = i0 * nj

        for j in 1:nj
            for ip in 1:it
                AP[ip0 + (j - 1) * MR + ip] = A[i0 + ip, j]
            end

            for ip in it + 1:MR
                AP[ip0 + (j - 1) * MR + ip] = z
            end
        end
    end

    @inbounds for k0 in 0:SGEMX_NR:nk - 1
        kt = min(SGEMX_NR, nk - k0); kp0 = k0 * nj

        for j in 1:nj
            for kp in 1:kt
                BP[kp0 + (j - 1) * SGEMX_NR + kp] = B[j, k0 + kp]
            end

            for kp in kt + 1:SGEMX_NR
                BP[kp0 + (j - 1) * SGEMX_NR + kp] = z
            end
        end
    end

    @inbounds for k0 in 0:SGEMX_NR:nk - 1
        kt = min(SGEMX_NR, nk - k0); kp0 = k0 * nj

        for i0 in 0:MR:ni - 1
            it = min(MR, ni - i0); ip0 = i0 * nj

            if it == MR && kt == SGEMX_NR
                sgemx_kernel!(s, C, i0, k0, AP, ip0 + 1, BP, kp0 + 1, nj, mr)
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
                    sgemx_kernel!(s, pointer(CP), MR, AP, ip0 + 1, BP, kp0 + 1, nj, mr)
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

function sgemx_kernel!(s::AbstractSemiring, pC::Ptr{T}, ldC::Int, AP::AbstractVector, ip0::Int, BP::AbstractVector, kp0::Int, nj::Int, ::Val{MR}) where {T, MR}
    w = ldC * sizeof(T)

    p1 = pC
    p2 = p1 + w
    p3 = p2 + w
    p4 = p3 + w

    c1 = vload(Vec{MR, T}, p1)
    c2 = vload(Vec{MR, T}, p2)
    c3 = vload(Vec{MR, T}, p3)
    c4 = vload(Vec{MR, T}, p4)

    @inbounds for jp in 1:nj
        a = vload(Vec{MR, T}, AP, ip0 + (jp - 1) * MR)
        kpj = kp0 + (jp - 1) * SGEMX_NR
        c1 = smuladd(s, a, BP[kpj],     c1)
        c2 = smuladd(s, a, BP[kpj + 1], c2)
        c3 = smuladd(s, a, BP[kpj + 2], c3)
        c4 = smuladd(s, a, BP[kpj + 3], c4)
    end

    vstore(c1, p1)
    vstore(c2, p2)
    vstore(c3, p3)
    vstore(c4, p4)
    return
end

function sgemx_kernel!(s::AbstractSemiring, C::AbstractMatrix{T}, i0::Int, k0::Int, AP::AbstractVector, ip0::Int, BP::AbstractVector, kp0::Int, nj::Int, mr::Val{MR}) where {T, MR}
    @preserve C begin
        pC = unsafe_convert(Ptr{T}, C) + (k0 * stride(C, 2) + i0) * sizeof(T)
        sgemx_kernel!(s, pC, stride(C, 2), AP, ip0, BP, kp0, nj, mr)
    end

    return
end

# ===== sgemv! =====

function sgemv!(s::AbstractSemiring, c::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    ni = size(A, 1)
    nj = size(A, 2)

    @inbounds for j in 1:nj
        bj = b[j]

        for i in 1:ni
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
