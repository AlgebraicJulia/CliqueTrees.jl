# ===== vecwidth =====

function vecwidth(::Type{T}) where {T}
    return 64 ÷ sizeof(T)
end

# ===== workspace pool =====

function spool_st(s::AbstractSemiring, ::Type{T}, ni::Integer, nj::Integer, nk::Integer) where {T}
    mr = SGEMX_MV * vecwidth(T)

    nic = min(ni, SGEMX_LEAF)
    njc = min(nj, SGEMX_LEAF)
    nkc = min(nk, SGEMX_LEAF)

    apn = cld(nic, mr) * mr * njc
    bpn = cld(nkc, SGEMX_NR) * SGEMX_NR * njc
    cpn = mr * SGEMX_NR

    AP = FVector{T}(undef, apn)
    BP = FVector{T}(undef, bpn)
    CP = FVector{T}(undef, cpn)

    return AP, BP, CP
end

function spool_st(s::AbstractSemiring, ::Type{T}) where {T}
    return spool_st(s, T, SGEMX_LEAF, SGEMX_LEAF, SGEMX_LEAF)
end

function spool_mt(s::AbstractSemiring, ::Type{T}, nt::Integer, ni::Integer, nj::Integer, nk::Integer) where {T}
    @assert nt >= 1

    pool = FVector{Tuple{FVector{T}, FVector{T}, FVector{T}}}(undef, nt)

    for t in 1:nt
        pool[t] = spool_st(s, T, ni, nj, nk)
    end

    return pool
end

function spool_mt(s::AbstractSemiring, ::Type{T}, nt::Integer) where {T}
    return spool_mt(s, T, nt, SGEMX_LEAF, SGEMX_LEAF, SGEMX_LEAF)
end

include("sdot.jl")
include("saxpy.jl")
include("sger.jl")
include("sgemx.jl")
include("strsx.jl")
include("strtri.jl")
include("sgetrs.jl")
include("sgetrf.jl")
include("sgetri.jl")
