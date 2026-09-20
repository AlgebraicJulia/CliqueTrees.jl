# ===== workspace pool =====

function spool(::Type{V}, nt::Integer) where {V}
    mr = sgemx_width(V)

    apn = cld(SGEMX_LEAF, mr) * mr * SGEMX_LEAF
    bpn = cld(SGEMX_LEAF, SGEMX_NR) * SGEMX_NR * SGEMX_LEAF
    cpn = mr * SGEMX_NR

    pool = Channel{Tuple{FVector{V}, FVector{V}, FVector{V}}}(nt)

    for _ in 1:nt
        AP = FVector{V}(undef, apn)
        BP = FVector{V}(undef, bpn)
        CP = FVector{V}(undef, cpn)
        put!(pool, (AP, BP, CP))
    end

    return pool
end

include("sgemx.jl")
include("strsx.jl")
include("slu.jl")
