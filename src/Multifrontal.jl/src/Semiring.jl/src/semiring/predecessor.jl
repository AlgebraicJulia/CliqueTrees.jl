struct Pred{S} <: AbstractQuantale
    s::S
end

struct Succ{S} <: AbstractQuantale
    s::S
end

struct UnsafePred{S} <: AbstractQuantale
    s::S
end

struct UnsafeSucc{S} <: AbstractQuantale
    s::S
end

const PredSucc{S} = Union{Pred{S}, Succ{S}}
const UnsafePredSucc{S} = Union{UnsafePred{S}, UnsafeSucc{S}}
const MaybeSafePredSucc{S} = Union{PredSucc{S}, UnsafePredSucc{S}}

# ----- integrality -----

function isintegral(::Type{Pred{S}}) where {S}
    return isintegral(S)
end

function isintegral(::Type{Succ{S}}) where {S}
    return isintegral(S)
end

function isintegral(::Type{UnsafePred{S}}) where {S}
    return isintegral(S)
end

function isintegral(::Type{UnsafeSucc{S}}) where {S}
    return isintegral(S)
end

# ----- pack / unpack -----

function flip(v::UInt32)
    if iszero(v & 0x80000000)
        w = 0x80000000
    else
        w = 0xffffffff
    end

    return v ⊻ w
end

@inline function flip(x::Vec{W, UInt64}) where {W}
    m = (reinterpret(Vec{W, UInt64}, reinterpret(Vec{W, Int64}, x) >> 63) & 0xffffffff00000000) | 0x8000000000000000
    return x ⊻ m
end

function unflip(v::UInt32)
    if iszero(v & 0x80000000)
        w = 0xffffffff
    else
        w = 0x80000000
    end

    return v ⊻ w
end

@inline function unflip(x::Vec{W, UInt64}) where {W}
    m = (~reinterpret(Vec{W, UInt64}, reinterpret(Vec{W, Int64}, x) >> 63) & 0xffffffff00000000) | 0x8000000000000000
    return x ⊻ m
end

function pack(s::MaybeSafePredSucc{S}, a) where {S <: Union{MinPlusLaw, MinProdLaw, MinProd}}
    return reinterpret(UInt64, a)
end

function pack(s::MaybeSafePredSucc{MinPlus}, a)
    w = reinterpret(UInt64, a)
    u = UInt32(w >> 32)

    if u == 0x80000000
        u = 0x00000000
    end

    v = UInt64(flip(u))
    return (v << 32) | (w & 0x00000000ffffffff)
end

function unpack(s::MaybeSafePredSucc{S}, w::UInt64) where {S <: Union{MinPlusLaw, MinProdLaw, MinProd}}
    if s isa PredSucc
        T = Tuple{UInt16, UInt16, Float32}
    else
        T = Tuple{UInt32, Float32}
    end

    return reinterpret(T, w)
end

function unpack(s::MaybeSafePredSucc{MinPlus}, w::UInt64)
    if s isa PredSucc
        T = Tuple{UInt16, UInt16, Float32}
    else
        T = Tuple{UInt32, Float32}
    end

    v = UInt64(unflip(UInt32(w >> 32)))
    w = (v << 32) | (w & 0x00000000ffffffff)
    return reinterpret(T, w)
end

# ----- slte -----

function slte(s::MaybeSafePredSucc{S}, a::UInt64, b::UInt64) where {S <: Union{MinPlusLaw, MinProdLaw, MinPlus, MinProd}}
    return b <= a
end

# ----- stop -----

function stop(s::MaybeSafePredSucc{S}, ::Type{UInt64}) where {S <: Union{MinPlusLaw, MinProd}}
    return 0x0000000000000000
end

function stop(s::MaybeSafePredSucc{MinProdLaw}, ::Type{UInt64})
    return 0x3f80000000000000
end

function stop(s::MaybeSafePredSucc{MinPlus}, ::Type{UInt64})
    return 0x007fffff00000000
end

# ----- szero -----

function szero(s::MaybeSafePredSucc{S}, ::Type{UInt64}) where {S <: Union{MinPlusLaw, MinProdLaw, MinProd}}
    return 0x7f80000000000000
end

function szero(s::MaybeSafePredSucc{MinPlus}, ::Type{UInt64})
    return 0xff80000000000000
end

# ----- sone -----

function sone(s::MaybeSafePredSucc{MinPlusLaw}, ::Type{UInt64})
    return 0x0000000000000000
end

function sone(s::MaybeSafePredSucc{S}, ::Type{UInt64}) where {S <: Union{MinProdLaw, MinProd}}
    return 0x3f80000000000000
end

function sone(s::MaybeSafePredSucc{MinPlus}, ::Type{UInt64})
    return 0x8000000000000000
end

# ----- splus -----

function splus(s::MaybeSafePredSucc{S}, a::UInt64, b::UInt64) where {S <: Union{MinPlusLaw, MinProdLaw, MinPlus, MinProd}}
    return min(a, b)
end

# ----- sprod -----

function sprod(s::PredSucc{S}, a::UInt64, b::UInt64) where {S <: Union{MinPlusLaw, MinProdLaw, MinPlus, MinProd}}
    W = 0x000000000000ffff
    H = 0x00000000ffff0000

    au = UInt32(a >> 32)
    bu = UInt32(b >> 32)

    if S <: MinPlus
        au = unflip(au)
        bu = unflip(bu)
    end

    av = reinterpret(Float32, au)
    bv = reinterpret(Float32, bu)

    if S <: Union{MinProdLaw, MinProd}
        cv = av * bv
    else
        cv = av + bv
    end

    cu = reinterpret(UInt32, cv)

    if S <: MinPlus
        cu = flip(cu)
    end

    c = (UInt64(cu) << 32) | (((a & H) + (b & H)) & H)

    a &= W
    b &= W

    if s isa Pred
        if iszero(b)
            c |= a
        else
            c |= b
        end
    else
        if iszero(a)
            c |= b
        else
            c |= a
        end
    end

    return c
end

function sprod(s::UnsafePredSucc{S}, a::UInt64, b::UInt64) where {S <: Union{MinPlusLaw, MinProdLaw, MinPlus, MinProd}}
    W = 0x00000000ffffffff

    au = UInt32(a >> 32)
    bu = UInt32(b >> 32)

    if S <: MinPlus
        au = unflip(au)
        bu = unflip(bu)
    end

    av = reinterpret(Float32, au)
    bv = reinterpret(Float32, bu)

    if S <: Union{MinProdLaw, MinProd}
        cv = av * bv
    else
        cv = av + bv
    end

    cu = reinterpret(UInt32, cv)

    if S <: MinPlus
        cu = flip(cu)
    end

    c = UInt64(cu) << 32

    a &= W
    b &= W

    if s isa UnsafePred
        if iszero(b)
            c |= a
        else
            c |= b
        end
    else
        if iszero(a)
            c |= b
        else
            c |= a
        end
    end

    return c
end

# ----- smuladd -----

function smuladd(s::PredSucc{S}, a::UInt64, b::UInt64, c::UInt64) where {S <: Union{MinPlusLaw, MinProdLaw, MinPlus, MinProd}}
    W = 0x000000000000ffff
    H = 0x00000000ffff0000

    au = UInt32(a >> 32)
    bu = UInt32(b >> 32)

    if S <: MinPlus
        au = unflip(au)
        bu = unflip(bu)
    end

    av = reinterpret(Float32, au)
    bv = reinterpret(Float32, bu)

    if S <: Union{MinProdLaw, MinProd}
        dv = av * bv
    else
        dv = av + bv
    end

    du = reinterpret(UInt32, dv)

    if S <: MinPlus
        du = flip(du)
    end

    d = (UInt64(du) << 32) | (((a & H) + (b & H)) & H)

    a &= W
    b &= W

    if s isa Pred
        if iszero(b)
            d |= a
        else
            d |= b
        end
    else
        if iszero(a)
            d |= b
        else
            d |= a
        end
    end

    return min(d, c)
end

function smuladd(s::UnsafePredSucc{S}, a::UInt64, b::UInt64, c::UInt64) where {S <: Union{MinPlusLaw, MinProdLaw, MinPlus, MinProd}}
    W = 0x00000000ffffffff

    au = UInt32(a >> 32)
    bu = UInt32(b >> 32)

    if S <: MinPlus
        au = unflip(au)
        bu = unflip(bu)
    end

    av = reinterpret(Float32, au)
    bv = reinterpret(Float32, bu)

    if S <: Union{MinProdLaw, MinProd}
        dv = av * bv
    else
        dv = av + bv
    end

    du = reinterpret(UInt32, dv)

    if S <: MinPlus
        du = flip(du)
    end

    d = UInt64(du) << 32

    a &= W
    b &= W

    if s isa UnsafePred
        if iszero(b)
            d |= a
        else
            d |= b
        end
    else
        if iszero(a)
            d |= b
        else
            d |= a
        end
    end

    return min(d, c)
end

@inline function smuladd(s::PredSucc{S}, a::Vec{W, UInt64}, b::UInt64, c::Vec{W, UInt64}) where {S <: Union{MinPlusLaw, MinProdLaw, MinPlus, MinProd}, W}
    V = 0xffffffff00000000
    H = 0x00000000ffff0000
    I = 0x000000000000ffff

    if b & V == szero(s, UInt64)
        d = c
    else
        au = a
        bu = Vec{W, UInt64}(b)

        if S <: MinPlus
            au = unflip(au)
            bu = unflip(bu)
        end

        av = reinterpret(Vec{2W, Float32}, au & V)
        bv = reinterpret(Vec{2W, Float32}, bu & V)

        if S <: Union{MinProdLaw, MinProd}
            dv = av * bv
        else
            dv = av + bv
        end

        du = reinterpret(Vec{W, UInt64}, dv)

        if S <: MinPlus
            du = flip(du)
        end

        d = (du & V) | (((a & H) + Vec{W, UInt64}(b & H)) & H)

        a &= I
        b &= I

        if s isa Pred
            if iszero(b)
                w = a
            else
                w = Vec{W, UInt64}(b)
            end
        else
            w = vifelse(a == Vec{W, UInt64}(0x0000000000000000), Vec{W, UInt64}(b), a)
        end

        d = min(d | w, c)
    end

    return d
end

# ----- sstar -----

#
#   a* = sone
#
function sstar(s::MaybeSafePredSucc{MinPlusLaw}, a::UInt64)
    return 0x0000000000000000
end

function sstar(s::MaybeSafePredSucc{MinProdLaw}, a::UInt64)
    return 0x3f80000000000000
end

function sstar(s::MaybeSafePredSucc{MinPlus}, a::UInt64)
    if a < 0x8000000000000000
        w = 0x007fffff00000000
    else
        w = 0x8000000000000000
    end

    return w
end

function sstar(s::MaybeSafePredSucc{MinProd}, a::UInt64)
    if a < 0x3f80000000000000
        w = 0x0000000000000000
    else
        w = 0x3f80000000000000
    end

    return w
end
