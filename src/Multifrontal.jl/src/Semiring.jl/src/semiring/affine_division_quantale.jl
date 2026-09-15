struct AffGCDProd <: AbstractQuantale end

function szero(::AffGCDProd, ::Type{NTuple{6, T}}) where {T}
    return affzero(T)
end

function sone(::AffGCDProd, ::Type{NTuple{6, T}}) where {T}
    return affone(T)
end

function stop(::AffGCDProd, ::Type{NTuple{6, T}}) where {T}
    return afftop(T)
end

function splus(::AffGCDProd, a, b)
    return affplus(a..., b...)
end

function sprod(::AffGCDProd, a, b)
    return affprod(a..., b...)
end

function smuladd(s::AffGCDProd, a::A, b::B, c::C) where {A, B, C}
    T = promote_type(A, B, C)

    ⊥ = szero(s, T)
    u =  sone(s, T)
    ⊤ =  stop(s, T)

    if c == ⊤ || a == ⊥ || b == ⊥
        d = convert(T, c)
    elseif a == u
        d = convert(T, splus(s, b, c))
    elseif b == u
        d = convert(T, splus(s, a, c))
    else
        d = affmuladd(a..., b..., c...)
    end

    return d
end

#
#   [ 0  0  0 ]
#   [    0  0 ]
#   [       0 ]
#
function affzero(::Type{T}) where {T}
    u = zero(T)
    return (u, u, u, u, u, u)
end

#
#   [ 1  0  1 ]
#   [    0  0 ]
#   [       0 ]
#
function affone(::Type{T}) where {T}
    u = zero(T)
    e = one(T)
    return (e, u, e, u, u, u)
end

#
#   [ 1  0  0 ]
#   [    1  0 ]
#   [       1 ]
#
function afftop(::Type{T}) where {T}
    u = zero(T)
    e = one(T)
    return (e, u, u, e, u, e)
end

function affistop(r₁₁, r₂₂, r₃₃)
    return isone(r₁₁) && isone(r₂₂) && isone(r₃₃)
end

@inline function affinsert(r₁₁::T, r₁₂::T, r₁₃::T, r₂₂::T, r₂₃::T, r₃₃::T, x₁::T, x₂::T, x₃::T) where {T}
    if !iszero(x₁)
        if iszero(r₁₁)
            r₁₁ = x₁
            r₁₂ = x₂
            r₁₃ = x₃

            x₂ = zero(T)
            x₃ = zero(T)
        elseif isone(r₁₁) || isone(-r₁₁)
            x₂ = x₂ - x₁ * r₁₁ * r₁₂
            x₃ = x₃ - x₁ * r₁₁ * r₁₃
        else
            g, u, v = gcdx(r₁₁, x₁)

            s = r₁₁ ÷ g
            t = x₁  ÷ g

            n₁ = u * r₁₂ + v * x₂
            n₂ = u * r₁₃ + v * x₃

            x₂ = s * x₂ - t * r₁₂
            x₃ = s * x₃ - t * r₁₃

            r₁₁ = g
            r₁₂ = n₁
            r₁₃ = n₂
        end
    end

    if !iszero(x₂)
        if iszero(r₂₂)
            r₂₂ = x₂
            r₂₃ = x₃

            x₃ = zero(T)
        elseif isone(r₂₂) || isone(-r₂₂)
            x₃ = x₃ - x₂ * r₂₂ * r₂₃
        else
            g, u, v = gcdx(r₂₂, x₂)

            s = r₂₂ ÷ g
            t = x₂  ÷ g

            n₃ = u * r₂₃ + v * x₃

            x₃ = s * x₃ - t * r₂₃

            r₂₂ = g
            r₂₃ = n₃
        end
    end

    if !iszero(x₃)
        if iszero(r₃₃)
            r₃₃ = x₃
        else
            r₃₃ = gcd(r₃₃, x₃)
        end
    end

    return (r₁₁, r₁₂, r₁₃, r₂₂, r₂₃, r₃₃)
end

@inline function affnorm(r₁₁::T, r₁₂::T, r₁₃::T, r₂₂::T, r₂₃::T, r₃₃::T) where {T}
    if signbit(r₁₁)
        r₁₁ = -r₁₁
        r₁₂ = -r₁₂
        r₁₃ = -r₁₃
    end

    if signbit(r₂₂)
        r₂₂ = -r₂₂
        r₂₃ = -r₂₃
    end

    if signbit(r₃₃)
        r₃₃ = -r₃₃
    end

    if !iszero(r₃₃)
        r₂₃ = mod(r₂₃, r₃₃)
        r₁₃ = mod(r₁₃, r₃₃)
    end

    if !iszero(r₂₂)
        q = fld(r₁₂, r₂₂)
        r₁₂ = r₁₂ - q * r₂₂
        r₁₃ = r₁₃ - q * r₂₃
    end

    if !iszero(r₃₃)
        r₁₃ = mod(r₁₃, r₃₃)
    end

    return (r₁₁, r₁₂, r₁₃, r₂₂, r₂₃, r₃₃)
end

function affplus(a₁₁, a₁₂, a₁₃, a₂₂, a₂₃, a₃₃,
                 b₁₁, b₁₂, b₁₃, b₂₂, b₂₃, b₃₃)

    T = promote_eltype(a₁₁, a₁₂, a₁₃, a₂₂, a₂₃, a₃₃,
                       b₁₁, b₁₂, b₁₃, b₂₂, b₂₃, b₃₃)

    u = zero(T)
    ⊤ = afftop(T)

    affistop(a₁₁, a₂₂, a₃₃) && return ⊤; c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(a₁₁, a₁₂, a₁₃, a₂₂, a₂₃, a₃₃, b₁₁, b₁₂, b₁₃)
    affistop(c₁₁, c₂₂, c₃₃) && return ⊤; c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃, u,   b₂₂, b₂₃)
    affistop(c₁₁, c₂₂, c₃₃) && return ⊤; c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃, u,   u,   b₃₃)

    return affnorm(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃)
end

function affprod(a₁₁, a₁₂, a₁₃, a₂₂, a₂₃, a₃₃,
                  b₁₁, b₁₂, b₁₃, b₂₂, b₂₃, b₃₃)

    T = promote_eltype(a₁₁, a₁₂, a₁₃, a₂₂, a₂₃, a₃₃,
                       b₁₁, b₁₂, b₁₃, b₂₂, b₂₃, b₃₃)

    u = zero(T)
    ⊥ = affzero(T)
    ⊤ = afftop(T)

                                         c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(⊥...,                         b₁₁ * a₁₁, b₁₁ * a₁₂ + b₁₂ * a₁₃, b₁₃ * a₁₃)
    affistop(c₁₁, c₂₂, c₃₃) && return ⊤; c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃, u,         b₂₂ * a₁₃,             b₂₃ * a₁₃)
    affistop(c₁₁, c₂₂, c₃₃) && return ⊤; c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃, u,         u,                     b₃₃ * a₁₃)
    affistop(c₁₁, c₂₂, c₃₃) && return ⊤; c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃, u,         b₁₁ * a₂₂ + b₁₂ * a₂₃, b₁₃ * a₂₃)
    affistop(c₁₁, c₂₂, c₃₃) && return ⊤; c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃, u,         b₂₂ * a₂₃,             b₂₃ * a₂₃)
    affistop(c₁₁, c₂₂, c₃₃) && return ⊤; c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃, u,         u,                     b₃₃ * a₂₃)
    affistop(c₁₁, c₂₂, c₃₃) && return ⊤; c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃, u,         b₁₂ * a₃₃,             b₁₃ * a₃₃)
    affistop(c₁₁, c₂₂, c₃₃) && return ⊤; c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃, u,         b₂₂ * a₃₃,             b₂₃ * a₃₃)
    affistop(c₁₁, c₂₂, c₃₃) && return ⊤; c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃, u,         u,                     b₃₃ * a₃₃)

    return affnorm(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃)
end

function affmuladd(a₁₁, a₁₂, a₁₃, a₂₂, a₂₃, a₃₃,
                    b₁₁, b₁₂, b₁₃, b₂₂, b₂₃, b₃₃,
                    c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃)

    T = promote_eltype(a₁₁, a₁₂, a₁₃, a₂₂, a₂₃, a₃₃,
                       b₁₁, b₁₂, b₁₃, b₂₂, b₂₃, b₃₃,
                       c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃)

    u = zero(T)
    ⊤ = afftop(T)

    affistop(c₁₁, c₂₂, c₃₃) && return ⊤; c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃, b₁₁ * a₁₁, b₁₁ * a₁₂ + b₁₂ * a₁₃, b₁₃ * a₁₃)
    affistop(c₁₁, c₂₂, c₃₃) && return ⊤; c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃, u,         b₂₂ * a₁₃,             b₂₃ * a₁₃)
    affistop(c₁₁, c₂₂, c₃₃) && return ⊤; c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃, u,         u,                     b₃₃ * a₁₃)
    affistop(c₁₁, c₂₂, c₃₃) && return ⊤; c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃, u,         b₁₁ * a₂₂ + b₁₂ * a₂₃, b₁₃ * a₂₃)
    affistop(c₁₁, c₂₂, c₃₃) && return ⊤; c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃, u,         b₂₂ * a₂₃,             b₂₃ * a₂₃)
    affistop(c₁₁, c₂₂, c₃₃) && return ⊤; c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃, u,         u,                     b₃₃ * a₂₃)
    affistop(c₁₁, c₂₂, c₃₃) && return ⊤; c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃, u,         b₁₂ * a₃₃,             b₁₃ * a₃₃)
    affistop(c₁₁, c₂₂, c₃₃) && return ⊤; c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃, u,         b₂₂ * a₃₃,             b₂₃ * a₃₃)
    affistop(c₁₁, c₂₂, c₃₃) && return ⊤; c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃ = affinsert(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃, u,         u,                     b₃₃ * a₃₃)

    return affnorm(c₁₁, c₁₂, c₁₃, c₂₂, c₂₃, c₃₃)
end
