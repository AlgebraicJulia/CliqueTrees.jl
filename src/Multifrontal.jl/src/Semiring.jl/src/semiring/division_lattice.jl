struct GCDLCM <: AbstractLattice end

const LCMGCD = DualLattice{GCDLCM}

function szero(::GCDLCM, ::Type{T}) where {T}
    return zero(T)
end

function sone(::GCDLCM, ::Type{T}) where {T <: Rational}
    return typemax(T)
end

function sone(::GCDLCM, ::Type{T}) where {T <: Integer}
    return one(T)
end

function splus(::GCDLCM, a, b)
    return safegcd(a, b)
end

function sprod(::GCDLCM, a, b)
    return safelcm(a, b)
end

function smuladd(s::GCDLCM, a::Rational, b::Rational, c::Rational)
    if isinf(c) || iszero(a) || iszero(b)
        d = c
    elseif iszero(c)
        d = sprod(s, a, b)
    else
        na, da = numerator(a), denominator(a)
        nb, db = numerator(b), denominator(b)
        nc, dc = numerator(c), denominator(c)

        nd, dd = gcdlcm(na, da, nb, db, nc, dc)
        d = unsafe_rational(nd, dd)
    end

    return d
end

function smuladd(s::LCMGCD, a::Rational, b::Rational, c::Rational)
    if iszero(c) || isinf(a) || isinf(b)
        d = c
    elseif isinf(c)
        d = sprod(s, a, b)
    else
        na, da = numerator(a), denominator(a)
        nb, db = numerator(b), denominator(b)
        nc, dc = numerator(c), denominator(c)

        dd, nd = gcdlcm(da, na, db, nb, dc, nc)
        d = unsafe_rational(nd, dd)
    end

    return d
end

function gcdlcm(na, da, nb, db, nc, dc)
    nd = gcdprod(na ÷ gcd(na, nb), nb, nc)
    dd = lcm(gcd(da, db), dc)
    return nd, dd
end
