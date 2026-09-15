struct GCDProd <: AbstractQuantale end

struct LCMProd <: AbstractQuantale end

const DivisionQuantale = Union{GCDProd, LCMProd}

function stop(::GCDProd, ::Type{T}) where {T}
    return one(T)
end

function stop(::LCMProd, ::Type{T}) where {T}
    return zero(T)
end

function szero(::GCDProd, ::Type{T}) where {T}
    return zero(T)
end

function szero(::LCMProd, ::Type{T}) where {T}
    return typemax(T)
end

function sone(::DivisionQuantale, ::Type{T}) where {T}
    return one(T)
end

function splus(::GCDProd, a, b)
    return safegcd(a, b)
end

function splus(::LCMProd, a, b)
    return safelcm(a, b)
end

function sprod(::GCDProd, a::Integer, b::Integer)
    return a * b
end

function sprod(s::DivisionQuantale, a::Rational, b::Rational)
    u = szero(s, promote_eltype(a, b))

    if a == u || b == u
        c = u
    else
        c = a * b
    end

    return c
end

function sstar(::GCDProd, a::T) where {T <: Integer}
    return one(T)
end

function sstar(::GCDProd, a::T) where {T <: Rational}
    if isinteger(a)
        b = one(T)
    else
        b = typemax(T)
    end

    return b
end

function sstar(::LCMProd, a::T) where {T}
    if isinteger(inv(a))
        b = one(T)
    else
        b = zero(T)
    end

    return b
end

function smuladd(s::GCDProd, a::T, b::T, c::T) where {T <: BitInteger}
    if isone(c)
        d = c
    elseif iszero(c)
        d = sprod(s, a, b)
    else
        d = gcdprod(a, b, c)
    end

    return d
end

function smuladd(s::GCDProd, a::Rational, b::Rational, c::Rational)
    if iszero(a) || iszero(b) || isinf(c)
        d = c
    elseif iszero(c)
        d = sprod(s, a, b)
    else
        na, da = numerator(a), denominator(a)
        nb, db = numerator(b), denominator(b)
        nc, dc = numerator(c), denominator(c)

        nd, dd = gcdprod(na, da, nb, db, nc, dc)
        d = unsafe_rational(nd, dd)
    end

    return d
end

function smuladd(s::LCMProd, a::Rational, b::Rational, c::Rational)
    if iszero(c) || isinf(a) || isinf(b)
        d = c
    elseif isinf(c)
        d = sprod(s, a, b)
    else
        na, da = numerator(a), denominator(a)
        nb, db = numerator(b), denominator(b)
        nc, dc = numerator(c), denominator(c)

        nd, dd = gcdprod(da, na, db, nb, dc, nc)
        d = unsafe_rational(dd, nd)
    end

    return d
end

function safegcd(a, b)
    return gcd(a, b)
end

function safegcd(a::Rational, b::Rational)
    return unsafe_rational(gcd(numerator(a), numerator(b)), lcm(denominator(a), denominator(b)))
end

function safelcm(a, b)
    return lcm(a, b)
end

function safelcm(a::Rational, b::Rational)
    return unsafe_rational(lcm(numerator(a), numerator(b)), gcd(denominator(a), denominator(b)))
end

function gcdprod(a, b, c)
    return gcd(a * b, c)
end

function gcdprod(a::T, b::T, c::T) where {T <: BitInteger}
    d, flag = mul_with_overflow(a, b)

    if flag
        d = convert(T, rem(widemul(a, b), widen(c)))
    else
        d = rem(d, c)
    end

    return gcd(d, c)
end

function gcdprod(na, da, nb, db, nc, dc)
    gab = gcd(na, db)
    gba = gcd(nb, da)

    na = na ÷ gab
    db = db ÷ gab
    nb = nb ÷ gba
    da = da ÷ gba

    nd = gcd(na * nb, nc)
    dd = lcm(da * db, dc)

    return nd, dd
end

function gcdprod(na::T, da::T, nb::T, db::T, nc::T, dc::T) where {T <: BitInteger}
    gab = gcd(na, db)
    gba = gcd(nb, da)

    na = na ÷ gab
    db = db ÷ gab
    nb = nb ÷ gba
    da = da ÷ gba

    nd, nflag = mul_with_overflow(na, nb)
    dd, dflag = mul_with_overflow(da, db)

    if nflag
        nd = convert(T, rem(widemul(na, nb), widen(nc)))
    else
        nd = rem(nd, nc)
    end

    if dflag
        wd = widemul(da, db)
        wd = (wd ÷ gcd(convert(T, rem(wd, widen(dc))), dc)) * widen(dc)
    else
        wd = widemul(dd ÷ gcd(rem(dd, dc), dc), dc)
    end

    nd = gcd(nd, nc)
    dd = convert(T, wd)

    return nd, dd
end

function sgemx_row!(s::GCDProd, C::AbstractMatrix{V}, A::AbstractMatrix{V}, B::AbstractMatrix{V}, AP::AbstractVector{V}, BP::AbstractVector{V}) where {V <: BitInteger}
    m = size(C, 1)
    n = size(C, 2)
    k = size(A, 2)

    rc = FVector{V}(undef, m)
    sc = FVector{V}(undef, n)

    rflag = false
    sflag = false

    @inbounds for i in 1:m
        p0 = (i - 1) * k

        for p in 1:k
            AP[p0 + p] = A[i, p]
        end

        ri = zero(V)

        for p in 1:k
            ri = gcd(ri, AP[p0 + p]); isone(ri) && break
        end

        if iszero(ri)
            ri = one(V)
        end

        rc[i] = ri

        if !isone(ri)
            rflag = true

            for p in 1:k
                AP[p0 + p] = AP[p0 + p] ÷ ri
            end
        end
    end

    @inbounds for j in 1:n
        p0 = (j - 1) * k

        for p in 1:k
            BP[p0 + p] = B[p, j]
        end

        sj = zero(V)

        for p in 1:k
            sj = gcd(sj, BP[p0 + p]); isone(sj) && break
        end

        if iszero(sj)
            sj = one(V)
        end

        sc[j] = sj

        if !isone(sj)
            sflag = true

            for p in 1:k
                BP[p0 + p] = BP[p0 + p] ÷ sj
            end
        end
    end

    if !rflag && !sflag
        j = 1

        @inbounds while j + SGEMX_NR - 1 <= n
            sgemx_row_kernel!(s, C, AP, BP, j, m, k); j += SGEMX_NR
        end

        @inbounds while j <= n
            bp0 = (j - 1) * k

            for i in 1:m
                ap0 = (i - 1) * k

                Cij = C[i, j]

                for p in 1:k
                    Cij = smuladd(s, AP[ap0 + p], BP[bp0 + p], Cij)
                end

                C[i, j] = Cij
            end

            j += 1
        end
    else
        @inbounds for j in 1:n
            sj = sc[j]

            bp0 = (j - 1) * k

            for i in 1:m
                ap0 = (i - 1) * k

                Gij = zero(V)

                for p in 1:k
                    Gij = smuladd(s, AP[ap0 + p], BP[bp0 + p], Gij); isone(Gij) && break
                end

                Pij = widemul(rc[i], sj) * widen(Gij)
                Cij = C[i, j]

                if !iszero(Cij)
                    Cij = gcd(convert(V, rem(Pij, widen(Cij))), Cij)
                elseif Pij <= typemax(V)
                    Cij = convert(V, Pij)
                else
                    Cij = zero(V)

                    for p in 1:k
                        Cij = smuladd(s, A[i, p], B[p, j], Cij)
                    end
                end

                C[i, j] = Cij
            end
        end
    end

    return C
end

function sgemx_row!(s::GCDProd, C::AbstractMatrix{Rational{T}}, A::AbstractMatrix{Rational{T}}, B::AbstractMatrix{Rational{T}}, AP::AbstractVector{Rational{T}}, BP::AbstractVector{Rational{T}}) where {T <: BitInteger}
    U = Rational{T}

    m = size(C, 1)
    n = size(C, 2)
    k = size(A, 2)

    rc = FVector{U}(undef, m)
    sc = FVector{U}(undef, n)

    AP = reinterpret(T, AP)
    BP = reinterpret(T, BP)

    @inbounds for i in 1:m
        p0 = (i - 1) * k

        ri = zero(U)

        for p in 1:k
            ri = safegcd(ri, A[i, p])
        end

        if iszero(ri)
            ri = one(U)
        end

        rc[i] = ri

        nr = numerator(ri)
        dr = denominator(ri)

        for p in 1:k
            Aip = A[i, p]

            na = numerator(Aip)
            da = denominator(Aip)

            AP[p0 + p] = widemul(na, dr) ÷ widemul(da, nr)
        end
    end

    @inbounds for j in 1:n
        p0 = (j - 1) * k

        sj = zero(U)

        for p in 1:k
            sj = safegcd(sj, B[p, j])
        end

        if iszero(sj)
            sj = one(U)
        end

        sc[j] = sj

        ns = numerator(sj)
        ds = denominator(sj)

        for p in 1:k
            Bpj = B[p, j]

            nb = numerator(Bpj)
            db = denominator(Bpj)

            BP[p0 + p] = widemul(nb, ds) ÷ widemul(db, ns)
        end
    end

    @inbounds for j in 1:n
        sj = sc[j]

        bp0 = (j - 1) * k

        for i in 1:m
            ap0 = (i - 1) * k

            Gij = zero(T)

            for p in 1:k
                Gij = smuladd(s, AP[ap0 + p], BP[bp0 + p], Gij); isone(Gij) && break
            end

            C[i, j] = safegcd(C[i, j], rc[i] * sj * Gij)
        end
    end

    return C
end
