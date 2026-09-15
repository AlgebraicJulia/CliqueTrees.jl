struct LAndPar <: IntegralQuantale end

struct LOrTens <: IntegralQuantale end

function slte(::LAndPar, a, b)
    return a >= b
end

function slte(::LOrTens, a, b)
    return a <= b
end

function szero(::LAndPar, ::Type{T}) where {T}
    return one(T)
end

function szero(::LOrTens, ::Type{T}) where {T}
    return zero(T)
end

function sone(::LAndPar, ::Type{T}) where {T}
    return zero(T)
end

function sone(::LOrTens, ::Type{T}) where {T}
    return one(T)
end

function splus(::LAndPar, a, b)
    return min(a, b)
end

function splus(::LOrTens, a, b)
    return max(a, b)
end

function sprod(::LAndPar, a, b)
    T = promote_eltype(a, b)
    return min(one(T), a + b)
end

function sprod(::LOrTens, a, b)
    T = promote_eltype(a, b)
    return max(zero(T), a + b - one(T))
end
