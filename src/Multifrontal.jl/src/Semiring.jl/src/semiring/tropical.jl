struct MinPlus <: AbstractQuantale end

struct MaxPlus <: AbstractQuantale end

struct MinProd <: AbstractQuantale end

struct MaxProd <: AbstractQuantale end

const TropicalSemiring = Union{MinPlus, MaxPlus, MinProd, MaxProd}

function slte(::Union{MinPlus, MinProd}, a, b)
    return a >= b
end

function slte(::Union{MaxPlus, MaxProd}, a, b)
    return a <= b
end

function stop(::MinPlus, ::Type{T}) where {T}
    return typemin(T)
end

function stop(::MaxPlus, ::Type{T}) where {T}
    return typemax(T)
end

function stop(::MinProd, ::Type{T}) where {T}
    return zero(T)
end

function stop(::MaxProd, ::Type{T}) where {T}
    return typemax(T)
end

function szero(::MinPlus, ::Type{T}) where {T}
    return typemax(T)
end

function szero(::MaxPlus, ::Type{T}) where {T}
    return typemin(T)
end

function szero(::MinProd, ::Type{T}) where {T}
    return typemax(T)
end

function szero(::MaxProd, ::Type{T}) where {T}
    return zero(T)
end

function sone(::Union{MinPlus, MaxPlus}, ::Type{T}) where {T}
    return zero(T)
end

function sone(::Union{MinProd, MaxProd}, ::Type{T}) where {T}
    return one(T)
end

function splus(::Union{MinPlus, MinProd}, a, b)
    return min(a, b)
end

function splus(::Union{MaxPlus, MaxProd}, a, b)
    return max(a, b)
end

function sprod(::Union{MinPlus, MaxPlus}, a, b)
    return a + b
end

function sprod(::Union{MinProd, MaxProd}, a, b)
    return a * b
end

function sstar(s::TropicalSemiring, a::T) where {T}
    if !slte(s, a, sone(s, T))
        b = stop(s, T)
    else
        b = sone(s, T)
    end

    return b
end
