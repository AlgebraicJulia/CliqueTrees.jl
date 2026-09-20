# The tropical semiring
#
#   ([-∞, ∞], min, +)
#
# - elements are extended real numbers
# - addition is minimization
# - multiplication is addition (+∞ + -∞ = +∞)
#
struct MinPlus <: AbstractQuantale end

# The dual tropical semiring
#
#   ([-∞, ∞], max, +)
#
# - elements are extended real numbers
# - addition is maximization
# - multiplication is addition (+∞ + -∞ = -∞)
#
const MaxPlus = DualQuantale{MinPlus}

# The Viterbi semiring
#
#   ([0, ∞], min, ×)
#
# - elements are non-negative extended real numbers
# - addition is minization
# - multiplication is as usual (+∞ × 0 = +∞)
#
struct MinProd <: AbstractQuantale end

# The max-times semiring
#
#   ([0, ∞], max, ×)
#
# - elements are non-negative extended real numbers
# - addition is maximization
# - multiplication is as usual (+∞ × 0 = 0)
#
const MaxProd = DualQuantale{MinProd}

const TropicalSemiring = Union{MinPlus, MaxPlus, MinProd, MaxProd}

function iscommutative(::Type{MinPlus})
    return true
end

function iscommutative(::Type{MinProd})
    return true
end

function slte(::Union{MinPlus, MinProd}, a, b)
    return a >= b
end

function szero(::MinPlus, ::Type{T}, ::Val{:N}) where {T}
    return typemax(T)
end

function szero(::MinPlus, ::Type{T}, ::Val{:C}) where {T}
    return typemin(T)
end

function szero(::MinProd, ::Type{T}, ::Val{:N}) where {T}
    return typemax(T)
end

function szero(::MinProd, ::Type{T}, ::Val{:C}) where {T}
    return zero(T)
end

function sone(::MinPlus, ::Type{T}, ::Val{:N}) where {T}
    return zero(T)
end

function sone(::MinPlus, ::Type{T}, ::Val{:C}) where {T}
    return zero(T)
end

function sone(::MinProd, ::Type{T}, ::Val{:N}) where {T}
    return one(T)
end

function sone(::MinProd, ::Type{T}, ::Val{:C}) where {T}
    return one(T)
end

function splus(::Union{MinPlus, MinProd}, a, b, ::Val{:N})
    return min(a, b)
end

function splus(::Union{MinPlus, MinProd}, a, b, ::Val{:C})
    return max(a, b)
end

function sprod(::MinPlus, a, b, ::Val{:N}, ::Val{:N})
    return a + b
end

function sprod(::MaxPlus, a, b, ::Val{:N}, ::Val{:N})
    return a + b
end

function sprod(::MinProd, a, b, ::Val{:N}, ::Val{:N})
    return a * b
end

function sprod(::MaxProd, a, b, ::Val{:N}, ::Val{:N})
    return a * b
end

function sprod(::Union{MinPlus, MaxPlus}, a, b, ::Val{:C}, ::Val{:N})
    return b - a
end

function sprod(::Union{MinProd, MaxProd}, a, b, ::Val{:C}, ::Val{:N})
    return b / a
end

#
#   a* = { 1  if a ≤ 1
#        { ⊤  otherwise
#
function sstar(s::TropicalSemiring, a::T) where {T}
    if slte(s, a, sone(s, T, Val(:N)))
        b = sone(s, T, Val(:N))
    else
        b = szero(s, T, Val(:C))
    end

    return b
end
