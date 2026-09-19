# The Lawvere quantale
#
#   ([0, ∞], min, +)
#
# - elements are non-negative extended real numbers
# - addition is minimization
# - multiplication is addition
#
struct MinPlusLaw <: IntegralQuantale end

# The dual Lawvere quantale
#
#   ([-∞, 0], max, +)
#
# - elements are non-positive extended real numbers
# - addition is maximization
# - multiplication is addition
#
struct MaxPlusLaw <: IntegralQuantale end

# The multiplicative Lawvere quantale
#
#   ([1, ∞], min, ×)
#
# - elements are extended real numbers at least 1
# - addition is minimization
# - multiplication is as usual
#
struct MinProdLaw <: IntegralQuantale end

# The dual multiplicative Lawvere quantale
#
#   ([0, 1], max, ×)
#
# - elements are real numbers between 0 and 1
# - addition is maximization
# - multiplication is as usual
#
struct MaxProdLaw <: IntegralQuantale end

const LawvereQuantale = Union{MinPlusLaw, MaxPlusLaw, MinProdLaw, MaxProdLaw}

function slte(::Union{MinPlusLaw, MinProdLaw}, a, b)
    return a >= b
end

function slte(::Union{MaxPlusLaw, MaxProdLaw}, a, b)
    return a <= b
end

function szero(::Union{MinPlusLaw, MinProdLaw}, ::Type{T}) where {T}
    return typemax(T)
end

function szero(::MaxPlusLaw, ::Type{T}) where {T}
    return typemin(T)
end

function szero(::MaxProdLaw, ::Type{T}) where {T}
    return zero(T)
end

function sone(::Union{MinPlusLaw, MaxPlusLaw}, ::Type{T}) where {T}
    return zero(T)
end

function sone(::Union{MinProdLaw, MaxProdLaw}, ::Type{T}) where {T}
    return one(T)
end

function splus(::Union{MinPlusLaw, MinProdLaw}, a, b)
    return min(a, b)
end

function splus(::Union{MaxPlusLaw, MaxProdLaw}, a, b)
    return max(a, b)
end

function sprod(::Union{MinPlusLaw, MaxPlusLaw}, a, b)
    return a + b
end

function sprod(::Union{MinProdLaw, MaxProdLaw}, a, b)
    return a * b
end
