# The Lawvere quantale
#
#   ([0, ∞], min, +)
#
# - elements are non-negative extended real numbers
# - addition is minimization
# - multiplication is addition
#
const MinPlusLaw = NegativeQuantale{MinPlus}

# The dual Lawvere quantale
#
#   ([-∞, 0], max, +)
#
# - elements are non-positive extended real numbers
# - addition is maximization
# - multiplication is addition
#
const MaxPlusLaw = NegativeQuantale{MaxPlus}

# The multiplicative Lawvere quantale
#
#   ([1, ∞], min, ×)
#
# - elements are extended real numbers at least 1
# - addition is minimization
# - multiplication is as usual
#
const MinProdLaw = NegativeQuantale{MinProd}

# The dual multiplicative Lawvere quantale
#
#   ([0, 1], max, ×)
#
# - elements are real numbers between 0 and 1
# - addition is maximization
# - multiplication is as usual
#
const MaxProdLaw = NegativeQuantale{MaxProd}

const LawvereQuantale = Union{MinPlusLaw, MaxPlusLaw, MinProdLaw, MaxProdLaw}

function sprod(n::Union{MinPlusLaw, MaxPlusLaw}, a, b, ::Val{:N}, ::Val{:N})
    return a + b
end

function sprod(n::Union{MinProdLaw, MaxProdLaw}, a, b, ::Val{:N}, ::Val{:N})
    return a * b
end

function sprod(n::MinPlusLaw, a, b, ::Val{:C}, ::Val{:N})
    return max(b - a, zero(b))
end

function sprod(n::MaxPlusLaw, a, b, ::Val{:C}, ::Val{:N})
    return min(b - a, zero(b))
end

function sprod(n::MinProdLaw, a, b, ::Val{:C}, ::Val{:N})
    return max(b / a, one(b))
end

function sprod(n::MaxProdLaw, a, b, ::Val{:C}, ::Val{:N})
    return min(b / a, one(b))
end

function sprod(n::MinPlusLaw, a::AbstractFloat, b::AbstractFloat, ::Val{:C}, ::Val{:N})
    return ifelse(b > a, b - a, zero(b))
end

function sprod(n::MaxPlusLaw, a::AbstractFloat, b::AbstractFloat, ::Val{:C}, ::Val{:N})
    return ifelse(b < a, b - a, zero(b))
end

function sprod(n::MinProdLaw, a::AbstractFloat, b::AbstractFloat, ::Val{:C}, ::Val{:N})
    return ifelse(b > a, b / a, one(b))
end

function sprod(n::MaxProdLaw, a::AbstractFloat, b::AbstractFloat, ::Val{:C}, ::Val{:N})
    return ifelse(b < a, b / a, one(b))
end

function smuladd(n::LawvereQuantale, a, b, c, ::Val{:N}, ::Val{:N})
    return splus(n, sprod(n, a, b, Val(:N), Val(:N)), c, Val(:N))
end

function smuladd(n::LawvereQuantale, a, b, c, ::Val{:C}, ::Val{:N})
    return splus(n, sprod(n, a, b, Val(:C), Val(:N)), c, Val(:C))
end

function sstar(n::LawvereQuantale, a)
    return sone(n, a, Val(:N))
end
