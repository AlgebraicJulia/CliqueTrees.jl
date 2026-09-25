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

function smuladd(n::LawvereQuantale, a, b, c, tA::N_OR_T, tB::N_OR_T)
    return splus(n, sprod(n, a, b, tA, tB), c, Val(:N))
end

function smuladd(n::LawvereQuantale, a, b, c, tA::Val, tB::Val)
    return splus(n, sprod(n, a, b, tA, tB), c, Val(:C))
end

function sstar(n::LawvereQuantale, a)
    return sone(n, a, Val(:N))
end
