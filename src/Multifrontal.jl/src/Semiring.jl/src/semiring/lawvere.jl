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

function sprod(n::LawvereQuantale, a, b, ::Val{:N}, ::Val{:N})
    return sprod_unsafe(n.s, a, b, Val(:N), Val(:N))
end
