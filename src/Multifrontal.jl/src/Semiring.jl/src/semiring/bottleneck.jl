# The bottleneck lattice
#
#   ([-∞, ∞], min, max)
#
# - elements are extended real numbers
# - addition is minimization
# - multiplication is maximization
#
const MinMax = Lattice{MinPlus}

# The dual bottleneck lattice
#
#   ([-∞, ∞], max, min)
#
# - elements are extended real numbers
# - addition is maximization
# - multiplication is minimization
#
const MaxMin = DualQuantale{MinMax}

function sprod(s::Union{MinMax, MaxMin}, a, b, ::Val{:C}, ::Val{:N})
    if slte(s, a, b)
        c = sone(s, b, Val(:N))
    else
        c = b
    end

    return c
end

function smuladd(s::Union{MinMax, MaxMin}, a, b, c, ::Val{:N}, ::Val{:N})
    return splus(s, sprod(s, a, b, Val(:N), Val(:N)), c, Val(:N))
end

function smuladd(s::Union{MinMax, MaxMin}, a, b, c, ::Val{:C}, ::Val{:N})
    return splus(s, sprod(s, a, b, Val(:C), Val(:N)), c, Val(:C))
end

function sstar(s::Union{MinMax, MaxMin}, a)
    return sone(s, a, Val(:N))
end
