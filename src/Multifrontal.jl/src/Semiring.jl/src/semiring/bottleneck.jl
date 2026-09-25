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

function smuladd(s::Union{MinMax, MaxMin}, a, b, c, tA::N_OR_T, tB::N_OR_T)
    return splus(s, sprod(s, a, b, tA, tB), c, Val(:N))
end

function smuladd(s::Union{MinMax, MaxMin}, a, b, c, tA::Val, tB::Val)
    return splus(s, sprod(s, a, b, tA, tB), c, Val(:C))
end

function sstar(s::Union{MinMax, MaxMin}, a)
    return sone(s, a, Val(:N))
end
