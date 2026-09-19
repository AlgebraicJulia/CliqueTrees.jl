# The bottleneck lattice
#
#   ([-∞, ∞], min, max)
#
# - elements are extended real numbers
# - addition is minimization
# - multiplication is maximization
#
struct MinMax <: AbstractLattice end

# The dual bottleneck lattice
#
#   ([-∞, ∞], max, min)
#
# - elements are extended real numbers
# - addition is maximization
# - multiplication is minimization
#
const MaxMin = DualLattice{MinMax}

function slte(::MinMax, a, b)
    return a >= b
end

function szero(::MinMax, ::Type{T}) where {T}
    return typemax(T)
end

function sone(::MinMax, ::Type{T}) where {T}
    return typemin(T)
end

function splus(::MinMax, a, b)
    return min(a, b)
end

function sprod(::MinMax, a, b)
    return max(a, b)
end
