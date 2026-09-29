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

@inline function sprod(s::Union{MinMax, MaxMin}, a, b, ::Val{:C}, ::Val{:N})
    if slte(s, a, b)
        c = sone(s, b, Val(:N))
    else
        c = b
    end

    return c
end

@inline function sprod(s::Union{MinMax, MaxMin}, a::Vec{W, T}, b::Vec{W, T}, ::Val{:C}, ::Val{:N}) where {W, T}
    return vifelse(slte(s, a, b), Vec{W, T}(sone(s, T, Val(:N))), b)
end

@inline function sprod(s::Union{MinMax, MaxMin}, a::Vec{W, T}, b::T, tA::Val{:C}, tB::Val{:N}) where {W, T}
    return sprod(s, a, Vec{W, T}(b), tA, tB)
end

@inline function sprod(s::Union{MinMax, MaxMin}, a::T, b::Vec{W, T}, tA::Val{:C}, tB::Val{:N}) where {W, T}
    return sprod(s, Vec{W, T}(a), b, tA, tB)
end

@inline function smuladd(s::Union{MinMax, MaxMin}, a, b, c, ::Val{:N}, ::Val{:N})
    return splus(s, sprod(s, a, b, Val(:N), Val(:N)), c, Val(:N))
end

@inline function smuladd(s::Union{MinMax, MaxMin}, a, b, c, ::Val{:C}, ::Val{:N})
    return splus(s, sprod(s, a, b, Val(:C), Val(:N)), c, Val(:C))
end

function sstar(s::Union{MinMax, MaxMin}, a)
    return sone(s, a, Val(:N))
end
