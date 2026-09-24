# The Boolean lattice
#
#   (2ⁿ, ∩, ∪)
#
# - elements are subsets
# - addition is intersection
# - multiplication is union
#
struct AndOr <: AbstractSemiring end

# The dual Boolean lattice
#
#   (2ⁿ, ∪, ∩)
#
# - elements are subsets
# - addition is union
# - multiplication is intersection
#
const OrAnd = DualQuantale{AndOr}

function islattice(::Type{AndOr})
    return true
end

function szero(::AndOr, ::Type{T}, ::Val{:N}) where {T}
    return typemax(T)
end

function sone(::AndOr, ::Type{T}, ::Val{:N}) where {T}
    return zero(T)
end

function splus(::AndOr, a, b, ::Val{:N})
    return a & b
end

function sprod(::AndOr, a, b, ::Val{:N}, ::Val{:N})
    return a | b
end

function sprod(s::Union{AndOr, OrAnd}, a, b, ::Val{:C}, ::Val{:N})
    return splus(s, ~a, b, Val(:N))
end
