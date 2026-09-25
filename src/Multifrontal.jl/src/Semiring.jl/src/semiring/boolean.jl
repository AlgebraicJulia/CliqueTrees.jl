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

function sid(::AndOr, a, ::Val{:C})
    return ~a
end

function slte(s::AndOr, a, b)
    return splus(s, a, b, Val(:N)) == b
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

function smuladd(s::Union{AndOr, OrAnd}, a, b, c, tA::N_OR_T, tB::N_OR_T)
    return splus(s, sprod(s, a, b, tA, tB), c, Val(:N))
end

function smuladd(s::Union{AndOr, OrAnd}, a, b, c, tA::Val, tB::Val)
    return splus(s, sprod(s, a, b, tA, tB), c, Val(:C))
end

function sstar(s::Union{AndOr, OrAnd}, a)
    return sone(s, a, Val(:N))
end

function issymmetric(::Type{AndOr})
    return true
end

function islattice(::Type{AndOr})
    return true
end
