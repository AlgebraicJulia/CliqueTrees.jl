# The Boolean lattice
#
#   ({0, 1}, &, |)
#
# - elements are truth values
# - addition is conjunction
# - multiplication disjunction
#
struct AndOr <: AbstractQuantale end

# The dual Boolean lattice
#
#   ({0, 1}, |, &)
#
# - elements are truth values
# - addition is disjunction
# - multiplication conjunction
#
const OrAnd = DualQuantale{AndOr}

function islattice(::Type{AndOr})
    return true
end

function iscommutative(::Type{AndOr})
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
