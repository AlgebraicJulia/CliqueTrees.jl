# The semiring of nonnegative real numbers
#
#   ([0, ∞], +, ×)
#
# - elements are extended non-negative real numbers
# - addition is as usual
# - multiplication is as usual
#
struct PlusProd <: AbstractSemiring end

function slte(::PlusProd, a, b)
    return a <= b
end

function stop(::PlusProd, ::Type{T}) where {T}
    return typemax(T)
end

function szero(::PlusProd, ::Type{T}) where {T}
    return zero(T)
end

function sone(::PlusProd, ::Type{T}) where {T}
    return one(T)
end

function splus(::PlusProd, a, b)
    return a + b
end

function sprod(::PlusProd, a, b)
    return a * b
end

#
#   a* = { (1 - a)⁻¹ if a < 1
#        {  ∞        if a ≥ 1
#
function sstar(::PlusProd, a::T) where {T}
    if a < one(T)
        b = inv(one(T) - a)
    else
        b = typemax(T)
    end

    return b
end

function smuladd(::PlusProd, a, b, c)
    return muladd(a, b, c)
end
