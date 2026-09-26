# The Minkowski quantale
#
#   (2ⁿ, ∪, ∨)
#
# where
#
#   - elements are subsets
#   - addition is union
#   - multiplication is the Minkowski maximum
#
struct Chain <: AbstractSemiring end

function slte(s::Chain, a, b)
    return splus(s, a, b, Val(:N)) == b
end

function szero(s::Chain, ::Type{T}, ::Val{:N}) where {T <: Unsigned}
    return zero(T)
end

function szero(s::Chain, ::Type{T}, ::Val{:C}) where {T <: Unsigned}
    return typemax(T)
end

function sone(s::Chain, ::Type{T}, ::Val{:N}) where {T <: Unsigned}
    return one(T)
end

function splus(s::Chain, a, b, ::Val{:N})
    return a | b
end

function splus(s::Chain, a, b, ::Val{:C})
    return a & b
end

@inline function sprod(s::Chain, a, b, ::Val{:N}, ::Val{:N})
    return (a & -(b & -b)) | (b & -(a & -a))
end

@inline function sprod(s::Chain, a, b, ::Val{:C}, ::Val{:N})
    c = (a & -a) - one(a)
    d = ~(csmear(a & ~b) >> 1)
    return (b | c) & d
end

@inline function smuladd(s::Chain, a, b, c, tA::N_OR_T, tB::N_OR_T)
    return splus(s, sprod(s, a, b, tA, tB), c, Val(:N))
end

@inline function smuladd(s::Chain, a, b, c, tA::Val, tB::Val)
    return splus(s, sprod(s, a, b, tA, tB), c, Val(:C))
end

function sstar(s::Chain, a::T) where {T}
    return a | one(T)
end

function issymmetric(::Type{Chain})
    return true
end

function iscommutative(::Type{Chain})
    return true
end

function isidempotent(::Type{Chain})
    return true
end

# ----- helpers -----

@inline function csmear(x)
    T = eltype(x)

    for k in (1, 2, 4, 8, 16, 32)
        if k < 8sizeof(T)
            x |= x >> k
        end
    end

    return x
end
