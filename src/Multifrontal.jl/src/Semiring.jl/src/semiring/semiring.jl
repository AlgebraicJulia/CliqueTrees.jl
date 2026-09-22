struct DualQuantale{S <: AbstractSemiring} <: AbstractSemiring
    s::S
end

function DualQuantale{S}() where {S <: AbstractSemiring}
    return DualQuantale{S}(S())
end

struct NegativeQuantale{S <: AbstractSemiring} <: AbstractSemiring
    s::S
end

function NegativeQuantale{S}() where {S <: AbstractSemiring}
    return NegativeQuantale{S}(S())
end

struct Lattice{S <: AbstractSemiring} <: AbstractSemiring
    s::S
end

function Lattice{S}() where {S <: AbstractSemiring}
    return Lattice{S}(S())
end

function slte(s::AbstractSemiring, a, b)
    if isidempotent(s)
        return splus(s, a, b, Val(:N)) == b
    else
        return error("not implemented")
    end
end

function slte(d::DualQuantale, a, b)
    return sgte(d.s, a, b)
end

function slte(n::NegativeQuantale, a, b)
    return slte(n.s, a, b)
end

function slte(s::Lattice, a, b)
    return slte(s.s, a, b)
end

function sgte(s::AbstractSemiring, a, b)
    return slte(s, b, a)
end

function szero(s::AbstractSemiring, a::T, op::Val) where {T}
    return szero(s, T, op)
end

function szero(s::AbstractSemiring, ::Type, ::Val)
    return error("not implemented")
end

function szero(s::AbstractSemiring, ::Type{T}, ::Val{:T}) where {T}
    return szero(s, T, Val(:N))
end

function szero(s::AbstractSemiring, ::Type{T}, ::Val{:C}) where {T}
    if isintegral(s)
        return sone(s, T, Val(:N))
    else
        return error("not implemented")
    end
end

function szero(s::AbstractSemiring, ::Type{T}, ::Val{:R}) where {T}
    return szero(s, T, Val(:C))
end

function szero(d::DualQuantale, ::Type{T}, ::Val{:N}) where {T}
    return szero(d.s, T, Val(:C))
end

function szero(d::DualQuantale, ::Type{T}, ::Val{:C}) where {T}
    return szero(d.s, T, Val(:N))
end

function szero(n::NegativeQuantale, ::Type{T}, ::Val{:N}) where {T}
    return szero(n.s, T, Val(:N))
end

function szero(n::NegativeQuantale, ::Type{T}, ::Val{:C}) where {T}
    return sone(n.s, T, Val(:N))
end

function szero(s::Lattice, ::Type{T}, ::Val{:N}) where {T}
    return szero(s.s, T, Val(:N))
end

function sone(s::AbstractSemiring, a::T, op::Val) where {T}
    return sone(s, T, op)
end

function sone(s::AbstractSemiring, ::Type, ::Val)
    return error("not implemented")
end

function sone(s::AbstractSemiring, ::Type{T}, ::Val{:T}) where {T}
    return sone(s, T, Val(:N))
end

function sone(s::AbstractSemiring, ::Type{T}, ::Val{:C}) where {T}
    if islattice(s)
        return szero(s, T, Val(:N))
    else
        return error("not implemented")
    end
end

function sone(s::AbstractSemiring, ::Type{T}, ::Val{:R}) where {T}
    return sone(s, T, Val(:C))
end

function sone(d::DualQuantale, ::Type{T}, ::Val{:N}) where {T}
    return sone(d.s, T, Val(:C))
end

function sone(d::DualQuantale, ::Type{T}, ::Val{:C}) where {T}
    return sone(d.s, T, Val(:N))
end

function sone(n::NegativeQuantale, ::Type{T}, ::Val{:N}) where {T}
    return sone(n.s, T, Val(:N))
end

function sone(s::Lattice, ::Type{T}, ::Val{:N}) where {T}
    return szero(s.s, T, Val(:C))
end

function splus(s::AbstractSemiring, a, b, ::Val)
    return error("not implemented")
end

function splus(s::AbstractSemiring, a, b, ::Val{:T})
    return splus(s, a, b, Val(:N))
end

function splus(s::AbstractSemiring, a, b, ::Val{:C})
    if islattice(s)
        return sprod(s, a, b, Val(:N), Val(:N))
    else
        return error("not implemented")
    end
end

function splus(s::AbstractSemiring, a, b, ::Val{:R})
    return splus(s, a, b, Val(:C))
end

function splus(d::DualQuantale, a, b, ::Val{:N})
    return splus(d.s, a, b, Val(:C))
end

function splus(d::DualQuantale, a, b, ::Val{:C})
    return splus(d.s, a, b, Val(:N))
end

function splus(n::NegativeQuantale, a, b, ::Val{:N})
    return splus(n.s, a, b, Val(:N))
end

function splus(n::NegativeQuantale, a, b, ::Val{:C})
    return splus(n.s, a, b, Val(:C))
end

function splus(s::Lattice, a, b, ::Val{:N})
    return splus(s.s, a, b, Val(:N))
end

function sprod(s::AbstractSemiring, a, b, ::Val, ::Val)
    return error("not implemented")
end

function sprod(s::AbstractSemiring, a, b, ::Val{:T}, tB::Val)
    return sprod(s, a, b, Val(:N), tB)
end

function sprod(s::AbstractSemiring, a, b, tA::Union{Val{:N}, Val{:C}}, ::Val{:T})
    return sprod(s, a, b, tA, Val(:N))
end

function sprod(s::AbstractSemiring, a, b, ::Val{:R}, tB::Val)
    return sprod(s, a, b, Val(:C), tB)
end

function sprod(s::AbstractSemiring, a, b, tA::Union{Val{:N}, Val{:C}}, ::Val{:R})
    return sprod(s, a, b, tA, Val(:C))
end

function sprod(s::AbstractSemiring, a, b, ::Val{:N}, ::Val{:C})
    if iscommutative(s)
        return sprod(s, b, a, Val(:C), Val(:N))
    else
        return error("not implemented")
    end
end

function sprod(n::NegativeQuantale, a, b, ::Val{:N}, ::Val{:N})
    return sprod(n.s, a, b, Val(:N), Val(:N))
end

function sprod(n::NegativeQuantale, a, b, ::Val{:C}, ::Val{:N})
    c = sprod(n.s, a, b, Val(:C), Val(:N))
    return splus(n.s, c, sone(n.s, c, Val(:N)), Val(:C))
end

function sprod(n::NegativeQuantale, a, b, ::Val{:N}, ::Val{:C})
    c = sprod(n.s, a, b, Val(:N), Val(:C))
    return splus(n.s, c, sone(n.s, c, Val(:N)), Val(:C))
end

function sprod(s::Lattice, a, b, ::Val{:N}, ::Val{:N})
    return splus(s.s, a, b, Val(:C))
end

function sprod(d::DualQuantale, a, b, ::Val{:N}, ::Val{:N})
    if islattice(d.s)
        return splus(d.s, a, b, Val(:N))
    else
        return error("not implemented")
    end
end

function sstar(s::AbstractSemiring, a)
    if isidempotent(s)
        b = sone(s, a, Val(:N))

        if !isintegral(s)
            a = splus(s, b, a, Val(:N))

            while b != a
                b = a
                a = sprod(s, b, b, Val(:N), Val(:N))
            end
        end

        return b
    else
        u = sone(s, a, Val(:N))
        b = u

        if !isintegral(s)
            c = splus(s, u, a, Val(:N))

            while b != c
                b = c
                c = smuladd(s, a, b, u, Val(:N), Val(:N))
            end
        end

        return b
    end
end

function smuladd(s::AbstractSemiring, a, b, c, tA::Val{TA}, tB::Val{TB}) where {TA, TB}
    if TA === TB
        op = Val(:N)
    else
        op = Val(:C)
    end

    return splus(s, sprod(s, a, b, tA, tB), c, op)
end

function smuladd(s::AbstractSemiring, a, b, c, ::Val{:T}, tB::Val)
    return smuladd(s, a, b, c, Val(:N), tB)
end

function smuladd(s::AbstractSemiring, a, b, c, tA::Union{Val{:N}, Val{:C}}, ::Val{:T})
    return smuladd(s, a, b, c, tA, Val(:N))
end

function smuladd(s::AbstractSemiring, a, b, c, ::Val{:R}, tB::Val)
    return smuladd(s, a, b, c, Val(:C), tB)
end

function smuladd(s::AbstractSemiring, a, b, c, tA::Union{Val{:N}, Val{:C}}, ::Val{:R})
    return smuladd(s, a, b, c, tA, Val(:C))
end

function isintegral(s::S) where {S <: AbstractSemiring}
    return isintegral(S)
end

function isintegral(::Type{S}) where {S <: AbstractSemiring}
    return islattice(S)
end

function isintegral(::Type{DualQuantale{S}}) where {S}
    return isintegral(S)
end

function isintegral(::Type{NegativeQuantale{S}}) where {S}
    return true
end

function islattice(s::S) where {S <: AbstractSemiring}
    return islattice(S)
end

function islattice(::Type{<:AbstractSemiring})
    return false
end

function islattice(::Type{DualQuantale{S}}) where {S}
    return islattice(S)
end

function islattice(::Type{NegativeQuantale{S}}) where {S}
    return islattice(S)
end

function islattice(::Type{Lattice{S}}) where {S}
    return true
end

function iscommutative(s::S) where {S <: AbstractSemiring}
    return iscommutative(S)
end

function iscommutative(::Type{S}) where {S <: AbstractSemiring}
    return islattice(S)
end

function iscommutative(::Type{DualQuantale{S}}) where {S}
    return iscommutative(S)
end

function iscommutative(::Type{NegativeQuantale{S}}) where {S}
    return iscommutative(S)
end

function isidempotent(s::S) where {S <: AbstractSemiring}
    return isidempotent(S)
end

function isidempotent(::Type{S}) where {S <: AbstractSemiring}
    return isintegral(S)
end

function isidempotent(::Type{DualQuantale{S}}) where {S}
    return isidempotent(S)
end

include("real.jl")
include("tropical.jl")
include("lawvere.jl")
include("bottleneck.jl")
include("boolean.jl")
include("predecessor.jl")
include("relative.jl")
