# ===== quantales =====

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

# ===== slte / sgte =====

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

# ===== compose =====

function compose(::Val{TA}, ::Val{TB}) where {TA, TB}
    tflag = (TA === :T) ⊻ (TA === :C) ⊻ (TB === :T) ⊻ (TB === :C)
    rflag = (TA === :R) ⊻ (TA === :C) ⊻ (TB === :R) ⊻ (TB === :C)

    if tflag
        if rflag
            tC = Val(:C)
        else
            tC = Val(:T)
        end
    else
        if rflag
            tC = Val(:R)
        else
            tC = Val(:N)
        end
    end

    return tC
end

# ===== sid =====

function sid(s::AbstractSemiring, a, ::Val{:N})
    return a
end

function sid(s::AbstractSemiring, a, ::Val{:T})
    if issymmetric(s)
        return a
    else
        return error("not implemented")
    end
end

function sid(d::DualQuantale, a, ::Val{:T})
    return sid(d.s, a, Val(:T))
end

function sid(d::DualQuantale, a, ::Val{:R})
    return sid(d.s, a, Val(:R))
end

function sid(d::DualQuantale, a, ::Val{:C})
    return sid(d.s, a, Val(:C))
end

# ===== szero =====

function szero(s::AbstractSemiring, a::T, op::Val) where {T}
    return szero(s, T, op)
end

function szero(s::AbstractSemiring, ::Type{Vec{W, T}}, op::Val{:N}) where {W, T}
    return Vec{W, T}(szero(s, T, op))
end

function szero(s::AbstractSemiring, ::Type{Vec{W, T}}, op::Val{:C}) where {W, T}
    return Vec{W, T}(szero(s, T, op))
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

function szero(s::Lattice, ::Type{T}, ::Val{:N}) where {T}
    return szero(s.s, T, Val(:N))
end

# ===== sone =====

function sone(s::AbstractSemiring, a::T, op::Val) where {T}
    return sone(s, T, op)
end

function sone(s::AbstractSemiring, ::Type{Vec{W, T}}, op::Val{:N}) where {W, T}
    return Vec{W, T}(sone(s, T, op))
end

function sone(s::AbstractSemiring, ::Type{Vec{W, T}}, op::Val{:C}) where {W, T}
    return Vec{W, T}(sone(s, T, op))
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

function sone(n::NegativeQuantale, ::Type{T}, ::Val{:N}) where {T}
    return sone(n.s, T, Val(:N))
end

function sone(s::Lattice, ::Type{T}, ::Val{:N}) where {T}
    return szero(s.s, T, Val(:C))
end

# ===== splus =====

function splus(s::AbstractSemiring, a, b, c, op::Val)
    return splus(s, splus(s, a, b, op), c, op)
end

function splus(s::AbstractSemiring, a, b, c, d, op::Val)
    return splus(s, splus(s, a, b, op), splus(s, c, d, op), op)
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

# ===== sprod =====

function sprod(s::AbstractSemiring, a, b, ::R_OR_C, ::R_OR_C)
    return error("not supported")
end

function sprod(s::AbstractSemiring, a, b, ::Val{:N}, ::Val{:C})
    if iscommutative(s)
        return sprod(s, b, a, Val(:C), Val(:N))
    else
        return error("not implemented")
    end
end

function sprod(s::AbstractSemiring, a, b, tA::Val{:T}, tB::Val)
    if issymmetric(s)
        return sprod(s, a, b, Val(:N), tB)
    else
        return error("not implemented")
    end
end

function sprod(s::AbstractSemiring, a, b, tA::N_OR_C, tB::Val{:T})
    if issymmetric(s)
        return sprod(s, a, b, tA, Val(:N))
    else
        return error("not implemented")
    end
end

function sprod(s::AbstractSemiring, a, b, tA::Val{:R}, tB::N_OR_T)
    if issymmetric(s)
        return sprod(s, a, b, Val(:C), tB)
    else
        return error("not implemented")
    end
end

function sprod(s::AbstractSemiring, a, b, tA::Val{:N}, tB::Val{:R})
    if issymmetric(s)
        return sprod(s, a, b, Val(:N), Val(:C))
    else
        return error("not implemented")
    end
end

function sprod(n::NegativeQuantale, a, b, ::Val{:N}, ::Val{:N})
    return sprod(n.s, a, b, Val(:N), Val(:N))
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

# ===== isintegral =====
#
# An idempotent semiring is *integral* if ever element
# is less-than-or-equal-to the multiplicative unit:
#
#   a ≤ 1
#
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

# ===== islattice =====
#
# An integral semiring is a *lattice* if its multiplication
# is idempotent:
#
#   aa = a
#
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

# ===== iscommutative =====
#
# A semiring is *commutative* if its multiplication
# commutes:
#
#   ab = ba
#
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

# ===== isidempotent =====
#
# A semiring is *idempotent* if its addition is
# idempotent:
#
#   a + a = a
#
function isidempotent(s::S) where {S <: AbstractSemiring}
    return isidempotent(S)
end

function isidempotent(::Type{S}) where {S <: AbstractSemiring}
    return isintegral(S)
end

function isidempotent(::Type{DualQuantale{S}}) where {S}
    return isidempotent(S)
end

# ===== issymmetric =====
#
# A semiring is *symmetric* if its transpose is the
# the identity
#
#   aᵀ = a.
#
function issymmetric(s::S) where {S <: AbstractSemiring}
    return issymmetric(S)
end

function issymmetric(::Type{S}) where {S <: AbstractSemiring}
    return isintegral(S)
end

function issymmetric(::Type{DualQuantale{S}}) where {S}
    return issymmetric(S)
end

# ===== includes =====

include("real.jl")
include("tropical.jl")
include("lawvere.jl")
include("bottleneck.jl")
include("boolean.jl")
include("predecessor.jl")
include("relative.jl")
include("minkowski/minkowski.jl")
include("chain.jl")
