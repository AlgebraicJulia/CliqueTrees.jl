# ----- generic fallbacks -----

function szero(s::AbstractSemiring, a::T) where {T}
    return szero(s, T)
end

function sone(s::AbstractSemiring, a::T) where {T}
    return sone(s, T)
end

function sstar(s::AbstractSemiring, a)
    u = sone(s, a)
    b = u
    c = splus(s, u, a)

    while b != c
        b = c
        c = smuladd(s, a, b, u)
    end

    return b
end

function sstar(s::AbstractQuantale, a)
    u = sone(s, a)
    a = splus(s, u, a)

    while u != a
        u = a
        a = sprod(s, u, u)
    end

    return a
end

function sstar(s::IntegralQuantale, a::T) where {T}
    return sone(s, T)
end

function smuladd(s::AbstractSemiring, a, b, c)
    return splus(s, sprod(s, a, b), c)
end

function slte(s::AbstractQuantale, a, b)
    return splus(s, a, b) == b
end

function sgte(s::AbstractSemiring, a, b)
    return slte(s, b, a)
end

function stop(s::IntegralQuantale, ::Type{T}) where {T}
    return sone(s, T)
end

# ----- dual lattice -----

struct DualLattice{S <: AbstractLattice} <: AbstractLattice
    s::S
end

function DualLattice{S}() where {S <: AbstractLattice}
    return DualLattice{S}(S())
end

function slte(d::DualLattice, a, b)
    return sgte(d.s, a, b)
end

function szero(d::DualLattice, ::Type{T}) where {T}
    return sone(d.s, T)
end

function sone(d::DualLattice, ::Type{T}) where {T}
    return szero(d.s, T)
end

function splus(d::DualLattice, a, b)
    return sprod(d.s, a, b)
end

function sprod(d::DualLattice, a, b)
    return splus(d.s, a, b)
end

# ----- semirings -----

include("real.jl")
include("tropical.jl")
include("bottleneck.jl")
include("division_quantale.jl")
include("division_lattice.jl")
include("boolean.jl")
include("relational.jl")
include("lukasiewicz.jl")
include("affine_division_quantale.jl")
include("predecessor.jl")
include("jet.jl")
include("best.jl")
