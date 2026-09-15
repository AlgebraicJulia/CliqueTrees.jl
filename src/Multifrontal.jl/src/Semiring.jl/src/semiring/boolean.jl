struct AndOr <: AbstractLattice end

const OrAnd = DualLattice{AndOr}

function szero(::AndOr, ::Type{T}) where {T}
    return typemax(T)
end

function sone(::AndOr, ::Type{T}) where {T}
    return zero(T)
end

function splus(::AndOr, a, b)
    return a & b
end

function sprod(::AndOr, a, b)
    return a | b
end
