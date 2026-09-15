struct MinMax <: AbstractLattice end

const MaxMin = DualLattice{MinMax}

function slte(::MinMax, a, b)
    return a >= b
end

function szero(::MinMax, ::Type{T}) where {T}
    return typemax(T)
end

function sone(::MinMax, ::Type{T}) where {T}
    return typemin(T)
end

function splus(::MinMax, a, b)
    return min(a, b)
end

function sprod(::MinMax, a, b)
    return max(a, b)
end
