"""
    Clique{V, E} <: AbstractVector{V}

A clique of a clique tree.
"""
struct Clique{V, E} <: AbstractVector{V}
    residual::UnitRange{V}
    separator::SubArray{V, 1, Vector{V}, Tuple{UnitRange{E}}, true}
end

"""
    residual(clique::Clique)

Get the residual of a clique.
"""
function residual(clique::Clique)
    return clique.residual
end

"""
    separator(clique::Clique)

Get the separator of a clique.
"""
function separator(clique::Clique)
    return clique.separator
end

#############################
# Abstract Vector Interface #
#############################

function Base.getindex(clique::Clique, i::Integer)
    if i in eachindex(residual(clique))
        return residual(clique)[i]
    else
        return separator(clique)[i - length(residual(clique))]
    end
end

function Base.IndexStyle(::Type{<:Clique})
    return IndexLinear()
end

function Base.size(clique::Clique)
    return (length(residual(clique)) + length(separator(clique)),)
end

function Base.in(v, clique::Clique)
    return v in residual(clique) || insorted(v, separator(clique))
end

function Base.hasfastin(::Type{<:Clique})
    return true
end
