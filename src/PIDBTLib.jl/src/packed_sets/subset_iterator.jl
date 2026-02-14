struct SubsetIterator{T <: AbstractPackedSet}
    set::T
end

function subsets(set::AbstractPackedSet)
    return SubsetIterator(set)
end

# ------------------- #
# Iteration Interface #
# ------------------- #

function Base.length(iter::SubsetIterator)
    return 1 << length(iter.set)
end

function Base.eltype(::Type{SubsetIterator{T}}) where {T <: AbstractPackedSet}
    return T
end

function Base.iterate(iter::SubsetIterator)
    return (iter.set, iter.set)
end

function Base.iterate(iter::SubsetIterator{T}, state::T) where {T <: AbstractPackedSet}
    set = iter.set
    next = nextset_nonempty(state, set)
    next == set && return
    return (next, next)
end

function Base.iterate(iter::Iterators.Reverse{SubsetIterator{T}}) where {T <: AbstractPackedSet}
    return (T(), T())
end

function Base.iterate(iter::Iterators.Reverse{SubsetIterator{T}}, state::T) where {T <: AbstractPackedSet}
    set = iter.itr.set
    state == set && return
    next = prevset_nonempty(state, set)
    return (next, next)
end
