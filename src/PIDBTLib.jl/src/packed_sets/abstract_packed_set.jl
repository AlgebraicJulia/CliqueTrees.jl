abstract type AbstractPackedSet{I} <: AbstractSet{Int} end

function domain(::T) where {T <: AbstractPackedSet}
    return domain(T)
end

function packedset(::Type{T}, iter) where {T <: AbstractPackedSet}
    return T() ∪ iter
end

function popfirst_nonempty(set::AbstractPackedSet)
    i = first_nonempty(set)
    return (i, nextset_nonempty(set, set))
end

function pop_nonempty(set::AbstractPackedSet)
    i = last_nonempty(set)
    return (i, setdiff(set, i))
end

# -------------------- #
# Iteration Interface  #
# -------------------- #

function Base.iterate(set::AbstractPackedSet)
    isempty(set) && return
    return popfirst_nonempty(set)
end

function Base.iterate(::T, set::T) where {T <: AbstractPackedSet}
    return iterate(set)
end

function Base.iterate(iter::Iterators.Reverse{T}) where {T <: AbstractPackedSet}
    set = iter.itr
    isempty(set) && return
    return pop_nonempty(set)
end

function Base.iterate(::Iterators.Reverse{T}, set::T) where {T <: AbstractPackedSet}
    isempty(set) && return
    return pop_nonempty(set)
end

# ---------------------- #
# Abstract Set Interface #
# ---------------------- #

function Base.in(i::Int, set::AbstractPackedSet)
    return !isempty(set ∩ i)
end

const IntegerOrRange = Union{Integer, AbstractUnitRange}

function Base.union(set::T, x::IntegerOrRange) where {T <: AbstractPackedSet}
    return set ∪ packedset(T, x)
end

function Base.union(x::IntegerOrRange, set::T) where {T <: AbstractPackedSet}
    return packedset(T, x) ∪ set
end

function Base.intersect(set::T, x::IntegerOrRange) where {T <: AbstractPackedSet}
    return set ∩ packedset(T, x)
end

function Base.intersect(x::IntegerOrRange, set::T) where {T <: AbstractPackedSet}
    return packedset(T, x) ∩ set
end

function Base.setdiff(set::T, x::IntegerOrRange) where {T <: AbstractPackedSet}
    return setdiff(set, packedset(T, x))
end

function Base.setdiff(x::IntegerOrRange, set::T) where {T <: AbstractPackedSet}
    return setdiff(packedset(T, x), set)
end

function Base.symdiff(set::T, x::IntegerOrRange) where {T <: AbstractPackedSet}
    return symdiff(set, packedset(T, x))
end

function Base.symdiff(x::IntegerOrRange, set::T) where {T <: AbstractPackedSet}
    return symdiff(packedset(T, x), set)
end

function Base.first(set::AbstractPackedSet)
    isempty(set) && error()
    return first_nonempty(set)
end

function Base.last(set::AbstractPackedSet)
    isempty(set) && error()
    return last_nonempty(set)
end

include("packed_set.jl")
include("n_packed_set.jl")
include("subset_iterator.jl")

# ---------------------- #
# Set Type Selection     #
# ---------------------- #

function settype(dom::Integer)
    N = 8sizeof(UInt)

    if dom ≤ N
        I = UInt8

        while 8sizeof(I) < dom
            I = widen(I)
        end

        T = PackedSet{I}
    else
        P = cld(dom, N) 
        T = NPackedSet{P, UInt}
    end

    return T
end

# ---------------------- #
# Type Aliases           #
# ---------------------- #

const PSet8 = settype(8)
const PSet16 = settype(16)
const PSet32 = settype(32)
const PSet64 = settype(64)
const PSet128 = settype(128)
const PSet256 = settype(256)
const PSet512 = settype(512)
