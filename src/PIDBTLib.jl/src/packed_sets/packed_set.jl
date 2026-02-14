struct PackedSet{I} <: AbstractPackedSet{I}
    data::I

    function PackedSet{I}(data::I) where {I}
        return new{I}(data)
    end
end

function PackedSet{I}() where {I}
    return PackedSet{I}(zero(I))
end

function domain(::Type{PackedSet{I}}) where {I}
    return 8sizeof(I)
end

function packedset(::Type{PackedSet{I}}, i::Int) where I
    return PackedSet{I}(one(I) << (i - 1))
end

function packedset(::Type{PackedSet{I}}, range::AbstractUnitRange{Int}) where I
    return interval(PackedSet{I}, first(range), last(range))
end

function packedset(::Type{PackedSet{I}}, range::OneTo{Int}) where I
    return downset(PackedSet{I}, length(range))
end

# upset: {start, start+1, ..., domain}
function upset(::Type{PackedSet{I}}, start::Int) where {I}
    return PackedSet{I}(typemax(I) << (start - 1))
end

# downset: {1, 2, ..., stop}
function downset(::Type{PackedSet{I}}, stop::Int) where {I}
    return PackedSet{I}(typemax(I) >> (8sizeof(I) - stop))
end

# interval: {start, start+1, ..., stop}
function interval(::Type{PackedSet{I}}, start::Int, stop::Int) where {I}
    return PackedSet{I}((typemax(I) >> (8sizeof(I) - (stop - start + 1))) << (start - 1))
end

function nextset_nonempty(subset::PackedSet{I}, set::PackedSet{I}) where {I}
    return PackedSet{I}((subset.data - one(I)) & set.data)
end

function prevset_nonempty(subset::PackedSet{I}, set::PackedSet{I}) where {I}
    return PackedSet{I}(((subset.data | ~set.data) + one(I)) & set.data)
end

function first_nonempty(set::PackedSet)
    return trailing_zeros(set.data) + 1
end

function last_nonempty(set::PackedSet{I}) where {I}
    return 8sizeof(I) - leading_zeros(set.data)
end

function Random.rand(rng::AbstractRNG, ::SamplerType{PackedSet{I}}) where {I}
    return PackedSet{I}(rand(rng, I))
end

# ------------------- #
# Iteration Interface #
# ------------------- #

function Base.length(set::PackedSet)
    return count_ones(set.data)
end

# ---------------------- #
# Abstract Set Interface #
# ---------------------- #

function Base.isempty(set::PackedSet)
    return iszero(set.data)
end

function Base.union(left::T, right::T) where {T <: PackedSet}
    return T(left.data | right.data)
end

function Base.intersect(left::T, right::T) where {T <: PackedSet}
    return T(left.data & right.data)
end

function Base.setdiff(left::T, right::T) where {T <: PackedSet}
    return T(left.data & ~right.data)
end

function Base.symdiff(left::T, right::T) where {T <: PackedSet}
    return T(left.data ⊻ right.data)
end

function Base.hash(set::PackedSet, h::UInt)
    return hash(set.data, h)
end

function Base.:(==)(left::T, right::T) where {T <: PackedSet}
    return left.data == right.data
end

