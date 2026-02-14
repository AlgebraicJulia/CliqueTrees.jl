struct NPackedSet{P, I} <: AbstractPackedSet{I}
    packs::NTuple{P, PackedSet{I}}
end

function NPackedSet{P, I}() where {P, I}
    return NPackedSet{P, I}(ntuple(_ -> PackedSet{I}(), Val(P)))
end

function domain(::Type{NPackedSet{P, I}}) where {P, I}
    return P * 8sizeof(I)
end

function getpack(set::NPackedSet{P, I}, i::Int) where {P, I}
    return set.packs[i]
end

function setpack(set::NPackedSet{P, I}, pack::PackedSet{I}, i::Int) where {P, I}
    return NPackedSet{P, I}(setindex(set.packs, pack, i))
end

function packedset(::Type{NPackedSet{P, I}}, i::Int) where {P, I}
    n = 8sizeof(I)
    p, b = divrem(i - 1, n) .+ 1
    return setpack(NPackedSet{P, I}(), packedset(PackedSet{I}, b), p)
end

function packedset(::Type{NPackedSet{P, I}}, range::OneTo{Int}) where {P, I}
    return downset(NPackedSet{P, I}, length(range))
end

function packedset(::Type{NPackedSet{P, I}}, range::AbstractUnitRange{Int}) where {P, I}
    return interval(NPackedSet{P, I}, first(range), last(range))
end

# upset: {start, start+1, ..., domain}
function upset(::Type{NPackedSet{P, I}}, start::Int) where {P, I}
    n = 8sizeof(I)
    packs = ntuple(p -> upset(PackedSet{I}, clamp(start - (p - 1)n, 1, n + 1)), Val(P))
    return NPackedSet{P, I}(packs)
end

# downset: {1, 2, ..., stop}
function downset(::Type{NPackedSet{P, I}}, stop::Int) where {P, I}
    n = 8sizeof(I)
    packs = ntuple(p -> downset(PackedSet{I}, clamp(stop - (p - 1)n, 0, n)), Val(P))
    return NPackedSet{P, I}(packs)
end

# interval: {start, start+1, ..., stop}
function interval(::Type{NPackedSet{P, I}}, start::Int, stop::Int) where {P, I}
    n = 8sizeof(I)
    packs = ntuple(p -> interval(PackedSet{I}, max(start - (p - 1)n, 1), min(stop - (p - 1)n, n)), Val(P))
    return NPackedSet{P, I}(packs)
end

function first_nonempty(set::NPackedSet{P, I}) where {P, I}
    i = 1

    for p in 1:P
        pack = getpack(set, p)
        i += first_nonempty(pack) - 1
        isempty(pack) || break
    end

    return i
end

function last_nonempty(set::NPackedSet{P, I}) where {P, I}
    n = 8sizeof(I); i = P * n

    for p in P:-1:1
        pack = getpack(set, p)
        i += last_nonempty(pack) - n
        isempty(pack) || break
    end

    return i
end

function nextset_nonempty(subset::NPackedSet{P, I}, set::NPackedSet{P, I}) where {P, I}
    for p in 1:P
        subpack = getpack(subset, p)
        pack = getpack(set, p)
        subset = setpack(subset, nextset_nonempty(subpack, pack), p)
        isempty(subpack) || break
    end

    return subset
end

function prevset_nonempty(subset::NPackedSet{P, I}, set::NPackedSet{P, I}) where {P, I}
    for p in 1:P
        subpack = getpack(subset, p)
        pack = getpack(set, p)
        subset = setpack(subset, prevset_nonempty(subpack, pack), p)
        subpack == pack || break
    end

    return subset
end

function Random.rand(rng::AbstractRNG, ::SamplerType{NPackedSet{P, I}}) where {P, I}
    return NPackedSet{P, I}(ntuple(_ -> rand(rng, PackedSet{I}), Val(P)))
end

# ------------------- #
# Iteration Interface #
# ------------------- #

function Base.length(set::NPackedSet)
    return sum(length, set.packs)
end

function Base.iterate(set::NPackedSet{P, I}, (p, pack)::Tuple{Int, PackedSet{I}}=(1, getpack(set, 1))) where {P, I}
    n = 8sizeof(I)

    while isempty(pack) && p < P
        p += 1; pack = getpack(set, p)
    end

    if !isempty(pack)
        i, pack = popfirst_nonempty(pack)
        return ((p - 1)n + i, (p, pack))
    end

    return
end

function Base.iterate(iter::Iterators.Reverse{NPackedSet{P, I}}, (p, pack)::Tuple{Int, PackedSet{I}}=(P, getpack(iter.itr, P))) where {P, I}
    n = 8sizeof(I)

    while isempty(pack) && p > 1
        p -= 1; pack = getpack(iter.itr, p)
    end

    if !isempty(pack)
        i, pack = pop_nonempty(pack)
        return ((p - 1) * n + i, (p, pack))
    end

    return
end

# ---------------------- #
# Abstract Set Interface #
# ---------------------- #

function Base.isempty(set::NPackedSet)
    return all(isempty, set.packs)
end

function Base.in(i::Int, set::NPackedSet{P, I}) where {P, I}
    n = 8sizeof(I)
    p, b = divrem(i - 1, n) .+ 1
    return b in getpack(set, p)
end

function Base.union(left::T, right::T) where {T <: NPackedSet}
    return T(left.packs .∪ right.packs)
end

function Base.intersect(left::T, right::T) where {T <: NPackedSet}
    return T(left.packs .∩ right.packs)
end

function Base.setdiff(left::T, right::T) where {T <: NPackedSet}
    return T(setdiff.(left.packs, right.packs))
end

function Base.symdiff(left::T, right::T) where {T <: NPackedSet}
    return T(symdiff.(left.packs, right.packs))
end

function Base.hash(set::NPackedSet, h::UInt)
    return hash(set.packs, h)
end

function Base.:(==)(left::NPackedSet{P, I}, right::NPackedSet{P, I}) where {P, I}
    return left.packs == right.packs
end
