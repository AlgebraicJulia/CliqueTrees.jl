struct LayeredSieve{PSet <: AbstractPackedSet}
    sieves::Vector{Sieve{PSet}}
    margins::Vector{Int}
    k::Int
    weights::Vector{Int}
    stack::Vector{Tuple{Int, Int, Int}}
    buffer::Vector{Tuple{Int, PSet, PSet}}   # (root, bag, vertices) tuples
    index::Dict{Int, Int} # root index -> index in buffer
end

function LayeredSieve{PSet}(k::Int, weights::Vector{Int}) where {PSet <: AbstractPackedSet}
    n = margin_index(k)
    sieves = Vector{Sieve{PSet}}(undef, n)
    margins = Vector{Int}(undef, n)

    for s in 1:n
        sieves[s] = Sieve{PSet}()
        margins[s] = 1 << (s - 2)
    end

    return LayeredSieve{PSet}(sieves, margins, k, weights, Tuple{Int, Int, Int}[], Tuple{Int, PSet, PSet}[], Dict{Int, Int}())
end

# Compute the smallest positive integer i such that
#
#    2ⁱ⁻² ≥ m.
#
function margin_index(m::Int)
    return 8sizeof(Int) - leading_zeros(max(2m - 1, 0)) + 1
end

function sieve_index(sieve::LayeredSieve{PSet}, B::PSet) where {PSet}
    return margin_index(sieve.k + 1 - wt(sieve.weights, B))
end

function Base.setindex!(sieve::LayeredSieve{PSet}, (B, V)::Tuple{PSet, PSet}, i::Int) where {PSet}
    push!(sieve.buffer, (i, B, V))
    sieve.index[i] = length(sieve.buffer)
    return sieve
end

function flush!(sieve::LayeredSieve{PSet}) where {PSet}
    n = length(sieve.sieves)

    for (i, B, V) in sieve.buffer
        if haskey(sieve.index, i)
            s = min(sieve_index(sieve, B), n)
            sieve.sieves[s][i] = V
            delete!(sieve.index, i)
        end
    end

    empty!(sieve.buffer)
end

function replace_subset!(sieve::LayeredSieve{PSet}, i::Int, B::PSet, j::Int, C::PSet, V::PSet) where {PSet}
    n = length(sieve.sieves)

    if haskey(sieve.index, i)
        s = sieve.index[j] = pop!(sieve.index, i)
        sieve.buffer[s] = (j, C, V)
    else
        s = min(sieve_index(sieve, B), n)
        t = min(sieve_index(sieve, C), n)

        if s == t
            replace_key!(sieve.sieves[s], i, j)
        else
            delete!(sieve.sieves[s], i)
            sieve.sieves[t][j] = V
        end        
    end

    return sieve
end

struct LayeredSieveQuery{PSet <: AbstractPackedSet}
    sieve::LayeredSieve{PSet}
    stack::Vector{Tuple{Int, Int, Int}}
    R::PSet
    S::PSet
    i::Int
end

function Base.eltype(::Type{<:LayeredSieveQuery})
    return Int
end

function Base.IteratorSize(::Type{<:LayeredSieveQuery})
    return Base.SizeUnknown()
end

function Base.iterate(iter::LayeredSieveQuery)
    init!(iter.stack)
    return (iter.i, 1)
end

function Base.iterate(iter::LayeredSieveQuery{PSet}, s::Int) where {PSet}
    sieve = iter.sieve
    stack = iter.stack
    R = iter.R
    S = iter.S

    i = 0; n = length(sieve.sieves)

    if s <= n
        i = next!(sieve.sieves[s], stack, R, S, sieve.margins[s], sieve.weights)
    end

    while iszero(i) && s < n
        s += 1
        i = next!(sieve.sieves[s], init!(stack), R, S, sieve.margins[s], sieve.weights)
    end

    iszero(i) || return (i, s)
    return
end

function query(sieve::LayeredSieve{PSet}, R::PSet, S::PSet, i::Int) where {PSet}
    return LayeredSieveQuery{PSet}(sieve, sieve.stack, R, S, i)
end
