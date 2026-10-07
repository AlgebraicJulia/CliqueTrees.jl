# A layered sieve stores PTDURs keyed by vertex set, partitioned by "margin":
# the remaining weight capacity k + 1 - w(bag). Layer s holds PTDURs with
# margin ≤ 2ˢ⁻², which groups PTDURs with similar margins together.
#
# Insertions are deferred until `flush!`, because the main loop of the DP adds
# PTDURs while it is iterating over the results of a query.
struct LayeredSieve{PSet <: AbstractPackedSet}
    sieves::Vector{Sieve{PSet}}
    k::Int
    weights::Weights{PSet}
    stack::Vector{Tuple{Int, Int}}
    adds::Vector{Tuple{Int, PSet, PSet}}     # pending insertions (root, bag, vertices)
end

function LayeredSieve{PSet}(k::Int, weights::Weights{PSet}) where {PSet <: AbstractPackedSet}
    sieves = [Sieve{PSet}() for _ in 1:margin_index(k)]
    return LayeredSieve{PSet}(sieves, k, weights, Tuple{Int, Int}[], Tuple{Int, PSet, PSet}[])
end

# Compute the smallest positive integer i such that
#
#    2ⁱ⁻² ≥ m.
#
function margin_index(m::Int)
    return 8sizeof(Int) - leading_zeros(max(2m - 1, 0)) + 1
end

# Schedule an insertion.
function Base.setindex!(sieve::LayeredSieve{PSet}, (B, V)::Tuple{PSet, PSet}, i::Int) where {PSet}
    push!(sieve.adds, (i, B, V))
    return sieve
end

function flush!(sieve::LayeredSieve{PSet}) where {PSet}
    for (i, B, V) in sieve.adds
        m = sieve.k + 1 - wt(sieve.weights, B)
        s = min(margin_index(m), length(sieve.sieves))
        sieve.sieves[s][i] = (V, m)
    end

    empty!(sieve.adds)
    return sieve
end

# Return all stored PTDURs whose vertex set V and bag B satisfy
#
#   - V ∩ R = ∅
#   - w(S - V) ≤ k + 1 - w(B)
#
# Pending insertions are not visible.
function query!(out::Vector{Int}, sieve::LayeredSieve{PSet}, R::PSet, S::PSet) where {PSet}
    empty!(out)

    for s in eachindex(sieve.sieves)
        query!(out, sieve.sieves[s], sieve.stack, R, S, sieve.weights)
    end

    return out
end
