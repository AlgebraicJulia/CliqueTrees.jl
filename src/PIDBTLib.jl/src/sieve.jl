# Sieve: a trie storing PTDURs, keyed by their root bags. Every PTDUR ρ is
# stored with its bag B, its vertex set V ⊇ B, and its margin m = k + 1 - w(B)
# (the remaining weight capacity of its bag). Keys need not be unique.
#
# A query with the inlet R and outlet S of a PTD τ asks for the PTDURs ρ with
#
#   (1) V ∩ R = ∅,
#   (2) w(S - B) ≤ m, i.e. w(B ∪ S) ≤ k + 1.
#
# (2) is the bag size test of `add_ptd_to_ptdur_check`. (A sieve keyed by V
# can only test the weaker w(S - V) ≤ m. The two agree unless S meets V - B,
# and then `add_ptd_to_ptdur_check` rejects ρ anyway.) Written as
#
#   (2') w(B - S) ≤ k + 1 - w(S),
#
# the same test bounds the part of B outside of S, so a query can prune with
# both forms at once: (2) with the largest margin below a node, and (2') with a
# bound that is fixed for the whole query.
#
# Every node v compares the keys below it on an interval [start(v), stop(v)]
# of vertices; the intervals along a root-to-leaf path partition 1:domain.
# Nodes with stop(v) == domain ("leafy" nodes) store PTDURs; all other nodes
# store child nodes. For every child, a node stores a key that agrees with the
# keys below it on [start(v), stop(v)], the largest margin below it, and the
# union and intersection of the keys below it. The last two bound what lies
# beyond stop(v), so that a query can often prune a child without visiting it.
#
# When a node has too many entries, it is split: its interval is shortened,
# and its entries are grouped under new nodes by their keys on the shortened
# interval.

const MAX_CHILDREN_PER_NODE = 128

# An entry of an inner node.
struct SieveChild{PSet <: AbstractPackedSet}
    key::PSet       # agrees with the keys below on the interval of the node
    any::PSet       # union of the keys below
    all::PSet       # intersection of the keys below
    margin::Int     # largest margin below
    node::Int
end

# An entry of a leafy node.
struct SieveLeaf{PSet <: AbstractPackedSet}
    key::PSet       # bag B
    set::PSet       # vertex set V
    margin::Int
    value::Int      # PTDUR root
end

struct Sieve{PSet <: AbstractPackedSet}
    start::Vector{Int}
    stop::Vector{Int}
    mask::Vector{PSet}                          # interval [start, stop] as a set
    rest::Vector{PSet}                          # interval [stop + 1, domain] as a set
    children::Vector{Vector{SieveChild{PSet}}}  # entries of inner nodes
    leaves::Vector{Vector{SieveLeaf{PSet}}}     # entries of leafy nodes
end

function Sieve{PSet}() where {PSet <: AbstractPackedSet}
    sieve = Sieve{PSet}(Int[], Int[], PSet[], PSet[], Vector{SieveChild{PSet}}[], Vector{SieveLeaf{PSet}}[])
    add_node!(sieve, 1, domain(PSet))
    return sieve
end

function add_node!(sieve::Sieve{PSet}, start::Int, stop::Int) where {PSet}
    push!(sieve.start, start)
    push!(sieve.stop, stop)
    push!(sieve.mask, interval(PSet, start, stop))
    push!(sieve.rest, interval(PSet, stop + 1, domain(PSet)))
    push!(sieve.children, SieveChild{PSet}[])
    push!(sieve.leaves, SieveLeaf{PSet}[])
    return length(sieve.start)
end

function isleafy(sieve::Sieve{PSet}, v::Int) where {PSet}
    return sieve.stop[v] == domain(PSet)
end

function nentries(sieve::Sieve, v::Int)
    return isleafy(sieve, v) ? length(sieve.leaves[v]) : length(sieve.children[v])
end

# ==================== Insertion ====================

# Store the PTDUR i with bag B, vertex set V, and margin m.
function Base.setindex!(sieve::Sieve{PSet}, (B, V, m)::Tuple{PSet, PSet, Int}, i::Int) where {PSet}
    v = 1

    @inbounds while !isleafy(sieve, v)
        M = sieve.mask[v]
        BM = B ∩ M
        children = sieve.children[v]
        found = 0

        for c in eachindex(children)
            if BM == children[c].key ∩ M
                found = c
                break
            end
        end

        if iszero(found)
            w = add_node!(sieve, sieve.stop[v] + 1, domain(PSet))
            push!(children, SieveChild{PSet}(B, B, B, m, w))
            push!(sieve.leaves[w], SieveLeaf{PSet}(B, V, m, i))
            maybe_split!(sieve, v)
            return sieve
        end

        x = children[found]
        children[found] = SieveChild{PSet}(x.key, x.any ∪ B, x.all ∩ B, max(x.margin, m), x.node)
        v = x.node
    end

    push!(sieve.leaves[v], SieveLeaf{PSet}(B, V, m, i))
    maybe_split!(sieve, v)
    return sieve
end

function maybe_split!(sieve::Sieve, v::Int)
    if nentries(sieve, v) > MAX_CHILDREN_PER_NODE && sieve.start[v] < sieve.stop[v]
        split_node!(sieve, v)
    end

    return
end

function split_node!(sieve::Sieve{PSet}, v::Int) where {PSet}
    start = sieve.start[v]
    oldstop = sieve.stop[v]
    newstop = split_index(start, oldstop)
    leafy = isleafy(sieve, v)
    oldchildren = sieve.children[v]
    oldleaves = sieve.leaves[v]

    sieve.stop[v] = newstop
    M = sieve.mask[v] = interval(PSet, start, newstop)
    sieve.rest[v] = interval(PSet, newstop + 1, domain(PSet))
    children = sieve.children[v] = SieveChild{PSet}[]
    sieve.leaves[v] = SieveLeaf{PSet}[]
    groups = Dict{PSet, Int}()   # key on M -> entry of v

    # Find (or make) the child of v for the key K, and record a subtree with
    # union `any`, intersection `all` and margin `m` below it.
    function place!(K::PSet, any::PSet, all::PSet, m::Int)
        e = get(groups, K ∩ M, 0)

        if iszero(e)
            push!(children, SieveChild{PSet}(K, any, all, m, add_node!(sieve, newstop + 1, oldstop)))
            e = groups[K ∩ M] = length(children)
        else
            x = children[e]
            children[e] = SieveChild{PSet}(x.key, x.any ∪ any, x.all ∩ all, max(x.margin, m), x.node)
        end

        return children[e].node
    end

    if leafy
        for x in oldleaves
            push!(sieve.leaves[place!(x.key, x.key, x.key, x.margin)], x)
        end
    else
        for x in oldchildren
            push!(sieve.children[place!(x.key, x.any, x.all, x.margin)], x)
        end
    end

    return
end

function split_index(start::Int, stop::Int)
    return start + (1 << floor(Int, log(stop - start))) - 1
end

# ==================== Query ====================

# Append to `out` the PTDURs i with bag B, vertex set V, and margin m such that
#
#   - V ∩ R = ∅
#   - w(S - B) ≤ m
#
# `slack` must be k + 1 - w(S).
function query!(out::Vector{Int}, sieve::Sieve{PSet}, stack::Vector{Tuple{Int, Int, Int}},
                R::PSet, S::PSet, slack::Int, weights::AbstractVector{Int}) where {PSet}
    push!(empty!(stack), (1, 0, 0))

    @inbounds while !isempty(stack)
        v, i, j = pop!(stack)
        M = sieve.mask[v]
        RM = R ∩ M
        SM = S ∩ M

        if isleafy(sieve, v)
            for x in sieve.leaves[v]
                W = x.key

                if isdisjoint(RM, W) && isdisjoint(R, x.set) &&
                        i + wt(weights, setdiff(SM, W)) <= x.margin &&
                        j + wt(weights, setdiff(W ∩ M, S)) <= slack
                    push!(out, x.value)
                end
            end
        else
            T = sieve.rest[v]
            ST = S ∩ T

            for x in sieve.children[v]
                W = x.key

                # the keys below contain x.all and are contained in x.any
                (isdisjoint(RM, W) && isdisjoint(R, x.all)) || continue
                ii = i + wt(weights, setdiff(SM, W))
                ii <= x.margin || continue
                jj = j + wt(weights, setdiff(W ∩ M, S))
                jj <= slack || continue
                ii + wt(weights, setdiff(ST, x.any)) <= x.margin || continue
                jj + wt(weights, setdiff(x.all ∩ T, S)) <= slack || continue
                push!(stack, (x.node, ii, jj))
            end
        end
    end

    return out
end
