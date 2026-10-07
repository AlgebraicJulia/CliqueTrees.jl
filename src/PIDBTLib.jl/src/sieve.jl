# Sieve: a trie storing key-value pairs i => V, where V is a vertex set (the
# vertex set of a PTDUR) and i is a PTDUR root. Keys need not be unique.
# Every value carries a margin m (the PTDUR's remaining weight capacity).
#
# Every inner node v compares the keys below it on an interval
# [start(v), stop(v)] of vertices; the intervals along a root-to-leaf path
# partition 1:domain. Inner nodes with stop(v) == domain have leaves as
# children; all other inner nodes have inner children. Every node stores its
# key and the largest margin in its subtree, so a query can prune a subtree as
# soon as the margin is exceeded.
#
# When a node has too many children, it is split: its interval is shortened,
# and its children are grouped under new nodes by their keys on the shortened
# interval.
#
# Nodes are stored as a struct of arrays; children form singly linked lists.

const MAX_CHILDREN_PER_NODE = 32

struct Sieve{PSet <: AbstractPackedSet}
    start::Vector{Int}
    stop::Vector{Int}
    mask::Vector{PSet}      # interval [start, stop] as a set (inner nodes)
    key::Vector{PSet}
    margin::Vector{Int}     # largest margin in subtree
    value::Vector{Int}      # PTDUR root (leaves); 0 (inner nodes)
    head::Vector{Int}       # first child
    next::Vector{Int}       # next sibling
    degree::Vector{Int}
end

function Sieve{PSet}() where {PSet <: AbstractPackedSet}
    sieve = Sieve{PSet}(Int[], Int[], PSet[], PSet[], Int[], Int[], Int[], Int[], Int[])
    add_node!(sieve, 1, domain(PSet), PSet(), -1, 0)
    return sieve
end

function add_node!(sieve::Sieve{PSet}, start::Int, stop::Int, K::PSet, m::Int, value::Int) where {PSet}
    push!(sieve.start, start)
    push!(sieve.stop, stop)
    push!(sieve.mask, start <= stop ? interval(PSet, start, stop) : PSet())
    push!(sieve.key, K)
    push!(sieve.margin, m)
    push!(sieve.value, value)
    push!(sieve.head, 0)
    push!(sieve.next, 0)
    push!(sieve.degree, 0)
    return length(sieve.start)
end

# Make v the first child of u.
function attach!(sieve::Sieve, u::Int, v::Int)
    sieve.next[v] = sieve.head[u]
    sieve.head[u] = v
    sieve.degree[u] += 1
    return
end

function isleafy(sieve::Sieve{PSet}, v::Int) where {PSet}
    return sieve.stop[v] == domain(PSet)
end

# ==================== Insertion ====================

function Base.setindex!(sieve::Sieve{PSet}, (V, m)::Tuple{PSet, Int}, i::Int) where {PSet}
    v = 1

    while !isleafy(sieve, v)
        sieve.margin[v] = max(sieve.margin[v], m)
        M = sieve.mask[v]
        VM = V ∩ M
        found = 0
        w = sieve.head[v]

        while !iszero(w)
            if VM == sieve.key[w] ∩ M
                found = w
                break
            end

            w = sieve.next[w]
        end

        if iszero(found)
            w = add_node!(sieve, sieve.stop[v] + 1, domain(PSet), V, m, 0)
            attach!(sieve, v, w)
            attach!(sieve, w, add_node!(sieve, 0, -1, V, m, i))
            maybe_split!(sieve, v)
            return sieve
        end

        v = found
    end

    sieve.margin[v] = max(sieve.margin[v], m)
    attach!(sieve, v, add_node!(sieve, 0, -1, V, m, i))
    maybe_split!(sieve, v)
    return sieve
end

function maybe_split!(sieve::Sieve, v::Int)
    if sieve.degree[v] > MAX_CHILDREN_PER_NODE && sieve.start[v] < sieve.stop[v]
        split_node!(sieve, v)
    end

    return
end

function split_node!(sieve::Sieve{PSet}, v::Int) where {PSet}
    start = sieve.start[v]
    oldstop = sieve.stop[v]
    newstop = split_index(start, oldstop)

    sieve.stop[v] = newstop
    M = sieve.mask[v] = interval(PSet, start, newstop)

    x = sieve.head[v]
    sieve.head[v] = 0
    sieve.degree[v] = 0
    groups = Dict{PSet, Int}()   # key on M -> new child

    while !iszero(x)
        y = sieve.next[x]
        X = sieve.key[x]
        g = get(groups, X ∩ M, 0)

        if iszero(g)
            g = groups[X ∩ M] = add_node!(sieve, newstop + 1, oldstop, X, sieve.margin[x], 0)
            attach!(sieve, v, g)
        else
            sieve.margin[g] = max(sieve.margin[g], sieve.margin[x])
        end

        attach!(sieve, g, x)
        x = y
    end

    return
end

function split_index(start::Int, stop::Int)
    return start + (1 << floor(Int, log(stop - start))) - 1
end

# ==================== Query ====================

# Append to `out` the values i of all key-value pairs i => V with margin m
# such that
#
#   - V ∩ R = ∅
#   - w(S - V) ≤ m
#
function query!(out::Vector{Int}, sieve::Sieve{PSet}, stack::Vector{Tuple{Int, Int}},
                R::PSet, S::PSet, weights::AbstractVector{Int}) where {PSet}
    push!(empty!(stack), (1, 0))

    @inbounds while !isempty(stack)
        v, i = pop!(stack)
        M = sieve.mask[v]
        RM = R ∩ M
        SM = S ∩ M
        leafy = isleafy(sieve, v)
        w = sieve.head[v]

        while !iszero(w)
            W = sieve.key[w]

            if isdisjoint(RM, W)
                j = i + wt(weights, setdiff(SM, W))

                if j <= sieve.margin[w]
                    if leafy
                        push!(out, sieve.value[w])
                    else
                        push!(stack, (w, j))
                    end
                end
            end

            w = sieve.next[w]
        end
    end

    return out
end
