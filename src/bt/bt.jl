#
# Helpers shared by the dynamic programs over potential maximal
# cliques and minimal separators. A packed set is a vector of
# words: the vertex v is the bit (v - 1) % 64 of the word
# (v - 1) ÷ 64 + 1.
#

# The word and the bit of the vertex v.
function bt_wrd(v::V) where {V}
    return (v - one(V)) >> 6 + one(V)
end

function bt_bit(v::V) where {V}
    return one(UInt64) << ((v - one(V)) & 63)
end

# The smallest vertex in the word i.
function bt_vtx(::Type{V}, i::Integer, word::UInt64) where {V}
    return convert(V, 64(i - 1) + trailing_zeros(word) + 1)
end

# The neighborhoods of a graph as packed sets, without loops.
function bt_adj!(adj::AbstractMatrix{UInt64}, graph::AbstractGraph)
    fill!(adj, zero(UInt64))

    @inbounds for v in vertices(graph)
        for x in neighbors(graph, v)
            if x != v
                adj[bt_wrd(x), v] |= bt_bit(x)
            end
        end
    end

    return adj
end

# The packed sets in out as a relation from sets to vertices.
function bt_out(out::Vector{UInt64}, nvtx::V, nwrd::Int) where {V}
    nlst = convert(V, length(out) ÷ max(nwrd, 1))
    lstptr = V[one(V)]; lsttgt = V[]

    @inbounds for f in oneto(nlst)
        p = (f - 1) * nwrd

        for i in oneto(nwrd)
            word = out[p + i]

            while !iszero(word)
                push!(lsttgt, bt_vtx(V, i, word)); word &= word - one(UInt64)
            end
        end

        push!(lstptr, convert(V, length(lsttgt)) + one(V))
    end

    return BipartiteGraph{V, V}(nvtx, nlst, length(lsttgt), lstptr, lsttgt)
end

# Remove the repeated sets from a list of packed sets.
function bt_unq!(lst::Vector{UInt64}, nwrd::Int)
    m = length(lst) ÷ max(nwrd, 1)

    function less(f, g)
        @inbounds for i in oneto(nwrd)
            x = lst[(f - 1) * nwrd + i]; y = lst[(g - 1) * nwrd + i]
            x == y || return x < y
        end

        return false
    end

    perm = sort!(collect(oneto(m)); lt = less)
    new = UInt64[]

    @inbounds for (j, f) in enumerate(perm)
        if isone(j) || less(perm[j - 1], f)
            for i in oneto(nwrd)
                push!(new, lst[(f - 1) * nwrd + i])
            end
        end
    end

    empty!(lst); append!(lst, new)
    return lst
end

# Sets in a pool: the set at offset p is the words pool[p + 1:p + nwrd].

function bt_new!(pool::Vector{UInt64}, nwrd::Int)
    p = length(pool)

    for _ in oneto(nwrd)
        push!(pool, zero(UInt64))
    end

    return p
end

function bt_len(pool::Vector{UInt64}, p::Int, nwrd::Int)
    c = 0

    @inbounds for i in oneto(nwrd)
        c += count_ones(pool[p + i])
    end

    return c
end

function bt_wgt(pool::Vector{UInt64}, p::Int, weights::AbstractVector{W}, nwrd::Int) where {W}
    w = zero(W)

    @inbounds for i in oneto(nwrd)
        word = pool[p + i]

        while !iszero(word)
            w += weights[bt_vtx(Int, i, word)]; word &= word - one(UInt64)
        end
    end

    return w
end

function bt_min(pool::Vector{UInt64}, p::Int, nwrd::Int)
    @inbounds for i in oneto(nwrd)
        word = pool[p + i]
        iszero(word) || return bt_vtx(Int, i, word)
    end

    return 0
end

function bt_eq(pool::Vector{UInt64}, p::Int, q::Int, nwrd::Int)
    @inbounds for i in oneto(nwrd)
        pool[p + i] == pool[q + i] || return false
    end

    return true
end

# Is the set p a subset of the set q?
function bt_sub(pool::Vector{UInt64}, p::Int, q::Int, nwrd::Int)
    @inbounds for i in oneto(nwrd)
        iszero(pool[p + i] & ~pool[q + i]) || return false
    end

    return true
end

# Is the set p disjoint from the set excl?
function bt_dsj(pool::Vector{UInt64}, p::Int, excl::AbstractVector{UInt64}, nwrd::Int)
    @inbounds for i in oneto(nwrd)
        iszero(pool[p + i] & excl[i]) || return false
    end

    return true
end

# Clear the bits of the set p beyond the last vertex.
function bt_cap!(pool::Vector{UInt64}, nvtx::Integer, p::Int, nwrd::Int)
    r = nvtx - 64(nwrd - 1)

    if r < 64
        @inbounds pool[p + nwrd] &= (one(UInt64) << r) - one(UInt64)
    end

    return
end

# The set p becomes the component of v in the set q.
function bt_cmp!(pool::Vector{UInt64}, adj::AbstractMatrix{UInt64}, p::Int, q::Int, v::Integer, nwrd::Int)
    top = length(pool)
    t = bt_new!(pool, nwrd)    # the frontier
    u = bt_new!(pool, nwrd)    # the next frontier

    @inbounds for i in oneto(nwrd)
        pool[p + i] = zero(UInt64)
    end

    @inbounds pool[p + bt_wrd(v)] = bt_bit(v)
    @inbounds pool[t + bt_wrd(v)] = bt_bit(v)

    @inbounds while true
        for i in oneto(nwrd)
            pool[u + i] = zero(UInt64)
        end

        for i in oneto(nwrd)
            word = pool[t + i]

            while !iszero(word)
                x = bt_vtx(Int, i, word); word &= word - one(UInt64)

                for j in oneto(nwrd)
                    pool[u + j] |= adj[j, x]
                end
            end
        end

        grow = false

        for i in oneto(nwrd)
            word = pool[u + i] & pool[q + i] & ~pool[p + i]
            pool[u + i] = word; pool[p + i] |= word; grow |= !iszero(word)
        end

        grow || break
        t, u = u, t
    end

    resize!(pool, top)
    return
end

function bt_nbr!(pool::Vector{UInt64}, adj::AbstractMatrix{UInt64}, p::Int, b::Int, nwrd::Int)
    @inbounds for i in oneto(nwrd)
        word = pool[b + i]

        while !iszero(word)
            u = bt_vtx(Int, i, word); word &= word - one(UInt64)

            for j in oneto(nwrd)
                pool[p + j] |= adj[j, u]
            end
        end
    end

    @inbounds for i in oneto(nwrd)
        pool[p + i] &= ~pool[b + i]
    end

    return
end

include("pmcdp.jl")
include("sepdp.jl")
include("sepenum.jl")
include("pmcenum.jl")
