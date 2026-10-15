function bt_wrd(v::V) where {V}
    return (v - one(V)) >> 6 + one(V)
end

function bt_bit(v::V) where {V}
    return 0x0000000000000001 << ((v - one(V)) & 63)
end

function bt_vtx(::Type{V}, i::Integer, word::UInt64) where {V}
    return convert(V, 64(i - 1) + trailing_zeros(word) + 1)
end

# The outcome of a search: it found what it looked for (a list,
# or a decomposition with bags of weight at most the bound), it
# showed that every decomposition has a heavier bag, or it ran out
# of time.
@enum BTStatus BT_FOUND BT_ABOVE BT_TIMEOUT

# A list of n packed sets: a matrix with a column of nwrd words for
# each set or, with one word, a vector of words, so that the methods
# for one word are chosen by its type. bt_mat(x, n) makes a list of
# the same kind as the list x.
function bt_mat(nwrd::Integer, n::Integer)
    if isone(nwrd)
        return FVector{UInt64}(undef, n)
    else
        return FMatrix{UInt64}(undef, nwrd, n)
    end
end

bt_mat(x::AbstractVector{UInt64}, n::Integer) = FVector{UInt64}(undef, n)
bt_mat(x::AbstractMatrix{UInt64}, n::Integer) = FMatrix{UInt64}(undef, size(x, 1), n)

# The word i of the set v, and setting it to w.
Base.@propagate_inbounds bt_get(x::AbstractMatrix{UInt64}, i::Integer, v::Integer) = x[i, v]
Base.@propagate_inbounds bt_get(x::AbstractVector{UInt64}, i::Integer, v::Integer) = x[v]
Base.@propagate_inbounds bt_put!(x::AbstractMatrix{UInt64}, w::UInt64, i::Integer, v::Integer) = (x[i, v] = w)
Base.@propagate_inbounds bt_put!(x::AbstractVector{UInt64}, w::UInt64, i::Integer, v::Integer) = (x[v] = w)

# The number of words of a set, and the number of sets.
bt_nwd(x::AbstractMatrix{UInt64}) = size(x, 1)
bt_nwd(x::AbstractVector{UInt64}) = 1
bt_nst(x::AbstractMatrix{UInt64}) = size(x, 2)
bt_nst(x::AbstractVector{UInt64}) = length(x)

function bt_adj!(adj::AbstractVector{UInt64}, graph::AbstractGraph)
    fill!(adj, zero(UInt64))

    @inbounds for v in vertices(graph)
        for x in neighbors(graph, v)
            if x != v
                adj[v] |= bt_bit(x)
            end
        end
    end

    return adj
end

function bt_adj!(adj::AbstractMatrix{UInt64}, graph::AbstractGraph)
    fill!(adj, zero(UInt64))

    @inbounds for v in vertices(graph)
        for x in neighbors(graph, v)
            if x != v
                w = bt_wrd(x)
                b = bt_bit(x)
                adj[w, v] |= b
            end
        end
    end

    return adj
end

function bt_out(out::Vector{UInt64}, nvtx::V, nwrd::V) where {V}
    nlst = convert(V, length(out) ÷ max(nwrd, one(V)))
    lstptr = V[one(V)]; lsttgt = V[]

    @inbounds for f in oneto(nlst)
        p = (f - one(V)) * nwrd

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
function bt_unq!(lst::Vector{UInt64}, nwrd::V) where {V}
    m = convert(V, length(lst) ÷ max(nwrd, one(V)))

    function less(f, g)
        @inbounds for i in oneto(nwrd)
            x = lst[(f - one(V)) * nwrd + i]; y = lst[(g - one(V)) * nwrd + i]
            x == y || return x < y
        end

        return false
    end

    perm = sort!(collect(oneto(m)); lt = less)
    new = UInt64[]

    @inbounds for (j, f) in enumerate(perm)
        if isone(j) || less(perm[j - 1], f)
            for i in oneto(nwrd)
                push!(new, lst[(f - one(V)) * nwrd + i])
            end
        end
    end

    empty!(lst); append!(lst, new)
    return lst
end

# A clock for a deadline, in the sense of time(). A search ticks it,
# and it reads the time on the first tick and every BT_TICK ticks
# after; once the deadline has passed, out is set, and the search
# unwinds. The tasks of a search
# each have their own ticks and share out.
const BT_TICK = 1024

function bt_clk(deadline::Real)
    return (; deadline = convert(Float64, deadline), tick = Ref(0), out = Threads.Atomic{Bool}(false))
end

function bt_clk(clk::NamedTuple)
    return (; deadline = clk.deadline, tick = Ref(0), out = clk.out)
end

function bt_tck!(clk::NamedTuple)
    t = clk.tick[] += 1

    if isone(t & (BT_TICK - 1)) && time() > clk.deadline
        clk.out[] = true
    end

    return clk.out[]
end

# Sets in a pool: the set at offset p is the words pool[p + 1:p + nwrd].

function bt_new!(pool::Vector{UInt64}, nwrd::V) where {V}
    p = convert(V, length(pool))

    for _ in oneto(nwrd)
        push!(pool, zero(UInt64))
    end

    return p
end

function bt_len(pool::Vector{UInt64}, p::V, nwrd::V) where {V}
    c = 0

    @inbounds for i in oneto(nwrd)
        c += count_ones(pool[p + i])
    end

    return c
end

function bt_wgt(pool::Vector{UInt64}, p::V, weights::AbstractVector{W}, nwrd::V) where {W, V}
    w = zero(W)

    @inbounds for i in oneto(nwrd)
        word = pool[p + i]

        while !iszero(word)
            w += weights[bt_vtx(Int, i, word)]; word &= word - one(UInt64)
        end
    end

    return w
end

function bt_min(pool::Vector{UInt64}, p::V, nwrd::V) where {V}
    @inbounds for i in oneto(nwrd)
        word = pool[p + i]
        iszero(word) || return bt_vtx(Int, i, word)
    end

    return 0
end

function bt_eq(pool::Vector{UInt64}, p::V, q::V, nwrd::V) where {V}
    @inbounds for i in oneto(nwrd)
        pool[p + i] == pool[q + i] || return false
    end

    return true
end

# Is the set p a subset of the set q?
function bt_sub(pool::Vector{UInt64}, p::V, q::V, nwrd::V) where {V}
    @inbounds for i in oneto(nwrd)
        iszero(pool[p + i] & ~pool[q + i]) || return false
    end

    return true
end

# Is the set p disjoint from the set excl?
function bt_dsj(pool::Vector{UInt64}, p::V, excl::AbstractVector{UInt64}, nwrd::V) where {V}
    @inbounds for i in oneto(nwrd)
        iszero(pool[p + i] & excl[i]) || return false
    end

    return true
end

# Clear the bits of the set p beyond the last vertex.
function bt_cap!(pool::Vector{UInt64}, nvtx::Integer, p::V, nwrd::V) where {V}
    r = nvtx - 64(nwrd - one(V))

    if r < 64
        @inbounds pool[p + nwrd] &= (one(UInt64) << r) - one(UInt64)
    end

    return
end

# The set p becomes the component of v in the set q.
function bt_cmp!(pool::Vector{UInt64}, adj::AbstractMatrix{UInt64}, p::V, q::V, v::Integer, nwrd::V) where {V}
    top = convert(V, length(pool))
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

# bt_cmp! with one word: the sets are words.
function bt_cmp!(pool::Vector{UInt64}, adj::AbstractVector{UInt64}, p::V, q::V, v::Integer, nwrd::V) where {V}
    @inbounds m = pool[q + one(V)]
    c = t = bt_bit(v)

    @inbounds while true
        u = zero(UInt64)

        while !iszero(t)
            u |= adj[trailing_zeros(t) + 1]; t &= t - one(UInt64)
        end

        t = u & m & ~c
        iszero(t) && break
        c |= t
    end

    @inbounds pool[p + one(V)] = c
    return
end

# bt_nbr! with one word.
function bt_nbr!(pool::Vector{UInt64}, adj::AbstractVector{UInt64}, p::V, b::V, nwrd::V) where {V}
    @inbounds word = b0 = pool[b + one(V)]
    @inbounds u = pool[p + one(V)]

    @inbounds while !iszero(word)
        u |= adj[trailing_zeros(word) + 1]; word &= word - one(UInt64)
    end

    @inbounds pool[p + one(V)] = u & ~b0
    return
end

function bt_nbr!(pool::Vector{UInt64}, adj::AbstractMatrix{UInt64}, p::V, b::V, nwrd::V) where {V}
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
include("septw.jl")
