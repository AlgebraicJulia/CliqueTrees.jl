# Fixed-width vertex sets as NTuple{W,UInt64}: isbits, so the same functions run on the
# CPU and inside CUDA kernels. Vertex v (1-based) is bit (v-1)&63 of word (v-1)>>6 + 1.

const BS{W} = NTuple{W, UInt64}

@inline bsval(::BS{W}) where {W} = Val(W)
@inline bs_zero(::Val{W}) where {W} = ntuple(_ -> zero(UInt64), Val(W))
@inline bs_bit(::Val{W}, v::Integer) where {W} = ntuple(i -> i == ((v - 1) >> 6) + 1 ? (one(UInt64) << ((v - 1) & 63)) : zero(UInt64), Val(W))
@inline bs_and(a::BS{W}, b::BS{W}) where {W} = ntuple(i -> a[i] & b[i], Val(W))
@inline bs_or(a::BS{W}, b::BS{W}) where {W} = ntuple(i -> a[i] | b[i], Val(W))
@inline bs_andnot(a::BS{W}, b::BS{W}) where {W} = ntuple(i -> a[i] & ~b[i], Val(W))
@inline function bs_isempty(a::BS{W}) where {W}
    x = zero(UInt64)
    for i in 1:W
        x |= a[i]
    end
    return iszero(x)
end
@inline function bs_count(a::BS{W}) where {W}
    c = 0
    for i in 1:W
        c += count_ones(a[i])
    end
    return c
end
@inline function bs_subset(a::BS{W}, b::BS{W}) where {W}   # a ⊆ b
    x = zero(UInt64)
    for i in 1:W
        x |= a[i] & ~b[i]
    end
    return iszero(x)
end
@inline function bs_disjoint(a::BS{W}, b::BS{W}) where {W}
    x = zero(UInt64)
    for i in 1:W
        x |= a[i] & b[i]
    end
    return iszero(x)
end
@inline bs_eq(a::BS{W}, b::BS{W}) where {W} = a === b
@inline function bs_min(a::BS{W}) where {W}   # smallest vertex, 0 if empty
    for i in 1:W
        if !iszero(a[i])
            return 64 * (i - 1) + trailing_zeros(a[i]) + 1
        end
    end
    return 0
end
@inline bs_has(a::BS{W}, v::Integer) where {W} = !iszero(a[((v - 1) >> 6) + 1] & (one(UInt64) << ((v - 1) & 63)))
@inline function bs_less(a::BS{W}, b::BS{W}) where {W}  # lexicographic on words, for sorting
    for i in W:-1:1
        a[i] == b[i] || return a[i] < b[i]
    end
    return false
end

# 64-bit hash of a set (splitmix-style mixing of the words).
@inline function bs_hash(a::BS{W}) where {W}
    h = 0x9e3779b97f4a7c15
    for i in 1:W
        x = a[i] + h + UInt64(i) * 0xbf58476d1ce4e5b9
        x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
        x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
        h = x ⊻ (x >> 31) ⊻ (h << 7) ⊻ (h >> 57)
    end
    return h
end

# The union of the adjacency rows of the vertices of s. adj is indexable by vertex.
@inline function bs_nbrs(adj, s::BS{W}) where {W}
    acc = bs_zero(Val(W))
    for i in 1:W
        word = s[i]
        while !iszero(word)
            v = 64 * (i - 1) + trailing_zeros(word) + 1
            word &= word - one(UInt64)
            @inbounds acc = bs_or(acc, adj[v])
        end
    end
    return acc
end
@inline bs_open(adj, s::BS{W}) where {W} = bs_andnot(bs_nbrs(adj, s), s)   # N(s)

# The component of v inside the allowed set (v ∈ allowed), by frontier expansion.
@inline function bs_comp(adj, allowed::BS{W}, v::Integer) where {W}
    comp = bs_bit(Val(W), v)
    front = comp
    while !bs_isempty(front)
        nb = bs_nbrs(adj, front)
        front = bs_andnot(bs_and(nb, allowed), comp)
        comp = bs_or(comp, front)
    end
    return comp
end

# Canonical weight of a set: vertices are grouped into weight classes, and the weight is
# sum_c wcls[c] * |s ∩ mask[c]| in class order. The value depends only on the set, never
# on an enumeration order, so every comparison is reproducible.
@inline function bs_weight(s::BS{W}, masks, wcls, ncls::Integer) where {W}
    w = 0.0
    for c in 1:ncls
        @inbounds w += wcls[c] * bs_count(bs_and(s, masks[c]))
    end
    return w
end
# The lightest weight among the vertices of s (Inf if empty).
@inline function bs_wmin(s::BS{W}, masks, wcls, ncls::Integer) where {W}
    w = Inf
    for c in 1:ncls
        @inbounds if !bs_disjoint(s, masks[c])
            w = min(w, wcls[c])
        end
    end
    return w
end
