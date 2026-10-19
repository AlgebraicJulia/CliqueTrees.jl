# Device-side analysis of a vertex set X (also callable on the CPU). Components of G - X are
# visited in increasing order of their smallest vertex (BFS from the smallest vertex left),
# which turns Tamaki's outbound test into a one-pass test with O(1) state:
#
#   D_j (neighbourhood N_j) is outbound  <=>  every earlier component D_i (smaller min vertex)
#   is confined to N_j and has N_i != N_j, and min(X \ N_j) > min(D_j).
#
# With all earlier N_i ⊆ N_j, some N_i equals N_j iff the largest earlier |N_i| is |N_j|,
# so the state is the union of the earlier neighbourhoods and their largest size.

# The first (outbound) full component of X, or the empty set. allowed = V \ X.
@inline function dv_full(adj, all::BS{W}, X::BS{W}) where {W}
    rest = bs_andnot(all, X)
    while !bs_isempty(rest)
        v = bs_min(rest)
        D = bs_comp(adj, rest, v)
        rest = bs_andnot(rest, D)
        N = bs_open(adj, D)
        N === X && return D
    end
    return bs_zero(Val(W))
end

# Rank of vertex v inside X (0-based position among the vertices of X).
@inline function dv_rank(X::BS{W}, v::Integer) where {W}
    wi = ((v - 1) >> 6) + 1; b = (v - 1) & 63
    r = 0
    for i in 1:(wi - 1)
        r += count_ones(X[i])
    end
    return r + count_ones(X[wi] & ((one(UInt64) << b) - one(UInt64)))
end

# s ⊆ X compressed to X-local coordinates: bit r of the result is the r-th vertex of X.
# Up to 128 vertices (two words).
@inline function dv_compress(X::BS{W}, s::BS{W}) where {W}
    lo = zero(UInt64); hi = zero(UInt64); base = 0
    for i in 1:W
        xw = X[i]; sw = s[i] & xw
        while !iszero(sw)
            b = trailing_zeros(sw); sw &= sw - one(UInt64)
            r = base + count_ones(xw & ((one(UInt64) << b) - one(UInt64)))
            if r < 64
                lo |= one(UInt64) << r
            else
                hi |= one(UInt64) << (r - 64)
            end
        end
        base += count_ones(xw)
    end
    return lo, hi
end

# Classify X: returns (kind, A) with kind 1 = has a full component (A = outbound one),
# 2 = PMC, 0 = neither, 3 = too large for the local test (|X| > 128). scratch(r) gives the
# r-th (lo, hi) row slot of this thread's uncovered-pair matrix via getrow/setrow.
@inline function dv_classify(adj, all::BS{W}, X::BS{W}, scr, sbase::Integer, sstride::Integer, pmc_only::Bool) where {W}
    nx = bs_count(X)
    nx > 128 && return (Int32(3), bs_zero(Val(W)))
    # pass 1: a full component? (cheap early exit for O-block candidates)
    rest = bs_andnot(all, X)
    # rows of the uncovered-pair matrix: non-neighbours inside X
    fulllo = nx >= 64 ? ~zero(UInt64) : (one(UInt64) << nx) - one(UInt64)
    fullhi = nx >= 128 ? ~zero(UInt64) : nx > 64 ? (one(UInt64) << (nx - 64)) - one(UInt64) : zero(UInt64)
    r = 0
    for i in 1:W
        word = X[i]
        while !iszero(word)
            v = 64 * (i - 1) + trailing_zeros(word) + 1; word &= word - one(UInt64)
            @inbounds lo, hi = dv_compress(X, adj[v])
            if r < 64
                lo |= one(UInt64) << r
            else
                hi |= one(UInt64) << (r - 64)
            end
            @inbounds scr[sbase + (2r) * sstride] = fulllo & ~lo
            @inbounds scr[sbase + (2r + 1) * sstride] = fullhi & ~hi
            r += 1
        end
    end
    while !bs_isempty(rest)
        v = bs_min(rest)
        D = bs_comp(adj, rest, v)
        rest = bs_andnot(rest, D)
        N = bs_open(adj, D)
        if N === X
            return (pmc_only ? Int32(0) : Int32(1), D)
        end
        mlo, mhi = dv_compress(X, N)
        m = mlo
        while !iszero(m)
            q = trailing_zeros(m); m &= m - one(UInt64)
            @inbounds scr[sbase + (2q) * sstride] &= ~mlo
            @inbounds scr[sbase + (2q + 1) * sstride] &= ~mhi
        end
        m = mhi
        while !iszero(m)
            q = trailing_zeros(m) + 64; m &= m - one(UInt64)
            @inbounds scr[sbase + (2q) * sstride] &= ~mlo
            @inbounds scr[sbase + (2q + 1) * sstride] &= ~mhi
        end
    end
    acc = zero(UInt64)
    for q in 0:(nx - 1)
        @inbounds acc |= scr[sbase + (2q) * sstride] | scr[sbase + (2q + 1) * sstride]
    end
    return (iszero(acc) ? Int32(2) : Int32(0), bs_zero(Val(W)))
end

# outlet(Ω) of a PMC Ω: the largest neighbourhood of an outbound component (empty if none).
@inline function dv_outlet(adj, all::BS{W}, Ω::BS{W}) where {W}
    rest = bs_andnot(all, Ω)
    U = bs_zero(Val(W)); M = 0
    outlet = bs_zero(Val(W)); best = -1
    while !bs_isempty(rest)
        v = bs_min(rest)
        D = bs_comp(adj, rest, v)
        rest = bs_andnot(rest, D)
        N = bs_open(adj, D); cN = bs_count(N)
        if bs_subset(U, N) && M < cN && bs_min(bs_andnot(Ω, N)) > v
            if cN > best
                best = cN; outlet = N
            end
        end
        U = bs_or(U, N); M = max(M, cN)
    end
    return outlet, best >= 0
end
