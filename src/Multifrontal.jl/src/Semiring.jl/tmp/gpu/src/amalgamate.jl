# Supernode amalgamation for the GPU solve.
#
# Most fronts of the elimination tree of a mesh or a road network have a single residual vertex
# (90–96% here), and every front gathers its whole separator once per right-hand side. Chains of
# such fronts re-gather almost the same separator over and over, and every front is a tree level
# (or a step of a thread's walk) of its own.
#
# This merges each chain f, f + 1, …, g in which every front is the last child of the next (so the
# residual vertices of the chain are contiguous) into one front with residual res(f) ∪ … ∪ res(g)
# and separator sep(g), up to `nmax` residual vertices, as long as the zero padding stays small.
# The merged blocks are filled from the existing factor and padded with the semiring zero, which
# contributes nothing: the sweeps perform the same products, over a coarser partition of the same
# elimination order (a relaxed supernode partition, as in sparse direct solvers). The CPU factor
# is not changed.
#
# By the clique-tree property, sep(m) ⊆ res(m + 1) ∪ sep(m + 1) for each member m, so every
# separator vertex of a member is either a residual vertex of the merged front or in sep(g).
#
# The merge depends only on the symbolic factorization. It is computed once per symbolic object as
# index maps (cached), and each numeric factor is then rearranged on the GPU by one gather per array,
# without leaving the device.

struct Amalgamation{I}
    nf::Int
    Rptr::Vector{I}
    Sptr::Vector{I}
    Stgt::Vector{I}
    Dptr::Vector{I}
    Lptr::Vector{I}
    pnt::Vector{I}
    idx::Vector{I}
    # new entry ← old entry: > 0 from the D array, < 0 from the L array (negated), 0 the semiring zero
    mLD::CuVector{Int32}
    mUD::CuVector{Int32}
    mLL::CuVector{Int32}
    mUL::CuVector{Int32}
    # a hash of the structure it was computed from: a cache hit must match it
    from::UInt
    group::Vector{Int}          # original front → merged front
end

# per symbolic factorization, (nmax, alpha) and device (the index maps live on the device). Keyed on the identity of an object that lives as long
# as the symbolic factorization (held weakly). Not a WeakKeyDict: that compares keys with isequal, so
# two factorizations whose separator targets are equal arrays (e.g. both empty) would share a merge.
const AMALGAMATIONS = Dict{UInt, Tuple{WeakRef, Dict{Tuple{Int, Float64, Int, UInt}, Any}}}()
const AMALGAMATIONS_LOCK = ReentrantLock()

function amalgamation(key, ::Type{I}, nmax::Integer, alpha::Real, Rptr, Sptr, Stgt, Dptr, Lptr, pnt, idx; allowed = nothing, compact::Bool = false) where {I}
    isnothing(key) && return amalgamate_fronts(I, nmax, alpha, Rptr, Sptr, Stgt, Dptr, Lptr, pnt, idx; allowed, compact)

    cache = lock(AMALGAMATIONS_LOCK) do
        filter!(kv -> !isnothing(kv[2][1].value), AMALGAMATIONS)           # forget collected factorizations
        id = objectid(key)
        entry = get(AMALGAMATIONS, id, nothing)

        if isnothing(entry) || entry[1].value !== key
            entry = (WeakRef(key), Dict{Tuple{Int, Float64, Int, UInt}, Any}())
            AMALGAMATIONS[id] = entry
        end

        entry[2]
    end

    k = (Int(nmax), Float64(alpha), CUDA.deviceid(CUDA.device()), isnothing(allowed) ? UInt(0) : hash((BitVector(allowed), compact)))
    A = lock(() -> get(cache, k, missing), AMALGAMATIONS_LOCK)

    if ismissing(A) || (!isnothing(A) && A.from != hash((Rptr, Sptr, Stgt, Dptr, Lptr, pnt, idx)))
        A = amalgamate_fronts(I, nmax, alpha, Rptr, Sptr, Stgt, Dptr, Lptr, pnt, idx; allowed, compact)
        lock(() -> (cache[k] = A), AMALGAMATIONS_LOCK)
    end

    return A::Union{Nothing, Amalgamation{I}}
end

# a mutable object that lives exactly as long as the symbolic factorization, or nothing (only a hint:
# empty arrays may share one Memory, so a cache hit is checked against the structure)
function amalgamation_key(x)
    ismutable(x) && return x
    hasfield(typeof(x), :mem) && ismutable(getfield(x, :mem)) && return getfield(x, :mem)
    return nothing
end

# |a ∩ b| for sorted vectors (separators are sorted), without allocating
function sorted_common(a::AbstractVector, b::AbstractVector)
    i = firstindex(a); j = firstindex(b); c = 0

    @inbounds while i <= lastindex(a) && j <= lastindex(b)
        x = a[i]; y = b[j]
        c += x == y
        i += x <= y
        j += y <= x
    end

    return c
end

# allowed: only fronts with allowed[f] are merged (all if nothing). compact: the merged arrays (and the
# maps) cover only the groups of allowed fronts; the others get empty blocks (for a merge that only the
# GPU top of the factorization uses)
function amalgamate_fronts(::Type{I}, nmax::Integer, alpha::Real, Rptr, Sptr, Stgt, Dptr, Lptr, pnt, idx; allowed = nothing, compact::Bool = false) where {I}
    nf = length(pnt)
    nn(f) = Int(Rptr[f + 1] - Rptr[f])
    na(f) = Int(Sptr[f + 1] - Sptr[f])
    sep(f) = view(Stgt, Sptr[f]:(Sptr[f + 1] - 1))
    #
    # decide the merges, in postorder: the chain ending at c = f - 1 joins f when c is the
    # last child of f, the merged width stays ≤ nmax, and the zero padding is small
    #
    width = [nn(f) for f in 1:nf]                       # residual width of the chain ending at f
    joins = falses(nf)                                   # joins[c]: c is merged into c + 1

    for f in 2:nf
        c = f - 1
        pnt[c] == f || continue
        (isnothing(allowed) || (allowed[c] && allowed[f])) || continue
        w = width[c] + nn(f)
        w <= nmax || continue
        common = sorted_common(sep(c), sep(f))
        zeros = width[c] * (na(f) - common)              # padded entries in the chain's separator columns
        zeros <= max(8, alpha * width[c] * max(na(c), 1)) || continue
        joins[c] = true
        width[f] = w
    end
    #
    # the merged fronts (groups h:g, in postorder of g)
    #
    groups = Tuple{Int, Int}[]
    h = 1

    for f in 1:nf
        if !joins[f]
            push!(groups, (h, f))
            h = f + 1
        end
    end

    ng = length(groups)
    group = zeros(Int, nf)

    for (q, (h, g)) in enumerate(groups), f in h:g
        group[f] = q
    end

    Rn = Vector{I}(undef, ng + 1); Sn = Vector{I}(undef, ng + 1); Dn = Vector{I}(undef, ng + 1); Ln = Vector{I}(undef, ng + 1)
    Tn = Vector{I}(undef, sum(((h, g),) -> na(g), groups; init = 0))     # (sized once: grown, it left ~15 MB of garbage per call)
    Rn[1] = Rptr[1]; Sn[1] = 1; Dn[1] = 1; Ln[1] = 1

    stored(h) = !compact || isnothing(allowed) || allowed[h]

    for (q, (h, g)) in enumerate(groups)
        w = Int(Rptr[g + 1] - Rptr[h]); a = na(g)
        Rn[q + 1] = Rptr[g + 1]
        copyto!(Tn, Sn[q], Stgt, Sptr[g], a)
        Sn[q + 1] = Sn[q] + a
        Dn[q + 1] = Dn[q] + (stored(h) ? w * w : 0)
        Ln[q + 1] = Ln[q] + (stored(h) ? w * a : 0)
    end

    max(Dn[end], Ln[end], Dptr[end], Lptr[end]) < typemax(Int32) || return nothing     # too large for Int32 maps: no merge
    #
    # the maps are built on the GPU (the merged factor is as large as the factor: tens of MB of maps per
    # call, which on the host would be garbage after one upload)
    #
    mLD = CUDA.zeros(Int32, Dn[end] - 1); mUD = CUDA.zeros(Int32, Dn[end] - 1)
    mLL = CUDA.zeros(Int32, Ln[end] - 1); mUL = CUDA.zeros(Int32, Ln[end] - 1)
    members = Int32[m for (h, g) in groups if stored(h) for m in h:g]
    amalgamate_maps_gpu!(mLD, mUD, mLL, mUL, members, group, groups, Rptr, Sptr, Stgt, Dptr, Lptr, Dn, Ln)

    pn = Vector{I}(undef, ng)

    for (q, (h, g)) in enumerate(groups)
        pn[q] = iszero(pnt[g]) ? zero(I) : I(group[pnt[g]])
    end

    idn = I[group[f] for f in idx]
    from = hash((Rptr, Sptr, Stgt, Dptr, Lptr, pnt, idx))
    return Amalgamation{I}(ng, Rn, Sn, Tn, Dn, Ln, pn, idn, mLD, mUD, mLL, mUL, from, group)
end

# the index maps of the merged blocks (see amalgamate_fronts), on the GPU: one block of threads per
# member front m of a stored group (h:g), which writes the entries of its own blocks into the merged
# blocks of its group (the members' entries are disjoint, the rest stays 0: the semiring zero)
function amalgamate_maps_gpu!(mLD, mUD, mLL, mUL, members, group, groups, Rptr, Sptr, Stgt, Dptr, Lptr, Dn, Ln)
    isempty(members) && return
    #
    # The work is flattened: one thread per map entry of all member fronts, which finds its front by
    # binary search in the prefix sums of the fronts' entries. Fronts range from one entry to millions,
    # so a fixed number of threads per front either idles (a block each) or serializes the largest ones
    # (a warp each: 5× slower). The inputs go up as Int32 in one upload (the maps are Int32, and
    # amalgamate_fronts checked that every offset fits): 64-bit inputs made every store a checked
    # conversion.
    #
    work = zeros(Int, length(members) + 1)

    for (w, m) in enumerate(members)
        nm = Rptr[m + 1] - Rptr[m]; am = Sptr[m + 1] - Sptr[m]
        work[w + 1] = work[w] + nm * nm + am * nm
    end

    total = work[end]
    (iszero(total) || total >= typemax(Int32)) && return
    parts = (members, group, first.(groups), last.(groups), Rptr, Sptr, Stgt, Dptr, Lptr, Dn, Ln, work)
    offs = cumsum([0; [length(p) for p in parts]])
    buf = Vector{Int32}(undef, offs[end])

    for (k, p) in enumerate(parts), (i, x) in enumerate(p)
        buf[offs[k] + i] = x % Int32
    end

    d = CuVector(buf)
    o = ntuple(k -> Int32(offs[k]), 12)
    nb = min(cld(total, 256), 2^16)
    @cuda threads = 256 blocks = nb amalgamate_maps_kernel!(mLD, mUD, mLL, mUL, d, o, Int32(length(members)), Int32(total))
    CUDA.unsafe_free!(d)
    return
end

# (e + 1)-th entry of a column-major matrix with d rows: its (row, column), 1-based, in 32 bits
@inline function cm_index32(e::Int32, d::Int32)
    a = e % UInt32; b = d % UInt32
    return Core.Intrinsics.urem_int(a, b) % Int32 + Int32(1), Core.Intrinsics.udiv_int(a, b) % Int32 + Int32(1)
end

function amalgamate_maps_kernel!(mLD, mUD, mLL, mUL, buf, o, nmem, total)
    one32 = Int32(1)
    E = (blockIdx().x - one32) * blockDim().x + threadIdx().x - one32

    @inbounds while E < total
        members(i) = buf[o[1] + i]; group(i) = buf[o[2] + i]; gh(i) = buf[o[3] + i]; gg(i) = buf[o[4] + i]
        Rptr(i) = buf[o[5] + i]; Sptr(i) = buf[o[6] + i]; Stgt(i) = buf[o[7] + i]
        Dptr(i) = buf[o[8] + i]; Lptr(i) = buf[o[9] + i]; Dn(i) = buf[o[10] + i]; Ln(i) = buf[o[11] + i]; work(i) = buf[o[12] + i]
        # the member w with work(w) ≤ E < work(w + 1)
        lo = one32; hi = nmem

        while lo < hi
            mid = (lo + hi + one32) >> one32
            work(mid) <= E ? (lo = mid) : (hi = mid - one32)
        end

        w = lo; e = E - work(w)
        m = members(w)
        q = group(m); h = gh(q); g = gg(q)
        r0 = Rptr(h); r1 = Rptr(g + one32) - one32; wd = r1 - r0 + one32
        s0 = Sptr(g); a = Sptr(g + one32) - s0                 # sep(g) = Stgt[s0:s0 + a - 1]
        off = Rptr(m) - r0                                     # offset of m in the merged residual
        nm = Rptr(m + one32) - Rptr(m); am = Sptr(m + one32) - Sptr(m); sm0 = Sptr(m)
        Dm = Dptr(m); Lm = Lptr(m); d0 = Dn(q) - one32; l0 = Ln(q) - one32

        if e < nm * nm
            #   D block of m (nm × nm, column-major) at (off + i, off + j) of the merged wd × wd block
            i, j = cm_index32(e, nm)
            v = Dm + (j - one32) * nm + i - one32
            pos = d0 + (off + i) + (off + j - one32) * wd
            mLD[pos] = v; mUD[pos] = v
        else
            #   separator row r of m (vertex u), entry c: L₂₁ is am × nm, U₁₂ is nm × am
            r, c = cm_index32(e - nm * nm, am)
            u = Stgt(sm0 + r - one32)

            if r0 <= u <= r1                                   # a later member's residual vertex: inside the merged block
                li = u - r0 + one32
                mLD[d0 + li + (off + c - one32) * wd] = -(Lm + (c - one32) * am + r - one32)
                mUD[d0 + (off + c) + (li - one32) * wd] = -(Lm + (r - one32) * nm + c - one32)
            else                                               # a vertex of sep(g): its position k, by binary search
                lo2 = Int32(0); hi2 = a
                while lo2 < hi2
                    mid = (lo2 + hi2) >> one32
                    Stgt(s0 + mid) < u ? (lo2 = mid + one32) : (hi2 = mid)
                end
                k = lo2 + one32
                mLL[l0 + k + (off + c - one32) * a] = Lm + (c - one32) * am + r - one32
                mUL[l0 + (off + c) + (k - one32) * wd] = Lm + (r - one32) * nm + c - one32
            end
        end

        E += blockDim().x * gridDim().x
    end

    return
end

function amalgamate_maps_gpu_ref!(mLD, mUD, mLL, mUL, members, group, groups, Rptr, Sptr, Stgt, Dptr, Lptr, Dn, Ln)
    isempty(members) && return
    gh = Int32[h for (h, _) in groups]; gg = Int32[g for (_, g) in groups]
    dev(x) = CuVector(x)                                   # (as they are: no host copies to Int32)
    @cuda threads = 256 blocks = length(members) amalgamate_maps_kernel_ref!(mLD, mUD, mLL, mUL, dev(members), dev(group),
        dev(gh), dev(gg), dev(Rptr), dev(Sptr), dev(Stgt), dev(Dptr), dev(Lptr), dev(Dn), dev(Ln))
    return
end

function amalgamate_maps_kernel_ref!(mLD, mUD, mLL, mUL, members, group, gh, gg, Rptr, Sptr, Stgt, Dptr, Lptr, Dn, Ln)
    m = members[blockIdx().x]
    t = threadIdx().x - Int32(1); nt = blockDim().x

    @inbounds begin
        q = group[m]; h = gh[q]; g = gg[q]
        r0 = Rptr[h]; r1 = Rptr[g + 1] - Int32(1); w = r1 - r0 + Int32(1)
        s0 = Sptr[g]; a = Sptr[g + 1] - s0                 # sep(g) = Stgt[s0:s0 + a - 1]
        o = Rptr[m] - r0                                   # offset of m in the merged residual
        nm = Rptr[m + 1] - Rptr[m]; am = Sptr[m + 1] - Sptr[m]; sm0 = Sptr[m]
        Dm = Dptr[m]; Lm = Lptr[m]; d0 = Dn[q] - Int32(1); l0 = Ln[q] - Int32(1)
        #
        #   D block of m (nm × nm, column-major) at (o + i, o + j) of the merged w × w block
        #
        e = t
        while e < nm * nm
            i, j = cm_index(e, nm)                         # 1-based
            v = Dm + (j - Int32(1)) * nm + i - Int32(1)
            pos = d0 + (o + i) + (o + j - Int32(1)) * w
            mLD[pos] = v; mUD[pos] = v
            e += nt
        end
        #
        #   separator row r of m (vertex u), entry c: L₂₁ is am × nm, U₁₂ is nm × am
        #
        e = t
        while e < am * nm
            r, c = cm_index(e, am)                         # 1-based: row r of sep(m), column (or row of U) c
            u = Stgt[sm0 + r - Int32(1)]

            if r0 <= u <= r1                               # a later member's residual vertex: inside the merged block
                li = u - r0 + Int32(1)
                mLD[d0 + li + (o + c - Int32(1)) * w] = -(Lm + (c - Int32(1)) * am + r - Int32(1))
                mUD[d0 + (o + c) + (li - Int32(1)) * w] = -(Lm + (r - Int32(1)) * nm + c - Int32(1))
            else                                           # a vertex of sep(g): its position k, by binary search
                lo = Int32(0); hi = a
                while lo < hi
                    mid = (lo + hi) >> 1
                    Stgt[s0 + mid] < u ? (lo = mid + Int32(1)) : (hi = mid)
                end
                k = lo + Int32(1)
                mLL[l0 + k + (o + c - Int32(1)) * a] = Lm + (c - Int32(1)) * am + r - Int32(1)
                mUL[l0 + (o + c) + (k - Int32(1)) * w] = Lm + (r - Int32(1)) * nm + c - Int32(1)
            end

            e += nt
        end
    end

    return
end

function amalgamate_gather_kernel!(out, D, L, from, z)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if i <= length(out)
        m = from[i]
        out[i] = ispositive(m) ? D[m] : (iszero(m) ? z : L[-m])
    end

    return
end

function amalgamate_gather(D::CuVector{T}, L::CuVector{T}, from::CuVector{Int32}, z::T) where {T}
    out = CuVector{T}(undef, length(from))
    isempty(out) || @cuda threads = 256 blocks = cld(length(out), 256) amalgamate_gather_kernel!(out, D, L, from, z)
    return out
end

# the merged factor (LD, LL, UD, UL) on the GPU from the original one
function amalgamate_values(A::Amalgamation, s::AbstractSemiring, LD::CuVector{T}, LL::CuVector{T}, UD::CuVector{T}, UL::CuVector{T}) where {T}
    z = szero(s, T, Val(:N))
    return (amalgamate_gather(LD, LL, A.mLD, z), amalgamate_gather(LL, LL, A.mLL, z),
            amalgamate_gather(UD, UL, A.mUD, z), amalgamate_gather(UL, UL, A.mUL, z))
end

# out ← gather (as amalgamate_gather, into a preallocated array: usable under graph capture)
function amalgamate_gather!(out::CuVector{T}, D::CuVector{T}, L::CuVector{T}, from::CuVector{Int32}, z::T) where {T}
    isempty(out) || @cuda threads = 256 blocks = cld(length(out), 256) amalgamate_gather_kernel!(out, D, L, from, z)
    return out
end

function amalgamate_scatter_kernel!(D, L, merged, from)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if i <= length(merged)
        m = from[i]
        if ispositive(m)
            D[m] = merged[i]
        elseif !iszero(m)
            L[-m] = merged[i]
        end
    end

    return
end

# the inverse of amalgamate_gather!: every original entry appears once in `from` (padding entries are 0)
function amalgamate_scatter!(D::CuVector{T}, L::CuVector{T}, merged::CuVector{T}, from::CuVector{Int32}) where {T}
    isempty(merged) || @cuda threads = 256 blocks = cld(length(merged), 256) amalgamate_scatter_kernel!(D, L, merged, from)
    return D
end
