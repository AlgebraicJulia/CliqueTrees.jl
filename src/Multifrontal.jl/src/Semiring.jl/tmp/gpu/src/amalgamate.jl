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
    # the structure it was computed from: a cache hit must match it
    from::NTuple{7, Vector{I}}
end

# per symbolic factorization and (nmax, alpha). Keyed on the identity of an object that lives as long
# as the symbolic factorization (held weakly). Not a WeakKeyDict: that compares keys with isequal, so
# two factorizations whose separator targets are equal arrays (e.g. both empty) would share a merge.
const AMALGAMATIONS = Dict{UInt, Tuple{WeakRef, Dict{Tuple{Int, Float64}, Any}}}()
const AMALGAMATIONS_LOCK = ReentrantLock()

function amalgamation(key, ::Type{I}, nmax::Integer, alpha::Real, Rptr, Sptr, Stgt, Dptr, Lptr, pnt, idx) where {I}
    isnothing(key) && return amalgamate_fronts(I, nmax, alpha, Rptr, Sptr, Stgt, Dptr, Lptr, pnt, idx)

    cache = lock(AMALGAMATIONS_LOCK) do
        filter!(kv -> !isnothing(kv[2][1].value), AMALGAMATIONS)           # forget collected factorizations
        id = objectid(key)
        entry = get(AMALGAMATIONS, id, nothing)

        if isnothing(entry) || entry[1].value !== key
            entry = (WeakRef(key), Dict{Tuple{Int, Float64}, Any}())
            AMALGAMATIONS[id] = entry
        end

        entry[2]
    end

    k = (Int(nmax), Float64(alpha))
    A = get(cache, k, missing)

    if ismissing(A) || (!isnothing(A) && A.from != (Rptr, Sptr, Stgt, Dptr, Lptr, pnt, idx))
        A = cache[k] = amalgamate_fronts(I, nmax, alpha, Rptr, Sptr, Stgt, Dptr, Lptr, pnt, idx)
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

function amalgamate_fronts(::Type{I}, nmax::Integer, alpha::Real, Rptr, Sptr, Stgt, Dptr, Lptr, pnt, idx) where {I}
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
        w = width[c] + nn(f)
        w <= nmax || continue
        common = length(intersect(sep(c), sep(f)))
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
    Tn = I[]
    Rn[1] = Rptr[1]; Sn[1] = 1; Dn[1] = 1; Ln[1] = 1

    for (q, (h, g)) in enumerate(groups)
        w = Int(Rptr[g + 1] - Rptr[h]); a = na(g)
        Rn[q + 1] = Rptr[g + 1]
        append!(Tn, sep(g))
        Sn[q + 1] = Sn[q] + a
        Dn[q + 1] = Dn[q] + w * w
        Ln[q + 1] = Ln[q] + w * a
    end

    max(Dn[end], Ln[end], Dptr[end], Lptr[end]) < typemax(Int32) || return nothing     # too large for Int32 maps: no merge
    mLD = zeros(Int32, Dn[end] - 1); mUD = zeros(Int32, Dn[end] - 1)
    mLL = zeros(Int32, Ln[end] - 1); mUL = zeros(Int32, Ln[end] - 1)

    for (q, (h, g)) in enumerate(groups)
        r0 = Int(Rptr[h]); r1 = Int(Rptr[g + 1]) - 1; w = r1 - r0 + 1
        sg = sep(g); a = length(sg)
        LDq = reshape(view(mLD, Dn[q]:(Dn[q + 1] - 1)), w, w); UDq = reshape(view(mUD, Dn[q]:(Dn[q + 1] - 1)), w, w)
        LLq = reshape(view(mLL, Ln[q]:(Ln[q + 1] - 1)), a, w); ULq = reshape(view(mUL, Ln[q]:(Ln[q + 1] - 1)), w, a)

        for m in h:g
            o = Int(Rptr[m]) - r0                        # offset of member m in the merged residual
            nm = nn(m); am = na(m); sm = sep(m)
            Dm = Int(Dptr[m]); Lm = Int(Lptr[m])

            for j in 1:nm, i in 1:nm                     # D block (nm × nm, column-major)
                LDq[o + i, o + j] = UDq[o + i, o + j] = Dm + (j - 1) * nm + i - 1
            end

            for (r, u) in enumerate(sm)                  # L₂₁ is am × nm, U₁₂ is nm × am
                if r0 <= u <= r1                         # a later member's residual vertex: inside the merged block
                    li = Int(u) - r0 + 1
                    for j in 1:nm; LDq[li, o + j] = -(Lm + (j - 1) * am + r - 1); end
                    for i in 1:nm; UDq[o + i, li] = -(Lm + (r - 1) * nm + i - 1); end
                else                                     # a vertex of sep(g)
                    k = searchsortedfirst(sg, u)
                    @assert k <= a && sg[k] == u "amalgamation: separator vertex outside the parent's bag"
                    for j in 1:nm; LLq[k, o + j] = Lm + (j - 1) * am + r - 1; end
                    for i in 1:nm; ULq[o + i, k] = Lm + (r - 1) * nm + i - 1; end
                end
            end
        end
    end

    pn = Vector{I}(undef, ng)

    for (q, (h, g)) in enumerate(groups)
        pn[q] = iszero(pnt[g]) ? zero(I) : I(group[pnt[g]])
    end

    idn = I[group[f] for f in idx]
    from = (copy(Rptr), copy(Sptr), copy(Stgt), copy(Dptr), copy(Lptr), copy(pnt), copy(idx))
    return Amalgamation{I}(ng, Rn, Sn, Tn, Dn, Ln, pn, idn, upload(mLD), upload(mUD), upload(mLL), upload(mUL), from)
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
