# CPU nested dissection with METIS vertex separators (METIS_ComputeVertexSeparator on induced subgraphs)
# down to leaves of at most Lb vertices; the leaves are ordered by `leafsolve` (e.g. one grounds_batch call),
# separators last (children first). Baseline "CPU METIS-ND + GPU batched leaves" for ND_NOTES.md.
using Metis

function metis_sep!(part::Vector{Metis.idx_t}, ptr::Vector{Metis.idx_t}, adj::Vector{Metis.idx_t}, n::Int, seed::Int)
    opts = fill(Metis.idx_t(-1), Metis.METIS_NOPTIONS)
    opts[Metis.METIS_OPTION_NUMBERING + 1] = 0
    opts[Metis.METIS_OPTION_SEED + 1] = seed
    vw = ones(Metis.idx_t, n)
    sepsize = Ref{Metis.idx_t}(0)
    Metis.@check Metis.METIS_ComputeVertexSeparator(Ref{Metis.idx_t}(n), ptr, adj, vw, opts, sepsize, part)
    return part
end

# induced subgraph of vertex set vs (global ids) as 0-based METIS CSR, using loc (global -> local, 0 = absent)
function induced0(ptr, adj, vs, loc)
    k = length(vs)
    for (i, v) in enumerate(vs); loc[v] = i; end
    xp = Vector{Metis.idx_t}(undef, k + 1); xp[1] = 0
    cnt = 0
    for (i, v) in enumerate(vs)
        for q in ptr[v]:(ptr[v + 1] - 1)
            loc[adj[q]] != 0 && (cnt += 1)
        end
        xp[i + 1] = cnt
    end
    xa = Vector{Metis.idx_t}(undef, cnt); c = 0
    for v in vs, q in ptr[v]:(ptr[v + 1] - 1)
        l = loc[adj[q]]
        l != 0 && (c += 1; xa[c] = l - 1)
    end
    for v in vs; loc[v] = 0; end
    return xp, xa
end

# local 1-based CSR (Int64 ptr, Int32 adj) of a leaf
function induced1(ptr, adj, vs, loc)
    xp, xa = induced0(ptr, adj, vs, loc)
    return Int64.(xp) .+ 1, Int32.(xa) .+ Int32(1)
end

function metis_nd_perm(ptr, adj; Lb::Int, leafsolve, timing = nothing, seed = 1)
    T0 = time()
    n = length(ptr) - 1
    perm = zeros(Int32, n)
    loc = zeros(Int32, n)
    leaves = Vector{Vector{Int32}}(); lpos = Int[]
    stack = Tuple{Vector{Int32}, Int}[(Int32.(1:n), 1)]
    nsep = 0
    while !isempty(stack)
        vs, start = pop!(stack)
        k = length(vs)
        if k <= Lb
            push!(leaves, vs); push!(lpos, start); continue
        end
        xp, xa = induced0(ptr, adj, vs, loc)
        part = Vector{Metis.idx_t}(undef, k)
        metis_sep!(part, xp, xa, k, seed)
        A = vs[part .== 0]; B = vs[part .== 1]; S = vs[part .== 2]
        if isempty(A) || isempty(B)
            push!(leaves, vs); push!(lpos, start); continue
        end
        s0 = start + length(A) + length(B)
        for (i, v) in enumerate(S); perm[s0 + i - 1] = v; end
        nsep += length(S)
        push!(stack, (A, start)); push!(stack, (B, start + length(A)))
    end
    T1 = time()
    csrs = [induced1(ptr, adj, vs, loc) for vs in leaves]
    T2 = time()
    orders = leafsolve(csrs)
    T3 = time()
    for (l, vs) in enumerate(leaves)
        o = orders[l]; s = lpos[l]
        for i in eachindex(o); perm[s + i - 1] = vs[o[i]]; end
    end
    T4 = time()
    if timing !== nothing
        timing[:nd] = T1 - T0; timing[:leafcut] = T2 - T1; timing[:leaves] = T3 - T2; timing[:assemble] = T4 - T3; timing[:total] = T4 - T0
        timing[:nleaves] = length(leaves); timing[:nsep] = nsep
    end
    return perm
end
