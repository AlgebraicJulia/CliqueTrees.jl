# The blocks (biconnected components) of a graph, for `ConnectedComponents{2}`
# and `ConnectedComponents{3}` (see components.jl).
#
# `blocktree` returns a clique tree in the format of `atomtree!`: `order` maps
# positions to vertices, the residual of each bag is a range of positions, and
# the separator of each bag lists positions in the residuals of its ancestors.

# The blocks of the graph G - skip (skip = 0: G itself), by the depth-first
# search of Hopcroft and Tarjan, without recursion. The search emits a block
# when it finishes the child w of a vertex u with low(w) >= num(u); u is the
# *head* of the block. Every vertex except the root of the search lies in
# exactly one block below its head: the block whose *residual* contains it.
# Blocks are emitted in postorder: the block whose residual contains the head
# of a block is emitted after it. The last block of each component contains
# the root in its residual, and the other blocks at the root become its
# children. An isolated vertex is a block.
#
# Returns the number of blocks q. For each vertex v, blk[v] is the block whose
# residual contains v; for each block b, hed[b] is its head (0 for the root
# block of a component) and cnt[b] is the size of its residual. The
# complexity is O(m), where m = |E|.
#
#   Hopcroft, John, and Robert Tarjan. "Algorithm 447: efficient algorithms
#   for graph manipulation." Communications of the ACM 16.6 (1973): 372-378.
function blocks!(
        num::AbstractVector{V},
        low::AbstractVector{V},
        nxt::AbstractVector{V},
        dfs::AbstractVector{V},
        stk::AbstractVector{V},
        blk::AbstractVector{V},
        hed::AbstractVector{V},
        cnt::AbstractVector{V},
        graph::AbstractGraph{V},
        skip::V,
    ) where {V}
    for v in vertices(graph)
        num[v] = zero(V)
    end

    t = zero(V); q = zero(V); sp = zero(V)

    for r in vertices(graph)
        (r == skip || ispositive(num[r])) && continue
        qq = q

        num[r] = low[r] = t += one(V); nxt[r] = one(V)
        dp = one(V); dfs[dp] = r
        sp += one(V); stk[sp] = r

        while ispositive(dp)
            v = dfs[dp]; nbr = neighbors(graph, v); k = nxt[v]

            if k <= length(nbr)
                nxt[v] = k + one(V); w = nbr[k]

                if w != skip
                    if iszero(num[w])
                        num[w] = low[w] = t += one(V); nxt[w] = one(V)
                        dp += one(V); dfs[dp] = w
                        sp += one(V); stk[sp] = w
                    else
                        low[v] = min(low[v], num[w])
                    end
                end
            else
                dp -= one(V)

                if ispositive(dp)
                    u = dfs[dp]
                    low[u] = min(low[u], low[v])

                    if low[v] >= num[u]
                        q += one(V); hed[q] = u; c = zero(V)

                        while true
                            y = stk[sp]; sp -= one(V)
                            blk[y] = q; c += one(V)
                            y == v && break
                        end

                        cnt[q] = c
                    end
                end
            end
        end

        # the root of the search
        sp -= one(V)

        if q == qq
            q += one(V); cnt[q] = zero(V)
        end

        blk[r] = q; hed[q] = zero(V); cnt[q] += one(V)
    end

    return q
end

# A clique tree from a partition of the vertices 1:n into the residuals of
# the nodes 1:q of a postordered forest. node[v] is the node whose residual
# contains v, prnt[i] is the parent of node i (0: a root), and sep1[i] and
# sep2[i] are the vertices of its separator (0: none).
function partitiontree(
        node::AbstractVector{V},
        prnt::AbstractVector{V},
        sep1::AbstractVector{V},
        sep2::AbstractVector{V},
        n::V,
        q::V,
        ::Type{E},
    ) where {V, E}
    resptr = FVector{V}(undef, q + one(V))
    index = FVector{V}(undef, n)
    order = Vector{V}(undef, n)

    for i in oneto(q)
        resptr[i] = zero(V)
    end

    for v in oneto(n)
        resptr[node[v]] += one(V)
    end

    p = one(V); m = zero(E)

    for i in oneto(q)
        c = resptr[i]; resptr[i] = p; p += c
        m += E(ispositive(sep1[i])) + E(ispositive(sep2[i]))
    end

    resptr[q + one(V)] = p

    for v in oneto(n)
        i = node[v]; j = resptr[i]; resptr[i] = j + one(V)
        order[j] = v; index[v] = j
    end

    for i in Iterators.reverse(oneto(q))
        resptr[i + one(V)] = resptr[i]
    end

    resptr[one(V)] = one(V)

    sepptr = FVector{E}(undef, q + one(V))
    septgt = FVector{V}(undef, m)
    s = one(E)

    for i in oneto(q)
        sepptr[i] = s
        a = ispositive(sep1[i]) ? index[sep1[i]] : zero(V)
        b = ispositive(sep2[i]) ? index[sep2[i]] : zero(V)

        if a > b
            a, b = b, a
        end

        if ispositive(a)
            septgt[s] = a; s += one(E)
        end

        if ispositive(b)
            septgt[s] = b; s += one(E)
        end
    end

    sepptr[q + one(V)] = s

    qree = Parent(q, prnt)
    qres = BipartiteGraph(n, q, n, resptr, oneto(n))
    qsep = BipartiteGraph(n, q, m, sepptr, septgt)
    return order, CliqueTree(SupernodeTree(Tree(qree), qres), qsep)
end

# The block tree of a graph: a clique tree whose bags are the blocks
# of the graph. The separator of a bag is the cut vertex it shares with its
# parent, or nothing at the root of a connected component.
function blocktree(graph::AbstractGraph{V}) where {V}
    E = etype(graph); n = nv(graph)

    num = FVector{V}(undef, n)
    low = FVector{V}(undef, n)
    nxt = FVector{V}(undef, n)
    dfs = FVector{V}(undef, n)
    stk = FVector{V}(undef, n)
    blk = FVector{V}(undef, n)
    hed = FVector{V}(undef, n)
    cnt = FVector{V}(undef, n)

    q = blocks!(num, low, nxt, dfs, stk, blk, hed, cnt, graph, zero(V))

    # the parent of a block is the block whose residual contains its head
    for b in oneto(q)
        h = hed[b]
        low[b] = iszero(h) ? zero(V) : blk[h]
        nxt[b] = zero(V)
    end

    return partitiontree(blk, low, hed, nxt, n, q, E)
end
