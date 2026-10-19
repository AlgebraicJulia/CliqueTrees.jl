#
# The minor-treewidth lower bound (Tamaki's NaiveContractionLB, from his
# RTW), with vertex weights: the widths of minors of the graph, found by
# exact searches.
#
# A contraction of a graph is no wider than the graph, if a contracted
# vertex weighs as little as the lightest vertex of its part (see
# rtw.jl). The edges are contracted one at a time, best first by
# Tamaki's edge values (rtw_vals), until the minor is dense: it has at
# most twice as many vertices as its least degree. Its width, by septw,
# is a lower bound. Then the contractions are undone, last first, and
# while a minor has no decomposition of width at most the bound (a
# round of septw: sepenum, then sepdp), the bound is raised. The
# search stops at the graph, at a minor with more than budget minimal
# separators lighter than the bound, or at the deadline dl, in the
# sense of time(). The minors near the bottom are small, and their
# exact searches are fast; higher up, they have more separators.
#
# The bound starts from the bound lo and from the weight of the
# heaviest vertex or edge, which every bag meets: the lightest vertices
# of the parts can be lighter than the graph needs.
#
# The minors are not kept. The parts of the current minor are kept as
# the label of each vertex of the graph and the vertices with each
# label; a contraction merges the smaller part into the larger, and is
# logged, so that it can be undone. Going down, the minor is contracted
# (rtw_contract); going up, each minor is made from the parts
# (mtw_quo).
#
#   - Tamaki, Hisao. "A contraction-recursive algorithm for treewidth."
#     *18th International Symposium on Parameterized and Exact
#     Computation (IPEC 2023)*. 2023.
#
function mtw(weights::AbstractVector, graph::AbstractGraph, lo, budget::Integer, dl::Real)
    g, wgt = hbt_graph(weights, graph); W = eltype(wgt); n = nv(g)
    lb = max(convert(W, lo), bt_bas(wgt, g))
    n <= 1 && return lb
    wmin = minimum(wgt)

    # the parts: the label of each vertex of the graph, the vertices with
    # each label, the label of each vertex of the minor, and the merges
    # (label kept, label merged, size of the part kept before)
    lab = collect(oneto(n)); mem = [[v] for v in oneto(n)]; mlab = collect(oneto(n))
    merges = Tuple{Int, Int, Int}[]

    # down
    h = g; wh = wgt

    while nv(h) > 2mtw_mindeg(h)
        time() > dl && return lb
        vals, edges = rtw_vals(h)
        isempty(edges) && break
        e = 1

        @inbounds for i in 2:length(vals)
            rtw_vlt(vals[i], vals[e]) && (e = i)
        end

        a, b = edges[e]; x = mlab[a]; y = mlab[b]
        length(mem[x]) < length(mem[y]) && ((x, y) = (y, x))
        push!(merges, (x, y, length(mem[x])))

        for v in mem[y]
            lab[v] = x
        end

        append!(mem[x], mem[y]); empty!(mem[y])
        h, wh, pmap = rtw_contract(h, wh, a, b)
        nlab = Vector{Int}(undef, nv(h))

        @inbounds for (u, z) in enumerate(pmap)
            nlab[z] = mlab[u]
        end

        nlab[pmap[a]] = x; mlab = nlab
    end

    # the bottom
    _, _, k, status = septw(wh, h, lb, typemax(W), dl, 1)
    status == BT_TIMEOUT && return lb
    lb = max(lb, k)

    # up
    for i in length(merges):-1:1
        x, y, len = merges[i]
        tail = view(mem[x], len + 1:length(mem[x]))

        for v in tail
            lab[v] = y
        end

        append!(mem[y], tail); resize!(mem[x], len)
        h, wh = mtw_quo(g, wgt, lab, mem)

        while true
            sep, status = sepenum(wh, h, lb, dl, 1)
            (status == BT_TIMEOUT || nv(sep) > budget) && return lb
            _, _, mink, status = sepdp(wh, h, sep, lb, dl, 1)
            status == BT_TIMEOUT && return lb
            status == BT_FOUND && break
            lb = min(mink, lb + wmin)
        end
    end

    return lb
end

function mtw_mindeg(graph::BipartiteGraph{Int, Int})
    d = typemax(Int)

    @inbounds for v in vertices(graph)
        d = min(d, length(neighbors(graph, v)))
    end

    return d
end

# The minor of the graph whose parts are the nonempty sets of mem (with
# the labels lab), each vertex as light as the lightest of its part.
function mtw_quo(graph::BipartiteGraph{Int, Int}, wgt::Vector{W}, lab::Vector{Int}, mem::Vector{Vector{Int}}) where {W}
    n = nv(graph); idx = zeros(Int, n); m = 0

    @inbounds for x in oneto(n)
        isempty(mem[x]) || (idx[x] = m += 1)
    end

    ptr = Vector{Int}(undef, m + 1); tgt = Int[]; mwgt = Vector{W}(undef, m); mark = zeros(Int, m)
    ptr[1] = 1

    @inbounds for x in oneto(n)
        p = idx[x]; iszero(p) && continue
        w = wgt[first(mem[x])]; lo = length(tgt) + 1; mark[p] = p

        for v in mem[x]
            w = min(w, wgt[v])

            for u in neighbors(graph, v)
                q = idx[lab[u]]

                if mark[q] != p
                    mark[q] = p; push!(tgt, q)
                end
            end
        end

        sort!(view(tgt, lo:length(tgt))); ptr[p + 1] = length(tgt) + 1; mwgt[p] = w
    end

    return BipartiteGraph{Int, Int}(m, m, length(tgt), ptr, tgt), mwgt
end
