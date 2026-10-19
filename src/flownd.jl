function permutation(weights::AbstractVector, graph::AbstractGraph, alg::ND{<:Any, <:EliminationAlgorithm, FlowND}; verbose::Bool = false)
    order = dissect(Int, weights, graph, alg)
    return order, invperm(order)
end

function separator!(sepsize::AbstractScalar{V}, part::AbstractVector{V}, weights::AbstractVector{V}, graph::BipartiteGraph{V, V}, imbalance::V, alg::FlowND) where {V}
    sepsize[] = flownd!(part, weights, graph, nv(graph), alg.tries, alg.balance, alg.seed)
    return
end

# Compute a vertex separator of the graph `graph` with vertex weights
# `weights`: see `fnd_impl!`. The vertices greater than `nreal` are hubs:
# they have infinite capacity and zero weight, so they are never cut, and
# the members of a hub are pairwise adjacent. On return, `part[v]` is 0 or
# 1 if `v` lies in one of the two parts and 2 if it lies in the separator.
# Returns the weight of the separator.
function flownd!(part::AbstractVector{V}, weights::AbstractVector{V}, graph::BipartiteGraph{V, V}, nreal::V, tries::Integer, balance::Real, seed::Integer) where {V}
    @assert nv(graph) <= length(part)
    @assert nv(graph) <= length(weights)
    @assert ispositive(tries)
    @assert 0 < balance < 0.5
    n = nv(graph); m = ne(graph); nn = n + one(V)

    target = FVector{V}(undef, m)
    invptr = FVector{V}(undef, m)
    begptr = FVector{V}(undef, nn)
    tmpptr = FVector{V}(undef, n)

    through = FVector{V}(undef, n)
    flow = FVector{V}(undef, m)
    status = FVector{UInt8}(undef, n)

    srcmark = FVector{Int}(undef, twice(n))
    snkmark = FVector{Int}(undef, twice(n))
    srcfrom = FVector{Int}(undef, twice(n))
    srcstep = FVector{V}(undef, twice(n))
    snkfrom = FVector{Int}(undef, twice(n))
    snkstep = FVector{V}(undef, twice(n))
    srcqueue = FVector{Int}(undef, twice(n))
    snkqueue = FVector{Int}(undef, twice(n))

    srccut = FVector{V}(undef, n)
    snkcut = FVector{V}(undef, n)
    srcdist = FVector{V}(undef, n)
    snkdist = FVector{V}(undef, n)
    label = FVector{V}(undef, n)
    cmpwgt = FVector{V}(undef, n)
    fallback = FVector{V}(undef, n)

    fill!(srcmark, 0)
    fill!(snkmark, 0)

    fnd_copy!(target, invptr, begptr, tmpptr, graph)

    return fnd_impl!(part, fallback, target, invptr, begptr, weights,
        through, flow, status, srcmark, snkmark, srcfrom, srcstep, snkfrom,
        snkstep, srcqueue, snkqueue, srccut, snkcut, srcdist, snkdist,
        label, cmpwgt, Xoshiro(seed), nreal, n, convert(Int, tries),
        convert(Float64, balance))
end

# the status of a vertex during a flow (see `fnd_cut!`)
const FND_NEARSOURCE = 0x01  # an arc of infinite capacity joins a source to the in-node of the vertex
const FND_NEARSINK = 0x02    # an arc of infinite capacity joins the out-node of the vertex to a sink
const FND_SRCLISTED = 0x04   # the vertex is in the list `srccut`
const FND_SNKLISTED = 0x08   # the vertex is in the list `snkcut`

# Copy the graph `graph` into the arrays `target` and `begptr`, and
# compute the reverse of every arc.
function fnd_copy!(
        target::AbstractVector{V},
        invptr::AbstractVector{V},
        begptr::AbstractVector{V},
        tmpptr::AbstractVector{V},
        graph::BipartiteGraph{V, V},
    ) where {V}
    n = nv(graph); nn = n + one(V)

    # `sorted` is true if every neighborhood is sorted
    sorted = true

    @inbounds for v in oneto(nn)
        begptr[v] = pointers(graph)[v]
    end

    @inbounds for v in oneto(n)
        # `prv` is the previous neighbor of `v`
        prv = zero(V)

        for p in begptr[v]:begptr[v + one(V)] - one(V)
            w = targets(graph)[p]; target[p] = w
            sorted = sorted && prv < w; prv = w
        end
    end

    # if the neighborhoods are not sorted, sort them by
    # transposing the graph
    @inbounds if !sorted
        for v in oneto(n)
            tmpptr[v] = begptr[v]
        end

        for v in oneto(n)
            for p in begptr[v]:begptr[v + one(V)] - one(V)
                # `p` is the arc (`v`, `w`)
                w = target[p]

                # `q` is the arc (`w`, `v`)
                q = tmpptr[w]; invptr[q] = v; tmpptr[w] = q + one(V)
            end
        end

        copyto!(target, one(V), invptr, one(V), begptr[nn] - one(V))
    end

    # compute the reverse of every arc: since the neighborhoods
    # are sorted, the reverse of the arc (`v`, `w`) is the first
    # unvisited arc incident to `w`
    @inbounds for v in oneto(n)
        tmpptr[v] = begptr[v]
    end

    @inbounds for v in oneto(n)
        for p in begptr[v]:begptr[v + one(V)] - one(V)
            # `p` is the arc (`v`, `w`)
            w = target[p]

            # `q` is the arc (`w`, `v`)
            q = tmpptr[w]; invptr[p] = q; tmpptr[w] = q + one(V)
        end
    end

    return
end

"""
    fnd_impl!(part, fallback, target, invptr, begptr, weights, through,
        flow, status, srcmark, snkmark, srcfrom, srcstep, snkfrom,
        snkstep, srcqueue, snkqueue, srccut, snkcut, srcdist, snkdist,
        label, cmpwgt, rng, n, tries, balance)

Graph Bisection with Pareto Optimization
Hamann and Strasser
FlowCutter

Computing Tree Decompositions with FlowCutter: PACE 2017 Submission
Strasser

Compute a vertex separator S of a graph with vertex weights, splitting
the remaining vertices into parts A and B. The separator minimizes the
expansion

    w(S) / min(w(A), w(B))

over a family of cuts, subject to

    min(w(A), w(B)) ≥ `balance` w(V).

If no cut in the family is balanced, then the cut of least expansion
is returned.

If the graph is disconnected, then a union of components is a
separator of weight zero; if it is balanced, it is returned. Otherwise,
the family consists of the cuts found by `tries` runs of
[`fnd_cut!`](@ref) on the heaviest component. Each run draws a source
`s` and a sink `t` from the component at random, with `t` not adjacent
to `s`, as FlowCutter does. A source and a sink farthest apart would be
corners of a grid, whose cuts are diagonals: these are small, but the
parts they leave are triangles, which are dissected poorly.

The state of the algorithm is a collection of arrays.

  - graph (see [`fnd_copy!`](@ref)):
    - `target`, `begptr`: adjacency lists, sorted
    - `invptr`: the reverse of each arc
    - `weights`: vertex weights
  - flows (see [`fnd_cut!`](@ref)):
    - `through`, `flow`: the flow through each vertex and on each arc
    - `status`: the status of each vertex
    - `srcmark`, `snkmark`, `srcfrom`, `srcstep`, `snkfrom`, `snkstep`,
      `srcqueue`, `snkqueue`: searches in the split graph
    - `srccut`, `snkcut`: the vertices of the cuts; before the flows,
      `srccut` serves as the queue of breadth-first searches
    - `srcdist`, `snkdist`: the distance of each vertex from the first
      source and the first sink
  - components:
    - `label`: the component of each vertex
    - `cmpwgt`: the weight of each component
  - separators:
    - `part`: the best balanced separator: `part[v]` is 0 if `v` is in
      A, 1 if `v` is in B, and 2 if `v` is in S
    - `fallback`: the best separator

input parameters:
  - `rng`: random number generator
  - `n`: number of vertices
  - `tries`: number of source-sink pairs
  - `balance`: minimum weight of the lighter part, as a fraction of
               the total weight

output parameters:
  - `sepsize`: weight of the separator
"""
function fnd_impl!(
        part::AbstractVector{V},
        fallback::AbstractVector{V},
        target::AbstractVector{V},
        invptr::AbstractVector{V},
        begptr::AbstractVector{V},
        weights::AbstractVector{V},
        through::AbstractVector{V},
        flow::AbstractVector{V},
        status::AbstractVector{UInt8},
        srcmark::AbstractVector{Int},
        snkmark::AbstractVector{Int},
        srcfrom::AbstractVector{Int},
        srcstep::AbstractVector{V},
        snkfrom::AbstractVector{Int},
        snkstep::AbstractVector{V},
        srcqueue::AbstractVector{Int},
        snkqueue::AbstractVector{Int},
        srccut::AbstractVector{V},
        snkcut::AbstractVector{V},
        srcdist::AbstractVector{V},
        snkdist::AbstractVector{V},
        label::AbstractVector{V},
        cmpwgt::AbstractVector{V},
        rng::AbstractRNG,
        nreal::V,
        n::V,
        tries::Int,
        balance::Float64,
    ) where {V}

    # `tot` is the total weight
    tot = zero(V)

    @inbounds for v in oneto(n)
        tot += weights[v]
    end

    # `bsep` and `bsmall` are the weights of the separator and of the
    # lighter part of the best balanced separator, and `fsep` and
    # `fsmall` are those of the best separator; `bsep` and `fsep` are
    # negative until a separator is found
    bsep = fsep = -one(V); bsmall = fsmall = zero(V)

    # `ncmp` is the number of components, and `big` is the
    # heaviest component
    ncmp = fnd_components!(label, cmpwgt, srccut, weights, target, begptr, n)
    big = one(V)

    @inbounds for c in oneto(ncmp)
        if cmpwgt[c] > cmpwgt[big]
            big = c
        end
    end

    # a union of components is a separator of weight zero
    @inbounds if ncmp > one(V)
        # `lim` is the last component in A: A is the heaviest
        # component if it is heavy enough, and otherwise the
        # first components, until they are heavy enough
        if cmpwgt[big] >= balance * tot
            wa = cmpwgt[big]; lim = zero(V)
        else
            wa = zero(V); lim = zero(V)

            while wa < balance * tot
                lim += one(V); wa += cmpwgt[lim]
            end
        end

        for v in oneto(n)
            c = label[v]
            fallback[v] = (iszero(lim) ? c == big : c <= lim) ? zero(V) : one(V)
        end

        fsep = zero(V); fsmall = min(wa, tot - wa)

        if fsmall >= balance * tot
            copyto!(part, one(V), fallback, one(V), n)
            return zero(V)
        end
    end

    # `tag` is the clock of the searches (see `fnd_cut!`)
    tag = 0

    @inbounds for _ in oneto(tries)
        # `s` is a random vertex of the heaviest component
        s = rand(rng, oneto(nreal))

        while label[s] != big
            s = s < nreal ? s + one(V) : one(V)
        end

        _, d = fnd_bfs!(srcdist, srccut, target, begptr, nreal, n, s)

        # if `s` is adjacent to every other vertex of the component,
        # then start from a vertex that is not; if there is none, then
        # the component is complete, and it has no separator
        if d < two(V)
            s = fnd_nonuniversal!(snkdist, label, target, begptr, big, nreal)
            iszero(s) && break
            fnd_bfs!(srcdist, srccut, target, begptr, nreal, n, s)
        end

        # `t` is a random vertex of the heaviest component that is not
        # adjacent to `s`
        t = rand(rng, oneto(nreal))

        while label[t] != big || srcdist[t] < two(V)
            t = t < nreal ? t + one(V) : one(V)
        end

        fnd_bfs!(snkdist, srccut, target, begptr, nreal, n, t)

        tag, bsep, bsmall, fsep, fsmall = fnd_cut!(part, fallback, target,
            invptr, begptr, weights, through, flow, status, srcmark,
            snkmark, srcfrom, srcstep, snkfrom, snkstep, srcqueue, snkqueue,
            srccut, snkcut, srcdist, snkdist, rng, tag, bsep, bsmall, fsep,
            fsmall, tot, balance, nreal, n, s, t)
    end

    # if no separator is balanced, then return the best one
    if isnegative(bsep)
        @assert !isnegative(fsep)
        copyto!(part, one(V), fallback, one(V), n)
        bsep = fsep
    end

    return bsep
end

# Label the components of the graph and compute their weights.
# Returns the number of components.
function fnd_components!(
        label::AbstractVector{V},
        cmpwgt::AbstractVector{V},
        queue::AbstractVector{V},
        weights::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{V},
        n::V,
    ) where {V}
    ncmp = zero(V)

    @inbounds for v in oneto(n)
        label[v] = zero(V)
    end

    @inbounds for r in oneto(n)
        if iszero(label[r])
            ncmp += one(V); label[r] = ncmp; cmpwgt[ncmp] = weights[r]
            queue[begin] = r; head = tail = one(V)

            while head <= tail
                v = queue[head]; head += one(V)

                for p in begptr[v]:begptr[v + one(V)] - one(V)
                    w = target[p]

                    if iszero(label[w])
                        label[w] = ncmp; cmpwgt[ncmp] += weights[w]
                        tail += one(V); queue[tail] = w
                    end
                end
            end
        end
    end

    return ncmp
end

# Compute the distance `dist[v]` of every vertex `v` from the vertex
# `s`, or -1 if `v` is unreachable. The members of a hub (a vertex
# greater than `nreal`) are at distance one from each other: a hub is at
# the distance of the first member reached, and is passed through at no
# cost. Returns the last vertex reached and its distance.
function fnd_bfs!(
        dist::AbstractVector{V},
        queue::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{V},
        nreal::V,
        n::V,
        s::V,
    ) where {V}

    @inbounds for v in oneto(n)
        dist[v] = -one(V)
    end

    @inbounds dist[s] = zero(V); queue[begin] = s; head = tail = one(V)

    @inbounds while head <= tail
        v = queue[head]; head += one(V)

        for p in begptr[v]:begptr[v + one(V)] - one(V)
            w = target[p]

            if isnegative(dist[w])
                if w > nreal
                    dist[w] = dist[v]

                    for q in begptr[w]:begptr[w + one(V)] - one(V)
                        u = target[q]

                        if isnegative(dist[u])
                            dist[u] = dist[v] + one(V)
                            tail += one(V); queue[tail] = u
                        end
                    end
                else
                    dist[w] = dist[v] + one(V)
                    tail += one(V); queue[tail] = w
                end
            end
        end
    end

    @inbounds v = queue[tail]
    @inbounds return v, dist[v]
end

# A vertex of the component `big` that is not adjacent to every other
# vertex of the component, or zero if there is none; `marker` is a work
# array.
function fnd_nonuniversal!(
        marker::AbstractVector{V},
        label::AbstractVector{V},
        target::AbstractVector{V},
        begptr::AbstractVector{V},
        big::V,
        nreal::V,
    ) where {V}
    # `size` is the number of vertices of the component
    size = zero(V)

    @inbounds for v in oneto(nreal)
        marker[v] = zero(V)

        if label[v] == big
            size += one(V)
        end
    end

    @inbounds for v in oneto(nreal)
        if label[v] == big
            # `deg` is the number of neighbors of `v`
            marker[v] = v; deg = zero(V)

            for p in begptr[v]:begptr[v + one(V)] - one(V)
                w = target[p]

                if w > nreal
                    for q in begptr[w]:begptr[w + one(V)] - one(V)
                        u = target[q]

                        if marker[u] != v
                            marker[u] = v; deg += one(V)
                        end
                    end
                elseif marker[w] != v
                    marker[w] = v; deg += one(V)
                end
            end

            if deg < size - one(V)
                return v
            end
        end
    end

    return zero(V)
end


"""
    fnd_cut!(part, fallback, target, invptr, begptr, weights, through,
        flow, status, srcmark, snkmark, srcfrom, srcstep, snkfrom,
        snkstep, srcqueue, snkqueue, srccut, snkcut, srcdist, snkdist,
        rng, tag, bsep, bsmall, fsep, fsmall, tot, balance, n, s, t)

Find a sequence of cuts between growing sets of sources and sinks,
starting from the source `s` and the sink `t`, and record the best
ones.

The flow is a flow in the split graph, where each vertex `x` is
replaced by an in-node 2`x` - 1 and an out-node 2`x`, joined by an arc
of capacity weight(`x`), and each arc (`a`, `b`) of the graph runs from
the out-node of `a` to the in-node of `b`, with infinite capacity. The
sources and sinks are nodes. The flow through `x` is `through[x]`, and
the flow on the arc `p` is `flow[p]`.

Let SR be the set of nodes reachable from the sources in the residual
graph, and let TR be the set of nodes from which the sinks are
reachable. If the flow is maximum, then there are two minimum cuts:

  - source side: A is the set of vertices whose out-node is in SR, and
    S is the set of vertices whose in-node, but not out-node, is in SR
  - sink side: B is the set of vertices whose in-node is in TR, and S
    is the set of vertices whose out-node, but not in-node, is in TR

In both, the weight of S is the value of the flow. The lighter side,
by w(A) on the source side and w(B) on the sink side, is then pierced:
every node of SR (or TR) becomes a source (or a sink), and so does the
out-node (or in-node) of a vertex of its cut, which must not be adjacent
to a vertex whose in-node (or out-node) is a sink (or source). A vertex
whose piercing creates no augmenting path is preferred: on the source
side, a vertex whose out-node is not in TR, and on the sink side, a
vertex whose in-node is not in SR. Ties are broken by the distance from
the first terminal of the other side, largest first, and then at random.
The piercing stops when neither side can be pierced, or when no later
cut can be better than the best balanced cut: the value of the flow
never decreases, and the lighter side of a cut weighs at most half of
the total weight.

Every node of SR that is not a source is reachable from the last node
that became a source, `srcextra`, and the searches that grow SR record
a tree of paths from it: the node `nd` is reached from the node
`srcfrom[nd]` by the step `srcstep[nd]`. Likewise, every node of TR
that is not a sink reaches `snkextra`, by the steps `snkstep` to the
nodes `snkfrom`. When the source side is pierced, if the out-node of the
pierced vertex is in TR, then the flow is augmented along its path in
the tree of TR. Then SR grows from `srcextra`; if the search meets a
sink, then the flow is augmented, the nodes of SR that are not sources
are removed, and the search starts again (see [`fnd_srcgrow!`](@ref)).
Finally, if the flow was augmented, then the nodes of TR that are not
sinks are removed, and TR grows from `snkextra`. The sink side is
pierced likewise.

A node is marked in `srcmark` (or `snkmark`) with the time at which it
joined SR (or TR). The time `tag` is shared by both sets. A node is in
SR if and only if its mark is at least `srcepoch`, and it is a source if
and only if its mark is also at most `srcassim`. The nodes of SR that are
not sources are listed in `srcqueue[1:srctail]`. The cut at an earlier
time can be read back until nodes are removed from the set: the best
cuts are written to `part` and `fallback` only then.

The lists `srccut` and `snkcut` hold the vertices of the cuts, along
with vertices that have left them.

input parameters:
  - `tot`: total weight
  - `balance`: minimum weight of the lighter part, as a fraction of
               the total weight
  - `n`: number of vertices
  - `s`: source
  - `t`: sink

updated parameters:
  - `tag`: time of the searches
  - `bsep`, `bsmall`, `fsep`, `fsmall`: see [`fnd_impl!`](@ref)
"""
function fnd_cut!(
        part::AbstractVector{V},
        fallback::AbstractVector{V},
        target::AbstractVector{V},
        invptr::AbstractVector{V},
        begptr::AbstractVector{V},
        weights::AbstractVector{V},
        through::AbstractVector{V},
        flow::AbstractVector{V},
        status::AbstractVector{UInt8},
        srcmark::AbstractVector{Int},
        snkmark::AbstractVector{Int},
        srcfrom::AbstractVector{Int},
        srcstep::AbstractVector{V},
        snkfrom::AbstractVector{Int},
        snkstep::AbstractVector{V},
        srcqueue::AbstractVector{Int},
        snkqueue::AbstractVector{Int},
        srccut::AbstractVector{V},
        snkcut::AbstractVector{V},
        srcdist::AbstractVector{V},
        snkdist::AbstractVector{V},
        rng::AbstractRNG,
        tag::Int,
        bsep::V,
        bsmall::V,
        fsep::V,
        fsmall::V,
        tot::V,
        balance::Float64,
        nreal::V,
        n::V,
        s::V,
        t::V,
    ) where {V}
    @inbounds m = begptr[n + one(V)] - one(V)

    # clear the flow
    @inbounds for x in oneto(n)
        through[x] = zero(V); status[x] = zero(UInt8)
    end

    @inbounds for p in oneto(m)
        flow[p] = zero(V)
    end

    # the nodes of `s` are sources, and the nodes of `t` are sinks;
    # `sout` is the out-node of `s`, and `tin` is the in-node of `t`
    sout = twice(Int(s)); tin = twice(Int(t)) - 1

    tag += 1; srcepoch = srcassim = tag
    @inbounds srcmark[sout - 1] = srcmark[sout] = tag
    fnd_near!(status, target, begptr, FND_NEARSOURCE, nreal, s)

    tag += 1; snkepoch = snkassim = tag
    @inbounds snkmark[tin] = snkmark[tin + 1] = tag
    fnd_near!(status, target, begptr, FND_NEARSINK, nreal, t)

    # `srcextra` and `snkextra` are the last nodes that became
    # sources and sinks
    srcextra = sout; snkextra = tin

    # `wsrc` is the weight of the vertices whose out-node is in SR,
    # and `wsnk` is the weight of the vertices whose in-node is in TR
    @inbounds wsrc = weights[s]; wsnk = weights[t]

    # `value` is the value of the flow
    value = zero(V); srctail = snktail = 0; nsrccut = nsnkcut = zero(V)

    # maximize the flow, and compute SR and TR
    value, wsrc, srctail, nsrccut, tag = fnd_srcgrow!(weights, target,
        invptr, begptr, through, flow, status, srcmark, snkmark, srcfrom,
        srcstep, srcqueue, srccut, srcepoch, srcassim, snkepoch, snkassim,
        srcextra, nreal, tag, value, wsrc, srctail, nsrccut)

    value, wsnk, snktail, nsnkcut, tag = fnd_snkgrow!(weights, target,
        invptr, begptr, through, flow, status, srcmark, snkmark, snkfrom,
        snkstep, snkqueue, snkcut, srcepoch, srcassim, snkepoch, snkassim,
        snkextra, nreal, tag, value, wsnk, snktail, nsnkcut)

    # `bpend` and `fpend` are 1 if the best balanced separator or the
    # best separator is the source-side cut at the time `btime` or
    # `ftime`, 2 if it is the sink-side cut, and 0 if it has been
    # written
    bpend = fpend = 0; btime = ftime = 0

    @inbounds while true
        # record the cuts
        for side in (1, 2)
            if isone(side)
                wa = wsrc; wb = tot - wsrc - value
            else
                wb = wsnk; wa = tot - wsnk - value
            end

            small = min(wa, wb)

            if small >= balance * tot && fndbetter(value, small, bsep, bsmall)
                bsep = value; bsmall = small; bpend = side; btime = tag
            end

            if fndbetter(value, small, fsep, fsmall)
                fsep = value; fsmall = small; fpend = side; ftime = tag
            end
        end

        # stop when both sides hold half of the weight
        min(wsrc, wsnk) >= half(tot) && break

        # stop when no later cut can be better than the best balanced
        # cut: a later cut is no lighter, and its lighter side weighs
        # at most half of the total weight
        if !isnegative(bsep) && ispositive(bsmall) && value * twice(bsmall) >= bsep * tot
            break
        end

        # the candidates for piercing
        x, nsrccut = fnd_candidate!(srccut, nsrccut, srcmark, snkmark,
            status, snkdist, rng, srcepoch, snkepoch, snkassim,
            FND_NEARSINK, FND_SRCLISTED, true)

        y, nsnkcut = fnd_candidate!(snkcut, nsnkcut, snkmark, srcmark,
            status, srcdist, rng, snkepoch, srcepoch, srcassim,
            FND_NEARSOURCE, FND_SNKLISTED, false)

        if ispositive(x) && (wsrc <= wsnk || iszero(y))
            # pierce the source side
            xout = twice(Int(x)); augment = snkmark[xout] >= snkepoch

            # if the flow will be augmented, then write the best cuts,
            # since nodes will be removed from SR and TR
            if augment
                bpend = fnd_flush!(part, srcmark, snkmark, srcepoch, snkepoch, bpend, btime, n)
                fpend = fnd_flush!(fallback, srcmark, snkmark, srcepoch, snkepoch, fpend, ftime, n)
            end

            # the nodes of SR become sources, and so does the out-node
            # of `x`
            srctail = fnd_assimilate!(status, target, begptr, srcqueue, srctail, FND_NEARSOURCE, nreal, false)
            tag += 1; srcassim = tag; srcmark[xout] = tag; srcextra = xout
            wsrc += weights[x]; fnd_near!(status, target, begptr, FND_NEARSOURCE, nreal, x)

            # if the out-node of `x` is in TR, then augment the flow
            # along the path to a sink in the search tree of TR
            if augment
                value += fnd_augment!(weights, through, flow, snkfrom, snkstep, nreal, xout, false)
            end

            value, wsrc, srctail, nsrccut, tag = fnd_srcgrow!(weights,
                target, invptr, begptr, through, flow, status, srcmark,
                snkmark, srcfrom, srcstep, srcqueue, srccut, srcepoch,
                srcassim, snkepoch, snkassim, srcextra, nreal, tag, value, wsrc,
                srctail, nsrccut)

            if augment
                snktail, wsnk = fnd_discard!(snkmark, snkqueue, weights, snkassim, snktail, wsnk, true)

                value, wsnk, snktail, nsnkcut, tag = fnd_snkgrow!(weights,
                    target, invptr, begptr, through, flow, status, srcmark,
                    snkmark, snkfrom, snkstep, snkqueue, snkcut, srcepoch,
                    srcassim, snkepoch, snkassim, snkextra, nreal, tag, value,
                    wsnk, snktail, nsnkcut)
            end
        elseif ispositive(y)
            # pierce the sink side
            yin = twice(Int(y)) - 1; augment = srcmark[yin] >= srcepoch

            # if the flow will be augmented, then write the best cuts,
            # since nodes will be removed from SR and TR
            if augment
                bpend = fnd_flush!(part, srcmark, snkmark, srcepoch, snkepoch, bpend, btime, n)
                fpend = fnd_flush!(fallback, srcmark, snkmark, srcepoch, snkepoch, fpend, ftime, n)
            end

            # the nodes of TR become sinks, and so does the in-node of
            # `y`
            snktail = fnd_assimilate!(status, target, begptr, snkqueue, snktail, FND_NEARSINK, nreal, true)
            tag += 1; snkassim = tag; snkmark[yin] = tag; snkextra = yin
            wsnk += weights[y]; fnd_near!(status, target, begptr, FND_NEARSINK, nreal, y)

            # if the in-node of `y` is in SR, then augment the flow
            # along the path from a source in the search tree of SR
            if augment
                value += fnd_augment!(weights, through, flow, srcfrom, srcstep, nreal, yin, true)
            end

            value, wsnk, snktail, nsnkcut, tag = fnd_snkgrow!(weights,
                target, invptr, begptr, through, flow, status, srcmark,
                snkmark, snkfrom, snkstep, snkqueue, snkcut, srcepoch,
                srcassim, snkepoch, snkassim, snkextra, nreal, tag, value, wsnk,
                snktail, nsnkcut)

            if augment
                srctail, wsrc = fnd_discard!(srcmark, srcqueue, weights, srcassim, srctail, wsrc, false)

                value, wsrc, srctail, nsrccut, tag = fnd_srcgrow!(weights,
                    target, invptr, begptr, through, flow, status, srcmark,
                    snkmark, srcfrom, srcstep, srcqueue, srccut, srcepoch,
                    srcassim, snkepoch, snkassim, srcextra, nreal, tag, value,
                    wsrc, srctail, nsrccut)
            end
        else
            break
        end
    end

    bpend = fnd_flush!(part, srcmark, snkmark, srcepoch, snkepoch, bpend, btime, n)
    fpend = fnd_flush!(fallback, srcmark, snkmark, srcepoch, snkepoch, fpend, ftime, n)
    return tag, bsep, bsmall, fsep, fsmall
end

# true if a separator of weight `sep` whose lighter part has weight
# `small` has smaller expansion than the separator (`bsep`, `bsmall`),
# or if there is no such separator (`bsep` < 0)
function fndbetter(sep::V, small::V, bsep::V, bsmall::V) where {V}
    if isnegative(bsep)
        better = true
    elseif iszero(bsmall)
        better = ispositive(small)
    else
        better = sep * bsmall < bsep * small
    end

    return better
end

# true if the node `nd` is a terminal: its mark is between `epoch`
# and `assim`
@inline function fndterminal(mark::AbstractVector{Int}, epoch::Int, assim::Int, nd::Int)
    @inbounds return epoch <= mark[nd] <= assim
end

# Mark with `near` the vertices joined to a node of the vertex `x` by
# arcs of infinite capacity, when that node becomes a terminal: the
# neighbors of `x`, and the members of the hubs among them. If `x` is a
# hub, then these are its members (see `fnd_hubnear!`).
function fnd_near!(
        status::AbstractVector{UInt8},
        target::AbstractVector{V},
        begptr::AbstractVector{V},
        near::UInt8,
        nreal::V,
        x::V,
    ) where {V}

    if x > nreal
        fnd_hubnear!(status, target, begptr, near, x)
    else
        @inbounds for p in begptr[x]:begptr[x + one(V)] - one(V)
            y = target[p]

            if y > nreal
                fnd_hubnear!(status, target, begptr, near, y)
            else
                status[y] |= near
            end
        end
    end

    return
end

# Mark the members of the hub `h` with `near`, unless the hub is marked:
# then they already are.
function fnd_hubnear!(
        status::AbstractVector{UInt8},
        target::AbstractVector{V},
        begptr::AbstractVector{V},
        near::UInt8,
        h::V,
    ) where {V}

    @inbounds if iszero(status[h] & near)
        status[h] |= near

        for p in begptr[h]:begptr[h + one(V)] - one(V)
            status[target[p]] |= near
        end
    end

    return
end

# Make the nodes `queue[1:tail]` terminals: since their marks are at
# most the current time, this only marks vertices with `near`, for the
# in-nodes (`odd`) or the out-nodes (see `fnd_near!`). Returns 0.
function fnd_assimilate!(
        status::AbstractVector{UInt8},
        target::AbstractVector{V},
        begptr::AbstractVector{V},
        queue::AbstractVector{Int},
        tail::Int,
        near::UInt8,
        nreal::V,
        odd::Bool,
    ) where {V}

    @inbounds for i in oneto(tail)
        nd = queue[i]

        if isodd(nd) == odd
            fnd_near!(status, target, begptr, near, nreal, convert(V, (nd + 1) >> 1))
        end
    end

    return 0
end

# Remove the nodes `queue[1:tail]` that are not terminals (their marks
# are greater than `assim`) from their set, and subtract the weights of
# their vertices from `wsum`, for the in-nodes (`odd`) or the out-nodes.
# Returns 0 and `wsum`.
function fnd_discard!(
        mark::AbstractVector{Int},
        queue::AbstractVector{Int},
        weights::AbstractVector{V},
        assim::Int,
        tail::Int,
        wsum::V,
        odd::Bool,
    ) where {V}

    @inbounds for i in oneto(tail)
        nd = queue[i]

        if mark[nd] > assim
            mark[nd] = 0

            if isodd(nd) == odd
                wsum -= weights[(nd + 1) >> 1]
            end
        end
    end

    return 0, wsum
end

# Choose a vertex to pierce from the list `cut[1:ncut]`, and remove the
# vertices that have left the cut. On the source side (`fwd`), a vertex
# `x` is in the cut if its in-node, but not its out-node, is marked at
# or after `epoch` in `mark`; on the sink side, the roles of the nodes
# are exchanged. A vertex marked with `near` is not chosen, nor is a
# vertex whose other node is a terminal of the other side, marked
# between `othepoch` and `othassim` in `othmark`. A vertex is preferred
# if its other node is not marked at or after `othepoch`, and then if
# `dist` is large. The vertices of the list are marked with `listed`.
# Returns the vertex, or zero if there is none.
function fnd_candidate!(
        cut::AbstractVector{V},
        ncut::V,
        mark::AbstractVector{Int},
        othmark::AbstractVector{Int},
        status::AbstractVector{UInt8},
        dist::AbstractVector{V},
        rng::AbstractRNG,
        epoch::Int,
        othepoch::Int,
        othassim::Int,
        near::UInt8,
        listed::UInt8,
        fwd::Bool,
    ) where {V}
    # `best` is the chosen vertex, `bgood` and `bdist` are its keys,
    # and `nbest` is the number of vertices with these keys
    best = zero(V); bgood = false; bdist = zero(V); nbest = 0

    # `k` is the length of the list after compaction
    k = zero(V)

    @inbounds for i in oneto(ncut)
        x = cut[i]; xin = twice(Int(x)) - 1; xout = xin + 1

        # `inner` is the node of `x` in the set, and `outer` the
        # node of `x` outside of it
        inner, outer = fwd ? (xin, xout) : (xout, xin)

        if mark[inner] >= epoch && mark[outer] < epoch
            k += one(V); cut[k] = x

            if iszero(status[x] & near) && !fndterminal(othmark, othepoch, othassim, outer)
                good = othmark[outer] < othepoch; d = dist[x]

                if iszero(best) || (good, d) > (bgood, bdist)
                    best = x; bgood = good; bdist = d; nbest = 1
                elseif (good, d) == (bgood, bdist)
                    nbest += 1

                    if isone(rand(rng, oneto(nbest)))
                        best = x
                    end
                end
            end
        else
            status[x] &= ~listed
        end
    end

    return best, k
end

# If `pend` is 1 or 2, write the source-side or sink-side cut at the
# time `time` to `labels`. Returns 0.
function fnd_flush!(
        labels::AbstractVector{V},
        srcmark::AbstractVector{Int},
        snkmark::AbstractVector{Int},
        srcepoch::Int,
        snkepoch::Int,
        pend::Int,
        time::Int,
        n::V,
    ) where {V}

    if isone(pend)
        @inbounds for x in oneto(n)
            xin = twice(Int(x)) - 1; xout = xin + 1

            if srcepoch <= srcmark[xout] <= time
                labels[x] = zero(V)
            elseif srcepoch <= srcmark[xin] <= time
                labels[x] = two(V)
            else
                labels[x] = one(V)
            end
        end
    elseif istwo(pend)
        @inbounds for x in oneto(n)
            xin = twice(Int(x)) - 1; xout = xin + 1

            if snkepoch <= snkmark[xin] <= time
                labels[x] = one(V)
            elseif snkepoch <= snkmark[xout] <= time
                labels[x] = two(V)
            else
                labels[x] = zero(V)
            end
        end
    end

    return 0
end

# Grow SR from the source `srcextra`. If the search meets a sink, then
# the flow is augmented along the path to it, the nodes of SR that are
# not sources are removed, and the search starts again.
#
# output parameters:
#   - `value`: value of the flow
#   - `wsrc`: weight of the vertices whose out-node is in SR
#   - `srctail`: number of nodes of SR that are not sources
#   - `nsrccut`: length of the list `srccut`
#   - `tag`: time of the searches
function fnd_srcgrow!(
        weights::AbstractVector{V},
        target::AbstractVector{V},
        invptr::AbstractVector{V},
        begptr::AbstractVector{V},
        through::AbstractVector{V},
        flow::AbstractVector{V},
        status::AbstractVector{UInt8},
        srcmark::AbstractVector{Int},
        snkmark::AbstractVector{Int},
        from::AbstractVector{Int},
        step::AbstractVector{V},
        srcqueue::AbstractVector{Int},
        srccut::AbstractVector{V},
        srcepoch::Int,
        srcassim::Int,
        snkepoch::Int,
        snkassim::Int,
        srcextra::Int,
        nreal::V,
        tag::Int,
        value::V,
        wsrc::V,
        srctail::Int,
        nsrccut::V,
    ) where {V}

    @inbounds while true
        tag += 1; head = srctail + 1
        srctail += 1; srcqueue[srctail] = srcextra; from[srcextra] = 0

        meet, srctail, wsrc, nsrccut = fnd_forward!(weights, target,
            invptr, begptr, through, flow, status, srcmark, snkmark, from,
            step, srcqueue, srccut, srcepoch, snkepoch, snkassim, nreal,
            tag, head, srctail, wsrc, nsrccut)

        iszero(meet) && break

        value += fnd_augment!(weights, through, flow, from, step, nreal, meet, true)
        srctail, wsrc = fnd_discard!(srcmark, srcqueue, weights, srcassim, srctail, wsrc, false)
    end

    return value, wsrc, srctail, nsrccut, tag
end

# Grow TR from the sink `snkextra`. If the search meets a source, then
# the flow is augmented along the path from it, the nodes of TR that are
# not sinks are removed, and the search starts again.
#
# output parameters:
#   - `value`: value of the flow
#   - `wsnk`: weight of the vertices whose in-node is in TR
#   - `snktail`: number of nodes of TR that are not sinks
#   - `nsnkcut`: length of the list `snkcut`
#   - `tag`: time of the searches
function fnd_snkgrow!(
        weights::AbstractVector{V},
        target::AbstractVector{V},
        invptr::AbstractVector{V},
        begptr::AbstractVector{V},
        through::AbstractVector{V},
        flow::AbstractVector{V},
        status::AbstractVector{UInt8},
        srcmark::AbstractVector{Int},
        snkmark::AbstractVector{Int},
        from::AbstractVector{Int},
        step::AbstractVector{V},
        snkqueue::AbstractVector{Int},
        snkcut::AbstractVector{V},
        srcepoch::Int,
        srcassim::Int,
        snkepoch::Int,
        snkassim::Int,
        snkextra::Int,
        nreal::V,
        tag::Int,
        value::V,
        wsnk::V,
        snktail::Int,
        nsnkcut::V,
    ) where {V}

    @inbounds while true
        tag += 1; head = snktail + 1
        snktail += 1; snkqueue[snktail] = snkextra; from[snkextra] = 0

        meet, snktail, wsnk, nsnkcut = fnd_backward!(weights, target,
            invptr, begptr, through, flow, status, srcmark, snkmark, from,
            step, snkqueue, snkcut, srcepoch, srcassim, snkepoch, nreal,
            tag, head, snktail, wsnk, nsnkcut)

        iszero(meet) && break

        value += fnd_augment!(weights, through, flow, from, step, nreal, meet, false)
        snktail, wsnk = fnd_discard!(snkmark, snkqueue, weights, snkassim, snktail, wsnk, true)
    end

    return value, wsnk, snktail, nsnkcut, tag
end

# Search forward from the nodes `srcqueue[head:tail]` in the residual
# graph, marking the nodes reached with the time `tag`; a node is
# already reached if its mark is at least `srcepoch`. The search stops
# when it meets a sink, and `from` and `step` give the node and the step
# that each node was reached from (see `fndresidual`). A saturated
# vertex whose in-node is reached is added to `srccut`.
#
# output parameters:
#   - `meet`: the sink that was met, or zero
#   - `tail`, `wsrc`, `nsrccut`: see `fnd_srcgrow!`
function fnd_forward!(
        weights::AbstractVector{V},
        target::AbstractVector{V},
        invptr::AbstractVector{V},
        begptr::AbstractVector{V},
        through::AbstractVector{V},
        flow::AbstractVector{V},
        status::AbstractVector{UInt8},
        srcmark::AbstractVector{Int},
        snkmark::AbstractVector{Int},
        from::AbstractVector{Int},
        step::AbstractVector{V},
        srcqueue::AbstractVector{Int},
        srccut::AbstractVector{V},
        srcepoch::Int,
        snkepoch::Int,
        snkassim::Int,
        nreal::V,
        tag::Int,
        head::Int,
        tail::Int,
        wsrc::V,
        nsrccut::V,
    ) where {V}

    @inbounds while head <= tail
        nd = srcqueue[head]; head += 1
        x = convert(V, (nd + 1) >> 1)

        if iseven(nd)
            # `nd` is the out-node of `x`: follow the arcs (`x`, `b`)
            for p in begptr[x]:begptr[x + one(V)] - one(V)
                bin = twice(Int(target[p])) - 1

                if srcmark[bin] < srcepoch
                    from[bin] = nd; step[bin] = p

                    if fndterminal(snkmark, snkepoch, snkassim, bin)
                        return bin, tail, wsrc, nsrccut
                    end

                    srcmark[bin] = tag; tail += 1; srcqueue[tail] = bin
                end
            end

            # undo flow through `x`
            xin = nd - 1

            if ispositive(through[x]) && srcmark[xin] < srcepoch
                from[xin] = nd; step[xin] = zero(V)

                if fndterminal(snkmark, snkepoch, snkassim, xin)
                    return xin, tail, wsrc, nsrccut
                end

                srcmark[xin] = tag; tail += 1; srcqueue[tail] = xin
            end
        else
            # `nd` is the in-node of `x`: send flow through `x`, or
            # else add `x` to the cut
            xout = nd + 1

            if srcmark[xout] < srcepoch
                if x > nreal || through[x] < weights[x]
                    from[xout] = nd; step[xout] = zero(V)

                    if fndterminal(snkmark, snkepoch, snkassim, xout)
                        return xout, tail, wsrc, nsrccut
                    end

                    srcmark[xout] = tag; wsrc += weights[x]
                    tail += 1; srcqueue[tail] = xout
                elseif iszero(status[x] & FND_SRCLISTED)
                    status[x] |= FND_SRCLISTED
                    nsrccut += one(V); srccut[nsrccut] = x
                end
            end

            # undo flow on the arcs (`a`, `x`)
            for q in begptr[x]:begptr[x + one(V)] - one(V)
                p = invptr[q]

                if ispositive(flow[p])
                    a = target[q]; aout = twice(Int(a))

                    if srcmark[aout] < srcepoch
                        from[aout] = nd; step[aout] = -p

                        if fndterminal(snkmark, snkepoch, snkassim, aout)
                            return aout, tail, wsrc, nsrccut
                        end

                        srcmark[aout] = tag; wsrc += weights[a]
                        tail += 1; srcqueue[tail] = aout
                    end
                end
            end
        end
    end

    return 0, tail, wsrc, nsrccut
end

# Search backward from the nodes `snkqueue[head:tail]` in the residual
# graph, marking the nodes reached with the time `tag`; a node is
# already reached if its mark is at least `snkepoch`. The search stops
# when it meets a source, and `from` and `step` give the node and the
# step that each node reaches (see `fndresidual`). A saturated vertex
# whose out-node is reached is added to `snkcut`.
#
# output parameters:
#   - `meet`: the source that was met, or zero
#   - `tail`, `wsnk`, `nsnkcut`: see `fnd_snkgrow!`
function fnd_backward!(
        weights::AbstractVector{V},
        target::AbstractVector{V},
        invptr::AbstractVector{V},
        begptr::AbstractVector{V},
        through::AbstractVector{V},
        flow::AbstractVector{V},
        status::AbstractVector{UInt8},
        srcmark::AbstractVector{Int},
        snkmark::AbstractVector{Int},
        from::AbstractVector{Int},
        step::AbstractVector{V},
        snkqueue::AbstractVector{Int},
        snkcut::AbstractVector{V},
        srcepoch::Int,
        srcassim::Int,
        snkepoch::Int,
        nreal::V,
        tag::Int,
        head::Int,
        tail::Int,
        wsnk::V,
        nsnkcut::V,
    ) where {V}

    @inbounds while head <= tail
        nd = snkqueue[head]; head += 1
        y = convert(V, (nd + 1) >> 1)

        if isodd(nd)
            # `nd` is the in-node of `y`: follow the arcs (`x`, `y`)
            # backward
            for q in begptr[y]:begptr[y + one(V)] - one(V)
                xout = twice(Int(target[q]))

                if snkmark[xout] < snkepoch
                    from[xout] = nd; step[xout] = invptr[q]

                    if fndterminal(srcmark, srcepoch, srcassim, xout)
                        return xout, tail, wsnk, nsnkcut
                    end

                    snkmark[xout] = tag; tail += 1; snkqueue[tail] = xout
                end
            end

            # undo flow through `y`
            yout = nd + 1

            if ispositive(through[y]) && snkmark[yout] < snkepoch
                from[yout] = nd; step[yout] = zero(V)

                if fndterminal(srcmark, srcepoch, srcassim, yout)
                    return yout, tail, wsnk, nsnkcut
                end

                snkmark[yout] = tag; tail += 1; snkqueue[tail] = yout
            end
        else
            # `nd` is the out-node of `y`: send flow through `y`, or
            # else add `y` to the cut
            yin = nd - 1

            if snkmark[yin] < snkepoch
                if y > nreal || through[y] < weights[y]
                    from[yin] = nd; step[yin] = zero(V)

                    if fndterminal(srcmark, srcepoch, srcassim, yin)
                        return yin, tail, wsnk, nsnkcut
                    end

                    snkmark[yin] = tag; wsnk += weights[y]
                    tail += 1; snkqueue[tail] = yin
                elseif iszero(status[y] & FND_SNKLISTED)
                    status[y] |= FND_SNKLISTED
                    nsnkcut += one(V); snkcut[nsnkcut] = y
                end
            end

            # undo flow on the arcs (`y`, `b`)
            for p in begptr[y]:begptr[y + one(V)] - one(V)
                if ispositive(flow[p])
                    b = target[p]; bin = twice(Int(b)) - 1

                    if snkmark[bin] < snkepoch
                        from[bin] = nd; step[bin] = -p

                        if fndterminal(srcmark, srcepoch, srcassim, bin)
                            return bin, tail, wsnk, nsnkcut
                        end

                        snkmark[bin] = tag; wsnk += weights[b]
                        tail += 1; snkqueue[tail] = bin
                    end
                end
            end
        end
    end

    return 0, tail, wsnk, nsnkcut
end

# Send flow along the path found by a search that met the node `meet`:
# the path runs back from `meet` along `from` if the search was forward
# (`fwd`), and on from `meet` along `from` otherwise. Returns the amount
# of flow sent.
function fnd_augment!(
        weights::AbstractVector{V},
        through::AbstractVector{V},
        flow::AbstractVector{V},
        from::AbstractVector{Int},
        step::AbstractVector{V},
        nreal::V,
        meet::Int,
        fwd::Bool,
    ) where {V}

    # find the bottleneck
    del = typemax(V); nd = meet

    @inbounds while ispositive(from[nd])
        # the step `step[nd]` runs from `from[nd]` to `nd` if the
        # search was forward, and from `nd` to `from[nd]` otherwise
        hd = fwd ? nd : from[nd]
        del = min(del, fndresidual(weights, through, flow, nreal, hd, step[nd])); nd = from[nd]
    end

    # augment
    nd = meet

    @inbounds while ispositive(from[nd])
        hd = fwd ? nd : from[nd]
        fnd_push!(through, flow, hd, step[nd], del); nd = from[nd]
    end

    return del
end

# the residual capacity of the step `a` to the node `nd`: a step
# follows an arc forward if `a` is positive, an arc backward if `a`
# is negative, and the internal arc of a vertex if `a` is zero; the
# internal arc of a hub (a vertex greater than `nreal`) has infinite
# capacity
function fndresidual(weights::AbstractVector{V}, through::AbstractVector{V}, flow::AbstractVector{V}, nreal::V, nd::Int, a::V) where {V}
    if iszero(a)
        x = convert(V, (nd + 1) >> 1)

        if iseven(nd)
            @inbounds res = x > nreal ? typemax(V) : weights[x] - through[x]
        else
            @inbounds res = through[x]
        end
    elseif isnegative(a)
        @inbounds res = flow[-a]
    else
        res = typemax(V)
    end

    return res
end

# send `del` units of flow along the step `a` to the node `nd`
function fnd_push!(through::AbstractVector{V}, flow::AbstractVector{V}, nd::Int, a::V, del::V) where {V}
    if iszero(a)
        x = convert(V, (nd + 1) >> 1)
        @inbounds through[x] = iseven(nd) ? through[x] + del : through[x] - del
    elseif ispositive(a)
        @inbounds flow[a] += del
    else
        @inbounds flow[-a] -= del
    end

    return
end
