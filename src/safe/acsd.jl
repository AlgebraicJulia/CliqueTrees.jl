# the clique tree and the weights by position, for `acsd_heavy!` (see below)
struct ACSDHeavy{V, W, A}
    at::A                       # the clique tree (an `ACSDTree`)
    rmq::RMQ{V, W}              # the weights, by position
    comps::Vector{Tuple{V, V}}  # the components of G - S
    ancs::Vector{V}
    subs::Vector{Tuple{V, V}}   # the ranges of the components that are subtrees
end

function ACSDHeavy(weights::AbstractVector{W}, order::AbstractVector{V}, tree::CliqueTree{V}) where {W, V}
    n = length(order); val = FVector{W}(undef, n)

    @inbounds for i in oneto(n)
        val[i] = weights[order[i]]
    end

    return ACSDHeavy(ACSDTree(tree, n), RMQ(V, val, false), Tuple{V, V}[], V[], Tuple{V, V}[])
end

function acsd_find!(
        head::AbstractVector{V},
        next::AbstractVector{V},
        mark::AbstractVector{V},
        stat::AbstractVector{V},
        cent::AbstractVector{V},
        wperm::AbstractVector{V},
        hv::Union{Nothing, ACSDHeavy},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        order::AbstractVector{V},
        tree::CliqueTree{V, E},
        pass::V,
    ) where {W, V, E}
    @assert nv(graph) <= length(head)
    @assert nv(graph) <= length(next)
    @assert nv(graph) <= length(mark)
    @assert nv(graph) <= length(weights)
    @assert nv(graph) <= length(order)
    @assert nv(graph) <= length(cent)
    @assert length(tree) <= length(stat)

    function set(v::V)
        h = view(head, v)
        return SinglyLinkedList(h, next)
    end

    minwgt = typemax(W)
    tol = tolerance(W)
    node = zero(V)
    totcnt = zero(E)
    numsep = zero(V)

    for v in vertices(graph)
        head[v] = zero(V)
        mark[v] = zero(V)
        minwgt = min(minwgt, weights[v])
    end

    for bag in tree
        node += one(V)
        sep = separator(bag)
        maxcnt = convert(V, length(sep))

        # skip cliques and separators already filled (both are cliques now)
        proc = !isnegative(stat[node])

        # after the first pass a separator can only change if it gained an
        # edge, and every new edge is incident to a center of the previous
        # pass; `cent[v]` is the last pass in which `v` was a center, and it
        # only grows
        if proc && pass > one(V)
            proc = false

            for i in sep
                if cent[order[i]] >= pass - one(V)
                    proc = true
                    break
                end
            end
        end

        if proc
            # pre-filter: a vertex of degree less than |S| - 2 is missing at
            # least two edges in S; two such vertices rule out an almost clique
            numlow = zero(V)

            for i in sep
                if eltypedegree(graph, order[i]) + two(V) < maxcnt
                    numlow += one(V)
                    numlow > one(V) && break
                end
            end

            if numlow <= one(V)
                for i in sep
                    mark[order[i]] = node
                end

                poscnt = negcnt = vert = alt1 = alt2 = zero(V)

                for i in sep
                    v = order[i]; cnt = maxcnt - one(V)

                    for w in neighbors(graph, v)
                        if w != v && mark[w] == node
                            cnt -= one(V)
                        end
                    end

                    if isone(cnt)
                        negcnt += one(V)

                        if iszero(alt1)
                            alt1 = v
                        else
                            alt2 = v
                        end
                    elseif ispositive(cnt)
                        if iszero(vert)
                            poscnt = cnt; vert = v
                        else
                            poscnt = -one(V)
                            break
                        end
                    end
                end

                if !isnegative(poscnt)
                    if iszero(vert) && iszero(negcnt)
                        # S is a clique
                        stat[node] = -one(V)
                    else
                        # the minimum weight of a vertex outside of S: the first
                        # vertex not in S in weight order (`wperm`)
                        minout = minwgt

                        for v in wperm
                            if mark[v] != node
                                minout = weights[v]
                                break
                            end
                        end

                        center = zero(V); k = zero(V)

                        # S is safe if the center is no heavier than every
                        # vertex outside S, or, failing that, than every
                        # vertex of two full components (see `acsd_heavy!`)
                        if ispositive(vert)
                            if poscnt == negcnt
                                if weights[vert] < minout + tol || acsd_heavy!(hv, tree, node, weights[vert])
                                    center = vert; k = poscnt
                                end
                            end
                        elseif negcnt == two(V)
                            if weights[alt1] < minout + tol
                                center = alt1; k = one(V)
                            elseif weights[alt2] < minout + tol
                                center = alt2; k = one(V)
                            elseif acsd_heavy!(hv, tree, node, weights[alt1])
                                center = alt1; k = one(V)
                            elseif acsd_heavy!(hv, tree, node, weights[alt2])
                                center = alt2; k = one(V)
                            end
                        end

                        if ispositive(center)
                            totcnt += convert(E, k)
                            numsep += one(V)
                            stat[node] = -one(V)
                            cent[center] = pass
                            pushfirst!(set(center), node)
                        end
                    end
                end
            end
        end
    end

    return de(graph) + twice(totcnt), numsep
end

# An almost-clique minimal separator S with center v is safe if, for each
# component C of G - S, some full component D ≠ C can be contracted into v:
# then G[C ∪ S] with S filled is a minor of G. With vertex weights, the
# contracted vertex must keep the weight of v, so no vertex of D may be
# lighter than v. This holds for every C exactly when S has two such full
# components.
#
# The components of G - S are read off the clique tree of the minimal
# triangulation (see `acsd_components!` in acsd_minor.jl). Their vertices are
# ranges of positions in the elimination order: the subtree below a node or,
# for a segment of the path from S to the root, the subtree below its top
# minus the subtree below the node that cuts it off (two ranges), minus the
# positions of S and the subtrees of the other components. So the least
# weight of a component is the least of a few range-minimum queries, one per
# gap between these positions, and a component is full if its neighborhood,
# which is a separator of the tree, is as large as S.
#
# Returns true if S = sep(`node`) has two full components with no vertex
# lighter than `wgt`.
function acsd_heavy!(hv::ACSDHeavy{V, W}, tree::CliqueTree{V}, node::V, wgt::W) where {V, W}
    tol = tolerance(W); at = hv.at; sep = separator(tree, node); k = length(sep)
    comps = acsd_components!(hv.comps, hv.ancs, at, tree, node)
    subs = hv.subs; empty!(subs)

    for (top, q) in comps
        top == q && push!(subs, (at.rfirst[at.fdesc[q]], at.rlast[q]))
    end

    sort!(subs)
    num = 0

    for (top, q) in comps
        length(separator(tree, q)) == k || continue

        if top == q
            # a subtree
            heavy = acsd_rmin(hv, at.rfirst[at.fdesc[q]], at.rlast[q]) + tol > wgt
        else
            # a segment of the path to the root, cut off by `q`
            lo = at.rfirst[at.fdesc[top]]; hi = at.rlast[top]
            qlo = at.rfirst[at.fdesc[q]]; qhi = at.rlast[q]

            heavy = acsd_gaps(hv, subs, sep, lo, qlo - one(V), wgt, tol) &&
                acsd_gaps(hv, subs, sep, qhi + one(V), hi, wgt, tol)
        end

        if heavy
            num += 1
            num > 1 && return true
        end
    end

    return false
end

# Is no position in `lo:hi`, other than those of `sep` and of the ranges
# `subs` (both sorted), lighter than `wgt`? Every range of `subs` is inside
# `lo:hi` or disjoint from it.
function acsd_gaps(hv::ACSDHeavy{V, W}, subs::Vector{Tuple{V, V}}, sep::AbstractVector{V}, lo::V, hi::V, wgt::W, tol::W) where {V, W}
    lo > hi && return true
    i = searchsortedfirst(sep, lo); s = searchsortedfirst(subs, (lo, zero(V)))
    cur = lo

    while true
        # the next position of `sep`, or the next range of `subs`, in `lo:hi`
        nxt = hi + one(V); skip = hi + one(V)

        if i <= length(sep) && sep[i] <= hi
            nxt = skip = sep[i]
        end

        if s <= length(subs) && first(subs[s]) < nxt
            nxt = first(subs[s]); skip = last(subs[s])
            s += 1
        elseif nxt <= hi
            i += 1
        end

        if cur < nxt
            acsd_rmin(hv, cur, nxt - one(V)) + tol > wgt || return false
        end

        nxt > hi && return true
        cur = skip + one(V)
    end
end

# with equal weights, the center is never heavier than a vertex outside S
acsd_heavy!(::Nothing, tree::CliqueTree, node::Integer, wgt::Number) = false

# the least weight of the positions `i:j`
@inline function acsd_rmin(hv::ACSDHeavy, i::Integer, j::Integer)
    return hv.rmq.val[rmq(hv.rmq, i, j)]
end

function acsd_complete!(
        pointer::AbstractVector{E},
        target::AbstractVector{V},
        head::AbstractVector{V},
        next::AbstractVector{V},
        mark::AbstractVector{V},
        graph::AbstractGraph{V},
        order::AbstractVector{V},
        tree::CliqueTree{V, E},
    ) where {V, E}   
    @assert nv(graph) < length(pointer)
    @assert de(graph) <= length(target)
    @assert nv(graph) <= length(head)
    @assert nv(graph) <= length(next)
    @assert nv(graph) <= length(mark)
    @assert nv(graph) <= length(order)

    n = nv(graph)
    
    function set(v::V)
        h = view(head, v)
        return SinglyLinkedList(h, next)
    end

    for v in vertices(graph)
        mark[v] = zero(V)
        pointer[v + one(V)] = zero(V)
    end

    for v in oneto(n - one(V))
        nodes = set(v); ndeg = pointer[v + two(V)] + eltypedegree(graph, v)
        
        if !isempty(nodes)         
            for w in neighbors(graph, v)
                mark[w] = v
            end

            for node in nodes, i in separator(tree, node)
                w = order[i]

                if v != w && mark[w] < v
                    mark[w] = v; ndeg += one(V)

                    if w < n
                        pointer[w + two(V)] += one(V)
                    end
                end
            end
        end

        pointer[v + two(V)] = ndeg
    end

    nodes = set(n)

    if !isempty(nodes)
        for w in neighbors(graph, n)
            mark[w] = n
        end

        for node in nodes, i in separator(tree, node)
            w = order[i]

            if n != w && mark[w] < n
                mark[w] = n

                if w < n
                    pointer[w + two(V)] += one(V)
                end
            end
        end
    end

    pointer[one(V)] = p = one(E) 
    
   for v in vertices(graph)
        mark[v] = zero(V)
        pointer[v + one(V)] = p += pointer[v + one(V)]
    end
    
    for v in vertices(graph)
        nodes = set(v); p = pointer[v + one(V)]
        
        if !isempty(nodes)            
            for w in neighbors(graph, v)
                target[p] = w; p += one(E)
                mark[w] = v
            end

            for node in nodes, i in separator(tree, node)
                w = order[i]

                if v != w && mark[w] < v
                    mark[w] = v
                    q = pointer[w + one(V)]
                    target[p] = w; p += one(E)
                    target[q] = v; q += one(E)
                    pointer[w + one(V)] = q
                end
            end
        else
            for w in neighbors(graph, v)
                target[p] = w; p += one(E)
            end
        end

        pointer[v + one(V)] = p
    end

    p = qstop = one(E)
    
    for v in vertices(graph)
        qstrt = qstop
        qstop = pointer[v + one(V)]

        for q in qstrt:qstop - one(E)
            w = target[q]

            if mark[w] < v + n
                mark[w] = v + n
                target[p] = w; p += one(V)
            end
        end

        pointer[v + one(V)] = p  
    end

    m = p - one(E)
    return BipartiteGraph(n, n, m, pointer, target)
end

# With `unweighted = true`, the minimal triangulation is made with unit
# weights (its separators are still tested with the true weights).
function acsd(weights::AbstractVector{W}, graph::AbstractGraph{V}, alg::MinimalAlgorithm; unweighted::Bool = false) where {W, V}
    E = etype(graph); n = nv(graph)

    order, tree = unweighted ? cliquetree(ones(W, n), graph, alg) : cliquetree(weights, graph, alg)
    t = length(tree)

    head = FVector{V}(undef, n)
    next = FVector{V}(undef, n)
    mark = FVector{V}(undef, n)
    stat = FVector{V}(undef, t)
    cent = FVector{V}(undef, n)

    for v in vertices(graph)
        cent[v] = zero(V)
    end

    for node in oneto(t)
        stat[node] = zero(V)
    end

    minwgt, maxwgt = extrema(view(weights, oneto(n)))

    # `wperm` lists the vertices in increasing weight order; when every weight
    # is equal, any order works, so skip the sort
    wperm = FVector{V}(undef, n)

    if maxwgt < minwgt + tolerance(W)
        copyto!(wperm, oneto(n))
    else
        sortperm!(wperm, view(weights, oneto(n)))
    end

    # the clique tree and the weights by position, for the weighted test (see
    # `acsd_heavy!`); with equal weights the test is never needed
    hv = maxwgt < minwgt + tolerance(W) ? nothing : ACSDHeavy(weights, order, tree)

    # fill the almost-clique separators of the triangulation, then re-check its
    # other separators: filling one can turn another into an almost clique. the
    # triangulation stays a minimal triangulation of every graph between the
    # input and itself, so each pass examines minimal separators of the current
    # graph
    pass = zero(V)

    while true
        pass += one(V)

        m, numsep = acsd_find!(head, next, mark, stat, cent,
            wperm, hv, weights, graph, order, tree, pass)

        iszero(numsep) && break

        # `pointer` and `target` are fresh each pass: the new graph is read out
        # of the old one, which lives in the previous pass's arrays
        pointer = FVector{E}(undef, n + one(V))
        target = FVector{V}(undef, m)
        graph = acsd_complete!(pointer, target, head, next, mark, graph, order, tree)
    end

    return graph, order, tree
end

function acsd(weights::AbstractVector, graph::AbstractGraph, alg::MinimalAlgorithm, rest::MinimalAlgorithm...)
    graph, order, tree = acsd(weights, graph, alg)

    for other in rest
        graph, order, tree = acsd(weights, graph, other)
    end

    return graph, order, tree
end

# Are the weights all equal? (Then a triangulation with unit weights is the
# same as the weighted one.)
function acsd_uniform(weights::AbstractVector, n::Integer)
    minwgt, maxwgt = extrema(view(weights, oneto(n)))
    return maxwgt < minwgt + tolerance(eltype(weights))
end

# With `tryunit`, also take the separators of the triangulation made with unit
# weights; the returned triangulation is then that one.
function acsd_both(weights::AbstractVector, graph::AbstractGraph, alg::MinimalAlgorithm, tryunit::Bool)
    graph, order, tree = acsd(weights, graph, alg)
    (tryunit && !acsd_uniform(weights, nv(graph))) || return graph, order, tree
    return acsd(weights, graph, alg; unweighted = true)
end

# Apply the minimal algorithms in `mins` in turn, filling the almost-clique
# separators of each triangulation. If `maxmissing` is positive, then also
# fill the separators of each triangulation that have at most `maxmissing`
# missing edges and pass the connectivity test of Medema and Lazovik (see
# acsd_flow.jl) or the clique-minor test of Bodlaender and Koster (see
# acsd_minor.jl), and repeat until a round fills nothing; the returned
# triangulation is then one of the returned graph.
function acsd(weights::AbstractVector, graph::AbstractGraph, mins::Tuple, maxmissing::Int, tryunit::Bool = true)
    if !ispositive(maxmissing)
        graph, order, tree = acsd_both(weights, graph, first(mins), tryunit)

        for other in Base.tail(mins)
            graph, order, tree = acsd_both(weights, graph, other, tryunit)
        end

        return graph, order, tree
    end

    while true
        graph, order, tree, changed = acsd_round(weights, graph, maxmissing, tryunit, false, mins...)
        changed || return graph, order, tree
    end
end

function acsd_round(weights::AbstractVector, graph::AbstractGraph, maxmissing::Int, tryunit::Bool, changed::Bool, alg::MinimalAlgorithm, rest::MinimalAlgorithm...)
    graph, order, tree = acsd(weights, graph, alg)
    graph, filled = acsd_minor(weights, graph, order, tree, maxmissing)
    changed = changed || filled

    # with `tryunit`, also take the separators of a triangulation made with
    # unit weights (unless the weights are all equal: then it is the same)
    if tryunit && !acsd_uniform(weights, nv(graph))
        graph, order, tree = acsd(weights, graph, alg; unweighted = true)
        graph, filled = acsd_minor(weights, graph, order, tree, maxmissing)
        changed = changed || filled
    end

    if isempty(rest)
        return graph, order, tree, changed
    end

    return acsd_round(weights, graph, maxmissing, tryunit, changed, rest...)
end

function acsd_ranks(weights::AbstractVector{<:Integer}, n::Integer)
    return Int[weights[v] for v in oneto(n)]
end

function acsd_ranks(weights::AbstractVector, n::Integer)
    perm = sortperm(view(weights, oneto(n)))
    rank = Vector{Int}(undef, n)
    r = 0

    for (i, v) in enumerate(perm)
        if i == 1 || weights[v] != weights[perm[i - 1]]
            r += 1
        end

        rank[v] = r
    end

    return rank
end
