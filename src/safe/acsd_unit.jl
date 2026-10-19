# The unit-minor test of `SafeSeparators(...; maxmissing)`, tried after the
# connectivity test and the minor test, when the weights are not all equal.
#
# Blow every vertex v up into a clique of w(v) twins (units). For a component
# C of G - S with N = N(C), the minor test of acsd_minor.jl contracts each
# connected set of R = V - C - N into one vertex of N, which must be no
# heavier than any vertex of the set. In the blow-up, the units of a vertex
# of N may instead be contracted into different sets, and a vertex q of R may
# lend its w(q) units to different sets. A *branch* is a triple (v, Q, μ): a
# label v ∈ N, a connected set Q ∋ v with Q - v ⊆ R, and a mass μ > 0 (that
# many units of v, each with its own copy of Q - v). The branches certify C
# if
#
#   (i)   the masses of the branches of each end of a missing edge sum to
#         its weight;
#   (ii)  the masses of the branches that contain a vertex q ∈ R sum to at
#         most w(q);
#   (iii) for every missing edge {u, v} of N, every branch of u touches every
#         branch of v (they share a vertex, or contain adjacent vertices).
#
# Then G[C ∪ N] + K(N), blown up, is a minor of the blow-up of G (contract
# the copies, which lie in G - C, into their twins of N), so its
# weighted treewidth is at most that of G. (Scale rational weights to
# integers; units of a shared vertex are twins, so they are adjacent. Real
# weights follow by continuity: the width is a minimum of maxima of linear
# forms in the weights, and for fixed sets Q, (i)-(ii) define a rational
# polyhedron in (w, μ), so rational weights arbitrarily close to w still admit
# a certificate, dropping any branch whose mass becomes 0.) Weights must be
# positive, and S a minimal separator (so it has a full component).
# If every component of G - S is certified, S is safe. The minor test is the
# case of one branch per label, whose vertices are no lighter than it.
#
# Search (greedy). Orient every missing edge from its lighter end to its
# heavier end. Every label with an incoming edge is a *target*, and gets a
# backbone: free vertices of R with capacity at least its weight, claimed by
# one breadth-first search from all the targets at once, heaviest first. Each branch of a label will contain its
# whole backbone, which costs w(v) on each of its vertices. An edge whose
# ends have touching backbones is covered. Every other edge u → v requires
# every branch of u to touch the backbone of v. A label u with such edges
# routes *fingers*: a breadth-first search from its backbone, through
# vertices with capacity left, to a vertex that touches the backbone of a
# target not yet touched, repeated until every target is touched. The finger
# takes the mass min(need, least capacity left on it), and u's branch is its
# backbone plus the finger. Fingers are routed, lightest label first, until
# the mass w(u) is placed; if no finger is found, the test fails. If it
# fails, it is tried again with backbones for every label.
#
# Region. A vertex of R has no neighbor in C (it would be in N), so a search
# that leaves N into R and never re-enters N stays in R: only the neighbors
# of the labels need a membership test.
#
# On the line graph of an MPS chain, the bonds of a cut lie on three rails
# (ket, bra, MPO) that are joined only by 1-bit physical indices, so no
# minor of the minor test exists. The fingers reach the rails through
# different physical indices, at different sites, which the blow-up allows.

mutable struct ACSDUnit{V, W}
    cid::Vector{Int}       # component of G - S (by tag), or -1 in S (standalone use)
    ctag::Vector{Int}
    capw::Vector{W}        # capacity left
    captag::Vector{Int}
    owner::Vector{Int}     # label whose backbone contains a vertex
    otag::Vector{Int}
    seen::Vector{Int}
    prev::Vector{V}
    ftag::Vector{Int}      # in the current finger
    queue::Vector{V}
    stack::Vector{V}
    comp::Vector{V}
    N::Vector{V}
    lidx::Vector{Int}      # label index of a vertex of N (by tag)
    ltag::Vector{Int}
    bbs::Vector{Vector{V}} # backbones
    tag::Int
    budget::Int
end

function ACSDUnit{V, W}(n::Integer) where {V, W}
    return ACSDUnit{V, W}(zeros(Int, n), zeros(Int, n), zeros(W, n), zeros(Int, n),
        zeros(Int, n), zeros(Int, n), zeros(Int, n), zeros(V, n), zeros(Int, n),
        V[], V[], V[], V[], zeros(Int, n), zeros(Int, n), Vector{V}[], 0, 0)
end

@inline function unit_newtag!(us::ACSDUnit)
    us.tag += 1
    return us.tag
end

# Standalone: is S (vertices) safe by the unit-minor test? The components of
# G - S are found by breadth-first search. `budget` bounds the arcs scanned
# by the search for certificates. With `bound = true`, only the necessary
# condition is checked: false means that some component has no certificate.
# With `percomp`, every component is tried and its outcome recorded.
function acsd_unitsafe!(us::ACSDUnit{V, W}, weights::AbstractVector{W}, graph::AbstractGraph{V}, Svert::AbstractVector{V}; budget::Int = 1_000_000, bound::Bool = false, percomp::Union{Nothing, Vector{Tuple{Vector{V}, Bool}}} = nothing) where {V, W}
    us.budget = budget
    ct = unit_newtag!(us)

    for s in Svert
        us.ctag[s] = ct; us.cid[s] = -1
    end

    ncomp = 0; allok = true

    for x in vertices(graph)
        us.ctag[x] == ct && continue
        ncomp += 1; c = ncomp
        empty!(us.comp); push!(us.comp, x); us.ctag[x] = ct; us.cid[x] = c
        head = 1

        while head <= length(us.comp)
            y = us.comp[head]; head += 1

            for z in neighbors(graph, y)
                if us.ctag[z] != ct
                    us.ctag[z] = ct; us.cid[z] = c; push!(us.comp, z)
                end
            end
        end

        # N = N(C)
        Nv = V[]; nt = unit_newtag!(us)

        for y in us.comp, z in neighbors(graph, y)
            if us.cid[z] == -1 && us.ltag[z] != nt
                us.ltag[z] = nt; push!(Nv, z)
            end
        end

        sort!(Nv)
        inC = let us = us, ct = ct, c = c
            x -> us.ctag[x] == ct && us.cid[x] == c
        end
        ok = unit_component!(us, weights, graph, Nv, inC, bound)

        if isnothing(percomp)
            ok || return false
        else
            push!(percomp, (copy(us.comp), ok)); allok &= ok
        end
    end

    return allok
end

# In the solver: the components of G - S for S = sep(j) are read off the clique
# tree, as by the minor test (`acsd_components!` must have been called for j).
function acsd_unitsafe!(us::ACSDUnit{V, W}, at::ACSDTree{V}, comps::Vector{Tuple{V, V}}, index::AbstractVector{V}, weights::AbstractVector{W}, graph::AbstractGraph{V}, order::AbstractVector{V}, tree::CliqueTree{V}; budget::Int = 1_000_000) where {V, W}
    us.budget = budget
    stag = at.tag[]

    for (top, q) in comps
        Nq = separator(tree, q)
        length(Nq) < 2 && continue
        Nv = V[order[i] for i in Nq]; sort!(Nv)
        inC = let at = at, index = index, top = top, stag = stag
            x -> (i = index[x]; at.smark[i] != stag && acsd_top!(at, at.home[i]) == top)
        end
        unit_component!(us, weights, graph, Nv, inC, false) || return false
    end

    return true
end

# Certify the component C with N = N(C) (sorted), given by its membership test.
function unit_component!(us::ACSDUnit{V, W}, weights::AbstractVector{W}, graph::AbstractGraph{V}, Nv::Vector{V}, inC::F, bound::Bool) where {V, W, F}
    lt = unit_newtag!(us); N = us.N; empty!(N); append!(N, Nv)
    k = length(N)

    for (i, v) in enumerate(N)
        us.ltag[v] = lt; us.lidx[v] = i
    end

    # missing edges, oriented lighter → heavier
    adjN = falses(k, k)

    for (i, v) in enumerate(N), z in neighbors(graph, v)
        us.ltag[z] == lt && (adjN[i, us.lidx[z]] = true)
    end

    outs = [Int[] for _ in 1:k]
    istarget = falses(k)
    anymissing = false

    for i in 1:k, j in i + 1:k
        adjN[i, j] && continue
        anymissing = true
        u, v = N[i], N[j]
        a, b = (weights[u], u) < (weights[v], v) ? (i, j) : (j, i)
        push!(outs[a], b); istarget[b] = true
    end

    anymissing || return true
    unit_obstructed(us, weights, graph, lt, inC, adjN) && return false
    bound && return true
    # first grow backbones for the targets only, then for every label
    unit_try!(us, weights, graph, lt, inC, outs, istarget, false) && return true
    us.budget < 0 && return false
    return unit_try!(us, weights, graph, lt, inC, outs, istarget, true)
end

# A necessary condition for any certificate (not only the greedy one). Let
# a(x) = w(N(x) ∩ R). For a missing edge {u, v}: if some branch of u is {u}
# alone, every branch of v contains a neighbor of u in R (u ∉ R, v ≁ u), and
# a neighbor of v in R, so a(u) ≥ w(v) and a(v) ≥ w(v); otherwise every
# branch of u contains a neighbor of u in R, so a(u) ≥ w(u). Returns true if
# some missing edge violates both alternatives (in both directions), or if
# the stronger test below fails.
function unit_obstructed(us::ACSDUnit{V, W}, weights::AbstractVector{W}, graph::AbstractGraph{V}, lt::Int, inC::F, adjN::BitMatrix) where {V, W, F}
    tol = tolerance(W); N = us.N; k = length(N)
    a = zeros(W, k)

    for (i, v) in enumerate(N), y in neighbors(graph, v)
        (us.ltag[y] != lt && !inC(y)) && (a[i] += weights[y])
    end

    for i in 1:k, j in i + 1:k
        adjN[i, j] && continue
        wu = weights[N[i]]; wv = weights[N[j]]
        au = a[i] > wu - tol; av = a[j] > wv - tol
        # from u: au, or (av and a(u) ≥ w(v)); from v: av, or (au and a(v) ≥ w(u))
        okfromu = au || (av && a[i] > wv - tol)
        okfromv = av || (au && a[j] > wu - tol)
        (okfromu && okfromv) || return true
    end

    # Stronger, for small N. The labels X with no branch {x} alone cover the
    # missing edges (two such branches would not touch). Every branch of x ∈ X
    # holds a unit of a neighbor of x in R, and branches share no units: the
    # weights of X must be routable into the neighbors in R, each q taking at
    # most w(q) (a transportation problem). A label y ∉ X has a branch {y}, so
    # every branch of each x with {x, y} missing holds a neighbor of y:
    # w(x) ≤ a(y). If no minimal cover X passes both, there is no certificate.
    k <= 12 || return false
    nbrR = [V[] for _ in 1:k]

    for (i, v) in enumerate(N), y in neighbors(graph, v)
        (us.ltag[y] != lt && !inC(y)) && push!(nbrR[i], y)
    end

    # bit masks: mis[i] = the labels j with {i, j} missing; a label y with
    # w(x) > a(y) for some missing {x, y} can have no branch {y}, so must be in X
    mis = zeros(Int, k); bad = 0

    for i in 1:k, j in 1:k
        if i != j && !adjN[i, j]
            mis[i] |= 1 << (j - 1)
            weights[N[j]] >= a[i] + tol && (bad |= 1 << (i - 1))
        end
    end

    # The transportation test only gets easier as X shrinks, and the test on
    # the labels outside X is bad ⊆ X: so it suffices to try the covers X ⊇ bad
    # that are minimal among those, where every i ∈ X - bad has a missing edge
    # to a label outside X.
    for X in 0:(1 << k) - 1
        bad & ~X == 0 || continue
        # X is a cover iff mis[i] ⊆ X for every i ∉ X
        cover = true

        for i in 1:k
            inX = (X >> (i - 1)) & 1 == 1
            inbad = (bad >> (i - 1)) & 1 == 1

            if (!inX && mis[i] & ~X != 0) || (inX && !inbad && mis[i] & ~X == 0)
                cover = false; break
            end
        end

        cover && unit_transport(weights, N, nbrR, X, k, tol) && return false
    end

    return true
end

# Can every label x in X send w(x) into its neighbors in R, each neighbor q
# receiving at most w(q)? (Max flow by augmenting paths on the bipartite graph.)
function unit_transport(weights::AbstractVector{W}, N::Vector{V}, nbrR::Vector{Vector{V}}, X::Int, k::Int, tol::W) where {V, W}
    labels = [i for i in 1:k if (X >> (i - 1)) & 1 == 1]
    qs = unique!(sort!(reduce(vcat, (nbrR[i] for i in labels); init = V[])))
    qi = Dict(q => t for (t, q) in enumerate(qs))
    nl = length(labels); nq = length(qs)
    flow = zeros(W, nl, nq); used = zeros(W, nq); sent = zeros(W, nl)
    adj = [[qi[q] for q in nbrR[i]] for i in labels]

    while true
        # BFS in the residual graph from labels with demand left to a q with room
        prevq = zeros(Int, nq); prevl = zeros(Int, nl); seenl = falses(nl); seenq = falses(nq)
        queue = Int[]

        for l in 1:nl
            if weights[N[labels[l]]] - sent[l] > tol / 2
                seenl[l] = true; push!(queue, l)
            end
        end

        isempty(queue) && return true
        hit = 0; head = 1

        while head <= length(queue) && iszero(hit)
            l = queue[head]; head += 1

            for t in adj[l]
                seenq[t] && continue
                seenq[t] = true; prevq[t] = l

                if weights[qs[t]] - used[t] > tol / 2
                    hit = t; break
                end

                # go back along an arc with flow
                for l2 in 1:nl
                    if !seenl[l2] && flow[l2, t] > tol / 2
                        seenl[l2] = true; prevl[l2] = t; push!(queue, l2)
                    end
                end
            end
        end

        iszero(hit) && return false
        # bottleneck
        t = hit; l = prevq[t]; delta = weights[qs[t]] - used[t]

        while true
            if iszero(prevl[l])
                delta = min(delta, weights[N[labels[l]]] - sent[l]); break
            end

            t2 = prevl[l]; delta = min(delta, flow[l, t2]); l = prevq[t2]
        end

        # augment
        t = hit; used[t] += delta

        while true
            l = prevq[t]; flow[l, t] += delta

            if iszero(prevl[l])
                sent[l] += delta; break
            end

            t2 = prevl[l]; flow[l, t2] -= delta; t = t2
        end
    end
end

function unit_try!(us::ACSDUnit{V, W}, weights::AbstractVector{W}, graph::AbstractGraph{V}, lt::Int, inC::F, outs0::Vector{Vector{Int}}, istarget::BitVector, allbb::Bool) where {V, W, F}
    tol = tolerance(W); N = us.N; k = length(N)
    outs = [copy(o) for o in outs0]
    ot = unit_newtag!(us); kt = unit_newtag!(us)

    # y ∈ R, reached from x ∈ N ∪ R? From a vertex of R, every neighbor not in
    # N is in R; from a label, test membership in C.
    @inline isN(y) = us.ltag[y] == lt
    @inline nextR(x, y) = !isN(y) && (!isN(x) || !inC(y))

    @inline function cap(x)
        if us.captag[x] != kt
            us.captag[x] = kt; us.capw[x] = weights[x]
        end

        return us.capw[x]
    end

    @inline owner(x) = us.otag[x] == ot ? us.owner[x] : (isN(x) ? us.lidx[x] : 0)

    # backbones: one breadth-first search from all the targets (or all the
    # labels) at once, heaviest first, so that each claims the free vertices
    # nearest to it with capacity at least its weight
    while length(us.bbs) < k
        push!(us.bbs, V[])
    end

    bbs = us.bbs

    for i in 1:k
        empty!(bbs[i]); push!(bbs[i], N[i])
    end

    tgts = [i for i in 1:k if istarget[i] || (allbb && !isempty(outs[i]))]
    sort!(tgts; by = i -> (-weights[N[i]], N[i]))
    q = us.queue; empty!(q)

    for i in tgts
        push!(q, N[i])
    end

    head = 1

    while head <= length(q)
        x = q[head]; head += 1
        i = owner(x); wv = weights[N[i]]

        for y in neighbors(graph, x)
            us.budget -= 1

            if nextR(x, y) && us.otag[y] != ot && cap(y) >= wv
                us.otag[y] = ot; us.owner[y] = i; us.capw[y] -= wv
                push!(q, y); push!(bbs[i], y)
            end
        end
    end

    us.budget < 0 && return false

    # touches(B_i, B_j)?
    function bbtouch(i::Int, j::Int)
        A = length(bbs[i]) <= length(bbs[j]) ? i : j
        B = A == i ? j : i

        for x in bbs[A], y in neighbors(graph, x)
            us.budget -= 1
            owner(y) == B && return true
        end

        return false
    end

    for i in 1:k
        filter!(j -> !bbtouch(i, j), outs[i])
    end

    us.budget < 0 && return false

    # fingers, lightest label first
    srcs = [i for i in 1:k if !isempty(outs[i])]
    sort!(srcs; by = i -> (weights[N[i]], N[i]))
    rem = falses(k)

    for i in srcs
        need = weights[N[i]]

        # (place the mass exactly: shortfalls within the tolerance would add up
        # over the labels of a bag)
        while ispositive(need)
            # a new finger
            ft = unit_newtag!(us)
            fill!(rem, false)

            for j in outs[i]
                rem[j] = true
            end

            nrem = length(outs[i])
            fverts = us.stack; empty!(fverts)

            while nrem > 0
                st = unit_newtag!(us)
                empty!(q); head = 1

                # sources: the backbone of i and the finger so far
                for src in (bbs[i], fverts), x in src, y in neighbors(graph, x)
                    us.budget -= 1

                    if nextR(x, y) && us.ftag[y] != ft && owner(y) != i && us.seen[y] != st && cap(y) >= tol
                        us.seen[y] = st; us.prev[y] = zero(V); push!(q, y)
                    end
                end

                hit = zero(V)

                while head <= length(q) && iszero(hit)
                    y = q[head]; head += 1

                    # does y touch a remaining target? (y ∈ R, so its
                    # neighbors are in R ∪ N)
                    o = owner(y)

                    if ispositive(o) && rem[o]
                        hit = y; break
                    end

                    for z in neighbors(graph, y)
                        us.budget -= 1
                        o = owner(z)

                        if ispositive(o) && rem[o]
                            hit = y; break
                        end
                    end

                    iszero(hit) || break

                    for z in neighbors(graph, y)
                        if !isN(z) && us.ftag[z] != ft && owner(z) != i && us.seen[z] != st && cap(z) >= tol
                            us.seen[z] = st; us.prev[z] = y; push!(q, z)
                        end
                    end
                end

                (iszero(hit) || us.budget < 0) && return false

                # add the path to the finger, and drop the targets it touches
                y = hit

                while !iszero(y)
                    us.ftag[y] = ft; push!(fverts, y)

                    for z in Iterators.flatten(((y,), neighbors(graph, y)))
                        o = owner(z)

                        if ispositive(o) && rem[o]
                            rem[o] = false; nrem -= 1
                        end
                    end

                    y = us.prev[y]
                end
            end

            # place the finger
            mu = need

            for y in fverts
                mu = min(mu, cap(y))
            end

            ispositive(mu) || return false

            for y in fverts
                us.capw[y] -= mu
            end

            need -= mu
        end
    end

    return true
end
