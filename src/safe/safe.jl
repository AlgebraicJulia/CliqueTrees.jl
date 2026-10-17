# ---------------------------------------------------------------------------
# Safe separators
#
# The solver splits a graph by safe separators, recursively. It merges the
# true twins of a graph, fills the safe separators of a minimal ordering of
# it (see acsd.jl), splits the filled graph into its atoms, and splits the
# torso of each atom (the atom, with its separators filled) in turn. A graph
# that is one atom is a *leaf*.
#
# The leaves are the torsos of the atoms of the *safe completion* G* of the
# input graph G: G with the separators of every level filled. Splitting a
# graph by its clique minimal separators ends at its atoms, in any order,
# and merging true twins makes no clique separator. The treewidth of G* is
# that of G, which is the largest treewidth of a leaf.
#
# The solver works depth-first on one global graph. A *node* is a set W of
# vertices of G, with a clique K ⊆ W: its separator with its parent. Its
# graph is G[W] + F[W], where F is the fill of its ancestors, with the
# twin classes of its ancestors merged: the classes of a node are unions of
# the classes of its parent. The graph of a node is realized when the node
# is reached, and its twins are merged; its vertices are numbered in the
# order of the input, so that ties are broken the same way at every level.
# The atoms of a node are pushed in postorder of its atom tree, rooted at an
# atom that contains K, so that the leaves come out in postorder: every leaf
# before the leaves that contain its separator. A leaf is ordered with its
# separator last (see safe_pull), so the orderings of the leaves, minus
# their separators, are stacked.
#
# A filled separator of a node needs only its missing edges, between merged
# vertices of the node, so the fill F is a stack of arcs between
# representatives: an arc between two vertices stands for every arc between
# their classes. These are kept in a linked list for each vertex.
#
# The state of the solver is a collection of arrays and scalars.
#
#   - the graph G (`graph00`) and its weights (`weights00`)
#   - the fill F: arc q goes from `fsrc[q]` to `ftgt[q]`, the next (older)
#     arc of its source is `fnext[q]`, and the newest arc of a vertex v is
#     `fhead[v]` (zero: none); there are `nfill` arcs
#   - the twin classes: a union-find with rollback; `ufp` is the parent of
#     each vertex (a root is its own parent), and `mlog[1:nlog]` are the
#     roots relinked so far, in order
#   - the pending nodes: the vertices of node i are `wtgt[wptr[i]:wptr[i + 1] - 1]`
#     (sorted), and its separator is `ktgt[kptr[i]:kptr[i + 1] - 1]`; the
#     nodes of level l are `nbeg[l]:nend[l]`, the next of them is `nnxt[l]`,
#     and `fmark[l]`, `lmark[l]` are `nfill`, `nlog` when it was entered;
#     the current level is `level`
#   - the leaves found so far (see SafeResult), with the weights and graphs
#     of those that are not cliques
#
# The realized graph of a node (V10) has a vertex for each class of the
# node, and is compressed (V20):
#
#   V00  →  V10  →  V20
#    vclass   project20
# ---------------------------------------------------------------------------

mutable struct SafeSolver{W <: Number, V <: Integer, E <: Integer, M <: Tuple}
    graph00    ::FBipartiteGraph{V, E}
    weights00  ::FVector{W}
    mins       ::M
    maxmissing ::Int
    fsrc       ::Vector{V}
    ftgt       ::Vector{V}
    fnext      ::Vector{Int}
    fhead      ::FVector{Int}
    nfill      ::Int
    ufp        ::FVector{V}
    mlog       ::Vector{V}
    nlog       ::Int
    wptr       ::Vector{Int}
    wtgt       ::Vector{V}
    kptr       ::Vector{Int}
    ktgt       ::Vector{V}
    nbeg       ::Vector{Int}
    nend       ::Vector{Int}
    nnxt       ::Vector{Int}
    fmark      ::Vector{Int}
    lmark      ::Vector{Int}
    level      ::Int
    leafptr    ::Vector{Int}
    prjptr     ::Vector{Int}
    prjtgt     ::Vector{V}
    sepptr     ::Vector{Int}
    septgt     ::Vector{V}
    lgraph     ::Vector{Int}
    weights30  ::Vector{FVector{W}}
    graphs30   ::Vector{FBipartiteGraph{V, E}}
    width30    ::W
    tag        ::Int
    stamp      ::FVector{Int}
    rmark      ::FVector{Int}
    rid        ::FVector{V}
    vclass     ::FVector{V}
    marker     ::FVector{V}
    lblptr     ::FVector{V}
    lbltgt     ::FVector{V}
    ptr10      ::FVector{E}
    tgt10      ::Vector{V}
    weights10  ::FVector{W}
    work00     ::FScalar{V}
    work01     ::FVector{V}
    work02     ::FVector{V}
    work03     ::FVector{V}
    work04     ::FVector{V}
    work05     ::FVector{V}
    work06     ::FVector{V}
    ptr20      ::FVector{E}
    tgt20      ::Vector{V}
    weights20  ::FVector{W}
end

# The leaves of a graph, in postorder: the classes of leaf k are
# `leafptr[k]:leafptr[k + 1] - 1`, the vertices of class c are
# `prjtgt[prjptr[c]:prjptr[c + 1] - 1]`, and the separator of leaf k is
# `septgt[sepptr[k]:sepptr[k + 1] - 1]`. Leaf k is ordered by the ordering
# of graph `lgraph[k]` or, if that is zero, it is a clique (one class). The
# largest weight of a clique, `width`, is a lower bound on the width.
struct SafeResult{W <: Number, V <: Integer}
    leafptr ::Vector{Int}
    prjptr  ::Vector{Int}
    prjtgt  ::Vector{V}
    sepptr  ::Vector{Int}
    septgt  ::Vector{V}
    lgraph  ::Vector{Int}
    width   ::W
    n       ::V
end

function SafeSolver(weights::AbstractVector{W}, graph::AbstractGraph{V}, mins::Tuple, maxmissing::Integer) where {W <: Number, V <: Integer}
    E = etype(graph); n = nv(graph); m = de(graph); nn = n + one(V)

    ptr00 = FVector{E}(undef, nn)
    tgt00 = FVector{V}(undef, m)
    graph00 = simplegraph!(ptr00, tgt00, graph)
    weights00 = FVector{W}(undef, n)
    fhead = FVector{Int}(undef, n)
    ufp = FVector{V}(undef, n)

    @inbounds for v in oneto(n)
        weights00[v] = weights[v]
        fhead[v] = 0
        ufp[v] = v
    end

    stamp = FVector{Int}(undef, n)
    rmark = FVector{Int}(undef, n)

    @inbounds for v in oneto(n)
        stamp[v] = 0
        rmark[v] = 0
    end

    # the root: every vertex, with no separator
    wtgt = collect(oneto(n))

    return SafeSolver{W, V, E, typeof(mins)}(
        graph00, weights00, mins, convert(Int, maxmissing),
        V[], V[], Int[], fhead, 0,
        ufp, V[], 0,
        Int[1, n + 1], wtgt, Int[1, 1], V[],
        Int[1], Int[1], Int[1], Int[0], Int[0], 1,
        Int[1], Int[1], V[], Int[1], V[], Int[],
        FVector{W}[], FBipartiteGraph{V, E}[], zero(W),
        0, stamp, rmark,
        FVector{V}(undef, n), FVector{V}(undef, n), FVector{V}(undef, n),
        FVector{V}(undef, nn), FVector{V}(undef, n),
        FVector{E}(undef, nn), V[], FVector{W}(undef, n),
        FScalar{V}(undef), FVector{V}(undef, nn), FVector{V}(undef, n),
        FVector{V}(undef, n), FVector{V}(undef, n),
        FVector{V}(undef, n), FVector{V}(undef, n),
        FVector{E}(undef, nn), V[], FVector{W}(undef, n))
end

function SafeResult(s::SafeSolver{W, V}) where {W, V}
    return SafeResult{W, V}(s.leafptr, s.prjptr, s.prjtgt, s.sepptr, s.septgt, s.lgraph, s.width30, nv(s.graph00))
end

# Split a graph into the leaves of its safe completion: returns the weights
# and graphs of the leaves that are not cliques, and the result.
function safesolve(weights::AbstractVector, graph::AbstractGraph, mins::Tuple, maxmissing::Integer; verbose::Bool = false)
    return safe_solve!(SafeSolver(weights, graph, mins, maxmissing); verbose)
end

function safe_solve!(s::SafeSolver; verbose::Bool = false)
    if verbose
        println()
        println("| iter | level |       n |       m | leaf |")
        println("|------|-------|---------|---------|------|")
    end

    iter = 0

    while true
        step = safe_step!(s)
        isnothing(step) && break
        iter += 1

        if verbose
            level, n, m, leaf = step
            println("| ", lpad(iter, 4), " | ", lpad(level, 5), " | ", lpad(n, 7), " | ", lpad(m, 7), " | ", lpad(leaf ? "✓" : "", 4), " |")
        end
    end

    return s.weights30, s.graphs30, SafeResult(s)
end

# Process the next pending node: a leaf, or the atoms of a split. Returns its
# level, the size of its compressed graph, and whether it is a leaf, or nothing
# if no node is left.
function safe_step!(s::SafeSolver)
    while ispositive(s.level) && s.nnxt[s.level] > s.nend[s.level]
        safe_pop!(s)
    end

    iszero(s.level) && return nothing
    node = s.nnxt[s.level]; s.nnxt[s.level] = node + 1
    level = s.level

    graph10, nc = safe_realize!(s, node)
    graph20, project20 = safe_compress!(s, graph10)
    compressweights_impl!(s.weights20, s.weights10, project20)
    n20 = nv(graph20); m20 = half(ne(graph20))

    if n20 > one(n20)
        weights20 = view(s.weights20, oneto(n20))
        graph22, order, tree = acsd(weights20, graph20, s.mins, s.maxmissing)
        order, tree = atomtree!(order, tree, graph22)

        if length(tree) > 1
            safe_split!(s, node, graph20, project20, graph22, order, tree)
            return level, n20, m20, false
        end
    end

    safe_leaf!(s, node, graph20, project20)
    return level, n20, m20, true
end

# Leave the current level: undo its fill and its merges, and drop its nodes.
function safe_pop!(s::SafeSolver)
    l = s.level

    @inbounds for q in s.nfill:-1:s.fmark[l] + 1
        s.fhead[s.fsrc[q]] = s.fnext[q]
    end

    s.nfill = s.fmark[l]

    @inbounds for p in s.nlog:-1:s.lmark[l] + 1
        r = s.mlog[p]; s.ufp[r] = r
    end

    s.nlog = s.lmark[l]

    nbeg = s.nbeg[l]
    resize!(s.wtgt, s.wptr[nbeg] - 1); resize!(s.wptr, nbeg)
    resize!(s.ktgt, s.kptr[nbeg] - 1); resize!(s.kptr, nbeg)
    s.level = l - 1
    return
end

# Realize the graph of a node (V10): a vertex for each class of the node,
# numbered by its first vertex, with the total weight of the class. Two
# classes are adjacent if a vertex of one is adjacent to a vertex of the
# other in G, or by an arc of the fill. On return, `vclass` maps each vertex
# of the node to its class, and `lblptr`, `lbltgt` list the vertices of each
# class, in increasing order.
function safe_realize!(s::SafeSolver{W, V, E}, node::Int) where {W, V, E}
    graph = s.graph00; weights = s.weights00
    stamp = s.stamp; rmark = s.rmark; rid = s.rid; vclass = s.vclass
    lblptr = s.lblptr; lbltgt = s.lbltgt; marker = s.marker
    ptr10 = s.ptr10; tgt10 = s.tgt10; weights10 = s.weights10
    wstrt = s.wptr[node]; wstop = s.wptr[node + 1] - 1

    s.tag += 1; t = s.tag; nc = zero(V)

    # the classes
    @inbounds for p in wstrt:wstop
        v = s.wtgt[p]; stamp[v] = t
        r = ufind(s.ufp, v)

        if rmark[r] != t
            rmark[r] = t; nc += one(V); rid[r] = nc
            lblptr[nc + one(V)] = zero(V)
        end

        vclass[v] = rid[r]
    end

    @inbounds lblptr[1] = one(V)

    @inbounds for p in wstrt:wstop
        c = vclass[s.wtgt[p]]
        lblptr[c + one(V)] += one(V)
    end

    @inbounds for c in oneto(nc)
        lblptr[c + one(V)] += lblptr[c]
        weights10[c] = zero(W)
        marker[c] = zero(V)
    end

    @inbounds for p in wstrt:wstop
        v = s.wtgt[p]; c = vclass[v]
        i = lblptr[c]; lbltgt[i] = v; lblptr[c] = i + one(V)
        weights10[c] += weights[v]
    end

    @inbounds for c in reverse(oneto(nc))
        lblptr[c + one(V)] = lblptr[c]
    end

    @inbounds lblptr[1] = one(V)

    # the arcs: the neighbors of every vertex of each class, in G and in F
    p = one(E)
    @inbounds ptr10[1] = p

    @inbounds for c in oneto(nc)
        for i in lblptr[c]:lblptr[c + one(V)] - one(V)
            v = lbltgt[i]

            for w in neighbors(graph, v)
                if stamp[w] == t
                    d = vclass[w]

                    if d != c && marker[d] != c
                        marker[d] = c
                        p > length(tgt10) && resize!(tgt10, max(16, twice(length(tgt10))))
                        tgt10[p] = d; p += one(E)
                    end
                end
            end

            q = s.fhead[v]

            while ispositive(q)
                w = s.ftgt[q]

                if stamp[w] == t
                    d = vclass[w]

                    if d != c && marker[d] != c
                        marker[d] = c
                        p > length(tgt10) && resize!(tgt10, max(16, twice(length(tgt10))))
                        tgt10[p] = d; p += one(E)
                    end
                end

                q = s.fnext[q]
            end
        end

        ptr10[c + one(V)] = p
    end

    return BipartiteGraph(nc, nc, p - one(E), ptr10, tgt10), nc
end

# Merge the true twins of the realized graph of a node (V10 → V20). The
# classes are numbered by their first vertex, so that the order of the
# vertices is kept: it is the order of the input. On return, `work02` maps
# each vertex of V10 to its class. Returns the compressed graph and the
# projection from V20 to V10.
function safe_compress!(s::SafeSolver{W, V, E}, graph10::AbstractGraph{V}) where {W, V, E}
    n10 = nv(graph10); marker = s.marker; svar = s.work02; newid = s.work04
    partition = twins_impl!(s.work00, s.work01, svar, s.work03, newid, graph10, Val(:T))
    n20 = nv(partition)

    @inbounds for x in oneto(n20)
        newid[x] = zero(V)
        marker[x] = zero(V)
    end

    # number the classes by their first vertex
    x = zero(V)

    @inbounds for i in oneto(n10)
        j = svar[i]

        if iszero(newid[j])
            x += one(V); newid[j] = x
        end
    end

    @inbounds for i in oneto(n10)
        svar[i] = newid[svar[i]]
    end

    # the projection V20 → V10: the vertices of each class, in increasing order
    prjptr = s.work01; prjtgt = s.work03

    @inbounds for x in oneto(n20 + one(V))
        prjptr[x] = zero(V)
    end

    @inbounds for i in oneto(n10)
        prjptr[svar[i] + one(V)] += one(V)
    end

    @inbounds prjptr[1] = one(V)

    @inbounds for x in oneto(n20)
        prjptr[x + one(V)] += prjptr[x]
    end

    @inbounds for i in oneto(n10)
        x = svar[i]; p = prjptr[x]
        prjtgt[p] = i; prjptr[x] = p + one(V)
    end

    @inbounds for x in reverse(oneto(n20))
        prjptr[x + one(V)] = prjptr[x]
    end

    @inbounds prjptr[1] = one(V)
    project20 = BipartiteGraph(n10, n20, n10, prjptr, prjtgt)

    # the compressed graph: the neighbors of the first vertex of each class
    if length(s.tgt20) < ne(graph10)
        resize!(s.tgt20, ne(graph10))
    end

    ptr20 = s.ptr20; tgt20 = s.tgt20; p = one(E)
    @inbounds ptr20[1] = p

    @inbounds for x in oneto(n20)
        i = prjtgt[prjptr[x]]

        for j in neighbors(graph10, i)
            y = svar[j]

            if y != x && marker[y] != x
                marker[y] = x; tgt20[p] = y; p += one(E)
            end
        end

        ptr20[x + one(V)] = p
    end

    return BipartiteGraph(n20, n20, p - one(E), ptr20, tgt20), project20
end

# The first vertex of a vertex of V20.
@inline function saferep(s::SafeSolver, project20::AbstractGraph, x::Integer)
    @inbounds c = first(neighbors(project20, x))
    @inbounds return s.lbltgt[s.lblptr[c]]
end

# Record a leaf: its classes (the vertices of V20), its separator, and its
# graph if it is not a clique.
function safe_leaf!(s::SafeSolver{W, V, E}, node::Int, graph20::AbstractGraph{V}, project20::AbstractGraph{V}) where {W, V, E}
    n20 = nv(graph20)

    @inbounds for x in oneto(n20)
        for c in neighbors(project20, x)
            append!(s.prjtgt, view(s.lbltgt, s.lblptr[c]:s.lblptr[c + one(V)] - one(V)))
        end

        push!(s.prjptr, length(s.prjtgt) + 1)
    end

    push!(s.leafptr, length(s.prjptr))
    append!(s.septgt, view(s.ktgt, s.kptr[node]:s.kptr[node + 1] - 1))
    push!(s.sepptr, length(s.septgt) + 1)

    if n20 > one(V)
        m20 = ne(graph20)
        ptr = FVector{E}(undef, n20 + one(V))
        tgt = FVector{V}(undef, m20)
        wgt = FVector{W}(undef, n20)

        @inbounds for x in oneto(n20 + one(V))
            ptr[x] = pointers(graph20)[x]
        end

        @inbounds for p in oneto(m20)
            tgt[p] = targets(graph20)[p]
        end

        @inbounds for x in oneto(n20)
            wgt[x] = s.weights20[x]
        end

        push!(s.graphs30, BipartiteGraph(n20, n20, m20, ptr, tgt))
        push!(s.weights30, wgt)
        push!(s.lgraph, length(s.graphs30))
    else
        push!(s.lgraph, 0)
        isone(n20) && (s.width30 = max(s.width30, s.weights20[1]))
    end

    return
end

# Split a node: push its fill and its merges, and its atoms in postorder of
# its atom tree, rooted at an atom that contains its separator K. graph22 is
# graph20 with its safe separators filled, and order and tree are its atom
# tree.
function safe_split!(
        s::SafeSolver{W, V, E},
        node::Int,
        graph20::AbstractGraph{V},
        project20::AbstractGraph{V},
        graph22::AbstractGraph{V},
        order::AbstractVector{V},
        tree::CliqueTree{V},
    ) where {W, V, E}
    n20 = nv(graph20); t = length(tree)
    marker = s.marker
    l = s.level + 1

    if length(s.nbeg) < l
        push!(s.nbeg, 0); push!(s.nend, 0); push!(s.nnxt, 0)
        push!(s.fmark, 0); push!(s.lmark, 0)
    end

    s.fmark[l] = s.nfill; s.lmark[l] = s.nlog

    # the fill: the arcs of graph22 that are not in graph20
    @inbounds for x in oneto(n20)
        marker[x] = zero(V)
    end

    @inbounds for x in oneto(n20)
        for y in neighbors(graph20, x)
            marker[y] = x
        end

        for y in neighbors(graph22, x)
            if x < y && marker[y] != x
                a = saferep(s, project20, x); b = saferep(s, project20, y)
                safe_arc!(s, a, b); safe_arc!(s, b, a)
            end
        end
    end

    # the merges: the classes of each vertex of V20 become one
    @inbounds for x in oneto(n20)
        r0 = zero(V)

        for c in neighbors(project20, x)
            r = ufind(s.ufp, s.lbltgt[s.lblptr[c]])

            if iszero(r0)
                r0 = r
            elseif r != r0
                s.ufp[r] = r0
                s.nlog += 1
                s.nlog > length(s.mlog) && resize!(s.mlog, max(16, twice(length(s.mlog))))
                s.mlog[s.nlog] = r
            end
        end
    end

    # the atom of each vertex of V20, and the first descendant of each atom:
    # the atoms are in postorder, so the subtree of atom a is fdesc[a]:a
    atom = s.work04; fdesc = s.work05; path = s.work06

    @inbounds for a in oneto(t)
        fdesc[a] = a

        for i in residual(tree[a])
            atom[order[i]] = a
        end
    end

    @inbounds for a in oneto(t)
        j = parentindex(tree, a)
        isnothing(j) || (fdesc[j] = min(fdesc[j], fdesc[a]))
    end

    # the atom r that receives K: the deepest atom whose residual meets K.
    # The atoms whose residuals meet K are ancestors of an atom that
    # contains K, and the deepest of them, the first in postorder, is one.
    r = t + 1

    @inbounds for p in s.kptr[node]:s.kptr[node + 1] - 1
        r = min(r, atom[s.work02[s.vclass[s.ktgt[p]]]])
    end

    # push the atoms in postorder of the tree, rooted at r. Re-rooting
    # reverses the path from r to the root: if c is the child of b on it,
    # then b becomes a child of c, with the separator of c, and the subtree
    # of b, in postorder, becomes (the subtree of c) minus (the subtree of b),
    # in postorder, after its new child.
    s.nbeg[l] = s.nnxt[l] = length(s.wptr)

    if r > t
        @inbounds for a in oneto(t)
            safe_atom!(s, node, project20, order, tree, a, a)
        end
    else
        np = 1; path[np] = r

        @inbounds while true
            j = parentindex(tree, path[np])
            isnothing(j) && break
            np += 1; path[np] = j
        end

        # the other trees
        @inbounds for a in oneto(t)
            fdesc[path[np]] <= a <= path[np] && continue
            safe_atom!(s, node, project20, order, tree, a, a)
        end

        @inbounds for k in np:-1:2
            b = path[k]; c = path[k - 1]

            for a in fdesc[b]:fdesc[c] - 1
                safe_atom!(s, node, project20, order, tree, a, a)
            end

            for a in c + 1:b - 1
                safe_atom!(s, node, project20, order, tree, a, a)
            end

            safe_atom!(s, node, project20, order, tree, b, c)
        end

        @inbounds for a in fdesc[r]:r - 1
            safe_atom!(s, node, project20, order, tree, a, a)
        end

        safe_atom!(s, node, project20, order, tree, r, 0)
    end

    s.nend[l] = length(s.wptr) - 1
    s.level = l
    return
end

# Push an atom a of a split: its vertices (sorted), and its separator: the
# separator of atom b or, if b is zero, the separator K of the node.
function safe_atom!(
        s::SafeSolver{W, V},
        node::Int,
        project20::AbstractGraph{V},
        order::AbstractVector{V},
        tree::CliqueTree{V},
        a::Integer,
        b::Integer,
    ) where {W, V}
    strt = length(s.wtgt) + 1

    @inbounds for i in tree[a], c in neighbors(project20, order[i])
        append!(s.wtgt, view(s.lbltgt, s.lblptr[c]:s.lblptr[c + one(V)] - one(V)))
    end

    sort!(view(s.wtgt, strt:length(s.wtgt)))
    push!(s.wptr, length(s.wtgt) + 1)

    if iszero(b)
        @inbounds for p in s.kptr[node]:s.kptr[node + 1] - 1
            push!(s.ktgt, s.ktgt[p])
        end
    else
        @inbounds for i in separator(tree[b]), c in neighbors(project20, order[i])
            append!(s.ktgt, view(s.lbltgt, s.lblptr[c]:s.lblptr[c + one(V)] - one(V)))
        end
    end

    push!(s.kptr, length(s.ktgt) + 1)
    return
end

# Push an arc of the fill.
function safe_arc!(s::SafeSolver{W, V}, a::V, b::V) where {W, V}
    s.nfill += 1; q = s.nfill

    if q > length(s.fsrc)
        len = max(16, twice(length(s.fsrc)))
        resize!(s.fsrc, len); resize!(s.ftgt, len); resize!(s.fnext, len)
    end

    @inbounds s.fsrc[q] = a; s.ftgt[q] = b
    @inbounds s.fnext[q] = s.fhead[a]; s.fhead[a] = q
    return
end

# An ordering of the input graph, from an ordering of each graph: the
# ordering of each leaf is rotated (Liu's composite rotations) so that the
# classes that meet its separator come last, without changing its fill, and
# its vertices, minus its separator, are written in the rotated order. The
# classes that meet the separator are a clique, since they are twins of its
# vertices, so the order of their vertices does not matter.
function safe_pull(result::SafeResult{W, V}, graphs::AbstractVector{<:BipartiteGraph{V, E}}, orders::AbstractVector) where {W, V, E}
    n = result.n; order = Vector{V}(undef, n); i = zero(V)
    mark = FVector{Int}(undef, n)

    @inbounds for v in oneto(n)
        mark[v] = 0
    end

    nmax = zero(V); mmax = zero(E)

    for g in graphs
        nmax = max(nmax, nv(g)); mmax = max(mmax, convert(E, half(de(g))))
    end

    alpha = FVector{V}(undef, nmax)
    work1 = FVector{V}(undef, nmax)
    work2 = FVector{V}(undef, nmax)
    work3 = FVector{V}(undef, nmax)
    work4 = FVector{V}(undef, nmax)
    index = FVector{V}(undef, nmax)
    clique = FVector{V}(undef, nmax)
    rotate = FVector{V}(undef, nmax)
    uppptr = FVector{E}(undef, nmax + one(V))
    lowptr = FVector{E}(undef, nmax + one(V))
    upptgt = FVector{V}(undef, mmax)
    lowtgt = FVector{V}(undef, mmax)

    @inbounds for k in eachindex(result.lgraph)
        for p in result.sepptr[k]:result.sepptr[k + 1] - 1
            mark[result.septgt[p]] = k
        end

        cstrt = result.leafptr[k]; g = result.lgraph[k]

        if iszero(g)
            for c in cstrt:result.leafptr[k + 1] - 1, p in result.prjptr[c]:result.prjptr[c + 1] - 1
                v = result.prjtgt[p]
                mark[v] == k || (i += one(V); order[i] = v)
            end
        else
            graph = graphs[g]; order1 = orders[g]; nc = convert(V, nv(graph)); nk = zero(V)

            for j in oneto(nc)
                index[order1[j]] = j
            end

            # the classes that meet the separator
            for x in oneto(nc)
                c = cstrt + x - 1

                for p in result.prjptr[c]:result.prjptr[c + 1] - 1
                    if mark[result.prjtgt[p]] == k
                        nk += one(V); clique[nk] = index[x]
                        break
                    end
                end
            end

            if ispositive(nk)
                upper = sympermute!_impl!(uppptr, upptgt, graph, view(index, oneto(nc)), Forward)
                lower = BipartiteGraph(nc, nc, de(upper), lowptr, lowtgt)
                tree = Parent(nc, work4)
                compositerotations_impl!(alpha, work1, work2, work3, lower, tree, upper, view(clique, oneto(nk)))

                for j in oneto(nc)
                    rotate[alpha[j]] = order1[j]
                end
            else
                for j in oneto(nc)
                    rotate[j] = order1[j]
                end
            end

            for j in oneto(nc)
                c = cstrt + rotate[j] - 1

                for p in result.prjptr[c]:result.prjptr[c + 1] - 1
                    v = result.prjtgt[p]
                    mark[v] == k || (i += one(V); order[i] = v)
                end
            end
        end
    end

    @assert i == n
    return order
end

function safeseparators(weights::AbstractVector, graph::AbstractGraph{V}, alg::EliminationAlgorithm, mins::Tuple, maxmissing::Int; verbose::Bool = false) where {V}
    nv(graph) > one(V) || return permutation(weights, graph, alg; verbose)
    weights30, graphs30, result = safesolve(weights, graph, mins, maxmissing; verbose)
    orders = batch_permutation(weights30, graphs30, alg; verbose)
    order = safe_pull(result, graphs30, orders)
    return order, invperm(order)
end

include("acsd.jl")
include("acsd_minor.jl")
include("acsd_flow.jl")
