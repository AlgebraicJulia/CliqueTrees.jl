struct ChordalSSymbolic{I}
    S::ChordalSymbolic{I}
    N::FBipartiteGraph{I, I}
    Bptr::FVector{I}
    Fptr::FVector{I}
    nBptr::I
end

function Base.size(S::ChordalSSymbolic)
    return size(S.S)
end

function Base.size(S::ChordalSSymbolic, d::Integer)
    return size(S.S, d)
end

function ncc(S::ChordalSSymbolic)
    return convert(Int, S.nBptr)
end

function components(S::ChordalSSymbolic)
    return oneto(ncc(S))
end

function ssymbolic(A::SparseMatrixCSC{<:Any, I}; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM) where {I}
    n = convert(I, size(A, 2))

    scc = sccs(A)

    nBptr = nv(scc)
    Bptr  = pointers(scc)
    invp  = targets(scc)
    stable_components!(invp, Bptr, nBptr)

    # (only the pattern of A is used from here on; one component: the labels are kept, no copy)
    graph = isone(nBptr) ? BipartiteGraph{I, I}(n, n, getcolptr(A)[n + one(I)] - one(I), getcolptr(A), rowvals(A)) : permute_pattern(A, invp, invp)
    perm, tree = scliquetree(graph, Bptr, nBptr; alg)
    S = ChordalSymbolic(tree)
    Fptr = sccfronts(S, Bptr, nBptr)
    # the coupling between components; with one component there is none
    N = isone(nBptr) ? nooffd(n) : soffd(graph, perm, Bptr, nBptr)

    @inbounds for a in oneto(n)
        perm[a] = invp[perm[a]]
    end

    @inbounds for a in oneto(n)
        invp[perm[a]] = a
    end

    return Permutation(perm, invp), ChordalSSymbolic(S, N, Bptr, Fptr, nBptr)
end

#
# the vertices of each component in increasing order (a counting sort by component): the components
# keep their (topological) order, but within one the labels keep the input's order, so that a column of
# an undirected graph, whose rows all lie in its component, stays sorted when relabelled
#
function stable_components!(tgt::AbstractVector{I}, Bptr::AbstractVector{I}, nBptr::I) where {I}
    n = length(tgt)
    comp = FVector{I}(undef, n)
    nxt = FVector{I}(undef, nBptr)

    @inbounds for c in oneto(nBptr)
        nxt[c] = Bptr[c]

        for p in Bptr[c]:(Bptr[c + one(I)] - one(I))
            comp[tgt[p]] = c
        end
    end

    @inbounds for v in oneto(I(n))
        c = comp[v]; tgt[nxt[c]] = v; nxt[c] += one(I)
    end

    return tgt
end

function scliquetree(A::SparseMatrixCSC{<:Any, I}, Bptr::AbstractVector{I}, nBptr::I; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM) where {I}
    return scliquetree(BipartiteGraph(A), Bptr, nBptr; alg)
end

function scliquetree(graph::BipartiteGraph{I}, Bptr::AbstractVector{I}, nBptr::I; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM) where {I}
    n = nv(graph)

    prnt = FVector{I}(undef, n)
    Rptr = FVector{I}(undef, n + one(I)); Rptr[one(I)] = one(I)
    Sptr = FVector{I}(undef, n + one(I)); Sptr[one(I)] = one(I)
    Rtgt = FVector{I}(undef, n)
    #
    # the clique tree of each component (one component: the whole graph, without copying it; components
    # of one or two vertices are always the same graph, so their clique trees are computed once), then
    # the separators of all of them in one array of the final size
    #
    trees = Vector{Any}(undef, nBptr)
    small1 = small2 = nothing; vstrt = one(I)

    @inbounds for c in oneto(nBptr)
        vstop = Bptr[c + one(I)]; nc = vstop - vstrt

        if isone(nc) && !isone(nBptr)
            isnothing(small1) && (small1 = cliquetree(symmetric(subgraph(graph, vstrt, vstop - one(I)), 'N'); alg))
            trees[c] = small1
        elseif nc == two(I) && !isone(nBptr)
            isnothing(small2) && (small2 = cliquetree(symmetric(subgraph(graph, vstrt, vstop - one(I)), 'N'); alg))
            trees[c] = small2
        else
            sub = isone(nBptr) ? graph : subgraph(graph, vstrt, vstop - one(I))
            trees[c] = cliquetree(symmetric_pattern(sub) ? sub : symmetric(sub, 'N'); alg)
        end

        vstrt = vstop
    end

    # one component: its clique tree is the whole graph's (no copy of the separators)
    if isone(nBptr)
        cperm, ctree = trees[1]
        copyto!(Rtgt, cperm)
        return Rtgt, ctree
    end

    Stgt = FVector{I}(undef, sum(t -> nseparated(t[2]), trees; init = 0))
    f = k = zero(I); vstrt = fstrt = one(I)

    @inbounds for c in oneto(nBptr)
        vstop = Bptr[c + one(I)]
        f, k = addcomponent!(prnt, Rptr, Sptr, Stgt, Rtgt, trees[c]..., f, k, vstrt, vstop, fstrt)::Tuple{I, I}
        vstrt = vstop; fstrt = f + one(I)
    end

    m = fstrt - one(I)

    res = BipartiteGraph{I, I}(n, m, n, Rptr, oneto(n))
    sep = BipartiteGraph{I, I}(n, m, k, Sptr, Stgt)

    etree = Tree(m, prnt)
    stree = SupernodeTree(etree, res)
    ctree = CliqueTree(stree, sep)

    return Rtgt, ctree
end

#
# whether the graph is its own symmetric(graph, 'N'): a symmetric pattern without loops, its adjacency
# lists sorted (then symmetric returns the same lists, in the same order). One read of the lists:
# visiting the columns j in order, the entries (j, i) of column i come in order too, so one pointer
# per column checks each mirror entry; false at the first one missing (or out of order).
#
function symmetric_pattern(graph::AbstractGraph{I}) where {I}
    n = nv(graph); ptr = pointers(graph); tgt = targets(graph)
    cur = FVector{I}(undef, n)

    @inbounds for i in oneto(n)
        cur[i] = ptr[i]
    end

    @inbounds for j in oneto(n), p in ptr[j]:(ptr[j + one(I)] - one(I))
        i = tgt[p]; q = cur[i]
        (i != j && q < ptr[i + one(I)] && tgt[q] == j) || return false
        cur[i] = q + one(I)
    end

    return true
end

# the number of separator entries of a clique tree (its separators' targets)
function nseparated(ctree)
    csep = separators(ctree)
    return Int(pointers(csep)[nv(csep) + 1]) - 1
end

# the fronts of one component's clique tree, appended to the whole graph's (vertices from vstrt)
function addcomponent!(prnt::AbstractVector{I}, Rptr::AbstractVector{I}, Sptr::AbstractVector{I}, Stgt::AbstractVector{I}, Rtgt::AbstractVector{I},
        cperm::AbstractVector, ctree, f::I, k::I, vstrt::I, vstop::I, fstrt::I) where {I}
    cres = residuals(ctree)
    csep = separators(ctree)
    cpnt = ctree.tree.tree.tree.prnt

    @inbounds for cf in vertices(cres)
        f += one(I); cg = cpnt[cf]

        if ispositive(cg)
            prnt[f] = cg + fstrt - one(I)
        else
            prnt[f] = cg
        end

        Rptr[f + one(I)] = Rptr[f] + eltypedegree(cres, cf)
        Sptr[f + one(I)] = Sptr[f] + eltypedegree(csep, cf)

        for cv in neighbors(csep, cf)
            k += one(I); Stgt[k] = cv + vstrt - one(I)
        end
    end

    @inbounds for i in oneto(vstop - vstrt)
        Rtgt[vstrt - one(I) + i] = vstrt - one(I) + cperm[i]
    end

    return f, k
end

function sccfronts(S::ChordalSymbolic{I}, Bptr::AbstractVector{I}, nBptr::I) where {I}
    Fptr = FVector{I}(undef, nBptr + one(I))

    @inbounds for c in oneto(nBptr)
        Fptr[c] = S.idx[Bptr[c]]
    end

    Fptr[nBptr + one(I)] = nv(S.res) + one(I)
    return Fptr
end

function nooffd(n::I) where {I}
    ptr = FVector{I}(undef, n + one(I))
    fill!(ptr, one(I))
    return BipartiteGraph{I, I}(n, n, zero(I), ptr, FVector{I}(undef, zero(I)))
end

#
# soffd(permute(graph, perm, perm), Bptr, nBptr) without the permuted copy: perm reorders vertices
# within components, so column j = invperm(perm)[b] keeps the rows a of column b that lie in earlier
# components (a < Bptr of the component of b), relabelled and sorted.
#
function soffd(graph::BipartiteGraph{I}, perm::AbstractVector{I}, Bptr::AbstractVector{I}, nBptr::I) where {I}
    n = nv(graph); gptr = pointers(graph); gtgt = targets(graph)
    ip = FVector{I}(undef, n)

    @inbounds for j in oneto(n)
        ip[perm[j]] = j
    end

    ptr = FVector{I}(undef, n + one(I))

    @inbounds for c in oneto(nBptr), b in Bptr[c]:(Bptr[c + one(I)] - one(I))
        d = zero(I)

        for p in gptr[b]:(gptr[b + one(I)] - one(I))
            gtgt[p] < Bptr[c] && (d += one(I))
        end

        ptr[ip[b] + one(I)] = d
    end

    ptr[one(I)] = p = one(I)

    @inbounds for j in oneto(n)
        ptr[j + one(I)] = p += ptr[j + one(I)]
    end

    m = p - one(I); tgt = FVector{I}(undef, m)

    @inbounds for c in oneto(nBptr), b in Bptr[c]:(Bptr[c + one(I)] - one(I))
        o = ptr[ip[b]]; q = o

        for p in gptr[b]:(gptr[b + one(I)] - one(I))
            a = gtgt[p]
            a < Bptr[c] || continue
            # insertion into the sorted list tgt[o:q - 1]
            i = ip[a]; u = q - one(I)

            while u >= o && tgt[u] > i
                tgt[u + one(I)] = tgt[u]; u -= one(I)
            end

            tgt[u + one(I)] = i; q += one(I)
        end
    end

    return BipartiteGraph{I, I}(n, n, m, ptr, tgt)
end

function soffd(A::SparseMatrixCSC{<:Any, I}, Bptr::AbstractVector{I}, nBptr::I) where {I}
    return soffd(BipartiteGraph(A), Bptr, nBptr)
end

function soffd(graph::BipartiteGraph{I}, Bptr::AbstractVector{I}, nBptr::I) where {I}
    n = nv(graph)

    ptr = FVector{I}(undef, n + one(I))

    p = one(I)

    @inbounds for c in oneto(nBptr)
        jstrt = Bptr[c]
        jstop = Bptr[c + one(I)] - one(I)

        for j in jstrt:jstop
            ptr[j] = p

            for i in neighbors(graph, j)
                if i < jstrt
                    p += one(I)
                end
            end
        end
    end

    ptr[n + one(I)] = p; m = p - one(I)

    tgt = FVector{I}(undef, m)

    @inbounds for c in oneto(nBptr)
        jstrt = Bptr[c]
        jstop = Bptr[c + one(I)] - one(I)

        for j in jstrt:jstop
            p = ptr[j]

            for i in neighbors(graph, j)
                if i < jstrt
                    tgt[p] = i; p += one(I)
                end
            end
        end
    end

    return BipartiteGraph{I, I}(n, n, m, ptr, tgt)
end
