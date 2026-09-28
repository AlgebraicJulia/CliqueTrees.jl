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

    A = permute(A, invp, invp)
    perm, tree = scliquetree(A, Bptr, nBptr; alg)
    S = ChordalSymbolic(tree)
    Fptr = sccfronts(S, Bptr, nBptr)

    A = permute(A, perm, perm)
    N = soffd(A, Bptr, nBptr)

    @inbounds for a in oneto(n)
        perm[a] = invp[perm[a]]
    end

    @inbounds for a in oneto(n)
        invp[perm[a]] = a
    end

    return Permutation(perm, invp), ChordalSSymbolic(S, N, Bptr, Fptr, nBptr)
end

function scliquetree(A::SparseMatrixCSC{<:Any, I}, Bptr::AbstractVector{I}, nBptr::I; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM) where {I}
    return scliquetree(BipartiteGraph(A), Bptr, nBptr; alg)
end

function scliquetree(graph::BipartiteGraph{I}, Bptr::AbstractVector{I}, nBptr::I; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM) where {I}
    n = nv(graph)

    prnt = FVector{I}(undef, n)
    Rptr = FVector{I}(undef, n + one(I)); Rptr[one(I)] = one(I)
    Sptr = FVector{I}(undef, n + one(I)); Sptr[one(I)] = one(I)
    Stgt = I[]
    Rtgt = FVector{I}(undef, n)

    f = k = zero(I); vstrt = fstrt = one(I)

    @inbounds for c in oneto(nBptr)
        vstop = Bptr[c + one(I)]
        cperm, ctree = cliquetree(symmetric(subgraph(graph, vstrt, vstop - one(I)), 'N'); alg)

        cres = residuals(ctree)
        csep = separators(ctree)
        cpnt = ctree.tree.tree.tree.prnt
        fstop = fstrt + nv(cres)

        for cf in vertices(cres)
            f += one(I); cg = cpnt[cf]

            if ispositive(cg)
                prnt[f] = cg + fstrt - one(I)
            else
                prnt[f] = cg
            end

            Rptr[f + one(I)] = Rptr[f] + eltypedegree(cres, cf)
            Sptr[f + one(I)] = Sptr[f] + eltypedegree(csep, cf)

            for cv in neighbors(csep, cf)
                k += one(I); push!(Stgt, cv + vstrt - one(I))
            end
        end

        for i in oneto(vstop - vstrt)
            Rtgt[vstrt - one(I) + i] = vstrt - one(I) + cperm[i]
        end

        vstrt = vstop; fstrt = fstop
    end

    m = fstrt - one(I)

    res = BipartiteGraph{I, I}(n, m, n, Rptr, oneto(n))
    sep = BipartiteGraph{I, I}(n, m, k, Sptr, FVector{I}(Stgt))

    etree = Tree(m, prnt)
    stree = SupernodeTree(etree, res)
    ctree = CliqueTree(stree, sep)

    return Rtgt, ctree
end

function sccfronts(S::ChordalSymbolic{I}, Bptr::AbstractVector{I}, nBptr::I) where {I}
    Fptr = FVector{I}(undef, nBptr + one(I))

    @inbounds for c in oneto(nBptr)
        Fptr[c] = S.idx[Bptr[c]]
    end

    Fptr[nBptr + one(I)] = nv(S.res) + one(I)
    return Fptr
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
