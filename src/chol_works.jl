"""
    CholWork{T, I}

A Cholesky factorization workspace.
"""
struct CholWork{T, I}
    updptr::FVector{I}
    nzval1::FVector{T}
    nzval2::FVector{T}
    updval::FVector{T}
    frtval::FVector{T}
    pattern1::BipartiteGraph{I, I, FVector{I}, FVector{I}}
    pattern2::BipartiteGraph{I, I, FVector{I}, FVector{I}} 
    mapping::BipartiteGraph{I, I, FVector{I}, FVector{I}}       
end

"""
    cholinit(matrix::SparseMatrixCSC, symbfact::SymbFact)

Initialize an cholesky factor and a factorization workspace.
"""
function cholinit(matrix::SparseMatrixCSC{T, I}, symbfact::SymbFact{I}) where {T, I}
    @argcheck size(matrix, 1) == size(matrix, 2)
    @argcheck size(matrix, 1) == nov(separators(symbfact.tree))

    tree = symbfact.tree
    residual = residuals(tree)
    separator = separators(tree)

    up = ns = nsmax = njmax = upmax = blkln = zero(I)

    @inbounds for j in vertices(separator)
        nn = eltypedegree(residual, j)
        na = eltypedegree(separator, j)
        nj = nn + na

        for i in childindices(tree, j)
            ma = eltypedegree(separator, i)

            ns -= one(I)
            up -= ma * ma
        end

        if !isnothing(parentindex(tree, j))
            ns += one(I)
            up += na * na 
        end

        nsmax = max(nsmax, ns)
        njmax = max(njmax, nj)
        upmax = max(upmax, up)

        blkln = blkln + nn * nj
    end

    neqns = nov(separators(tree))
    adjln = half(convert(I, nnz(matrix)) - neqns) + neqns

    treln = nv(separator)
    relln = ne(separator)
    frtln = njmax * njmax

    colptr1 = FVector{I}(undef, neqns + one(I))
    colptr2 = FVector{I}(undef, neqns + one(I))
    blkptr  = FVector{I}(undef, treln + one(I))
    updptr  = FVector{I}(undef, nsmax + one(I))
    relptr = pointers(separator)

    rowval1 = FVector{I}(undef, adjln)
    rowval2 = FVector{I}(undef, adjln)
    relidx  = FVector{I}(undef, relln)

    nzval1 = FVector{T}(undef, adjln)
    nzval2 = FVector{T}(undef, adjln)
    blkval = FVector{T}(undef, blkln)
    updval = FVector{T}(undef, upmax)
    frtval = FVector{T}(undef, frtln)

    status = FScalar{Bool}(undef)

    pattern1 = BipartiteGraph(neqns, neqns, adjln, colptr1, rowval1)
    pattern2 = BipartiteGraph(neqns, neqns, adjln, colptr2, rowval2)
    mapping = BipartiteGraph(njmax, treln, relln, relptr, relidx)

    cholfact = CholFact{T, I}(symbfact, frtln, blkptr, blkval, status) 
    cholwork = CholWork{T, I}(updptr, nzval1, nzval2, updval, frtval, pattern1, pattern2, mapping)
    return cholfact, cholwork
end

   
