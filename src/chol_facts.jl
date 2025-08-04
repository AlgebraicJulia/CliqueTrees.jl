"""
    CholFact{T, I}

A Cholesky factorization object.
"""
struct CholFact{T, I}
    fact::SymbFact{I}
    width::I
    blkptr::FVector{I}
    blkval::FVector{T}
    status::Bool
end

"""
    cholesky(matrix::AbstractMatrix;
        alg::EliminationAlgorithm=DEFAULT_ELIMINATION_ALGORITHM,
        snd::SupernodeType=DEFAULT_SUPERNODE_TYPE,
    )

Compute the Cholesky factorization of a sparse positive definite matrix.
The factorization occurs in two phases: symbolic and numeric. During the
symbolic phase, a *tree decomposition* is constructed which will control
the numeric phase. The speed of the numeric phase is dependant on the
quality of this tree decomposition.

The symbolic phase is controlled by the parameters `alg` and `snd`.
See the function [`cliquetree`](@ref) for more information.

### Parameters

  - `alg`: elimination algorithm
  - `snd`: supernode type

```julia
julia> import CliqueTrees

julia> M = [
           1.5   94.2    0.8 0.0
           94.2  15080.4 0.0 0.0
           0.8   0.0     3.1 0.0
           0.0   0.0     0.0 1.6
       ];

julia> b = [1.0, 2.0, 1.0, 2.0];

julia> F = CliqueTrees.cholesky(M)
CholFact{Float64, Int64}:
    success: true

julia> x = F \\ b; # solve M x = b
```
"""
function cholesky(matrix::AbstractMatrix; alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM, snd::SupernodeType=DEFAULT_SUPERNODE_TYPE)
    fact = cholesky(matrix, alg, snd)
    return fact
end

function cholesky(matrix::AbstractMatrix, alg::PermutationOrAlgorithm, snd::SupernodeType)
    fact = cholesky(matrix, symbolic(matrix, alg, snd))
    return fact
end

function cholesky(matrix::AbstractMatrix, fact::SymbFact)
    return cholesky!(sparse(matrix), fact)
end

function cholesky(matrix::SparseMatrixCSC{T, I}, fact::SymbFact{I}) where {T, I}
    @argcheck size(matrix, 1) == size(matrix, 2)
    @argcheck size(matrix, 1) == nov(separators(fact.tree))
    tree = fact.tree
    perm = fact.perm 
    invp = fact.invp

    neqns = nov(separators(tree))
    adjln = half(convert(I, nnz(matrix)) - neqns) + neqns

    colptr0 = matrix.colptr
    colptr1 = FVector{I}(undef, neqns + one(I))
    colptr2 = FVector{I}(undef, neqns + one(I))

    rowval0 = matrix.rowval
    rowval1 = FVector{I}(undef, adjln)
    rowval2 = FVector{I}(undef, adjln)
    
    nzval0 = matrix.nzval
    nzval1 = FVector{T}(undef, adjln)
    nzval2 = FVector{T}(undef, adjln)

    cholesky_permute!(colptr0, colptr1, rowval0, rowval1, nzval0, nzval1, neqns, invp)
    cholesky_reverse!(colptr1, colptr2, rowval1, rowval2, nzval1, nzval2, neqns)

    width, mapping, blkptr, updptr, blkval,
        updval, frtval = cholesky_alloc(T, fact)

    status = cholesky_impl!(mapping, blkptr, updptr,
        blkval, updval, frtval, tree, colptr2, rowval2, nzval2)

    return CholFact(fact, width, blkptr, blkval, status)
end

function cholesky!(matrix::SparseMatrixCSC; alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM, snd::SupernodeType=DEFAULT_SUPERNODE_TYPE)
    fact = cholesky!(matrix, alg, snd)
    return fact
end

function cholesky!(matrix::SparseMatrixCSC{T, I}, alg::PermutationOrAlgorithm, snd::SupernodeType) where {T, I}
    fact = cholesky!(matrix, symbolic(matrix, alg, snd))
    return fact
end

function cholesky!(matrix::SparseMatrixCSC{T, I}, fact::SymbFact{I}) where {T, I}
    @argcheck size(matrix, 1) == size(matrix, 2)
    @argcheck size(matrix, 1) == nov(separators(fact.tree))
    tree = fact.tree
    perm = fact.perm 
    invp = fact.invp

    neqns = nov(separators(tree))
    adjln = half(convert(I, nnz(matrix)) - neqns) + neqns

    colptr0 = matrix.colptr
    colptr1 = FVector{I}(undef, neqns + one(I))
    colptr2 = matrix.colptr

    rowval0 = matrix.rowval
    rowval1 = FVector{I}(undef, adjln)
    rowval2 = matrix.rowval
    
    nzval0 = matrix.nzval
    nzval1 = FVector{T}(undef, adjln)
    nzval2 = matrix.nzval

    cholesky_permute!(colptr0, colptr1, rowval0, rowval1, nzval0, nzval1, neqns, invp)
    cholesky_reverse!(colptr1, colptr2, rowval1, rowval2, nzval1, nzval2, neqns)

    width, mapping, blkptr, updptr, blkval,
        updval, frtval = cholesky_alloc(T, fact)

    status = cholesky_impl!(mapping, blkptr, updptr,
        blkval, updval, frtval, tree, colptr2, rowval2, nzval2)

    return CholFact(fact, width, blkptr, blkval, status)
end 

function cholesky_alloc(::Type{T}, fact::SymbFact{I}) where {T, I}
    tree = fact.tree
    residual = residuals(tree)
    separator = separators(tree)

    up = ns = nsmax = njmax = upmax = blkln = zero(I)

    for j in vertices(separator)
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

    treln = nv(separator)
    relln = ne(separator)
    frtln = njmax * njmax

    blkptr = FVector{I}(undef, treln + one(I))
    updptr = FVector{I}(undef, nsmax + one(I))
    relidx = FVector{I}(undef, relln)

    blkval = FVector{T}(undef, blkln)
    updval = FVector{T}(undef, upmax)
    frtval = FVector{T}(undef, frtln)

    relptr = pointers(separator)
    mapping = BipartiteGraph(njmax, treln, relln, relptr, relidx)
    return njmax, mapping, blkptr, updptr, blkval, updval, frtval
end

function cholesky_permute!(
        colptr1::AbstractVector{I},
        colptr2::AbstractVector{I},
        rowval1::AbstractVector{I},
        rowval2::AbstractVector{I},
        nzval1::AbstractVector{T},
        nzval2::AbstractVector{T},
        neqns::I,
        invp::AbstractVector{I},
    ) where {T, I}
    @inbounds for i2 in oneto(neqns)
        colptr2[i2 + one(I)] = zero(I)
    end

    @inbounds for j1 in oneto(neqns)
        j2 = invp[j1]
        pstrt1 = colptr1[j1]
        pstop1 = colptr1[j1 + one(I)]

        for p1 in pstrt1:pstop1 - one(I)
            i1 = rowval1[p1]

            if i1 <= j1
                i2 = invp[i1]
                k2 = j2

                if k2 < i2
                    k2 = i2
                end

                if k2 < neqns
                    colptr2[k2 + two(I)] += one(I)
                end
            end
        end
    end

    @inbounds colptr2[begin] = p2 = one(I)

    @inbounds for i2 in oneto(neqns)
        colptr2[i2 + one(I)] = p2 += colptr2[i2 + one(I)]
    end

    @inbounds for j1 in oneto(neqns)
        j2 = invp[j1]
        pstrt1 = colptr1[j1]
        pstop1 = colptr1[j1 + one(I)]

        for p1 in pstrt1:pstop1 - one(I)
            i1 = rowval1[p1]

            if i1 <= j1
                x1 = nzval1[p1]
                i2 = invp[i1]
                k2 = j2

                if k2 < i2
                    i2, k2 = k2, i2
                end

                p2 = colptr2[k2 + one(I)]

                colptr2[k2 + one(I)] = p2 + one(I)
                rowval2[p2] = i2
                nzval2[p2] = x1
            end
        end
    end

    return
end

function cholesky_reverse!(
        colptr1::AbstractVector{I},
        colptr2::AbstractVector{I},
        rowval1::AbstractVector{I},
        rowval2::AbstractVector{I},
        nzval1::AbstractVector{T},
        nzval2::AbstractVector{T},
        neqns::I,
    ) where {T, I}
    @inbounds for i2 in oneto(neqns)
        colptr2[i2 + one(I)] = zero(I)
    end

    @inbounds for j1 in oneto(neqns)
        pstrt1 = colptr1[j1]
        pstop1 = colptr1[j1 + one(I)]

        for p1 in pstrt1:pstop1 - one(I)
            i1 = rowval1[p1]

            if i1 < neqns
                colptr2[i1 + two(I)] += one(I)
            end
        end
    end

    @inbounds colptr2[begin] = p2 = one(I)

    @inbounds for i2 in oneto(neqns)
        colptr2[i2 + one(I)] = p2 += colptr2[i2 + one(I)]
    end

    @inbounds for j1 in oneto(neqns)
        pstrt1 = colptr1[j1]
        pstop1 = colptr1[j1 + one(I)]

        for p1 in pstrt1:pstop1 - one(I)
            i1 = rowval1[p1]
            x1 = nzval1[p1]
            p2 = colptr2[i1 + one(I)]

            colptr2[i1 + one(I)] = p2 + one(I)
            rowval2[p2] = j1
            nzval2[p2] = x1
        end
    end

    return
end

function cholesky_impl!(
        mapping::BipartiteGraph{I, I},        
        blkptr::AbstractVector{I},
        updptr::AbstractVector{I},
        blkval::AbstractVector{T},
        updval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I},
        colptr::AbstractVector{I},
        rowval::AbstractVector{I},
        nzval::AbstractVector{T},
    ) where {T, I}
    separator = separators(tree)
    relidx = targets(mapping)
    status = true; ns = zero(I)

    cholesky_init!(relidx, blkptr, updptr,
        blkval, tree, colptr, rowval, nzval)

    for j in vertices(separator)
        iterstatus, ns = cholesky_loop!(mapping, blkptr,
            updptr, blkval, updval, frtval, tree, ns, j)

        status = status && iterstatus
    end

    return status
end

function cholesky_init!(
        relidx::AbstractVector{I},
        blkptr::AbstractVector{I},
        updptr::AbstractVector{I},
        blkval::AbstractVector{T},
        tree::CliqueTree{I, I},
        colptr::AbstractVector{I},
        rowval::AbstractVector{I},
        nzval::AbstractVector{T},
    ) where {T, I}
    residual = residuals(tree)
    separator = separators(tree)
    treln = nv(separator)

    @inbounds updptr[one(I)] = p = q = one(I)

    @inbounds for j in vertices(separator)
        blkptr[j] = p

        res = neighbors(residual, j)
        sep = neighbors(separator, j)
        bag = Clique(res, sep)

        nn = eltypedegree(residual, j)
        na = eltypedegree(separator, j)
        nj = nn + na

        for v in res
            k = one(I)
            qstrt = colptr[v]
            qstop = colptr[v + one(I)]

            for q in qstrt:qstop - one(I)
                w = rowval[q]

                while bag[k] < w
                    blkval[p] = zero(T)
                    k += one(I)
                    p += one(I)
                end

                blkval[p] = nzval[q]
                k += one(I)
                p += one(I) 
            end

            while k <= nj
                blkval[p] = zero(T)
                k += one(I)
                p += one(I)
            end
        end

        pj = parentindex(tree, j)

        if !isnothing(pj)
            pbag = tree[pj]    

            k = one(I)

            for w in sep
                while pbag[k] < w
                    k += one(I)
                end

                relidx[q] = k
                q += one(I)
                k += one(I)
            end
        end
    end

    @inbounds blkptr[treln + one(I)] = p
    return
end

function cholesky_loop!(
        mapping::BipartiteGraph{I, I},        
        blkptr::AbstractVector{I},
        updptr::AbstractVector{I},
        blkval::AbstractVector{T},
        updval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I},
        ns::I,
        j::I,
    ) where {T, I}
    residual = residuals(tree)
    separator = separators(tree)

    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    nn = eltypedegree(residual, j)

    # na is the size of the separator at node j
    #
    #     na = | sep(j) |
    #
    na = eltypedegree(separator, j)

    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na

    # F is the frontal matrix at node j
    #
    #     F = [ F₁  F₂  ]
    #
    #       = [ F₁₁ F₁₂ ]
    #         [ F₂₁ F₂₂ ]
    #
    # only the lower triangular part is used
    F = reshape(view(frtval, oneto(nj * nj)), nj, nj)
    F₁ =  view(F, oneto(nj),      oneto(nn))
    F₁₁ = view(F, oneto(nn),      oneto(nn))
    F₂₁ = view(F, nn + one(I):nj, oneto(nn))
    F₂₂ = view(F, nn + one(I):nj, nn + one(I):nj)

    # B is part of the Cholesky factor
    #
    #          res(j)
    #     B = [ B₁₁  ] res(j)
    #         [ B₂₁  ] sep(j)
    #
    pstrt = blkptr[j]
    pstop = blkptr[j + one(I)]
    B = reshape(view(blkval, pstrt:pstop - one(I)), nj, nn)

    # copy B into F₁
    #
    #     F₁₁ ← B₁₁
    #     F₂₁ ← B₂₁
    #
    lacpy!(F, B); fill!(F₂₂, zero(T))

    for i in Iterators.reverse(childindices(tree, j))
        cholesky_add_update!(F, ns, i, mapping, updptr, updval)
        ns -= one(I)
    end

    # factorize F₁₁ as
    #
    #     F₁₁ = L₁₁ L₁₁ᴴ
    #
    # and store F₁₁ ← L₁₁
    status = potrf!(F₁₁)

    if ispositive(na)
        # solve for L₂₁ in
        #
        #     L₂₁ L₁₁ᴴ = F₂₁
        #
        # and store F₂₁ ← L₂₁
        trsm!(F₁₁, F₂₁, Val(true), Val(true))
    
        # compute
        #
        #    U₂₂ = F₂₂ - L₂₁ L₂₁ᴴ
        #
        # and store F₂₂ ← U₂₂
        syrk!(F₂₁, F₂₂)

        ns += one(I)
        pstrt = updptr[ns]
        pstop = updptr[ns + one(I)] = pstrt + na * na
        U₂₂ = reshape(view(updval, pstrt:pstop - one(I)), na, na)
        lacpy!(U₂₂, F₂₂)
    end
 
    # copy F₁  into B
    #
    #     B₁₁ ← F₁₁
    #     B₂₁ ← F₂₁
    #
    lacpy!(B, F₁)
    return status, ns
end

function cholesky_add_update!(
        F::AbstractMatrix{T},
        ns::I,
        i::I,
        mapping::BipartiteGraph{I, I},
        updptr::AbstractVector{I},
        updval::AbstractVector{T},
    ) where {T, I}
    # na is the size of the separator at node i.
    #
    #     na = | sep(i) |
    #
    na = eltypedegree(mapping, i)

    # ind is the subset inclusion
    #
    #     ind: sep(i) → sep(parent(i))
    #
    ind = neighbors(mapping, i)

    # U is the na × na update matrix for node i.
    pstrt = updptr[ns]
    pstop = pstrt + na * na
    U = reshape(view(updval, pstrt:pstop - one(I)), na, na)

    # for all uj in sep(i) ...
    @inbounds for uj in oneto(na)
        # let fj = ind(uj)
        fj = ind[uj]

        # for all ui ≥ uj in sep(i) ...
        for ui in uj:na
            # let fi = ind(ui)
            fi = ind[ui]

            # compute the sum
            #
            #     F[fi, fj] + U[ui, uj]
            #
            # and assign it to F[fi, fj]
            F[fi, fj] += U[ui, uj]
        end
    end

    return
end

function Base.:\(F::CholFact, B)
    return ldiv(F, B)
end

function LinearAlgebra.ldiv(F::CholFact{T}, B::Union{AbstractVector, AbstractMatrix}) where {T}
    @argcheck nov(separators(F.fact.tree)) == size(B, 1)
    X = Array{T}(undef, size(B)); copyto!(X, B)
    return ldiv!(F, X)
end

function LinearAlgebra.ldiv!(F::CholFact{T}, B::Union{AbstractVector, AbstractMatrix}) where {T}
    @argcheck nov(separators(F.fact.tree)) == size(B, 1)
    tree = F.fact.tree
    perm = F.fact.perm
    width = F.width
    blkptr = F.blkptr
    blkval = F.blkval
    frtval = FVector{T}(undef, width * size(B, 2))

    residual = residuals(tree)
    separator = separators(tree)

    X = FArray{T}(undef, size(B))

    for w in axes(B, 2), v in axes(B, 1)
        X[v, w] = B[perm[v], w]
    end
    
    for j in vertices(separator)
        ldiv!_loop_fwd!(blkptr, blkval, frtval, tree, X, j)
    end
    
    for j in reverse(vertices(separator))
        ldiv!_loop_bwd!(blkptr, blkval, frtval, tree, X, j)
    end

    for w in axes(B, 2), v in axes(B, 1)
        B[perm[v], w] = X[v, w]
    end
    
    return B
end

function ldiv!_loop_fwd!(
        blkptr::AbstractVector{I},
        blkval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I}, 
        B::AbstractVector{T},
        j::I,
    ) where {T, I}
    residual = residuals(tree)
    separator = separators(tree)

    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    @inbounds nn = eltypedegree(residual, j)

    # na is the size of the separator at node j.
    #
    #     na = | sep(j) |
    #
    @inbounds na = eltypedegree(separator, j)

    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na

    # bag is the bag at node j
    @inbounds bag = tree[j] 
    
    # L is part of the Cholesky factor
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    @inbounds pstrt = blkptr[j]
    @inbounds pstop = blkptr[j + one(I)]
    @inbounds L = reshape(view(blkval, pstrt:pstop - one(I)), nj, nn)
    
    @inbounds L₁₁ = view(L, oneto(nn),      oneto(nn))
    @inbounds L₂₁ = view(L, nn + one(I):nj, oneto(nn))

    # X is part of B
    #
    #     X = [ X₁ ] res(j)
    #         [ X₂ ] sep(j)
    #
    @inbounds X = view(frtval, oneto(nj))
    @inbounds X₁ = view(X, oneto(nn))
    @inbounds X₂ = view(X, nn + one(I):nj)

    @inbounds for k in oneto(nj)
        X[k] = B[bag[k]]
    end

    # solve for Y₁ in
    #
    #     L₁₁ Y₁ = X₁
    #
    # and store X₁ ← Y₁
    trsv!(L₁₁, X₁, Val(false))
    
    if ispositive(na)
        # compute the difference
        #
        #     Y₂ = X₂ - L₂₁ Y₁
        #
        # and store X₂ ← Y₂ 
        gemv!(L₂₁, X₁, X₂, Val(false))
    end

    # copy X into b
    @inbounds for k in oneto(nj)
        B[bag[k]] = X[k]
    end

    return
end

function ldiv!_loop_fwd!(
        blkptr::AbstractVector{I},
        blkval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I}, 
        B::AbstractMatrix{T},
        j::I,
    ) where {T, I}
    residual = residuals(tree)
    separator = separators(tree)

    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    @inbounds nn = eltypedegree(residual, j)

    # na is the size of the separator at node j.
    #
    #     na = | sep(j) |
    #
    @inbounds na = eltypedegree(separator, j)

    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na

    # bag is the bag at node j
    @inbounds bag = tree[j] 
    
    # L is part of the Cholesky factor
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    @inbounds pstrt = blkptr[j]
    @inbounds pstop = blkptr[j + one(I)]
    @inbounds L = reshape(view(blkval, pstrt:pstop - one(I)), nj, nn)
    
    @inbounds L₁₁ = view(L, oneto(nn),      oneto(nn))
    @inbounds L₂₁ = view(L, nn + one(I):nj, oneto(nn))

    # X is part of B
    #
    #     X = [ X₁ ] res(j)
    #         [ X₂ ] sep(j)
    #
    @inbounds X = reshape(view(frtval, oneto(nj * size(B, 2))), nj, size(B, 2))
    @inbounds X₁ = view(X, oneto(nn),      axes(B, 2))
    @inbounds X₂ = view(X, nn + one(I):nj, axes(B, 2))

    @inbounds for w in axes(B, 2), k in oneto(nj)
        X[k, w] = B[bag[k], w]
    end

    # solve for Y₁ in
    #
    #     L₁₁ Y₁ = X₁
    #
    # and store X₁ ← Y₁
    trsm!(L₁₁, X₁, Val(false), Val(false))
    
    if ispositive(na)
        # compute the difference
        #
        #     Y₂ = X₂ - L₂₁ Y₁
        #
        # and store X₂ ← Y₂ 
        gemm!(L₂₁, X₁, X₂, Val(false))
    end

    # copy X into b

    @inbounds for w in axes(B, 2), k in oneto(nj)
        B[bag[k], w] = X[k, w]
    end

    return
end

function ldiv!_loop_bwd!(
        blkptr::AbstractVector{I},
        blkval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I}, 
        B::AbstractVector{T},
        j::I,
    ) where {T, I}
    residual = residuals(tree)
    separator = separators(tree)

    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    @inbounds nn = eltypedegree(residual, j)

    # na is the size of the separator at node j.
    #
    #     na = | sep(j) |
    #
    @inbounds na = eltypedegree(separator, j)

    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na

    # bag is the bag at node j
    @inbounds bag = tree[j]        
 
    # L is part of the Cholesky factor
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    @inbounds pstrt = blkptr[j]
    @inbounds pstop = blkptr[j + one(I)]
    @inbounds L = reshape(view(blkval, pstrt:pstop - one(I)), nj, nn)
    
    @inbounds L₁₁ = view(L, oneto(nn),      oneto(nn))
    @inbounds L₂₁ = view(L, nn + one(I):nj, oneto(nn))

    # X is part of B
    #
    #     X = [ X₁ ] res(j)
    #         [ X₂ ] sep(j)
    #
    @inbounds X = view(frtval, oneto(nj))
    @inbounds X₁ = view(X, oneto(nn))
    @inbounds X₂ = view(X, nn + one(I):nj)

    @inbounds for k in oneto(nj)
        X[k] = B[bag[k]]
    end
    
    if ispositive(na)
        # compute the difference
        #
        #     Y₁ = X₁ - L₂₁ᴴ X₂
        #
        # and store X₁ ← Y₁ 
        gemv!(L₂₁, X₂, X₁, Val(true))
    end

    # solve for Z₁ in
    #
    #     L₁₁ᴴ Z₁ = Y₁
    #
    # and store X₁ ← Z₁
    trsv!(L₁₁, X₁, Val(true))

    # copy X into b
    @inbounds for k in oneto(nj)
        B[bag[k]] = X[k]
    end

    return
end

function ldiv!_loop_bwd!(
        blkptr::AbstractVector{I},
        blkval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I}, 
        B::AbstractMatrix{T},
        j::I,
    ) where {T, I}
    residual = residuals(tree)
    separator = separators(tree)

    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    @inbounds nn = eltypedegree(residual, j)

    # na is the size of the separator at node j.
    #
    #     na = | sep(j) |
    #
    @inbounds na = eltypedegree(separator, j)

    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na

    # bag is the bag at node j
    @inbounds bag = tree[j]        
 
    # L is part of the Cholesky factor
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    @inbounds pstrt = blkptr[j]
    @inbounds pstop = blkptr[j + one(I)]
    @inbounds L = reshape(view(blkval, pstrt:pstop - one(I)), nj, nn)
    
    @inbounds L₁₁ = view(L, oneto(nn),      oneto(nn))
    @inbounds L₂₁ = view(L, nn + one(I):nj, oneto(nn))

    # X is part of B
    #
    #     X = [ X₁ ] res(j)
    #         [ X₂ ] sep(j)
    #
    @inbounds X = reshape(view(frtval, oneto(nj * size(B, 2))), nj, size(B, 2))
    @inbounds X₁ = view(X, oneto(nn),      axes(B, 2))
    @inbounds X₂ = view(X, nn + one(I):nj, axes(B, 2))

    @inbounds for w in axes(B, 2), k in oneto(nj)
        X[k, w] = B[bag[k], w]
    end
    
    if ispositive(na)
        # compute the difference
        #
        #     Y₁ = X₁ - L₂₁ᴴ X₂
        #
        # and store X₁ ← Y₁ 
        gemm!(L₂₁, X₂, X₁, Val(true))
    end

    # solve for Z₁ in
    #
    #     L₁₁ᴴ Z₁ = Y₁
    #
    # and store X₁ ← Z₁
    trsm!(L₁₁, X₁, Val(true), Val(false))

    # copy X into b
    @inbounds for w in axes(B, 2), k in oneto(nj)
        B[bag[k], w] = X[k, w]
    end

    return
end

function Base.show(io::IO, ::MIME"text/plain", F::CholFact{T, I}) where {T, I}
    println(io, "CholFact{$T, $I}:")
    print(io,   "    success: $(F.status)")
end


##################################
# Dense Numerical Linear Algebra #
##################################


# copy the lower triangular part of B to A
function lacpy!(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    m = size(B, 1)

    @inbounds for j in axes(B, 2)
        for i in j:m
            A[i, j] = B[i, j]
        end
    end

    return
end

@static if VERSION >= v"1.11"

function lacpy!(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: BLAS.BlasFloat}
    LAPACK.lacpy!(A, B, 'L')
    return
end

end

# factorize A as
#
#     A = L Lᴴ
#
# and store A ← L
function potrf!(A::AbstractMatrix)
    F = LinearAlgebra.cholesky!(Hermitian(A, :L), NoPivot())
    status = iszero(F.info)
    return status
end

# compute the lower triangular part of the difference
#
#     C = L - A * Aᴴ
#
# and store L ← C
function syrk!(A::AbstractMatrix{T}, L::AbstractMatrix{T}) where {T}
    m = size(A, 1)

    @inbounds for j in axes(A, 1), k in axes(A, 2)
        Ajk = A[j, k]

        for i in j:m
            L[i, j] -= A[i, k] * Ajk
        end
    end

    return
end

function syrk!(A::AbstractMatrix{T}, L::AbstractMatrix{T}) where {T <: Complex}
    m = size(A, 1)

    @inbounds for j in axes(A, 1), k in axes(A, 2)
        Ajk = conj(A[j, k])

        for i in j:m
            L[i, j] -= A[i, k] * Ajk
        end
    end

    return
end

function syrk!(A::AbstractMatrix{T}, L::AbstractMatrix{T}) where {T <: BLAS.BlasReal}
    BLAS.syrk!('L', 'N', -one(T), A, one(T), L)
    return
end

function syrk!(A::AbstractMatrix{T}, L::AbstractMatrix{T}) where {T <: BLAS.BlasComplex}
    BLAS.herk!('L', 'N', -one(real(T)), A, one(real(T)), L)
    return
end

# solve for x in
#
#     L x = b
#
# and store b ← x
function trsv!(L::AbstractMatrix{T}, b::AbstractVector{T}, tL::Val{false}) where {T}
    ldiv!(LowerTriangular(L), b)
    return
end

# solve for x in
#
#     Lᴴ x = b
#
# and store b ← x
function trsv!(L::AbstractMatrix{T}, b::AbstractVector{T}, tL::Val{true}) where {T}
    ldiv!(LowerTriangular(L) |> adjoint, b)
    return
end

# solve for X in
#
#     L X = B
#
# and store B ← X
function trsm!(L::AbstractMatrix{T}, B::AbstractMatrix{T}, tL::Val{false}, side::Val{false}) where {T}
    ldiv!(LowerTriangular(L), B)
    return
end

# solve for X in
#
#     Lᴴ X = B
#
# and store B ← X
function trsm!(L::AbstractMatrix{T}, B::AbstractMatrix{T}, tL::Val{true}, side::Val{false}) where {T}
    ldiv!(LowerTriangular(L) |> adjoint, B)
    return
end

# solve for X in
#
#     X Lᴴ = B
#
# and store B ← X
function trsm!(L::AbstractMatrix{T}, B::AbstractMatrix{T}, tL::Val{true}, side::Val{true}) where {T}
    rdiv!(B, LowerTriangular(L) |> adjoint)
    return
end

# compute the difference
#
#     z = y - A x
#
# and store y ← z
function gemv!(A::AbstractMatrix{T}, x::AbstractVector{T}, y::AbstractVector{T}, tA::Val{false}) where {T}
    mul!(y, A, x, -one(T), one(T))
    return
end

# compute the difference
#
#     z = y - Aᴴ x
#
# and store y ← z
function gemv!(A::AbstractMatrix{T}, x::AbstractVector{T}, y::AbstractVector{T}, tA::Val{true}) where {T}
    mul!(y, A |> adjoint, x, -one(T), one(T))
    return
end

# compute the difference
#
#     Z = Y - A X
#
# and store Y ← Z
function gemm!(A::AbstractMatrix{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}, tA::Val{false}) where {T}
    mul!(Y, A, X, -one(T), one(T))
    return
end

# compute the difference
#
#     Z = Y - Aᴴ X
#
# and store Y ← Z
function gemm!(A::AbstractMatrix{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}, tA::Val{true}) where {T}
    mul!(Y, A |> adjoint, X, -one(T), one(T))
    return
end
