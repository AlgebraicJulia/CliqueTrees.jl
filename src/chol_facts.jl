"""
    CholFact{T, I} <: Factorization{T}

A Cholesky factorization object.
"""
struct CholFact{T, I} <: Factorization{T}
    symbfact::SymbFact{I}
    width::I
    blkptr::FVector{I}
    blkval::FVector{T}
    status::FScalar{Bool}
end

"""
    SymbFact(cholfact::CholFact)

Get the underlying symbolic factorization of a Cholesky
factorization.
"""
function SymbFact(cholfact::CholFact)
    return cholfact.symbfact
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

```julia-repl
julia> import CliqueTrees

julia> matrix = [
           3 1 0 0 0 0 0 0
           1 3 1 0 0 2 0 0
           0 1 3 1 0 1 2 1
           0 0 1 3 0 0 0 0
           0 0 0 0 3 1 1 0
           0 2 1 0 1 3 0 0
           0 0 2 0 1 0 3 1
           0 0 1 0 0 0 1 3
       ];

julia> cholfact = CliqueTrees.cholesky(matrix)
CholFact{Float64, Int64}:
    nnz: 19
    success: true
```

### Parameters

  - `matrix`: sparse positive-definite matrix
  - `alg`: elimination algorithm
  - `snd`: supernode type

"""
function cholesky(matrix::AbstractMatrix; alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM, snd::SupernodeType=DEFAULT_SUPERNODE_TYPE)
    cholfact = cholesky(matrix, alg, snd)
    return cholfact
end

function cholesky(matrix::AbstractMatrix, alg::PermutationOrAlgorithm, snd::SupernodeType)
    cholfact = cholesky(matrix, symbolic(matrix, alg, snd))
    return cholfact
end

function cholesky(matrix::AbstractMatrix, symbfact::SymbFact)
    cholfact = cholesky(sparse(matrix), symbfact)
    return cholfact
end

function cholesky(matrix::SparseMatrixCSC{<:Any, I}, symbfact::SymbFact{I}) where {I}
    @argcheck size(matrix, 1) == size(matrix, 2)
    @argcheck size(matrix, 1) == nov(separators(symbfact.tree))
    return cholesky!(cholinit(matrix, symbfact)..., matrix, symbfact) 
end

"""
    cholesky!(cholfact::CholFact, cholwork::CholWork, matrix::AbstractMatrix, symbfact::SymbFact)

A non-allocating version of [`cholesky`](@ref).

```julia-repl
julia> import CliqueTrees

julia> matrix = [
           3 1 0 0 0 0 0 0
           1 3 1 0 0 2 0 0
           0 1 3 1 0 1 2 1
           0 0 1 3 0 0 0 0
           0 0 0 0 3 1 1 0
           0 2 1 0 1 3 0 0
           0 0 2 0 1 0 3 1
           0 0 1 0 0 0 1 3
       ];

julia> symbfact = CliqueTrees.symbolic(matrix)
SymbFact{Int64}:
    nnz: 19

julia> cholfact, cholwork = CliqueTrees.cholinit(matrix, symbfact);

julia> CliqueTrees.cholesky!(cholfact, cholwork, matrix, symbfact)
CholFact{Float64, Int64}:
    nnz: 19
    success: true
```

### Parameters

  - `cholfact`: Cholesky factor
  - `cholwork`: workspace
  - `matrix`: sparse positive-definite matrix
  - `symbfact`: symbolic factorization 
"""
function cholesky!(cholfact::CholFact{T, I}, cholwork::CholWork{T, I}, matrix::AbstractMatrix, symbfact::SymbFact{I}) where {T, I}
    return cholesky!(cholfact, cholwork, sparse(matrix), symbfact)
end

function cholesky!(cholfact::CholFact{T, I}, cholwork::CholWork{T, I}, matrix::SparseMatrixCSC{<:Any, I}, symbfact::SymbFact{I}) where {T, I}
    tree = symbfact.tree
    invp = symbfact.invp

    blkptr = cholfact.blkptr
    blkval = cholfact.blkval
    
    mapping = cholwork.mapping
    updptr = cholwork.updptr
    updval = cholwork.updval
    frtval = cholwork.frtval

    pattern0 = BipartiteGraph(matrix)
    pattern1 = cholwork.pattern1
    pattern2 = cholwork.pattern2

    nzval0 = matrix.nzval
    nzval1 = cholwork.nzval1
    nzval2 = cholwork.nzval2

    cholesky_permute!(pattern0, pattern1, nzval0, nzval1, invp)
    cholesky_reverse!(pattern1, pattern2, nzval1, nzval2)

    cholfact.status[] = cholesky_impl!(mapping, blkptr, updptr, blkval, updval, frtval, tree, pattern2, nzval2) 

    return cholfact    
end

function cholesky_permute!(
        pattern1::BipartiteGraph{I, I},
        pattern2::BipartiteGraph{I, I},
        nzval1::AbstractVector,
        nzval2::AbstractVector,
        invp::AbstractVector{I},
    ) where {I}
    neqns = nv(pattern1)

    @inbounds for i2 in vertices(pattern1)
        pointers(pattern2)[i2 + one(I)] = zero(I)
    end

    @inbounds for j1 in vertices(pattern1)
        j2 = invp[j1]
        j2 == neqns && continue

        for i1 in neighbors(pattern1, j1)
            i1 > j1 && continue

            i2 = invp[i1]
            i2 == neqns && continue

            k2 = max(i2, j2)
            pointers(pattern2)[k2 + two(I)] += one(I)
        end
    end

    @inbounds pointers(pattern2)[begin] = p2 = one(I)

    @inbounds for i2 in vertices(pattern1)
        pointers(pattern2)[i2 + one(I)] = p2 += pointers(pattern2)[i2 + one(I)]
    end

    @inbounds for j1 in vertices(pattern1)
        j2 = invp[j1]

        for p1 in incident(pattern1, j1)
            i1 = targets(pattern1)[p1]
            i1 > j1 && continue

            x1 = nzval1[p1]
            i2 = invp[i1]
            k2 = j2

            if k2 < i2
                i2, k2 = k2, i2
            end

            p2 = pointers(pattern2)[k2 + one(I)]
            pointers(pattern2)[k2 + one(I)] = p2 + one(I)
            targets(pattern2)[p2] = i2
            nzval2[p2] = x1
        end
    end

    return
end

function cholesky_reverse!(
        pattern1::BipartiteGraph{I, I},
        pattern2::BipartiteGraph{I, I},
        nzval1::AbstractVector,
        nzval2::AbstractVector,
    ) where {I}
    @inbounds for i2 in vertices(pattern1)
        pointers(pattern2)[i2 + one(I)] = zero(I)
    end

    @inbounds for j1 in vertices(pattern1), i1 in neighbors(pattern1, j1)
        i1 == nv(pattern1) && continue
        pointers(pattern2)[i1 + two(I)] += one(I)
    end

    @inbounds pointers(pattern2)[begin] = p2 = one(I)

    @inbounds for i2 in vertices(pattern1)
        pointers(pattern2)[i2 + one(I)] = p2 += pointers(pattern2)[i2 + one(I)]
    end

    @inbounds for j1 in vertices(pattern1), p1 in incident(pattern1, j1)
        i1 = targets(pattern1)[p1]
        x1 = nzval1[p1]
        p2 = pointers(pattern2)[i1 + one(I)]

        pointers(pattern2)[i1 + one(I)] = p2 + one(I)
        targets(pattern2)[p2] = j1
        nzval2[p2] = x1
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
        pattern::BipartiteGraph{I, I},
        nzval::AbstractVector,
    ) where {T, I}
    separator = separators(tree)
    relidx = targets(mapping)
    status = true; ns = zero(I)

    cholesky_init!(relidx, blkptr, updptr,
        blkval, tree, pattern, nzval)

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
        pattern::BipartiteGraph{I, I},
        nzval::AbstractVector,
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

            for q in incident(pattern, v)
                w = targets(pattern)[q]

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

function LinearAlgebra.ldiv!(cholfact::CholFact{T}, B::Union{AbstractVector, AbstractMatrix}) where {T}
    @argcheck nov(separators(cholfact.symbfact.tree)) == size(B, 1)
    tree = cholfact.symbfact.tree
    perm = cholfact.symbfact.perm
    width = cholfact.width
    blkptr = cholfact.blkptr
    blkval = cholfact.blkval
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

function LinearAlgebra.rdiv!(B::AbstractMatrix, cholfact::CholFact{T}) where {T}
    @argcheck size(B, 2) == nov(separators(cholfact.symbfact.tree))
    tree = cholfact.symbfact.tree
    perm = cholfact.symbfact.perm
    width = cholfact.width
    blkptr = cholfact.blkptr
    blkval = cholfact.blkval
    frtval = FVector{T}(undef, width * size(B, 1))

    residual = residuals(tree)
    separator = separators(tree)

    X = FArray{T}(undef, size(B))

    for w in axes(B, 1), v in axes(B, 2)
        X[w, v] = B[w, perm[v]]
    end
    
    for j in vertices(separator)
        rdiv!_loop_fwd!(blkptr, blkval, frtval, tree, X, j)
    end
    
    for j in reverse(vertices(separator))
        rdiv!_loop_bwd!(blkptr, blkval, frtval, tree, X, j)
    end

    for w in axes(B, 1), v in axes(B, 2)
        B[w, perm[v]] = X[w, v]
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
        gemm!(L₂₁, X₁, X₂, Val(false), Val(false))
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
        gemm!(L₂₁, X₂, X₁, Val(true), Val(false))
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

function rdiv!_loop_fwd!(
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

    # na is the size of the separator at node j
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
    @inbounds X = reshape(view(frtval, oneto(nj * size(B, 1))), size(B, 1), nj)
    @inbounds X₁ = view(X, axes(B, 1), oneto(nn))
    @inbounds X₂ = view(X, axes(B, 1), nn + one(I):nj)

    @inbounds for k in oneto(nj), w in axes(B, 1)
        X[w, k] = B[w, bag[k]]
    end

    # solve for Y₁ in
    #
    #     Y₁ L₁₁ᴴ = X₁
    #
    # and store X₁ ← Y₁
    trsm!(L₁₁, X₁, Val(true), Val(true))
    
    if ispositive(na)
        # compute the difference
        #
        #     Y₂ = X₂ - Y₁ L₂₁ᴴ
        #
        # and store X₂ ← Y₂ 
        gemm!(X₁, L₂₁, X₂, Val(false), Val(true))
    end

    # copy X into B
    @inbounds for k in oneto(nj), w in axes(B, 1)
        B[w, bag[k]] = X[w, k]
    end

    return
end

function rdiv!_loop_bwd!(
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

    # na is the size of the separator at node j
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
    @inbounds X = reshape(view(frtval, oneto(nj * size(B, 1))), size(B, 1), nj)
    @inbounds X₁ = view(X, axes(B, 1), oneto(nn))
    @inbounds X₂ = view(X, axes(B, 1), nn + one(I):nj)

    @inbounds for k in oneto(nj), w in axes(B, 1)
        X[w, k] = B[w, bag[k]]
    end
    
    if ispositive(na)
        # compute the difference
        #
        #     Y₁ = X₁ - X₂ L₂₁
        #
        # and store X₁ ← Y₁ 
        gemm!(X₂, L₂₁, X₁, Val(false), Val(false))
    end

    # solve for Z₁ in
    #
    #     Z₁ L₁₁ = Y₁
    #
    # and store X₁ ← Z₁
    trsm!(L₁₁, X₁, Val(false), Val(true))

    # copy X into B
    @inbounds for k in oneto(nj), w in axes(B, 1)
        B[w, bag[k]] = X[w, k]
    end

    return
end

function LinearAlgebra.issuccess(cholfact::CholFact)
    return cholfact.status[]
end

function LinearAlgebra.isposdef(cholfact::CholFact)
    return issuccess(cholfact)
end

function LinearAlgebra.det(cholfact::CholFact{T, I}) where {T, I}
    tree = cholfact.symbfact.tree
    blkptr = cholfact.blkptr
    blkval = cholfact.blkval

    residual = residuals(tree)
    separator = separators(tree)

    det = one(real(T))

    @inbounds for j in vertices(separator)
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

        # L is part of the Cholesky factor
        #
        #          res(j)
        #     L = [ L₁₁  ] res(j)
        #         [ L₂₁  ] sep(j)
        #
        pstrt = blkptr[j]
        pstop = blkptr[j + one(I)]
        L = reshape(view(blkval, pstrt:pstop - one(I)), nj, nn)
 
        for k in oneto(nn)
            Lkk = real(L[k, k])
            det *= Lkk * Lkk
        end
    end

    return det
end

function LinearAlgebra.logdet(cholfact::CholFact{T, I}) where {T, I}
    tree = cholfact.symbfact.tree
    blkptr = cholfact.blkptr
    blkval = cholfact.blkval

    residual = residuals(tree)
    separator = separators(tree)

    logdet = zero(real(T))

    @inbounds for j in vertices(separator)
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

        # L is part of the Cholesky factor
        #
        #          res(j)
        #     L = [ L₁₁  ] res(j)
        #         [ L₂₁  ] sep(j)
        #
        pstrt = blkptr[j]
        pstop = blkptr[j + one(I)]
        L = reshape(view(blkval, pstrt:pstop - one(I)), nj, nn)
 
        for k in oneto(nn)
            Lkk = log(real(L[k, k]))
            logdet += Lkk + Lkk
        end
    end

    return logdet
end

function SparseArrays.nnz(cholfact::CholFact)
    return nnz(cholfact.symbfact)
end

function Base.show(io::IO, ::MIME"text/plain", cholfact::CholFact{T, I}) where {T, I}
    println(io, "CholFact{$T, $I}:")
    println(io, "    nnz: $(nnz(cholfact))")
    print(io,   "    success: $(issuccess(cholfact))")
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
#     X L = B
#
# and store B ← X
function trsm!(L::AbstractMatrix{T}, B::AbstractMatrix{T}, tL::Val{false}, side::Val{true}) where {T}
    rdiv!(B, LowerTriangular(L))
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
function gemm!(A::AbstractMatrix{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}, tA::Val{false}, tX::Val{false}) where {T}
    mul!(Y, A, X, -one(T), one(T))
    return
end

# compute the difference
#
#     Z = Y - Aᴴ X
#
# and store Y ← Z
function gemm!(A::AbstractMatrix{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}, tA::Val{true}, tX::Val{false}) where {T}
    mul!(Y, A |> adjoint, X, -one(T), one(T))
    return
end

# compute the difference
#
#     Z = Y - A Xᴴ
#
# and store Y ← Z
function gemm!(A::AbstractMatrix{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}, tA::Val{false}, tX::Val{true}) where {T}
    mul!(Y, A, X |> adjoint, -one(T), one(T))
    return
end

# compute the difference
#
#     Z = Y - Aᴴ Xᴴ
#
# and store Y ← Z
function gemm!(A::AbstractMatrix{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}, tA::Val{true}, tX::Val{true}) where {T}
    mul!(Y, A |> adjoint, X |> adjoint, -one(T), one(T))
    return
end
