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
    residual = residuals(tree)
    separator = separators(tree)

    tril!(permute!(matrix, perm, perm))
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

    status = cholesky!_impl!(mapping, blkptr,
        updptr, blkval, updval, frtval, tree, matrix)

    return CholFact(fact, njmax, blkptr, blkval, status)
end 

function cholesky!_impl!(
        mapping::BipartiteGraph{I, I},        
        blkptr::AbstractVector{I},
        updptr::AbstractVector{I},
        blkval::AbstractVector{T},
        updval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I},
        matrix::SparseMatrixCSC{T, I},
    ) where {T, I}
    separator = separators(tree)
    relidx = targets(mapping)
    status = true; ns = zero(I)

    cholesky!_init!(relidx, blkptr,
        updptr, blkval, tree, matrix)

    for j in vertices(separator)
        iterstatus, ns = cholesky!_loop!(mapping, blkptr,
            updptr, blkval, updval, frtval, tree, ns, j)

        status = status && iterstatus
    end

    return status
end

function cholesky!_init!(
        relidx::AbstractVector{I},
        blkptr::AbstractVector{I},
        updptr::AbstractVector{I},
        blkval::AbstractVector{T},
        tree::CliqueTree{I, I},
        matrix::SparseMatrixCSC{T, I},
    ) where {T, I}
    residual = residuals(tree)
    separator = separators(tree)
    treln = nv(separator)

    updptr[one(I)] = p = q = one(I)

    for j in vertices(separator)
        blkptr[j] = p

        res = neighbors(residual, j)
        sep = neighbors(separator, j)
        bag = Clique(res, sep)

        nn = eltypedegree(residual, j)
        na = eltypedegree(separator, j)
        nj = nn + na

        for v in res
            k = one(I)

            for q in nzrange(matrix, v)
                w = rowvals(matrix)[q]

                while bag[k] < w
                    blkval[p] = zero(T)
                    k += one(I)
                    p += one(I)
                end

                blkval[p] = nonzeros(matrix)[q]
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

    blkptr[treln + one(I)] = p
    return
end

function cholesky!_loop!(
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
        cholesky!_add_update!(F, ns, i, mapping, updptr, updval)
        ns -= one(I)
    end

    # factorize F₁₁ as
    #
    #     F₁₁ = L₁₁ L₁₁ᵀ
    #
    # and store F₁₁ ← L₁₁
    status = potrf!(F₁₁)

    if ispositive(na)
        # solve for L₂₁ in
        #
        #     L₂₁ L₁₁ᵀ = F₂₁
        #
        # and store F₂₁ ← L₂₁
        trsm!(F₁₁, F₂₁, Val(true), Val(true))
    
        # compute
        #
        #    U₂₂ = F₂₂ - L₂₁ L₂₁ᵀ
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

function cholesky!_add_update!(
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

        # for all ui in sep(i) ...
        for ui in oneto(na)
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
    X = Array{T}(undef, size(B))

    for w in axes(B, 2), v in axes(B, 1)
        X[v, w] = B[v, w]
    end

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
    nn = eltypedegree(residual, j)

    # na is the size of the separator at node j.
    #
    #     na = | sep(j) |
    #
    na = eltypedegree(separator, j)

    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na

    # bag is the bag at node j
    bag = tree[j] 
    
    # L is part of the Cholesky factor
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    pstrt = blkptr[j]
    pstop = blkptr[j + one(I)]
    L = reshape(view(blkval, pstrt:pstop - one(I)), nj, nn)
    
    L₁₁ = view(L, oneto(nn),      oneto(nn))
    L₂₁ = view(L, nn + one(I):nj, oneto(nn))

    # X is part of B
    #
    #     X = [ X₁ ] res(j)
    #         [ X₂ ] sep(j)
    #
    X = view(frtval, oneto(nj))
    X₁ = view(X, oneto(nn))
    X₂ = view(X, nn + one(I):nj)

    for k in oneto(nj)
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
    for k in oneto(nj)
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
    nn = eltypedegree(residual, j)

    # na is the size of the separator at node j.
    #
    #     na = | sep(j) |
    #
    na = eltypedegree(separator, j)

    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na

    # bag is the bag at node j
    bag = tree[j] 
    
    # L is part of the Cholesky factor
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    pstrt = blkptr[j]
    pstop = blkptr[j + one(I)]
    L = reshape(view(blkval, pstrt:pstop - one(I)), nj, nn)
    
    L₁₁ = view(L, oneto(nn),      oneto(nn))
    L₂₁ = view(L, nn + one(I):nj, oneto(nn))

    # X is part of B
    #
    #     X = [ X₁ ] res(j)
    #         [ X₂ ] sep(j)
    #
    X = reshape(view(frtval, oneto(nj * size(B, 2))), nj, size(B, 2))
    X₁ = view(X, oneto(nn),      axes(B, 2))
    X₂ = view(X, nn + one(I):nj, axes(B, 2))

    for w in axes(B, 2), k in oneto(nj)
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

    for w in axes(B, 2), k in oneto(nj)
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
    nn = eltypedegree(residual, j)

    # na is the size of the separator at node j.
    #
    #     na = | sep(j) |
    #
    na = eltypedegree(separator, j)

    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na

    # bag is the bag at node j
    bag = tree[j]        
 
    # L is part of the Cholesky factor
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    pstrt = blkptr[j]
    pstop = blkptr[j + one(I)]
    L = reshape(view(blkval, pstrt:pstop - one(I)), nj, nn)
    
    L₁₁ = view(L, oneto(nn),      oneto(nn))
    L₂₁ = view(L, nn + one(I):nj, oneto(nn))

    # X is part of B
    #
    #     X = [ X₁ ] res(j)
    #         [ X₂ ] sep(j)
    #
    X = view(frtval, oneto(nj))
    X₁ = view(X, oneto(nn))
    X₂ = view(X, nn + one(I):nj)

    for k in oneto(nj)
        X[k] = B[bag[k]]
    end
    
    if ispositive(na)
        # compute the difference
        #
        #     Y₁ = X₁ - L₂₁ᵀ X₂
        #
        # and store X₁ ← Y₁ 
        gemv!(L₂₁, X₂, X₁, Val(true))
    end

    # solve for Z₁ in
    #
    #     L₁₁ᵀ Z₁ = Y₁
    #
    # and store X₁ ← Z₁
    trsv!(L₁₁, X₁, Val(true))

    # copy X into b
    for k in oneto(nj)
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
    nn = eltypedegree(residual, j)

    # na is the size of the separator at node j.
    #
    #     na = | sep(j) |
    #
    na = eltypedegree(separator, j)

    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na

    # bag is the bag at node j
    bag = tree[j]        
 
    # L is part of the Cholesky factor
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    pstrt = blkptr[j]
    pstop = blkptr[j + one(I)]
    L = reshape(view(blkval, pstrt:pstop - one(I)), nj, nn)
    
    L₁₁ = view(L, oneto(nn),      oneto(nn))
    L₂₁ = view(L, nn + one(I):nj, oneto(nn))

    # X is part of B
    #
    #     X = [ X₁ ] res(j)
    #         [ X₂ ] sep(j)
    #
    X = reshape(view(frtval, oneto(nj * size(B, 2))), nj, size(B, 2))
    X₁ = view(X, oneto(nn),      axes(B, 2))
    X₂ = view(X, nn + one(I):nj, axes(B, 2))

    for w in axes(B, 2), k in oneto(nj)
        X[k, w] = B[bag[k], w]
    end
    
    if ispositive(na)
        # compute the difference
        #
        #     Y₁ = X₁ - L₂₁ᵀ X₂
        #
        # and store X₁ ← Y₁ 
        gemm!(L₂₁, X₂, X₁, Val(true))
    end

    # solve for Z₁ in
    #
    #     L₁₁ᵀ Z₁ = Y₁
    #
    # and store X₁ ← Z₁
    trsm!(L₁₁, X₁, Val(true), Val(false))

    # copy X into b
    for w in axes(B, 2), k in oneto(nj)
        B[bag[k], w] = X[k, w]
    end

    return
end

function Base.show(io::IO, ::MIME"text/plain", F::CholFact{T, I}) where {T, I}
    println(io, "CholFact{$T, $I}:")
    print(io,   "    success: $(F.status)")
end
