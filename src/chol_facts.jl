"""
    CholFact{T, I} <: Factorization{T}

A Cholesky factorization object.
"""
struct CholFact{T, I} <: Factorization{T}
    symbfact::SymbFact{I}
    blkptr::FVector{I}
    blkval::FVector{T}
    status::FScalar{Bool}
    mapping::BipartiteGraph{I, I, FVector{I}, FVector{I}}
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
    return cholesky!(cholinit(matrix, symbfact)..., matrix) 
end

"""
    cholesky!(cholfact::CholFact, cholwork::CholWork, matrix::AbstractMatrix)

Compute the Cholesky factorization of a sparse positive definite matrix
using a pre-allocated workspace. See [`cholesky`](@ref).

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
"""
function cholesky!(cholfact::CholFact{T, I}, cholwork::CholWork{T, I}, matrix::AbstractMatrix) where {T, I}
    return cholesky!(cholfact, cholwork, sparse(matrix))
end

function cholesky!(cholfact::CholFact{T, I}, cholwork::CholWork{T, I}, matrix::SparseMatrixCSC{<:Any, I}) where {T, I}
    symbfact = cholfact.symbfact

    tree = symbfact.tree
    invp = symbfact.invp

    mapping = cholfact.mapping

    blkptr = cholfact.blkptr
    blkval = cholfact.blkval

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
        nzval2[p2] = conj(x1)
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

    @inbounds for j in vertices(separator)
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

    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    @inbounds nn = eltypedegree(residual, j)

    if isone(nn)
        status, ns = cholesky_loop_nod!(mapping, blkptr,
            updptr, blkval, updval, frtval, tree, ns, j)
    else
        status, ns = cholesky_loop_snd!(mapping, blkptr,
            updptr, blkval, updval, frtval, tree, ns, j)
    end

    return status, ns
end


function cholesky_loop_snd!(
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

    # F is the frontal matrix at node j
    #
    #           nn  na
    #     F = [ F₁  F₂  ] nj
    #
    #       = [ F₁₁ F₁₂ ] nn
    #         [ F₂₁ F₂₂ ] na
    #
    # only the lower triangular part is used
    @inbounds F = reshape(view(frtval, oneto(nj * nj)), nj, nj)
    @inbounds F₁ =  view(F, oneto(nj),      oneto(nn))
    @inbounds F₁₁ = view(F, oneto(nn),      oneto(nn))
    @inbounds F₂₁ = view(F, nn + one(I):nj, oneto(nn))
    @inbounds F₂₂ = view(F, nn + one(I):nj, nn + one(I):nj)

    # B is part of the Cholesky factor
    #
    #          res(j)
    #     B = [ B₁₁  ] res(j)
    #         [ B₂₁  ] sep(j)
    #
    @inbounds pstrt = blkptr[j]
    @inbounds pstop = blkptr[j + one(I)]
    @inbounds B = reshape(view(blkval, pstrt:pstop - one(I)), nj, nn)

    # copy B into F₁
    #
    #     F₁₁ ← B₁₁
    #     F₂₁ ← B₂₁
    #
    lacpy!(F, B); fill!(F₂₂, zero(T))

    @inbounds for i in Iterators.reverse(childindices(tree, j))
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
        ns += one(I)

        # U₂₂ is the update matrix for node j
        @inbounds pstrt = updptr[ns]
        @inbounds pstop = updptr[ns + one(I)] = pstrt + na * na
        @inbounds U₂₂ = reshape(view(updval, pstrt:pstop - one(I)), na, na)
        lacpy!(U₂₂, F₂₂)

        # solve for L₂₁ in
        #
        #     L₂₁ L₁₁ᴴ = F₂₁
        #
        # and store F₂₁ ← L₂₁
        rdiv!(F₂₁, LowerTriangular(F₁₁) |> adjoint)
    
        # compute the difference
        #
        #    L₂₂ = U₂₂ - L₂₁ L₂₁ᴴ
        #
        # and store U₂₂ ← L₂₂
        syrk!(F₂₁, U₂₂)
    end
 
    # copy F₁ into B
    #
    #     B₁₁ ← F₁₁
    #     B₂₁ ← F₂₁
    #
    lacpy!(B, F₁)
    return status, ns
end

function cholesky_loop_nod!(
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
    nn = one(I)

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

    # F is the frontal matrix at node j
    #
    #           nn  na
    #     F = [ F₁  F₂  ] nj
    #
    #       = [ F₁₁ F₁₂ ] nn
    #         [ F₂₁ F₂₂ ] na
    #
    # only the lower triangular part is used
    @inbounds F = reshape(view(frtval, oneto(nj * nj)), nj, nj)
    @inbounds F₁ =  view(F, oneto(nj),      nn)
    @inbounds F₁₁ = view(F, nn,             nn)
    @inbounds F₂₁ = view(F, nn + one(I):nj, nn)
    @inbounds F₂₂ = view(F, nn + one(I):nj, nn + one(I):nj)

    # B is part of the Cholesky factor
    #
    #          res(j)
    #     B = [ B₁₁  ] res(j)
    #         [ B₂₁  ] sep(j)
    #
    @inbounds pstrt = blkptr[j]
    @inbounds pstop = blkptr[j + one(I)]
    @inbounds B = view(blkval, pstrt:pstop - one(I))
    @inbounds B₁ = view(B, nn)
    @inbounds B₂ = view(B, nn + one(I):nj)

    # copy B into F₁
    #
    #     F₁₁ ← B₁₁
    #     F₂₁ ← B₂₁
    #
    F₁₁[] = B₁[]; copyto!(F₂₁, B₂); fill!(F₂₂, zero(T))

    @inbounds for i in Iterators.reverse(childindices(tree, j))
        cholesky_add_update!(F, ns, i, mapping, updptr, updval)
        ns -= one(I)
    end

    # factorize F₁₁ as
    #
    #     F₁₁ = L₁₁ L₁₁ᴴ
    #
    # and store F₁₁ ← L₁₁
    f₁₁ = F₁₁[]

    if ispositive(real(f₁₁))
        f₁₁ = sqrt(f₁₁); status = true
    else
        f₁₁ = one(T);    status = false
    end

    if ispositive(na)
        ns += one(I)

        # U₂₂ is the update matrix for node j
        @inbounds pstrt = updptr[ns]
        @inbounds pstop = updptr[ns + one(I)] = pstrt + na * na
        @inbounds U₂₂ = reshape(view(updval, pstrt:pstop - one(I)), na, na)
        lacpy!(U₂₂, F₂₂)

        # solve for L₂₁ in
        #
        #     L₂₁ L₁₁ᴴ = F₂₁
        #
        # and store F₂₁ ← L₂₁
        F₂₁ ./= conj(f₁₁)
    
        # compute the difference
        #
        #    L₂₂ = U₂₂ - L₂₁ L₂₁ᴴ
        #
        # and store U₂₂ ← L₂₂
        syr!(F₂₁, U₂₂)
    end

    # copy F₁ into B
    #
    #     B₁₁ ← F₁₁
    #     B₂₁ ← F₂₁
    #
    B₁[] = f₁₁; copyto!(B₂, F₂₁)
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

function rdiv!_fwd_add_update!(
        F::AbstractVector{T},
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
    @inbounds na = eltypedegree(mapping, i)

    # ind is the subset inclusion
    #
    #     ind: sep(i) → sep(parent(i))
    #
    @inbounds ind = neighbors(mapping, i)

    # U is the update vector for node i.
    @inbounds pstrt = updptr[ns]
    pstop = pstrt + na
    @inbounds U = view(updval, pstrt:pstop - one(I))

    # for all u in sep(i) ...
    @inbounds for u in oneto(na)
        # let f = ind(u)
        f = ind[u]

        # compute the sum
        #
        #     F[f] + U[u]
        #
        # and assign it to F[f]
        F[f] += U[u]
    end

    return
end

function rdiv!_fwd_add_update!(
        F::AbstractMatrix{T},
        ns::I,
        i::I,
        mapping::BipartiteGraph{I, I},
        updptr::AbstractVector{I},
        updval::AbstractVector{T},
    ) where {T, I}
    nrhs = convert(I, size(F, 1))

    # na is the size of the separator at node i.
    #
    #     na = | sep(i) |
    #
    @inbounds na = eltypedegree(mapping, i)

    # ind is the subset inclusion
    #
    #     ind: sep(i) → sep(parent(i))
    #
    @inbounds ind = neighbors(mapping, i)

    # U is the update vector for node i.
    @inbounds pstrt = updptr[ns]
    pstop = pstrt + na * nrhs
    @inbounds U = reshape(view(updval, pstrt:pstop - one(I)), nrhs, na)

    # for all u in sep(i) ...
    @inbounds for u in oneto(na)
        # let f = ind(u)
        f = ind[u]

        # for all rows c ...
        for c in oneto(nrhs)
            # compute the sum
            #
            #     F[c, f] + U[c, u]
            #
            # and assign it to F[c, f]
            F[c, f] += U[c, u]
        end
    end

    return
end

function rdiv!_bwd_add_update!(
        F::AbstractVector{T},
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
    @inbounds na = eltypedegree(mapping, i)

    # ind is the subset inclusion
    #
    #     ind: sep(i) → sep(parent(i))
    #
    @inbounds ind = neighbors(mapping, i)

    # U is the update vector for node i.
    @inbounds pstrt = updptr[ns]
    @inbounds pstop = updptr[ns + one(I)] = pstrt + na
    @inbounds U = view(updval, pstrt:pstop - one(I))

    # for all u in sep(i) ...
    @inbounds for u in oneto(na)
        # let f = ind(u)
        f = ind[u]

        # assign U[u] ← F[f]
        U[u] = F[f]
    end

    return
end

function rdiv!_bwd_add_update!(
        F::AbstractMatrix{T},
        ns::I,
        i::I,
        mapping::BipartiteGraph{I, I},
        updptr::AbstractVector{I},
        updval::AbstractVector{T},
    ) where {T, I}
    nrhs = convert(I, size(F, 1))

    # na is the size of the separator at node i.
    #
    #     na = | sep(i) |
    #
    @inbounds na = eltypedegree(mapping, i)

    # ind is the subset inclusion
    #
    #     ind: sep(i) → sep(parent(i))
    #
    @inbounds ind = neighbors(mapping, i)

    # U is the update vector for node i.
    @inbounds pstrt = updptr[ns]
    @inbounds pstop = updptr[ns + one(I)] = pstrt + na * nrhs
    @inbounds U = reshape(view(updval, pstrt:pstop - one(I)), nrhs, na)

    # for all u in sep(i) ...
    @inbounds for u in oneto(na)
        # let f = ind(u)
        f = ind[u]

        # for all rows c...
        for c in oneto(nrhs)
            # assign U[c, u] ← F[c, f]
            U[c, u] = F[c, f]
        end
    end

    return
end

function rdiv!_loop_fwd!(
        mapping::BipartiteGraph{I, I},
        blkptr::AbstractVector{I},
        updptr::AbstractVector{I},
        blkval::AbstractVector{T},
        updval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I}, 
        B::AbstractArray{T},
        ns::I,
        j::I,
    ) where {T, I}
    residual = residuals(tree)

    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    @inbounds nn = eltypedegree(residual, j)

    if isone(nn)
        ns = rdiv!_loop_fwd_nod!(mapping, blkptr, updptr,
            blkval, updval, frtval, tree, B, ns, j)
    else
        ns = rdiv!_loop_fwd_snd!(mapping, blkptr, updptr,
            blkval, updval, frtval, tree, B, ns, j)
    end

    return ns
end

function rdiv!_loop_fwd_snd!(
        mapping::BipartiteGraph{I, I},
        blkptr::AbstractVector{I},
        updptr::AbstractVector{I},
        blkval::AbstractVector{T},
        updval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I}, 
        B::AbstractVector{T},
        ns::I,
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

    # res is the residual at node j
    @inbounds res = neighbors(residual, j)
    
    # L is part of the Cholesky factor
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    @inbounds pstrt = blkptr[j]
    @inbounds pstop = blkptr[j + one(I)]
    @inbounds L = reshape(view(blkval, pstrt:pstop - one(I)), nj, :)
    @inbounds L₁₁ = view(L, oneto(nn),      :)
    @inbounds L₂₁ = view(L, nn + one(I):nj, :)

    # F is part of B
    #
    #     F = [ F₁ ] res(j)
    #         [ F₂ ] sep(j)
    #
    @inbounds F = view(frtval, oneto(nj))
    @inbounds F₁ = view(F, oneto(nn))
    @inbounds F₂ = view(F, nn + one(I):nj)

    @inbounds F₁ .= view(B, res)
    F₂ .= zero(T)

    @inbounds for i in Iterators.reverse(childindices(tree, j))
        rdiv!_fwd_add_update!(F, ns, i, mapping, updptr, updval)
        ns -= one(I)
    end

    # solve for Y₁ in
    #
    #     L₁₁ Y₁ = F₁
    #
    # and store F₁ ← Y₁
    ldiv!(LowerTriangular(L₁₁), F₁)
 
    if ispositive(na)
        # U₂ is the update matrix for node j
        ns += one(I)
        @inbounds pstrt = updptr[ns]
        @inbounds pstop = updptr[ns + one(I)] = pstrt + na
        @inbounds U₂ = view(updval, pstrt:pstop - one(I))
        U₂ .= F₂

        # compute the difference
        #
        #     Y₂ = U₂ - L₂₁ F₁
        #
        # and store U₂ ← Y₂ 
        mul!(U₂, L₂₁, F₁, -one(T), one(T))
    end
   
    # copy B ← F₁
    @inbounds B[res] .= F₁
    return ns
end

function rdiv!_loop_fwd_snd!(
        mapping::BipartiteGraph{I, I},
        blkptr::AbstractVector{I},
        updptr::AbstractVector{I},
        blkval::AbstractVector{T},
        updval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I}, 
        B::AbstractMatrix{T},
        ns::I,
        j::I,
    ) where {T, I}
    nrhs = convert(I, size(B, 1))

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

    # res is the residual at node j
    @inbounds res = neighbors(residual, j)
    
    # L is part of the Cholesky factor
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    @inbounds pstrt = blkptr[j]
    @inbounds pstop = blkptr[j + one(I)]
    @inbounds L = reshape(view(blkval, pstrt:pstop - one(I)), nj, :)
    @inbounds L₁₁ = view(L, oneto(nn),      :)
    @inbounds L₂₁ = view(L, nn + one(I):nj, :)

    # F is part of B
    #
    #         res(j) sep(j)
    #     F = [ F₁    F₂ ]
    #
    @inbounds F = reshape(view(frtval, oneto(nj * nrhs)), :, nj)
    @inbounds F₁ = view(F, :, oneto(nn))
    @inbounds F₂ = view(F, :, nn + one(I):nj)

    @inbounds F₁ .= view(B, :, res)
    F₂ .= zero(T)

    @inbounds for i in Iterators.reverse(childindices(tree, j))
        rdiv!_fwd_add_update!(F, ns, i, mapping, updptr, updval)
        ns -= one(I)
    end

    # solve for Y₁ in
    #
    #     Y₁ L₁₁ᴴ = F₁
    #
    # and store F₁ ← Y₁
    rdiv!(F₁, LowerTriangular(L₁₁) |> adjoint)
 
    if ispositive(na)
        # U₂ is the update matrix for node j
        ns += one(I)
        @inbounds pstrt = updptr[ns]
        @inbounds pstop = updptr[ns + one(I)] = pstrt + na * nrhs
        @inbounds U₂ = reshape(view(updval, pstrt:pstop - one(I)), nrhs, na)
        U₂ .= F₂

        # compute the difference
        #
        #     Y₂ = U₂ - Y₁ L₂₁ᴴ
        #
        # and store U₂ ← Y₂ 
        mul!(U₂, F₁, L₂₁ |> adjoint, -one(T), one(T))
    end
   
    # copy B ← F₁
    @inbounds B[:, res] .= F₁
    return ns
end

function rdiv!_loop_fwd_nod!(
        mapping::BipartiteGraph{I, I},
        blkptr::AbstractVector{I},
        updptr::AbstractVector{I},
        blkval::AbstractVector{T},
        updval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I}, 
        B::AbstractVector{T},
        ns::I,
        j::I,
    ) where {T, I}
    residual = residuals(tree)
    separator = separators(tree)

    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    @inbounds nn = one(I)

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

    # res is the residual at node j
    @inbounds res = only(neighbors(residual, j))
    
    # L is part of the Cholesky factor
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    @inbounds pstrt = blkptr[j]
    @inbounds pstop = blkptr[j + one(I)]
    @inbounds L = view(blkval, pstrt:pstop - one(I))
    @inbounds L₁₁ = view(L, nn)
    @inbounds L₂₁ = view(L, nn + one(I):nj)

    # F is part of B
    #
    #     F = [ F₁ ] res(j)
    #         [ F₂ ] sep(j)
    #
    @inbounds F = view(frtval, oneto(nj))
    @inbounds F₁ = view(F, nn)
    @inbounds F₂ = view(F, nn + one(I):nj)

    @inbounds F₁[] = B[res]
    F₂ .= zero(T)

    @inbounds for i in Iterators.reverse(childindices(tree, j))
        rdiv!_fwd_add_update!(F, ns, i, mapping, updptr, updval)
        ns -= one(I)
    end

    # solve for Y₁ in
    #
    #     L₁₁ Y₁ = F₁
    #
    # and store F₁ ← Y₁
    f₁ = F₁[] / L₁₁[]
 
    if ispositive(na)
        # U₂ is the update matrix for node j
        ns += one(I)
        @inbounds pstrt = updptr[ns]
        @inbounds pstop = updptr[ns + one(I)] = pstrt + na
        @inbounds U₂ = view(updval, pstrt:pstop - one(I))
        U₂ .= F₂

        # compute the difference
        #
        #     Y₂ = U₂ - L₂₁ F₁
        #
        # and store U₂ ← Y₂
        mul!(U₂, L₂₁, f₁, -one(T), one(T))
    end
   
    # copy B ← F₁
    @inbounds B[res] = f₁

    return ns
end

function rdiv!_loop_fwd_nod!(
        mapping::BipartiteGraph{I, I},
        blkptr::AbstractVector{I},
        updptr::AbstractVector{I},
        blkval::AbstractVector{T},
        updval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I}, 
        B::AbstractMatrix{T},
        ns::I,
        j::I,
    ) where {T, I}
    nrhs = convert(I, size(B, 1))

    residual = residuals(tree)
    separator = separators(tree)

    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    @inbounds nn = one(I)

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

    # res is the residual at node j
    @inbounds res = only(neighbors(residual, j))
    
    # L is part of the Cholesky factor
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    @inbounds pstrt = blkptr[j]
    @inbounds pstop = blkptr[j + one(I)]
    @inbounds L = view(blkval, pstrt:pstop - one(I))
    @inbounds L₁₁ = view(L, nn)
    @inbounds L₂₁ = view(L, nn + one(I):nj)

    # F is part of B
    #
    #         res(j) sep(j)
    #     F = [ F₁    F₂ ]
    #
    @inbounds F = reshape(view(frtval, oneto(nj * nrhs)), :, nj)
    @inbounds F₁ = view(F, :, nn)
    @inbounds F₂ = view(F, :, nn + one(I):nj)

    @inbounds F₁ .= view(B, :, res)

    F₂ .= zero(T)

    @inbounds for i in Iterators.reverse(childindices(tree, j))
        rdiv!_fwd_add_update!(F, ns, i, mapping, updptr, updval)
        ns -= one(I)
    end

    # solve for Y₁ in
    #
    #     Y₁ L₁₁ᴴ = F₁
    #
    # and store F₁ ← Y₁
    F₁ ./= conj(L₁₁[])
 
    if ispositive(na)
        # U₂ is the update matrix for node j
        ns += one(I)
        @inbounds pstrt = updptr[ns]
        @inbounds pstop = updptr[ns + one(I)] = pstrt + na * nrhs
        @inbounds U₂ = reshape(view(updval, pstrt:pstop - one(I)), :, na)
        U₂ .= F₂

        # compute the difference
        #
        #     Y₂ = U₂ - Y₁ L₂₁ᴴ
        #
        # and store U₂ ← Y₂ 
        mul!(U₂, F₁, L₂₁ |> adjoint, -one(T), one(T))
    end
   
    # copy B ← F₁
    @inbounds B[:, res] .= F₁
    return ns
end

function rdiv!_loop_bwd!(
        mapping::BipartiteGraph{I, I},
        blkptr::AbstractVector{I},
        updptr::AbstractVector{I},
        blkval::AbstractVector{T},
        updval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I}, 
        B::AbstractArray{T},
        ns::I,
        j::I,
    ) where {T, I}
    residual = residuals(tree)

    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    @inbounds nn = eltypedegree(residual, j)

    if isone(nn)
        ns = rdiv!_loop_bwd_nod!(mapping, blkptr, updptr,
            blkval, updval, frtval, tree, B, ns, j)
    else
        ns = rdiv!_loop_bwd_snd!(mapping, blkptr, updptr,
            blkval, updval, frtval, tree, B, ns, j)
    end

    return ns
end

function rdiv!_loop_bwd_snd!(
        mapping::BipartiteGraph{I, I},
        blkptr::AbstractVector{I},
        updptr::AbstractVector{I},
        blkval::AbstractVector{T},
        updval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I}, 
        B::AbstractVector{T},
        ns::I,
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

    # res is the res at node j
    @inbounds res = neighbors(residual, j)
 
    # L is part of the Cholesky factor
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    @inbounds pstrt = blkptr[j]
    @inbounds pstop = blkptr[j + one(I)]
    @inbounds L = reshape(view(blkval, pstrt:pstop - one(I)), nj, :)
    @inbounds L₁₁ = view(L, oneto(nn),      :)
    @inbounds L₂₁ = view(L, nn + one(I):nj, :)

    # F is part of B
    #
    #     F = [ F₁ ] res(j)
    #         [ F₂ ] sep(j)
    #
    @inbounds F = view(frtval, oneto(nj))
    @inbounds F₁ = view(F, oneto(nn))
    @inbounds F₂ = view(F, nn + one(I):nj)

    @inbounds F₁ .= view(B, res)
    
    if ispositive(na)
        # U₂ is the update matrix for node j
        @inbounds pstrt = updptr[ns]
        pstop = pstrt + na
        @inbounds U₂ = view(updval, pstrt:pstop - one(I))
        ns -= one(I)

        # compute the difference
        #
        #     Y₁ = F₁ - L₂₁ᴴ U₂
        #
        # and store F₁ ← Y₁ 
        mul!(F₁, L₂₁ |> adjoint, U₂, -one(T), one(T))

        # copy F₂ ← U₂
        F₂ .= U₂
    end

    # solve for Z₁ in
    #
    #     L₁₁ᴴ Z₁ = Y₁
    #
    # and store F₁ ← Z₁
    ldiv!(LowerTriangular(L₁₁) |> adjoint, F₁)

    @inbounds for i in childindices(tree, j)
        ns += one(I)
        rdiv!_bwd_add_update!(F, ns, i, mapping, updptr, updval)
    end

    # copy B ← F₁
    @inbounds B[res] .= F₁
    return ns
end

function rdiv!_loop_bwd_snd!(
        mapping::BipartiteGraph{I, I},
        blkptr::AbstractVector{I},
        updptr::AbstractVector{I},
        blkval::AbstractVector{T},
        updval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I}, 
        B::AbstractMatrix{T},
        ns::I,
        j::I,
    ) where {T, I}
    nrhs = convert(I, size(B, 1))

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

    # res is the res at node j
    @inbounds res = neighbors(residual, j)
 
    # L is part of the Cholesky factor
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    @inbounds pstrt = blkptr[j]
    @inbounds pstop = blkptr[j + one(I)]
    @inbounds L = reshape(view(blkval, pstrt:pstop - one(I)), nj, :)
    @inbounds L₁₁ = view(L, oneto(nn),      :)
    @inbounds L₂₁ = view(L, nn + one(I):nj, :)

    # F is part of B
    #
    #         res(j) sep(j)
    #     F = [ F₁    F₂ ]
    #
    @inbounds F = reshape(view(frtval, oneto(nj * nrhs)), :, nj)
    @inbounds F₁ = view(F, :, oneto(nn))
    @inbounds F₂ = view(F, :, nn + one(I):nj)

    @inbounds F₁ .= view(B, :, res)

    if ispositive(na)
        # U₂ is the update matrix for node j
        @inbounds pstrt = updptr[ns]
        pstop = pstrt + na * nrhs
        @inbounds U₂ = reshape(view(updval, pstrt:pstop - one(I)), :, na)
        ns -= one(I)

        # compute the difference
        #
        #     Y₁ = F₁ - U₂ L₂₁
        #
        # and store F₁ ← Y₁ 
        mul!(F₁, U₂, L₂₁, -one(T), one(T))

        # copy F₂ ← U₂
        F₂ .= U₂
    end

    # solve for Z₁ in
    #
    #     Z₁ L₁₁ = Y₁
    #
    # and store F₁ ← Z₁
    rdiv!(F₁, LowerTriangular(L₁₁))

    @inbounds for i in childindices(tree, j)
        ns += one(I)
        rdiv!_bwd_add_update!(F, ns, i, mapping, updptr, updval)
    end

    # copy B ← F₁
    @inbounds B[:, res] .= F₁
    return ns
end

function rdiv!_loop_bwd_nod!(
        mapping::BipartiteGraph{I, I},
        blkptr::AbstractVector{I},
        updptr::AbstractVector{I},
        blkval::AbstractVector{T},
        updval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I}, 
        B::AbstractVector{T},
        ns::I,
        j::I,
    ) where {T, I}
    residual = residuals(tree)
    separator = separators(tree)

    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    @inbounds nn = one(I)

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

    # res is the res at node j
    @inbounds res = only(neighbors(residual, j))
 
    # L is part of the Cholesky factor
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    @inbounds pstrt = blkptr[j]
    @inbounds pstop = blkptr[j + one(I)]
    @inbounds L = view(blkval, pstrt:pstop - one(I))
    @inbounds L₁₁ = view(L, nn)
    @inbounds L₂₁ = view(L, nn + one(I):nj)

    # F is part of B
    #
    #     F = [ F₁ ] res(j)
    #         [ F₂ ] sep(j)
    #
    @inbounds F = view(frtval, oneto(nj))
    @inbounds F₁ = view(F, nn)
    @inbounds F₂ = view(F, nn + one(I):nj)

    @inbounds f₁ = F₁[] = B[res]
    
    if ispositive(na)
        # U₂ is the update matrix for node j
        @inbounds pstrt = updptr[ns]
        pstop = pstrt + na
        @inbounds U₂ = view(updval, pstrt:pstop - one(I))
        ns -= one(I)

        # compute the difference
        #
        #     Y₁ = F₁ - L₂₁ᴴ U₂
        #
        # and store F₁ ← Y₁ 
        f₁ -= dot(L₂₁, U₂)

        # copy F₂ ← U₂
        F₂ .= U₂
    end

    # solve for Z₁ in
    #
    #     L₁₁ᴴ Z₁ = Y₁
    #
    # and store F₁ ← Z₁
    F₁[] = f₁ / conj(L₁₁[])

    @inbounds for i in childindices(tree, j)
        ns += one(I)
        rdiv!_bwd_add_update!(F, ns, i, mapping, updptr, updval)
    end

    # copy B ← F₁
    @inbounds B[res] = F₁[]
    return ns
end

function rdiv!_loop_bwd_nod!(
        mapping::BipartiteGraph{I, I},
        blkptr::AbstractVector{I},
        updptr::AbstractVector{I},
        blkval::AbstractVector{T},
        updval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I}, 
        B::AbstractMatrix{T},
        ns::I,
        j::I,
    ) where {T, I}
    nrhs = convert(I, size(B, 1))

    residual = residuals(tree)
    separator = separators(tree)

    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    @inbounds nn = one(I)

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

    # res is the res at node j
    @inbounds res = only(neighbors(residual, j))
 
    # L is part of the Cholesky factor
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    @inbounds pstrt = blkptr[j]
    @inbounds pstop = blkptr[j + one(I)]
    @inbounds L = view(blkval, pstrt:pstop - one(I))
    @inbounds L₁₁ = view(L, nn)
    @inbounds L₂₁ = view(L, nn + one(I):nj)

    # F is part of B
    #
    #         res(j) sep(j)
    #     F = [ F₁    F₂ ]
    #
    @inbounds F = reshape(view(frtval, oneto(nj * nrhs)), :, nj)
    @inbounds F₁ = view(F, :, nn)
    @inbounds F₂ = view(F, :, nn + one(I):nj)

    F₁ .= view(B, :, res)
    
    if ispositive(na)
        # U₂ is the update matrix for node j
        @inbounds pstrt = updptr[ns]
        pstop = pstrt + na * nrhs
        @inbounds U₂ = reshape(view(updval, pstrt:pstop - one(I)), :, na)
        ns -= one(I)

        # compute the difference
        #
        #     Y₁ = F₁ - U₂ L₂₁
        #
        # and store F₁ ← Y₁
        mul!(F₁, U₂, L₂₁, -one(T), one(T))
 
        # copy F₂ ← U₂
        F₂ .= U₂
    end

    # solve for Z₁ in
    #
    #     Z₁ L₁₁ = Y₁
    #
    # and store F₁ ← Z₁
    F₁ ./= L₁₁[]

    @inbounds for i in childindices(tree, j)
        ns += one(I)
        rdiv!_bwd_add_update!(F, ns, i, mapping, updptr, updval)
    end

    # copy B ← F₁
    @inbounds B[:, res] .= F₁
    return ns
end

function SparseArrays.nnz(cholfact::CholFact)
    return nnz(cholfact.symbfact)
end

function Base.show(io::IO, ::MIME"text/plain", cholfact::CholFact{T, I}) where {T, I}
    println(io, "CholFact{$T, $I}:")
    println(io, "    nnz: $(nnz(cholfact))")
    print(io,   "    success: $(issuccess(cholfact))")
end

###########################
# Factorization Interface #
###########################

function LinearAlgebra.ldiv!(cholfact::CholFact, B::AbstractArray)
    return linsolve!(B, cholfact, Val(false))
end

function LinearAlgebra.rdiv!(B::AbstractArray, cholfact::CholFact)
    return linsolve!(B, cholfact, Val(true))
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
    fact = LinearAlgebra.cholesky!(Hermitian(A, :L), NoPivot(); check = false)
    status = iszero(fact.info)
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

# compute the lower triangular part of the difference
#
#     C = L - a * aᴴ
#
# and store L ← C
function syr!(a::AbstractVector{T}, L::AbstractMatrix{T}) where {T}
    m = length(a)

    @inbounds for j in eachindex(a)
        aj = conj(a[j])

        for i in j:m
            L[i, j] -= a[i] * aj
        end
    end

    return
end

function syr!(a::AbstractVector{T}, L::AbstractMatrix{T}) where {T <: BLAS.BlasReal}
    BLAS.syr!('L', -one(T), a, L)
    return
end

function syr!(a::AbstractVector{T}, L::AbstractMatrix{T}) where {T <: BLAS.BlasComplex}
    BLAS.her!('L', -one(real(T)), a, L)
    return
end
