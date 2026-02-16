"""
    LDLTFact{T, I} <: Factorization{T}

An LDLt factorization object.
"""
struct LDLTFact{T, I} <: Factorization{T}
    symbfact::SymbFact{I}
    blkptr::FVector{I}
    blkval::FVector{T}
    diaval::FVector{T}
    status::FScalar{Bool}
    mapping::BipartiteGraph{I, I, FVector{I}, FVector{I}}
end

"""
    SymbFact(ldltfact::LDLTFact)

Get the underlying symbolic factorization of an LDLT
factorization.
"""
function SymbFact(ldltfact::LDLTFact)
    return ldltfact.symbfact
end

"""
    ldlt(matrix::AbstractMatrix;
        alg::EliminationAlgorithm=DEFAULT_ELIMINATION_ALGORITHM,
        snd::SupernodeType=DEFAULT_SUPERNODE_TYPE,
    )

Compute the LDLt factorization of a sparse quasi definite matrix.
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

julia> ldltfact = CliqueTrees.ldlt(matrix)
LDLTFact{Float64, Int64}:
    nnz: 19
    success: true
```

### Parameters

  - `matrix`: sparse quasi-definite matrix
  - `alg`: elimination algorithm
  - `snd`: supernode type

"""
function ldlt(matrix::AbstractMatrix; alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM, snd::SupernodeType=DEFAULT_SUPERNODE_TYPE)
    ldltfact = ldlt(matrix, alg, snd)
    return ldltfact
end

function ldlt(matrix::AbstractMatrix, alg::PermutationOrAlgorithm, snd::SupernodeType)
    ldltfact = ldlt(matrix, symbolic(matrix, alg, snd))
    return ldltfact
end

function ldlt(matrix::AbstractMatrix, symbfact::SymbFact)
    ldltfact = ldlt(sparse(matrix), symbfact)
    return ldltfact
end

function ldlt(matrix::SparseMatrixCSC{<:Any, I}, symbfact::SymbFact{I}) where {I}
    @assert size(matrix, 1) == size(matrix, 2)
    @assert size(matrix, 1) == nov(separators(symbfact.tree))
    return ldlt!(ldltinit(matrix, symbfact)..., matrix) 
end

"""
    ldlt!(ldltfact::LDLTFact, ldltwork::LDLTWork, matrix::AbstractMatrix)

Compute the LDLT factorization of a sparse quasi definite matrix
using a pre-allocated workspace. See [`ldlt`](@ref).

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

julia> ldltfact, ldltwork = CliqueTrees.ldltinit(matrix, symbfact);

julia> CliqueTrees.ldlt!(ldltfact, ldltwork, matrix, symbfact)
LDLTFact{Float64, Int64}:
    nnz: 19
    success: true
```

### Parameters

  - `ldltfact`: LDLT factor
  - `ldltwork`: workspace
  - `matrix`: sparse quasi-definite matrix
"""
function ldlt!(ldltfact::LDLTFact{T, I}, ldltwork::LDLTWork{T, I}, matrix::AbstractMatrix) where {T, I}
    return ldlt!(ldltfact, ldltwork, sparse(matrix))
end

function ldlt!(ldltfact::LDLTFact{T, I}, ldltwork::LDLTWork{T, I}, matrix::SparseMatrixCSC{<:Any, I}) where {T, I}
    symbfact = ldltfact.symbfact

    tree = symbfact.tree
    invp = symbfact.invp

    mapping = ldltfact.mapping

    blkptr = ldltfact.blkptr
    blkval = ldltfact.blkval
    diaval = ldltfact.diaval

    updptr = ldltwork.updptr
    updval = ldltwork.updval
    frtval = ldltwork.frtval
    wrkval = ldltwork.wrkval

    pattern0 = BipartiteGraph(matrix)
    pattern1 = ldltwork.pattern1
    pattern2 = ldltwork.pattern2

    nzval0 = matrix.nzval
    nzval1 = ldltwork.nzval1
    nzval2 = ldltwork.nzval2

    cholesky_permute!(pattern0, pattern1, nzval0, nzval1, invp)
    cholesky_reverse!(pattern1, pattern2, nzval1, nzval2)

    ldltfact.status[] = ldlt_impl!(mapping, blkptr, updptr, blkval, diaval, updval, frtval, wrkval, tree, pattern2, nzval2) 

    return ldltfact
end

function ldlt_impl!(
        mapping::BipartiteGraph{I, I},        
        blkptr::AbstractVector{I},
        updptr::AbstractVector{I},
        blkval::AbstractVector{T},
        diaval::AbstractVector{T},
        updval::AbstractVector{T},
        frtval::AbstractVector{T},
        wrkval::AbstractVector{T},
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
        iterstatus, ns = ldlt_loop!(mapping, blkptr, updptr,
            blkval, diaval, updval, frtval, wrkval, tree, ns, j)

        status = status && iterstatus
    end

    return status
end

function ldlt_loop!(
        mapping::BipartiteGraph{I, I},        
        blkptr::AbstractVector{I},
        updptr::AbstractVector{I},
        blkval::AbstractVector{T},
        diaval::AbstractVector{T},
        updval::AbstractVector{T},
        frtval::AbstractVector{T},
        wrkval::AbstractVector{T},
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
        status, ns = ldlt_loop_nod!(mapping, blkptr, updptr,
            blkval, diaval, updval, frtval, wrkval, tree, ns, j)
    else
        status, ns = ldlt_loop_snd!(mapping, blkptr, updptr,
            blkval, diaval, updval, frtval, wrkval, tree, ns, j)
    end

    return status, ns
end


function ldlt_loop_snd!(
        mapping::BipartiteGraph{I, I},        
        blkptr::AbstractVector{I},
        updptr::AbstractVector{I},
        blkval::AbstractVector{T},
        diaval::AbstractVector{T},
        updval::AbstractVector{T},
        frtval::AbstractVector{T},
        wrkval::AbstractVector{T},
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

    # res is the residual at node j
    @inbounds res = neighbors(residual, j)

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

    # W has the same dimensions as F.
    #
    #           nn  na
    #     W = [ W₁₁ W₁₂ ] nn
    #         [ W₂₁ W₂₂ ] na
    #
    @inbounds W = reshape(view(wrkval, oneto(nj * nj)), nj, nj)
    @inbounds W₁₁ = view(W, oneto(nn),      oneto(nn))
    @inbounds W₂₁ = view(W, nn + one(I):nj, oneto(nn))

    # B is part of the lower triangular factor
    #
    #          res(j)
    #     B = [ B₁₁  ] res(j)
    #         [ B₂₁  ] sep(j)
    #
    @inbounds pstrt = blkptr[j]
    @inbounds pstop = blkptr[j + one(I)]
    @inbounds B = reshape(view(blkval, pstrt:pstop - one(I)), nj, nn)

    # D₁₁ is part of the diagonal factor
    #
    #          res(j)
    #         [ D₁₁ ] res(j)
    #
    @inbounds D₁₁ = view(diaval, res) 

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
    #     F₁₁ = L₁₁ D₁₁ L₁₁ᴴ
    #
    # and store F₁₁ ← L₁₁
    status = qdtrf!(W₁₁, F₁₁, D₁₁)

    if ispositive(na)
        ns += one(I)

        # U₂₂ is the update matrix for node j
        @inbounds pstrt = updptr[ns]
        @inbounds pstop = updptr[ns + one(I)] = pstrt + na * na
        @inbounds U₂₂ = reshape(view(updval, pstrt:pstop - one(I)), na, na)
        lacpy!(U₂₂, F₂₂)

        # solve for L₂₁ in
        #
        #     L₂₁ D₁₁ L₁₁ᴴ = F₂₁
        #
        # and store F₂₁ ← L₂₁
        rdiv!(F₂₁, UnitLowerTriangular(F₁₁) |> adjoint)
        copyto!(W₂₁, F₂₁)
        rdiv!(F₂₁, Diagonal(D₁₁))

        # compute the difference
        #
        #    L₂₂ = U₂₂ - L₂₁ D₁₁ L₂₁ᴴ
        #
        # and store U₂₂ ← L₂₂
        trrk!(W₂₁, F₂₁, U₂₂)
    end
 
    # copy F₁ into B
    #
    #     B₁₁ ← F₁₁
    #     B₂₁ ← F₂₁
    #
    lacpy!(B, F₁)
    return status, ns
end

function ldlt_loop_nod!(
        mapping::BipartiteGraph{I, I},        
        blkptr::AbstractVector{I},
        updptr::AbstractVector{I},
        blkval::AbstractVector{T},
        diaval::AbstractVector{T},
        updval::AbstractVector{T},
        frtval::AbstractVector{T},
        wrkval::AbstractVector{T},
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

    # res is the residual at node j
    @inbounds res = only(neighbors(residual, j))

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

    # W has the same dimensions as F.
    #
    #           nn  na
    #     W = [ W₁₁ W₁₂ ] nn
    #         [ W₂₁ W₂₂ ] na
    #
    @inbounds W = reshape(view(wrkval, oneto(nj * nj)), nj, nj)
    @inbounds W₁₁ = view(W, nn,             nn)
    @inbounds W₂₁ = view(W, nn + one(I):nj, nn)

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

    # D₁₁ is part of the diagonal factor
    #
    #          res(j)
    #         [ D₁₁ ] res(j)
    #
    @inbounds D₁₁ = view(diaval, res)

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
    #     F₁₁ = L₁₁ D₁₁ L₁₁ᴴ
    #
    # and store F₁₁ ← L₁₁
    d₁₁ = D₁₁[] = F₁₁[]

    if !iszero(d₁₁)
                      status = true
    else
        d₁₁ = one(T); status = false
    end

    if ispositive(na)
        ns += one(I)

        # U₂₂ is the update matrix for node j
        @inbounds pstrt = updptr[ns]
        @inbounds pstop = updptr[ns + one(I)] = pstrt + na * na
        @inbounds U₂₂ = reshape(view(updval, pstrt:pstop - one(I)), na, na)
        lacpy!(U₂₂, F₂₂)

        # compute the difference
        #
        #    L₂₂ = U₂₂ - F₂₁ D₁₁⁻¹ F₂₁ᴴ
        #
        # and store U₂₂ ← L₂₂
        syr!(d₁₁, F₂₁, U₂₂)

        # solve for L₂₁ in
        #
        #     L₂₁ D₁₁ L₁₁ᴴ = F₂₁
        #
        # and store F₂₁ ← L₂₁
        F₂₁ ./= d₁₁
    end

    # copy F₁ into B
    #
    #     B₁₁ ← F₁₁
    #     B₂₁ ← F₂₁
    #
    copyto!(B₂, F₂₁)
    return status, ns
end

function rdiv!_ldlt_loop_fwd!(
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
        ns = rdiv!_ldlt_loop_fwd_nod!(mapping, blkptr, updptr,
            blkval, updval, frtval, tree, B, ns, j)
    else
        ns = rdiv!_ldlt_loop_fwd_snd!(mapping, blkptr, updptr,
            blkval, updval, frtval, tree, B, ns, j)
    end

    return ns
end

function rdiv!_ldlt_loop_fwd_snd!(
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
    
    # L is part of the lower triangular factor
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
    ldiv!(UnitLowerTriangular(L₁₁), F₁)
 
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

function rdiv!_ldlt_loop_fwd_snd!(
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
    rdiv!(F₁, UnitLowerTriangular(L₁₁) |> adjoint)
 
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

function rdiv!_ldlt_loop_fwd_nod!(
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
    f₁ = F₁[]
 
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

function rdiv!_ldlt_loop_fwd_nod!(
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
    F₁
 
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

function rdiv!_ldlt_loop_bwd!(
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
        ns = rdiv!_ldlt_loop_bwd_nod!(mapping, blkptr, updptr,
            blkval, updval, frtval, tree, B, ns, j)
    else
        ns = rdiv!_ldlt_loop_bwd_snd!(mapping, blkptr, updptr,
            blkval, updval, frtval, tree, B, ns, j)
    end

    return ns
end

function rdiv!_ldlt_loop_bwd_snd!(
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
    ldiv!(UnitLowerTriangular(L₁₁) |> adjoint, F₁)

    @inbounds for i in childindices(tree, j)
        ns += one(I)
        rdiv!_bwd_add_update!(F, ns, i, mapping, updptr, updval)
    end

    # copy B ← F₁
    @inbounds B[res] .= F₁
    return ns
end

function rdiv!_ldlt_loop_bwd_snd!(
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
    rdiv!(F₁, UnitLowerTriangular(L₁₁))

    @inbounds for i in childindices(tree, j)
        ns += one(I)
        rdiv!_bwd_add_update!(F, ns, i, mapping, updptr, updval)
    end

    # copy B ← F₁
    @inbounds B[:, res] .= F₁
    return ns
end

function rdiv!_ldlt_loop_bwd_nod!(
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
    F₁[] = f₁

    @inbounds for i in childindices(tree, j)
        ns += one(I)
        rdiv!_bwd_add_update!(F, ns, i, mapping, updptr, updval)
    end

    # copy B ← F₁
    @inbounds B[res] = F₁[]
    return ns
end

function rdiv!_ldlt_loop_bwd_nod!(
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
    F₁

    @inbounds for i in childindices(tree, j)
        ns += one(I)
        rdiv!_bwd_add_update!(F, ns, i, mapping, updptr, updval)
    end

    # copy B ← F₁
    @inbounds B[:, res] .= F₁
    return ns
end

function Base.size(ldltfact::LDLTFact)
    tree = ldltfact.symbfact.tree
    separator = separators(tree)
    neqns = convert(Int, nov(separator))
    return (neqns, neqns)
end

function SparseArrays.nnz(ldltfact::LDLTFact)
    return nnz(ldltfact.symbfact)
end

function Base.show(io::IO, ::MIME"text/plain", ldltfact::LDLTFact{T, I}) where {T, I}
    println(io, "LDLTFact{$T, $I}:")
    println(io, "    nnz: $(nnz(ldltfact))")
    print(io,   "    success: $(issuccess(ldltfact))")
end

###########################
# Factorization Interface #
###########################

function LinearAlgebra.ldiv!(ldltfact::LDLTFact, B::AbstractArray)
    return linsolve!(B, ldltfact, Val(false))
end

function LinearAlgebra.rdiv!(B::AbstractArray, ldltfact::LDLTFact)
    return linsolve!(B, ldltfact, Val(true))
end

function LinearAlgebra.issuccess(ldltfact::LDLTFact)
    return ldltfact.status[]
end

function LinearAlgebra.isposdef(ldltfact::LDLTFact)
    tree = ldltfact.symbfact.tree
    diaval = ldltfact.diaval
    separator = separators(tree)

    if !issuccess(ldltfact)
        return false
    end

    for j in outvertices(separator)
        if !ispositive(diaval[j])
            return false
        end
    end

    return true
end

function LinearAlgebra.det(ldltfact::LDLTFact{T, I}) where {T, I}
    tree = ldltfact.symbfact.tree
    diaval = ldltfact.diaval
    separator = separators(tree)

    det = one(real(T))

    for j in outvertices(separator)
        det *= real(diaval[j])
    end

    return det
end

function LinearAlgebra.logdet(ldltfact::LDLTFact{T, I}) where {T, I}
    tree = ldltfact.symbfact.tree
    diaval = ldltfact.diaval
    separator = separators(tree)

    logdet = zero(real(T))

    for j in outvertices(separator)
        logdet += log(real(diaval[j]))
    end

    return logdet
end

##################################
# Dense Numerical Linear Algebra #
##################################

# factorize A as
#
#     A = L D Lᴴ
#
# and store A ← L
function qdtrf2!(A::AbstractMatrix{T}, D::AbstractVector{T}) where {T}
    n = size(A, 1)
    
    @inbounds @fastmath for j in axes(A, 1)
        Ajj = real(A[j, j])

        for k in 1:j - 1
            Ajj -= abs2(A[j, k]) * D[k]
        end

        if iszero(Ajj)
            return false
        end

        Djj = D[j] = Ajj; iDjj = inv(Djj)

        for i in j + 1:n
            for k in 1:j - 1
                A[i, j] -= A[i, k] * D[k] * conj(A[j, k])
            end

            A[i, j] *= iDjj
        end
    end
    
    return true
end

# factorize A as
#
#     A = L D Lᴴ
#
# and store A ← L
function qdtrf!(W::AbstractMatrix{T}, A::AbstractMatrix{T}, D::AbstractVector{T}; blocksize::Int = 64) where {T}
    n = size(A, 1)
    
    for bstrt in 1:blocksize:n
        bsize = min(blocksize, n - bstrt + 1)
        bstop = bstrt + bsize - 1

        A11 = view(A, bstrt:bstop, bstrt:bstop)
        D11 = view(D, bstrt:bstop)
        status = qdtrf2!(A11, D11)
        
        if !status
            return false
        end
        
        if bstop < n
            A21 = view(A, bstop + 1:n, bstrt:bstop)
            W21 = view(W, bstop + 1:n, 1:bsize)
            A22 = view(A, bstop + 1:n, bstop + 1:n)
            
            rdiv!(A21, UnitLowerTriangular(A11) |> adjoint)
            copyto!(W21, A21)
            rdiv!(A21, Diagonal(D11))
            trrk!(W21, A21, A22)
        end
    end
    
    return true
end

# compute the lower triangular part of the difference
#
#     C = L - A * Bᴴ
#
# and store L ← C
function trrk!(A::AbstractMatrix{T}, B::AbstractMatrix{T}, L::AbstractMatrix{T}) where {T <: BLAS.BlasReal}
    BLAS.syr2k!('L', 'N', convert(T, -1/2), A, B, one(T), L)
    return
end

function trrk!(A::AbstractMatrix{T}, B::AbstractMatrix{T}, L::AbstractMatrix{T}) where {T <: BLAS.BlasComplex}
    BLAS.her2k!('L', 'N', convert(T, -1/2), A, B, one(real(T)), L)
    return
end

# compute the lower triangular part of the difference
#
#     C = L - a * aᴴ / alpha
#
# and store L ← C
function syr!(alpha::T, a::AbstractVector{T}, L::AbstractMatrix{T}) where {T <: BLAS.BlasReal}
    BLAS.syr!('L', -inv(alpha), a, L)
    return
end

function syr!(alpha::T, a::AbstractVector{T}, L::AbstractMatrix{T}) where {T <: BLAS.BlasComplex}
    BLAS.her!('L', -inv(real(alpha)), a, L)
    return
end
