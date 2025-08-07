"""
    SymbFact{I}

A symbolic factorization object.
"""
struct SymbFact{I}
    tree::CliqueTree{I, I}
    perm::FVector{I}
    invp::FVector{I}
    adjln::I
    blkln::I
    njmax::I
    nsmax::I
    upmax::I
    lsmax::I
end

"""
    symbolic(matrix::AbstractMatrix;
        alg::EliminationAlgorithm=DEFAULT_ELIMINATION_ALGORITHM,
        snd::SupernodeType=DEFAULT_SUPERNODE_TYPE,
    )

Compute a symbolic factorization of a sparse symmetric matrix.
See the function [`cliquetree`](@ref) for more information about
the parameters `alg` and `snd`.

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
```

### Parameters

  - `alg`: elimination algorithm
  - `snd`: supernode type
"""
function symbolic(matrix::AbstractMatrix; alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM, snd::SupernodeType=DEFAULT_SUPERNODE_TYPE)
    fact = symbolic(matrix, alg, snd)
    return fact
end

function symbolic(matrix::AbstractMatrix, alg::PermutationOrAlgorithm, snd::SupernodeType)
    return symbolic(sparse(matrix), alg, snd)
end

function symbolic(matrix::SparseMatrixCSC{<:Any, I}, alg::PermutationOrAlgorithm, snd::SupernodeType) where {I}
    perm, tree = cliquetree(matrix, alg, snd)
    residual = residuals(tree)
    separator = separators(tree)

    neqns = nov(separator)
    adjln = half(convert(I, nnz(matrix)) - neqns) + neqns
    up = ls = ns = nsmax = njmax = upmax = lsmax = blkln = zero(I)

    @inbounds for j in vertices(separator)
        nn = eltypedegree(residual, j)
        na = eltypedegree(separator, j)
        nj = nn + na

        for i in childindices(tree, j)
            ma = eltypedegree(separator, i)

            ns -= one(I)
            up -= ma * ma
            ls -= ma
        end

        if !isnothing(parentindex(tree, j))
            ns += one(I)
            up += na * na
            ls += na
        end

        nsmax = max(nsmax, ns)
        njmax = max(njmax, nj)
        upmax = max(upmax, up)
        lsmax = max(lsmax, ls)

        blkln = blkln + nn * nj
    end

    perm1 = FVector{I}(undef, neqns)
    invp1 = FVector{I}(undef, neqns)

    @inbounds for v in outvertices(separator)
        w = perm[v]
        perm1[v] = w
        invp1[w] = v        
    end

    return SymbFact{I}(tree, perm1, invp1, adjln, blkln, njmax, nsmax, upmax, lsmax)
end

function SparseArrays.nnz(fact::SymbFact{I}) where {I}
    n = nov(separators(fact.tree))
    weights = Ones{I}(n)
    return treefill(weights, fact.tree)
end

function Base.show(io::IO, ::MIME"text/plain", fact::SymbFact{I}) where {I}
    println(io, "SymbFact{$I}:")
    print(io,   "    nnz: $(nnz(fact))")
end
