"""
    SymbFact{I}

A symbolic factorization object.
"""
struct SymbFact{I}
    tree::CliqueTree{I, I}
    perm::FVector{I}
    invp::FVector{I}
end

function SymbFact(tree::CliqueTree{I, I}, perm::AbstractVector{I}) where {I}
    n = nov(separators(tree))
    perm1 = FVector{I}(undef, n)
    invp1 = FVector{I}(undef, n)

    @inbounds for v in oneto(n)
        w = perm[v]
        perm1[v] = w
        invp1[w] = v        
    end

    return SymbFact(tree, perm1, invp1)
end

"""
    symbolic(matrix::AbstractMatrix;
        alg::EliminationAlgorithm=DEFAULT_ELIMINATION_ALGORITHM,
        snd::SupernodeType=DEFAULT_SUPERNODE_TYPE,
    )

Compute a symbolic factorization of a sparse symmetric matrix.
See the function [`cliquetree`](@ref) for more information about
the parameters `alg` and `snd`.

### Parameters

  - `alg`: elimination algorithm
  - `snd`: supernode type

```julia
julia> import CliqueTrees

julia> matrix = [
           1.5   94.2    0.8 0.0
           94.2  15080.4 0.0 0.0
           0.8   0.0     3.1 0.0
           0.0   0.0     0.0 1.6
       ];

julia> CliqueTrees.symbolic(matrix)
SymbFact{Int64}:
    nnz: 6
```
"""
function symbolic(matrix::AbstractMatrix; alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM, snd::SupernodeType=DEFAULT_SUPERNODE_TYPE)
    fact = symbolic(matrix, alg, snd)
    return fact
end

function symbolic(matrix::AbstractMatrix, alg::PermutationOrAlgorithm, snd::SupernodeType)
    perm, tree = cliquetree(matrix, alg, snd)
    return SymbFact(tree, perm)
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
