"""
    SymbFact{I}

A symbolic factorization object.
"""
struct SymbFact{I}
    tree::CliqueTree{I, I}
    perm::Vector{I}
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

julia> M = [
           1.5   94.2    0.8 0.0
           94.2  15080.4 0.0 0.0
           0.8   0.0     3.1 0.0
           0.0   0.0     0.0 1.6
       ];

julia> F = CliqueTrees.symbolic(M)
SymbFact{Int64}:
    nnz: 6
    flop: 14

julia> CliqueTrees.nnz(F)
6

julia> CliqueTrees.flop(F)
14
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

function flop(fact::SymbFact{I}) where {I}
    tree = fact.tree
    residual = residuals(tree)
    separator = separators(tree)
    total = zero(I)

    @inbounds for i in vertices(residual)
        nn = eltypedegree(residual, i)
        na = eltypedegree(separator, i)
        flop = half(nn * (nn + one(I))) + nn * na
        total += flop * flop
    end

    return total
end

function Base.show(io::IO, ::MIME"text/plain", fact::SymbFact{I}) where {I}
    println(io, "SymbFact{$I}:")
    println(io,   "    nnz: $(nnz(fact))")
    print(io,     "    flop: $(flop(fact))")
end
