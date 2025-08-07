"""
    CholWork{T, I}

A workspace for the function [`cholesky!`](@ref).
"""
struct CholWork{T, I}
    updptr::FVector{I}
    nzval1::FVector{T}
    nzval2::FVector{T}
    updval::FVector{T}
    frtval::FVector{T}
    pattern1::BipartiteGraph{I, I, FVector{I}, FVector{I}}
    pattern2::BipartiteGraph{I, I, FVector{I}, FVector{I}} 
end

"""
    cholinit([T::Type, ]matrix::AbstractMatrix, symbfact::SymbFact)

Initialize a cholesky factor and a factorization workspace.

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
```

### Parameters

  - `T`: element type (optional)
  - `matrix`: sparse positive definite matrix
  - `symbfact`: symbolic factorization
"""
function cholinit(matrix::AbstractMatrix{T}, symbfact::SymbFact) where {T}
    return cholinit(T, matrix, symbfact)
end

function cholinit(matrix::AbstractMatrix{T}, symbfact::SymbFact) where {T <: Integer}
    return cholinit(float(T), matrix, symbfact)
end

function cholinit(::Type{T}, matrix::AbstractMatrix, symbfact::SymbFact) where {T}
    return cholinit(T, sparse(matrix), symbfact)
end

function cholinit(::Type{T}, matrix::SparseMatrixCSC{<:Any, I}, symbfact::SymbFact{I}) where {T, I}
    @argcheck size(matrix, 1) == size(matrix, 2)
    @argcheck size(matrix, 1) == nov(separators(symbfact.tree))

    tree = symbfact.tree
    separator = separators(tree)

    adjln = symbfact.adjln
    blkln = symbfact.blkln
    njmax = symbfact.njmax
    nsmax = symbfact.nsmax
    upmax = symbfact.upmax

    neqns = nov(separator)
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

    mapping = BipartiteGraph(njmax, treln, relln, relptr, relidx)
    pattern1 = BipartiteGraph(neqns, neqns, adjln, colptr1, rowval1)
    pattern2 = BipartiteGraph(neqns, neqns, adjln, colptr2, rowval2)

    cholfact = CholFact{T, I}(symbfact, blkptr, blkval, status, mapping) 
    cholwork = CholWork{T, I}(updptr, nzval1, nzval2, updval, frtval, pattern1, pattern2)
    return cholfact, cholwork
end

function Base.show(io::IO, ::MIME"text/plain", cholwork::CholWork{T, I}) where {T, I}
    print(io, "CholWork{$T, $I}:")
end
