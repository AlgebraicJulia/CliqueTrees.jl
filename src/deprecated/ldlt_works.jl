"""
    LDLTWork{T, I}

A workspace for the function [`ldlt!`](@ref).
"""
struct LDLTWork{T, I}
    updptr::FVector{I}
    nzval1::FVector{T}
    nzval2::FVector{T}
    updval::FVector{T}
    frtval::FVector{T}
    wrkval::FVector{T}
    pattern1::BipartiteGraph{I, I, FVector{I}, FVector{I}}
    pattern2::BipartiteGraph{I, I, FVector{I}, FVector{I}} 
end

"""
    ldltinit([T::Type, ]matrix::AbstractMatrix, symbfact::SymbFact)

Initialize an LDLt factor and a factorization workspace.

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
```

### Parameters

  - `T`: element type (optional)
  - `matrix`: sparse quasi definite matrix
  - `symbfact`: symbolic factorization
"""
function ldltinit(matrix::AbstractMatrix{T}, symbfact::SymbFact) where {T}
    return ldltinit(T, matrix, symbfact)
end

function ldltinit(matrix::AbstractMatrix{T}, symbfact::SymbFact) where {T <: Integer}
    return ldltinit(float(T), matrix, symbfact)
end

function ldltinit(::Type{T}, matrix::AbstractMatrix, symbfact::SymbFact) where {T}
    return ldltinit(T, sparse(matrix), symbfact)
end

function ldltinit(::Type{T}, matrix::SparseMatrixCSC{<:Any, I}, symbfact::SymbFact{I}) where {T, I}
    @assert size(matrix, 1) == size(matrix, 2)
    @assert size(matrix, 1) == nov(separators(symbfact.tree))

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
    diaval = FVector{T}(undef, neqns)
    updval = FVector{T}(undef, upmax)
    frtval = FVector{T}(undef, frtln)
    wrkval = FVector{T}(undef, frtln)

    status = FScalar{Bool}(undef)

    mapping = BipartiteGraph(njmax, treln, relln, relptr, relidx)
    pattern1 = BipartiteGraph(neqns, neqns, adjln, colptr1, rowval1)
    pattern2 = BipartiteGraph(neqns, neqns, adjln, colptr2, rowval2)

    ldltfact = LDLTFact{T, I}(symbfact, blkptr, blkval, diaval, status, mapping) 
    ldltwork = LDLTWork{T, I}(updptr, nzval1, nzval2, updval, frtval, wrkval, pattern1, pattern2)
    return ldltfact, ldltwork
end

function Base.show(io::IO, ::MIME"text/plain", ldltwork::LDLTWork{T, I}) where {T, I}
    print(io, "LDTWork{$T, $I}:")
end
