"""
    LinWork{T, I}

A workspace for the function [`linsolve!`](@ref).
"""
struct LinWork{T, I}
    updptr::FVector{I}
    updval::FVector{T}
    frtval::FVector{T}
    vecval::FVector{T}
end

"""
    lininit(nrhs::Integer, cholfact::CholFact)

Initialize a linear solve workspace.

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

julia> linwork = CliqueTrees.lininit(2, cholfact)
LinWork{Float64}:
```

### Paramerers

  - `nrhs`: number of right-hand sides
  - `cholfact`: factorized coefficient matrix
"""
function lininit(nrhs::Integer, cholfact::CholFact{T, I}) where {T, I}
    @argcheck !isnegative(nrhs)
    symbfact = cholfact.symbfact
    tree = symbfact.tree
    separator = separators(tree)

    treln = nv(separator)
    updln = symbfact.lsmax * nrhs
    frtln = symbfact.njmax * nrhs
    vecln = nov(separators(symbfact.tree)) * nrhs

    updptr = FVector{I}(undef, treln + one(I))
    updval = FVector{T}(undef, updln)
    frtval = FVector{T}(undef, frtln)
    vecval = FVector{T}(undef, vecln)

    return LinWork{T, I}(updptr, updval, frtval, vecval)
end

"""
    linsolve!(rhs::AbstractArray, cholfact::CholFact, side::Val)

Solve a linear system of equations.

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

julia> rhs = rand(8, 2);

julia> cholfact = CliqueTrees.cholesky(matrix)
CholFact{Float64, Int64}:
    nnz: 19
    success: true

julia> sol = CliqueTrees.linsolve!(rhs, cholfact, Val(false)) # sol = inv(matrix) * rhs
8×2 Matrix{Float64}:
  0.339907    0.0202252
 -0.364903    0.573497
 -0.243223    0.354763
  0.293368   -0.00477056
 -0.336252    0.507332
  0.600361   -0.655479
  0.452218   -0.266566
 -0.0620433   0.0528108
```

### Parameters

  - `rhs`: right-hand side
  - `cholfact`: factorized coefficient matrix
  - `side`: left or right division
    - `Val(false)`: left division
    - `Val(true)`: right division
"""
linsolve!(rhs::AbstractArray, cholfact::CholFact, side::Val)

function linsolve!(rhs::AbstractMatrix, cholfact::CholFact, side::Val{false})
    return linsolve!(rhs, lininit(size(rhs, 2), cholfact), cholfact, side)
end

function linsolve!(rhs::AbstractMatrix, cholfact::CholFact, side::Val{true})
    return linsolve!(rhs, lininit(size(rhs, 1), cholfact), cholfact, side)
end

function linsolve!(rhs::AbstractVector, cholfact::CholFact, side::Val)
    return linsolve!(rhs, lininit(1, cholfact), cholfact, side)
end

"""
    linsolve!(rhs::AbstractArray, linwork::LinWork{T}, cholfact::CholFact, side::Val)

Solve a linear system of equations using a pre-allocated workspace.

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

julia> rhs = rand(8, 2);

julia> cholfact = CliqueTrees.cholesky(matrix)
CholFact{Float64, Int64}:
    nnz: 19
    success: true

julia> linwork = CliqueTrees.lininit(2, cholfact)
LinWork{Float64}:

julia> sol = CliqueTrees.linsolve!(rhs, linwork, cholfact, Val(false)) # sol = inv(matrix) * rhs
8×2 Matrix{Float64}:
  0.339907    0.0202252
 -0.364903    0.573497
 -0.243223    0.354763
  0.293368   -0.00477056
 -0.336252    0.507332
  0.600361   -0.655479
  0.452218   -0.266566
 -0.0620433   0.0528108
```

### Parameters

  - `rhs`: right-hand side
  - `linwork`: linear solve workspace
  - `cholfact`: factorized coefficient matrix
  - `side`: left or right division
    - `Val(false)`: left division
    - `Val(true)`: right division

"""
function linsolve!(rhs::AbstractVector, linwork::LinWork, cholfact::CholFact, side::Val{false})
    return linsolve!(rhs, linwork, cholfact, Val(true))
end

function linsolve!(rhs::AbstractMatrix, linwork::LinWork{T, I}, cholfact::CholFact{T, I}, side::Val{false}) where {T, I}
    @argcheck nov(separators(cholfact.symbfact.tree)) == size(rhs, 1)
    @argcheck length(linwork.vecval) >= length(rhs)

    tree = cholfact.symbfact.tree
    neqns = nov(separators(tree))
    nrhs = convert(I, size(rhs, 2))

    adj = reshape(rhs, nrhs, neqns)
    tmp = view(linwork.vecval, oneto(nrhs * neqns))

    copyto!(adj, adjoint!(reshape(tmp, nrhs, neqns), rhs))
    linsolve!(adj, linwork, cholfact, Val(true))    
    copyto!(rhs, adjoint!(reshape(tmp, neqns, nrhs), adj))

    return rhs
end

function linsolve!(rhs::AbstractVector, linwork::LinWork{T, I}, cholfact::CholFact{T, I}, side::Val{true}) where {T, I}
    @argcheck nov(separators(cholfact.symbfact.tree)) == length(rhs)
    @argcheck length(linwork.frtval) >= cholfact.symbfact.njmax
    @argcheck length(linwork.vecval) >= length(rhs)

    tree = cholfact.symbfact.tree
    perm = cholfact.symbfact.perm

    mapping = cholfact.mapping

    blkptr = cholfact.blkptr
    blkval = cholfact.blkval

    updptr = linwork.updptr
    updval = linwork.updval
    frtval = linwork.frtval
    vecval = linwork.vecval

    residual = residuals(tree)
    separator = separators(tree)

    neqns = nov(separators(tree))
    vec = view(vecval, oneto(neqns))

    updptr[begin] = one(I); ns = zero(I)

    @inbounds for v in oneto(neqns)
        vec[v] = rhs[perm[v]]
    end

    @inbounds for j in vertices(separator)
        ns = rdiv!_loop_fwd!(mapping, blkptr, updptr,
            blkval, updval, frtval, tree, vec, ns, j)
    end

    @inbounds for j in reverse(vertices(separator))
        ns = rdiv!_loop_bwd!(mapping, blkptr, updptr,
            blkval, updval, frtval, tree, vec, ns, j)
    end

    @inbounds for v in oneto(neqns)
        rhs[perm[v]] = vec[v]
    end

    return rhs
end

function linsolve!(rhs::AbstractMatrix, linwork::LinWork{T, I}, cholfact::CholFact{T, I}, side::Val{true}) where {T, I}
    @argcheck nov(separators(cholfact.symbfact.tree)) == size(rhs, 2)
    @argcheck length(linwork.frtval) >= cholfact.symbfact.njmax * size(rhs, 1)
    @argcheck length(linwork.vecval) >= length(rhs)

    tree = cholfact.symbfact.tree
    perm = cholfact.symbfact.perm

    mapping = cholfact.mapping

    blkptr = cholfact.blkptr
    blkval = cholfact.blkval

    updptr = linwork.updptr
    updval = linwork.updval
    frtval = linwork.frtval
    vecval = linwork.vecval

    residual = residuals(tree)
    separator = separators(tree)

    neqns = nov(separators(tree))
    nrhs = convert(I, size(rhs, 1))
    vec = reshape(view(vecval, oneto(nrhs * neqns)), nrhs, neqns)

    updptr[begin] = one(I); ns = zero(I)

    @inbounds for c in oneto(nrhs), v in oneto(neqns)
        vec[c, v] = rhs[c, perm[v]]
    end

    @inbounds for j in vertices(separator)
        ns = rdiv!_loop_fwd!(mapping, blkptr, updptr,
            blkval, updval, frtval, tree, vec, ns, j)
    end

    @inbounds for j in reverse(vertices(separator))
        ns = rdiv!_loop_bwd!(mapping, blkptr, updptr,
            blkval, updval, frtval, tree, vec, ns, j)
    end

    @inbounds for c in oneto(nrhs), v in oneto(neqns)
        rhs[c, perm[v]] = vec[c, v]
    end

    return rhs
end

function Base.show(io::IO, ::MIME"text/plain", linwork::LinWork{T}) where {T}
    print(io, "LinWork{$T}:")
end
