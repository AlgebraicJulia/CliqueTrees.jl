"""
    LinWork{T}

A workspace for the function [`linsolve!`](@ref).
"""
struct LinWork{T}
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
function lininit(nrhs::Integer, cholfact::CholFact{T}) where {T}
    @argcheck !isnegative(nrhs)
    symbfact = cholfact.symbfact

    frtln = symbfact.njmax * nrhs
    vecln = nov(separators(symbfact.tree)) * nrhs

    frtval = FVector{T}(undef, frtln)
    vecval = FVector{T}(undef, vecln)

    return LinWork{T}(frtval, vecval)
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
function linsolve!(rhs::AbstractArray, cholfact::CholFact{T}, side::Val{false}) where {T}
    return linsolve!(rhs, lininit(size(rhs, 2), cholfact), cholfact, side)
end

function linsolve!(rhs::AbstractArray, cholfact::CholFact{T}, side::Val{true}) where {T}
    return linsolve!(rhs, lininit(size(rhs, 1), cholfact), cholfact, side)
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
function linsolve!(rhs::Union{AbstractVector, AbstractMatrix}, linwork::LinWork{T}, cholfact::CholFact{T}, side::Val{false}) where {T}
    @argcheck nov(separators(cholfact.symbfact.tree)) == size(rhs, 1)
    @argcheck length(linwork.frtval) >= cholfact.symbfact.njmax * size(rhs, 2)
    @argcheck length(linwork.vecval) >= length(rhs)

    tree = cholfact.symbfact.tree
    perm = cholfact.symbfact.perm

    blkptr = cholfact.blkptr
    blkval = cholfact.blkval
    frtval = linwork.frtval
    vecval = linwork.vecval

    separator = separators(tree)

    vec = reshape(view(vecval, eachindex(rhs)), size(rhs))

    @inbounds for w in axes(rhs, 2), v in axes(rhs, 1)
        vec[v, w] = rhs[perm[v], w]
    end

    @inbounds for j in vertices(separator)
        ldiv!_loop_fwd!(blkptr, blkval, frtval, tree, vec, j)
    end

    @inbounds for j in reverse(vertices(separator))
        ldiv!_loop_bwd!(blkptr, blkval, frtval, tree, vec, j)
    end

    @inbounds for w in axes(rhs, 2), v in axes(rhs, 1)
        rhs[perm[v], w] = vec[v, w]
    end

    return rhs
end

function linsolve!(rhs::AbstractMatrix, linwork::LinWork{T}, cholfact::CholFact{T}, side::Val{true}) where {T}
    @argcheck nov(separators(cholfact.symbfact.tree)) == size(rhs, 2)
    @argcheck length(linwork.frtval) >= cholfact.symbfact.njmax * size(rhs, 1)
    @argcheck length(linwork.vecval) >= length(rhs)

    tree = cholfact.symbfact.tree
    perm = cholfact.symbfact.perm

    blkptr = cholfact.blkptr
    blkval = cholfact.blkval
    frtval = linwork.frtval
    vecval = linwork.vecval

    residual = residuals(tree)
    separator = separators(tree)

    vec = reshape(view(vecval, eachindex(rhs)), size(rhs))

    @inbounds for w in axes(rhs, 1), v in axes(rhs, 2)
        vec[w, v] = rhs[w, perm[v]]
    end

    @inbounds for j in vertices(separator)
        rdiv!_loop_fwd!(blkptr, blkval, frtval, tree, vec, j)
    end

    @inbounds for j in reverse(vertices(separator))
        rdiv!_loop_bwd!(blkptr, blkval, frtval, tree, vec, j)
    end

    @inbounds for w in axes(rhs, 1), v in axes(rhs, 2)
        rhs[w, perm[v]] = vec[w, v]
    end

    return rhs
end

function Base.show(io::IO, ::MIME"text/plain", linwork::LinWork{T}) where {T}
    print(io, "LinWork{$T}:")
end
