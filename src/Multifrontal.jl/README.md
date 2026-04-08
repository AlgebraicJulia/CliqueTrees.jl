# Multifrontal.jl

The Multifrontal.jl submodule implements sparse Cholesky and LDLt factorization in pure Julia.
The algorithms implemented in Multifrontal.jl are *supernodal*, partitioning the input matrix into
dense blocks on which they call BLAS level 3 matrix kernels. This distinguishes Multifrontal.jl from
similar libraries like [QDLDL.jl](https://github.com/oxfordcontrol/QDLDL.jl) and
[LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl), which
implement *simplicial* algorithms that proceed column-by-column. Simplicial algorithms are faster on extremely
sparse matrices, whereas supernodal algorithms are faster on everything else.

## Algorithms

### Cholesky Factorization

A Cholesky factorization represents a symmetric positive-definite matrix
$A$ as a product

```math
A = P^\mathsf{T} L L^\mathsf{T} P
```

where $L$ is lower triangular and $P$ is a permutation. In order
to compute a Cholesky factorization using Multifrontal.jl, construct
a `ChordalCholesky` object and call the function `cholesky!`.

```julia-repl
julia> using CliqueTrees.Multifrontal, LinearAlgebra

julia> A = [
           4  2  0  0  2
           2  5  0  0  3
           0  0  4  2  0
           0  0  2  5  2
           2  3  0  2  7
       ];

julia> F = cholesky!(ChordalCholesky(A))
5×5 FChordalCholesky{:L, Float64, Int64} with 10 stored entries:
 2.0   ⋅    ⋅    ⋅    ⋅
 0.0  2.0   ⋅    ⋅    ⋅
 0.0  1.0  2.0   ⋅    ⋅
 1.0  0.0  0.0  2.0   ⋅
 0.0  1.0  1.0  1.0  2.0
```

`ChordalCholesky` objects behave like factorizations in LinearAlgebra.jl.
You can solve linear systems using `/` and `\`.

```julia-repl
julia> b = [4, 2, 0, 0, 2];

julia> F \ b
5-element Vector{Float64}:
 1.0
 0.0
 0.0
 0.0
 0.0
```

Alternatively, you can access each part of the factorization as a field:
`F.L` and `F.P`.

```julia-repl
julia> F.P \ (F.L' \ (F.L \ (F.P' \ b)))
5-element Vector{Float64}:
 1.0
 0.0
 0.0
 0.0
 0.0
```

### LDLt Factorization

An LDLt factorization represents a symmetric quasi-definite matrix $A$ as a product

```math
A = P^\mathsf{T} L D L^\mathsf{T} P
```

where $D$ is diagonal, $L$ is unit lower triangular, and $P$ is a permutation.
In order to compute an LDLt factorization using Multifrontal.jl, construct
a `ChordalLDLt` object and call the function `ldlt!`.

```julia-repl
julia> using CliqueTrees.Multifrontal, LinearAlgebra

julia> A = [
          -4 -2  0  0  2
          -2 -5  0  0  3
           0  0  4  2  0
           0  0  2  5  2
           2  3  0  2  7
       ];

julia> F = ldlt!(ChordalLDLt(A))
5×5 FChordalLDLt{:L, Float64, Int64} with 12 stored entries:
 1.0    ⋅     ⋅    ⋅    ⋅ 
 0.0   1.0    ⋅    ⋅    ⋅ 
 0.0   0.5   1.0   ⋅    ⋅ 
 0.5   0.0   0.0  1.0   ⋅ 
 0.0  -0.5  -0.5  0.5  1.0

 4.0    ⋅     ⋅    ⋅    ⋅ 
  ⋅   -4.0    ⋅    ⋅    ⋅ 
  ⋅     ⋅   -4.0   ⋅    ⋅ 
  ⋅     ⋅     ⋅   4.0   ⋅ 
  ⋅     ⋅     ⋅    ⋅   8.0
```

`ChordalLDLt` objects behave like factorizations in LinearAlgebra.jl.
You can solve linear systems using `/` and `\`.

```julia-repl
julia> b = [-4, -2, 0, 0, 2];

julia> F \ b
5-element Vector{Float64}:
  1.0
 -0.0
  0.0
  0.0
  0.0
```

Alternatively, you can access each part of the factorization as a field:
`F.D`, `F.L`, and `F.P`.

```julia-repl
julia> F.P \ (F.L' \ (F.D \ (F.L \ (F.P' \ b))))
5-element Vector{Float64}:
  1.0
 -0.0
  0.0
  0.0
  0.0
```

The diagonal elements of $D$ are called *pivots*. Sometimes, it is possible to
know the sign of each pivot before performing the factorization. This data
can be provided to the factorization algorithm using the keyword argument `signs`.

```julia-repl
julia> F = ldlt!(ChordalLDLt(A); signs=[-1, -1, 1, 1, 1])
5×5 FChordalLDLt{:L, Float64, Int64} with 12 stored entries:
 1.0    ⋅     ⋅    ⋅    ⋅ 
 0.0   1.0    ⋅    ⋅    ⋅ 
 0.0   0.5   1.0   ⋅    ⋅ 
 0.5   0.0   0.0  1.0   ⋅ 
 0.0  -0.5  -0.5  0.5  1.0

 4.0    ⋅     ⋅    ⋅    ⋅ 
  ⋅   -4.0    ⋅    ⋅    ⋅ 
  ⋅     ⋅   -4.0   ⋅    ⋅ 
  ⋅     ⋅     ⋅   4.0   ⋅ 
  ⋅     ⋅     ⋅    ⋅   8.0
```

### Benchmarks

Here are runtimes for the matrix [oilpan](https://sparse.tamu.edu/GHS_psdef/oilpan),
which has 73,752 columns and 2,148,558 nonzero entries.

| library                    | time (ms) |
| -------------------------- | --------- |
| CHOLMOD                    | 113       |
| Multifrontal.jl (LDLt)     | 126       |
| Multifrontal.jl (Cholesky) | 139       |
| QDLDL.jl                   | 582       |
| LDLFactorizations.jl       | 587       |
