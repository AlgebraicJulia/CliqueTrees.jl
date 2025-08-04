# CliqueTrees.jl

CliqueTrees.jl implements *clique trees* in Julia. You can use it to construct [tree decompositions](https://en.wikipedia.org/wiki/Tree_decomposition) and [chordal completions](https://en.wikipedia.org/wiki/Chordal_completion) of graphs.

## Getting Help

If you have a question about the library, feel free to open an [issue](https://github.com/AlgebraicJulia/CliqueTrees.jl/issues) or leave a message in the [cliquetrees.jl](https://julialang.zulipchat.com/#narrow/channel/513749-cliquetrees.2Ejl) Zulip channel.

## Projects using CliqueTrees

- [BandedMatrices.jl](https://github.com/JuliaLinearAlgebra/BandedMatrices.jl)
- [BayesNets.jl](https://github.com/sisl/BayesNets.jl)
- [CausalInference.jl](https://github.com/mschauer/CausalInference.jl)
- [EinExprs.jl](https://github.com/bsc-quantic/EinExprs.jl)
- [IncrementalInference.jl](https://github.com/JuliaRobotics/IncrementalInference.jl)
- [OMEinsumContractionOrders.jl](https://github.com/TensorBFS/OMEinsumContractionOrders.jl)
- [Scruff.jl](https://github.com/charles-river-analytics/Scruff.jl)
- [SparseMatrixColorings.jl](https://github.com/gdalle/SparseMatrixColorings.jl)
- [SumOfSquares.jl](https://github.com/jump-dev/SumOfSquares.jl)
- [TSSOS.jl](https://github.com/wangjie212/TSSOS)

## Installation

To install CliqueTrees.jl, enter the Pkg REPL by typing `]` and run the following command.

```julia-repl
pkg> add CliqueTrees
```

## Basic Usage

### Tree Decompositions

The function `cliquetree` computes tree decompositions.

```julia-repl
julia> using CliqueTrees, LinearAlgebra, SparseArrays

julia> graph = [
           0 1 0 0 0 0 0 0
           1 0 1 0 0 1 0 0
           0 1 0 1 0 1 1 1
           0 0 1 0 0 0 0 0
           0 0 0 0 0 1 1 0
           0 1 1 0 1 0 0 0
           0 0 1 0 1 0 0 1
           0 0 1 0 0 0 1 0
       ];

julia> label, tree = cliquetree(graph);

julia> tree
6-element CliqueTree{Int64, Int64}:
 [6, 7, 8]
 └─ [5, 7, 8]
    ├─ [1, 5]
    ├─ [3, 5, 7]
    │  └─ [2, 3]
    └─ [4, 5, 8]
```

The clique tree `tree` is a tree decomposition of the permuted graph `graph[label, label]`.
A clique tree is a vector of cliques, so you can retrieve the clique at node 4 by typing `tree[4]`.

```julia-repl
julia> tree[4]
3-element Clique{Int64, Int64}:
 4
 5
 8
```

!!! warning
    The numbers in each clique are vertices of the permuted graph `graph[label, label]`.
    You can see the vertices of the original graph by typing
    ```julia-repl
    julia> label[tree[4]]
    3-element Vector{Int64}:
    8
    3
    7
    ```
    Notice that the clique is no longer sorted.

The width of a clique tree is computed by the function `treewidth`.

```julia-repl
julia> treewidth(tree)
2
```

### Chordal Completions

Clique trees can be used to construct chordal completions.

```julia-repl
julia> filledgraph = FilledGraph(tree)
{8, 11} FilledGraph{Int64, Int64}

julia> sparse(filledgraph)
8×8 SparseMatrixCSC{Bool, Int64} with 11 stored entries:
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 1  ⋅  1  1  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  1  1  ⋅  ⋅
 ⋅  ⋅  ⋅  1  1  1  1  ⋅
```

The graph `filledgraph` is ordered: its edges are directed from lower to higher vertices. The underlying undirected graph is a chordal completion of the permuted graph `graph[label, label]`.

```julia-repl
julia> chordalgraph = Symmetric(sparse(filledgraph), :L)
8×8 Symmetric{Bool, SparseMatrixCSC{Bool, Int64}}:
 ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  1  ⋅  ⋅  1  ⋅  1  ⋅
 ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  1
 1  ⋅  1  1  ⋅  ⋅  1  1
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  1
 ⋅  ⋅  1  ⋅  1  1  ⋅  1
 ⋅  ⋅  ⋅  1  1  1  1  ⋅

julia> ischordal(graph)
false

julia> ischordal(chordalgraph)
true

julia> all(graph[label, label] .<= chordalgraph)
true
```

### Cholesky Factorization

The function `cholesky` computes Cholesky factorizations of sparse positive-definite matrices.

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

You can solve linear systems of equations with the operators
`/` and `\`.

```julia-repl
julia> lhs = rand(2, 8); rhs = copy(transpose(lhs));

julia> lhs / cholfact # lhs * inv(matrix)
2×8 Matrix{Float64}:
 -0.202009  0.661177   0.173183  0.110932   0.375653  -0.556495  -0.0751984  0.0793129
 -0.164852  0.665989  -0.126911  0.0915613  0.187998  -0.378656   0.0536805  0.127395

julia> cholfact \ rhs # inv(matrix) * rhs
8×2 Matrix{Float64}:
 -0.202009   -0.164852
  0.661177    0.665989
  0.173183   -0.126911
  0.110932    0.0915613
  0.375653    0.187998
 -0.556495   -0.378656
 -0.0751984   0.0536805
  0.0793129   0.127395
```

The function `symbolic` computes symbolic factorizations.

```julia-repl
julia> symbfact = CliqueTrees.symbolic(matrix)
SymbFact{Int64}:
    nnz: 19
```

Symbolic factorizations can be reused to factorize matrices with
the same sparsity pattern.

```julia-repl
julia> matrix[1, 2] = matrix[2, 1] = 2;

julia> cholfact = CliqueTrees.cholesky(matrix, symbfact)
CholFact{Float64, Int64}:
    nnz: 19
    success: true
```

## Graphs

Users can input graphs as adjacency matrices. Additionally, CliqueTrees.jl supports the `HasGraph` type from [Catlab.jl](https://github.com/AlgebraicJulia/Catlab.jl) and the `AbstractGraph` type from [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl). Instances of the latter should implement the following subset of the [abstract graph interface](https://juliagraphs.org/Graphs.jl/stable/core_functions/interface/).

  - `is_directed`
  - `ne`
  - `nv`
  - `outneighbors`
  - `vertices`

Self-edges are always ignored.

## Citation

If you use CliqueTrees.jl for a publication, please cite it as follows.

```bibtex
@misc{cliquetrees2025samuelson,
  author = {Samuelson, Richard and Fairbanks, James},
  url = {https://github.com/AlgebraicJulia/CliqueTrees.jl},
  title = {CliqueTrees.jl: A Julia library for computing tree decompositions and chordal completions of graphs},
  year = {2025}
}
```
