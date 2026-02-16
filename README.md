<div align="center">
    <img src="logo.png" alt="CliqueTrees.jl" width="200">
</div>
<h1 align="center">
    CliqueTrees.jl
    <p align="center">
        <a href="https://algebraicjulia.github.io/CliqueTrees.jl/stable">
            <img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="documentation (stable)">
        </a>
        <a href="https://algebraicjulia.github.io/CliqueTrees.jl/dev">
            <img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="documentation (development)">
        </a>
        <a href="https://github.com/AlgebraicJulia/CliqueTrees.jl/actions/workflows/julia_ci.yml">
            <img src="https://github.com/AlgebraicJulia/CliqueTrees.jl/actions/workflows/julia_ci.yml/badge.svg" alt="build status">
        </a>
        <a href="https://codecov.io/gh/AlgebraicJulia/CliqueTrees.jl">
            <img src="https://codecov.io/gh/AlgebraicJulia/CliqueTrees.jl/branch/main/graph/badge.svg" alt="code coverage">
        </a>
        <a href="https://github.com/fredrikekre/Runic.jl">
            <img src="https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black" alt="Runic">
        </a>
        <a href="https://github.com/JuliaTesting/Aqua.jl">
            <img src="https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg" alt="Aqua">
        </a>
        <a href="https://github.com/aviatesk/JET.jl">
            <img src="https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a" alt="JET">
        </a>
    </p>
</h2>

CliqueTrees.jl implements *clique trees* in Julia. You can use it to construct [clique trees](https://en.wikipedia.org/wiki/Tree_decomposition) and [chordal completions](https://en.wikipedia.org/wiki/Chordal_completion) of graphs. Additionally, you can use the submodule `CliqueTrees.Multifrontal` to compute Cholesky and LDLt factorizations of sparse matrices.

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

```
pkg> add CliqueTrees
```

## Clique Trees

A clique tree (also tree decomposition, junction tree, or join tree) partitions a graph into a tree of overlapping subgraphs.
The key invariant of a clique tree is the *running intersection property*: if two subgraphs in the tree contain the same
vertex, then so do all the subgraphs on the unique path between them.

<div align="center">
    <img src="tree.svg" alt="clique tree" width="400">
</div>

Clique trees play an important role in algorithms for

- graph coloring
- probabilistic inference
- tensor network contraction
- matrix factorization
- semidefinite programming
- polynomial optimization

and more. In all of these applications, it is important that the subgraphs in a clique tree be as small as possible.
Consider, for example graph coloring. 3-coloring is NP-Hard, and the fastest known algorithm for deciding if a graph is
3-colorable runs in $O(1.3289^n)$ time. However, if we are given tree decomposition of a graph with $m$ subgraphs,
and whose largest subgraph contains $k$ vertices, then we can decide if it is 3-colorable in $O(m3^k)$ time. This
is very powerful, but only if $k$ is very small.

## Constructing Clique Trees

Clique trees can be constructed using the function `cliquetree`. The function returns two objects: a vertex permutation
and a clique tree.

```julia-repl
julia> using CliqueTrees

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

julia> perm, tree = cliquetree(graph; alg=MF());

julia> tree
6-element CliqueTree{Int64, Int64}:
 [6, 7, 8]
 └─ [5, 7, 8]
    ├─ [1, 5]
    ├─ [3, 5, 7]
    │  └─ [2, 3]
    └─ [4, 5, 8]
```

The clique tree object behaves like a vector of vectors, with `perm[tree[i]]` containing the vertices in the i-th
subgraph. The tree also implements the [indexed tree interface](http://juliacollections.github.io/AbstractTrees.jl/stable/#The-Indexed-Tree-Interface)
from AbstractTrees.jl.

## Algorithms

The `alg` keyword argument specifies which algorithm is used to construct the clique tree. CliqueTrees.jl implements
a large number of algorithms for solving problems. It also interfaces with libraries like TreeWidthSolver.jl, Metis.jl,
and AMD.jl, so that users can use the algorithms implemented there as well. Algorithms for computing clique trees
can be divided into two categories: exact and approximate.

### Exact Algorithms

Exact algorithms construct optimal clique trees. CliqueTrees.jl exports two exact algorithms.

- `BT`: Bouchite-Toudinca
- `PIDBT`: positive-instance driven Bouchitte-Toudinca

Beware! These algorithms are solving an NP-Hard problem. Users are advised to wrap them in the
pre-processing algorithms `SafeSeparators` and `SafeRules`.

```julia-repl
julia> alg = SafeRules(SafeSeparators(PIDBT()));
```

If a graph is [chordal](https://en.wikipedia.org/wiki/Chordal_graph), then an optimal clique tree can be computed in
linear time using either of the following algorithms.

- `MCS`: maximum cardinality search
- `LexBFS`: lexicographic breadth-first search

Users can detect whether a graph is chordal using the function `ischordal`.

### Heuristic Algorithms

Because computing optimal clique trees is NP-Hard, a large number of approximate algorithms have been developed for
quickly computing non-optimal ones. These include the following.

- `RCM`: reverse Cuthill-McKee
- `MMD`: multiple minimum degree
- `MF`: minimum fill
- `AMD`: approximate minimum degree
- `AMF`: approximate minimum fill
- `METIS`: nested dissection
- `ND`: nested dissection

For large graphs, the algorithms `AMD` and `METIS` are the state-of-the practice. They are implemented in the C libraries
SuiteSparse and METIS. The current default algorithm is `MF`: a slower but more reliable alternative to `AMD`.

### Pre-Processing Algorithms

The performance of clique tree algorithms can be improved by wrapping them one or more of the following pre-processing
algorithms.

- `ConnectedComponents`
- `Compression`
- `SimplicialRule`
- `SafeRules`
- `SafeSeparators`

## Matrix Factorization

An important application of clique trees is sparse matrix factorization. The multifrontal Cholesky factorization algorithm
uses a clique tree to schedule computations, performing a dense matrix factorization at each subgraph. This algorithm is
implemented in the submodule `CliqueTrees.Multifrontal`.

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

The multifrontal LDLt factorization algorithm is implemented as well. It can be used to factorize quasi-definite matrices.

```julia-repl
julia> F = ldlt!(ChordalLDLt(A))
5×5 FChordalLDLt{:L, Float64, Int64} with 10 stored entries:
 1.0   ⋅    ⋅    ⋅    ⋅
 0.0  1.0   ⋅    ⋅    ⋅
 0.0  0.5  1.0   ⋅    ⋅
 0.5  0.0  0.0  1.0   ⋅
 0.0  0.5  0.5  0.5  1.0

 4.0   ⋅    ⋅    ⋅    ⋅
  ⋅   4.0   ⋅    ⋅    ⋅
  ⋅    ⋅   4.0   ⋅    ⋅
  ⋅    ⋅    ⋅   4.0   ⋅
  ⋅    ⋅    ⋅    ⋅   4.0
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
