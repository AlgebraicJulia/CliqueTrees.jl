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

CliqueTrees.jl implements *clique trees* in Julia. You can use it to construct [tree decompositions](https://en.wikipedia.org/wiki/Tree_decomposition) and [chordal completions](https://en.wikipedia.org/wiki/Chordal_completion) of graphs.

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

> [!WARNING]
> The numbers in each clique are vertices of the permuted graph `graph[label, label]`.
> You can see the vertices of the original graph by typing
> ```julia-repl
> julia> label[tree[4]]
> 3-element Vector{Int64}:
>  8
>  3
>  7
> ```
> Notice that the clique is no longer sorted.

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

## Graphs

Users can input graphs as adjacency matrices. Additionally, CliqueTrees.jl supports the `HasGraph` type from [Catlab.jl](https://github.com/AlgebraicJulia/Catlab.jl) and the `AbstractGraph` type from [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl). Instances of the latter should implement the following subset of the [abstract graph interface](https://juliagraphs.org/Graphs.jl/stable/core_functions/interface/).

  - `is_directed`
  - `ne`
  - `nv`
  - `outneighbors`
  - `vertices`

Weights and self-edges are always ignored.

## References

CliqueTrees.jl was inspired by the book [Chordal Graphs and Semidefinite Optimization](https://www.nowpublishers.com/article/Details/OPT-006) by Vandenberghe and Andersen.
