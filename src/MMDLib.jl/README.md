# MMDLib.jl

The multiple minimum degree algorithm.

## Basic Usage

The function `mmd` computes a fill-reducing permutation of a simple graph.

```julia-repl
julia> using CliqueTrees.MMDLib, SparseArrays

julia> graph = sparse([
           0 1 0 0 0 0 0 0
           1 0 1 0 0 1 0 0
           0 1 0 1 0 1 1 1
           0 0 1 0 0 0 0 0
           0 0 0 0 0 1 1 0
           0 1 1 0 1 0 0 0
           0 0 1 0 1 0 0 1
           0 0 1 0 0 0 1 0
       ]);

julia> invp = mmd(8, graph.colptr, graph.rowval)
8-element Vector{Int64}:
 2
 3
 6
 1
 5
 7
 8
 4
```
