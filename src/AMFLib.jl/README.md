# AMFLib.jl

The approximate minimum fill algorithm.

## Basic Usage

The function `amf` computes a fill-reducing permutation of a simple graph.

```julia-repl
julia> using CliqueTrees.AMFLib, SparseArrays

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

julia> perm, invp = amf(graph.colptr, graph.rowval);

julia> perm
8-element Vector{Int64}:
 4
 1
 2
 8
 3
 5
 6
 7
```
