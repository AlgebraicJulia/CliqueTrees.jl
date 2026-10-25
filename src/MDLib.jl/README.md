# MDLib.jl

The exact minimum degree algorithm.

## Basic Usage

The function `md` computes a minimum degree permutation of a graph with n
vertices and m edges in O(mn) time and O(m + n) space. It returns the inverse
permutation `invp`.

```julia-repl
julia> using CliqueTrees.MDLib, SparseArrays

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

julia> invp = md(8, graph.colptr, graph.rowval)
8-element Vector{Int64}:
 2
 3
 8
 1
 5
 4
 7
 6
```

## Parameters

    md(neqns[, vwght], xadj, adjncy; external=true, dense=10.0, mass=true, twin=true, compress=true)

  - `neqns`: the number of vertices
  - `vwght`: vertex weights, positive integers; a vertex of weight w stands
    for w indistinguishable vertices (default 1)
  - `xadj`, `adjncy`: the graph; the neighbors of `v` are
    `adjncy[xadj[v]:xadj[v + 1] - 1]`, and self loops are ignored
  - `external`: minimize the external degree instead of the degree
  - `dense`: vertices with more than `max(16, dense * sqrt(neqns))` neighbors
    are handled as dense (a negative value disables this)
  - `mass`: mass elimination
  - `twin`: false-twin detection
  - `compress`: merge the vertices with the same closed neighborhood first
