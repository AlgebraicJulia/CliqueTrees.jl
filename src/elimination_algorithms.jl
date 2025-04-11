"""
    EliminationAlgorithm

A *graph elimination algorithm* computes a permutation of the vertices of a graph, which induces a chordal completion
of the graph. The algorithms below generally seek to minimize the *fill* (number of edges) or *width* (largest clique)
of the completed graph.

| type                          | name                                         | time     | space    |
|:----------------------------- |:-------------------------------------------- |:-------- |:-------- |
| [`BFS`](@ref)                 | breadth-first search                         | O(m + n) | O(n)     |
| [`MCS`](@ref)                 | maximum cardinality search                   | O(m + n) | O(n)     |
| [`LexBFS`](@ref)              | lexicographic breadth-first search           | O(m + n) | O(m + n) |
| [`RCMMD`](@ref)               | reverse Cuthill-Mckee (minimum degree)       | O(m + n) | O(m + n) |
| [`RCMGL`](@ref)               | reverse Cuthill-Mckee (George-Liu)           | O(m + n) | O(m + n) |
| [`MCSM`](@ref)                | maximum cardinality search (minimal)         | O(mn)    | O(n)     |
| [`LexM`](@ref)                | lexicographic breadth-first search (minimal) | O(mn)    | O(n)     |
| [`AMF`](@ref)                 | approximate minimum fill                     | O(mn)    | O(m + n) |
| [`MF`](@ref)                  | minimum fill                                 | O(mn²)   |          |
| [`MMD`](@ref)                 | multiple minimum degree                      | O(mn²)   | O(m + n) |
| [`MinimalChordal`](@ref)      | MinimalChordal                               |          |          |
| [`CompositeRotations`](@ref)  | elimination tree rotation                    | O(m + n) | O(m + n) |
| [`SafeRules`](@ref)           | treewith-safe rule-based reduction           |          |          |    
| [`SafeSeparators`](@ref)      | treewith-safe separator decomposition        |          |          |    
| [`ConnectedComponents`](@ref) | connected component decomposition            |          |          |

The following additional algorithms are implemented as package extensions and require loading an additional package.

| type                             | name                              | time  | space    | package                                                                                   |
|:-------------------------------- |:--------------------------------- |:----- |:-------- |:----------------------------------------------------------------------------------------- |
| [`AMD`](@ref)                    | approximate minimum degree        | O(mn) | O(m + n) | [AMD.jl](https://github.com/JuliaSmoothOptimizers/AMD.jl)                                 |
| [`SymAMD`](@ref)                 | column approximate minimum degree | O(mn) | O(m + n) | [AMD.jl](https://github.com/JuliaSmoothOptimizers/AMD.jl)                                 |
| [`METIS`](@ref)                  | multilevel nested dissection      |       |          | [Metis.jl](https://github.com/JuliaSparse/Metis.jl)                                       |
| [`Spectral`](@ref)               | spectral ordering                 |       |          | [Laplacians.jl](https://github.com/danspielman/Laplacians.jl)                             |
| [`FlowCutter`](@ref)             | FlowCutter                        |       |          | [FlowCutterPACE17_jll.jl](https://github.com/JuliaBinaryWrappers/FlowCutterPACE17_jll.jl) |
| [`BT`](@ref)                     | Bouchitte-Todinca                 |       |          | [TreeWidthSolver.jl](https://github.com/ArrogantGao/TreeWidthSolver.jl)                   |
| [`SAT{libpicosat_jll}`](@ref)    | SAT encoding (picosat)            |       |          | [libpicosat_jll.jl](https://github.com/JuliaBinaryWrappers/libpicosat_jll.jl)             |
| [`SAT{PicoSAT_jll}`](@ref)       | SAT encoding (picosat)            |       |          | [PicoSAT_jll.jl](https://github.com/JuliaBinaryWrappers/PicoSAT_jll.jl)                   |
| [`SAT{Lingeling_jll}`](@ref)     | SAT encoding (lingeling)          |       |          | [Lingeling_jll.jl](https://github.com/JuliaBinaryWrappers/Lingeling_jll.jl)               |
| [`SAT{CryptoMiniSat_jll}`](@ref) | SAT encoding (cryptominisat)      |       |          | [CryptoMiniSat_jll.jl](https://github.com/JuliaBinaryWrappers/CryptoMiniSat_jll.jl)       |

# Triangulation Recognition Heuristics

  - [`MCS`](@ref)
  - [`LexBFS`](@ref)
  - [`MCSM`](@ref)
  - [`LexM`](@ref)

These algorithms are guaranteed to compute perfect elimination orderings for chordal graphs. [`MCSM`](@ref) and [`LexM`](@ref)
are variants of [`MCS`](@ref) and [`LexBFS`](@ref) that compute minimal orderings. The Lex algorithms were pubished first,
and the MCS algorithms were introducd later as simplications. In practice, these algorithms work poorly on non-chordal graphs.

# Bandwidth and Envelope Reduction Heuristics

  - [`RCMMD`](@ref)
  - [`RCMGL`](@ref)
  - [`Spectral`](@ref)

These algorithms seek to minimize the *bandwidth* and *profile* of the permuted graph, quantities that upper bound the width
and fill of theinduced chordal completion. [`RCMMD`](@ref) and [`RCMGL`](@ref) are two variants of the reverse Cuthill-McKee
algorithm, a type of breadth-first search. They differ in in their choice of starting vertex. In practice, these algorithms work
better than the triangulation recognition heuristics and worse than the greedy heuristics.

# Greedy Heuristics

  - [`MMD`](@ref)
  - [`MF`](@ref)
  - [`AMD`](@ref)
  - [`SymAMD`](@ref)
  - [`AMF`](@ref)

These algorithms simulate the elimination process, greedity selecting vertices to eliminate. [`MMD`](@ref) selects a vertex
of minimum degree, and [`MF`](@ref) selects a vertex that induces the least fill. Updating the degree or fill of every vertex
after elimination is costly; the algorithms [`AMD`](@ref), [`SymAMD`](@ref), and [`AMF`](@ref) are relaxations that work by
approximating these values. The [`AMD`](@ref) algorithm is the state-of-the-practice for sparse matrix ordering.

# Exact Treewidth Algorithms

  - [`BT`](@ref)
  - [`SAT`](@ref)

These algorithm minimizes the treewidth of the completed graph.

!!! warning
    This is an NP-hard problem. I recommend wrapping exact treewidth algorithms with preprocessors like
    [`SafeRules`](@ref) or [`ConnectedComponents`](@ref). 

# Meta Algorithms

  - [`MinimalChordal`](@ref)
  - [`CompositeRotations`](@ref)
  - [`SafeRules`](@ref)
  - [`ConnectedComponents`](@ref)

These algorithms are parametrized by another algorithm and work by transforming its input or output. 

"""
abstract type EliminationAlgorithm end

"""
    PermutationOrAlgorithm = Union{AbstractVector, EliminationAlgorithm}

Either a permutation or an algorithm.
"""
const PermutationOrAlgorithm = Union{AbstractVector, EliminationAlgorithm}

"""
    BFS <: EliminationAlgorithm

    BFS()

The [breadth-first search algorithm](https://en.wikipedia.org/wiki/Breadth-first_search).

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

julia> alg = BFS()
BFS

julia> treewidth(graph; alg)
2
```
"""
struct BFS <: EliminationAlgorithm end

"""
    MCS <: EliminationAlgorithm

    MCS()

The maximum cardinality search algorithm.

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

julia> alg = MCS()
MCS

julia> treewidth(graph; alg)
3
```

### References

  - Tarjan, Robert E., and Mihalis Yannakakis. "Simple linear-time algorithms to test chordality of graphs, test acyclicity of hypergraphs, and selectively reduce acyclic hypergraphs." *SIAM Journal on Computing* 13.3 (1984): 566-579.
"""
struct MCS <: EliminationAlgorithm end

"""
    LexBFS <: EliminationAlgorithm

    LexBFS()

The [lexicographic breadth-first-search algorithm](https://en.wikipedia.org/wiki/Lexicographic_breadth-first_search).

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

julia> alg = LexBFS()
LexBFS

julia> treewidth(graph; alg)
2
```

### References

  - Rose, Donald J., R. Endre Tarjan, and George S. Lueker. "Algorithmic aspects of vertex elimination on graphs." *SIAM Journal on Computing* 5.2 (1976): 266-283.
"""
struct LexBFS <: EliminationAlgorithm end

"""
    RCMMD{A} <: EliminationAlgorithm

    RCMMD(alg::Algorithm)

    RCMMD()

The [reverse Cuthill-McKee algorithm](https://en.wikipedia.org/wiki/Cuthill%E2%80%93McKee_algorithm).
An initial vertex is selected using the minimum degree heuristic.

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

julia> alg = RCMMD(QuickSort)
RCMMD:
    Base.Sort.QuickSortAlg()

julia> treewidth(graph; alg)
3
```

### Parameters

  - `alg`: sorting algorithm

### References

  - Cuthill, Elizabeth, and James McKee. "Reducing the bandwidth of sparse symmetric matrices." *Proceedings of the 1969 24th National Conference.* 1969.
"""
struct RCMMD{A <: SortingAlgorithm} <: EliminationAlgorithm
    alg::A
end

function RCMMD()
    return RCMMD(DEFAULT_UNSTABLE)
end

"""
    RCMGL{A} <: EliminationAlgorithm

    RCMGL(alg::Algorithm)

    RCMGL()

The [reverse Cuthill-McKee algorithm](https://en.wikipedia.org/wiki/Cuthill%E2%80%93McKee_algorithm).
An initial vertex is selected using George and Liu's variant of the GPS algorithm.

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

julia> alg = RCMGL(QuickSort)
RCMGL:
    Base.Sort.QuickSortAlg()

julia> treewidth(graph; alg)
3
```

### Parameters

  - `alg`: sorting algorithm

### References

  - Cuthill, Elizabeth, and James McKee. "Reducing the bandwidth of sparse symmetric matrices." *Proceedings of the 1969 24th National Conference.* 1969.
  - George, Alan, and Joseph WH Liu. "An implementation of a pseudoperipheral node finder." *ACM Transactions on Mathematical Software (TOMS)* 5.3 (1979): 284-295.
"""
struct RCMGL{A <: SortingAlgorithm} <: EliminationAlgorithm
    alg::A
end

function RCMGL()
    return RCMGL(DEFAULT_UNSTABLE)
end

"""
    LexM <: EliminationAlgorithm

    LexM()

A minimal variant of the [lexicographic breadth-first-search algorithm](https://en.wikipedia.org/wiki/Lexicographic_breadth-first_search).

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

julia> alg = LexM()
LexM

julia> treewidth(graph; alg)
2
```

### References

  - Rose, Donald J., R. Endre Tarjan, and George S. Lueker. "Algorithmic aspects of vertex elimination on graphs." *SIAM Journal on Computing* 5.2 (1976): 266-283.
"""
struct LexM <: EliminationAlgorithm end

"""
    MCSM <: EliminationAlgorithm

    MCSM()

A minimal variant of the maximal cardinality search algorithm.

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

julia> alg = MCSM()
MCSM

julia> treewidth(graph; alg)
2
```

### References

  - Berry, Anne, et al. "Maximum cardinality search for computing minimal triangulations of graphs." *Algorithmica* 39 (2004): 287-298.
"""
struct MCSM <: EliminationAlgorithm end

"""
    AMF <: EliminationAlgorithm

    AMF(; speed=1)

The approximate minimum fill algorithm.

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

julia> alg = AMF(; speed=2)
AMF:
    speed: 2

julia> treewidth(graph; alg)
2
```

### Parameters

  - `speed`: fill approximation strategy (`1`, `2`, or, `3`)

### References

  - Rothberg, Edward, and Stanley C. Eisenstat. "Node selection strategies for bottom-up sparse matrix ordering." SIAM Journal on Matrix Analysis and Applications 19.3 (1998): 682-695.
"""
@kwdef struct AMF <: EliminationAlgorithm
    speed::Int = 1
end

"""
    MF <: EliminationAlgorithm

    MF()

The greedy minimum fill algorithm.

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

julia> alg = MF
MF

julia> treewidth(graph; alg)
2
```

### References

  - Tinney, William F., and John W. Walker. "Direct solutions of sparse network equations by optimally ordered triangular factorization." *Proceedings of the IEEE* 55.11 (1967): 1801-1809.
"""
struct MF <: EliminationAlgorithm end

"""
    MMD <: EliminationAlgorithm

    MMD(; delta=0)

The [multiple minimum degree algorithm](https://en.wikipedia.org/wiki/Minimum_degree_algorithm).

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

julia> alg = MMD(; delta=1)
MMD:
    delta: 1

julia> treewidth(graph; alg)
2
```

### Parameters

  - `delta`: tolerance for multiple elimination

### References

  - Liu, Joseph WH. "Modification of the minimum-degree algorithm by multiple elimination." *ACM Transactions on Mathematical Software (TOMS)* 11.2 (1985): 141-153.
"""
@kwdef struct MMD <: EliminationAlgorithm
    delta::Int = 0
end

"""
    AMD <: EliminationAlgorithm

    AMD(; dense=10.0, aggressive=1.0)

The approximate minimum degree algorithm.

```julia-repl
julia> using CliqueTrees

julia> import AMD as AMDLib

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

julia> alg = AMD(; dense=5.0, aggressive=2.0)
AMD:
    dense: 5.0
    aggressive: 2.0

julia> treewidth(graph; alg)
2
```

### Parameters

  - `dense`: dense row parameter
  - `aggressive`: aggressive absorption

### References

  - Amestoy, Patrick R., Timothy A. Davis, and Iain S. Duff. "An approximate minimum degree ordering algorithm." *SIAM Journal on Matrix Analysis and Applications* 17.4 (1996): 886-905.
"""
@kwdef struct AMD <: EliminationAlgorithm
    dense::Float64 = 10.0
    aggressive::Float64 = 1.0
end

"""
    SymAMD <: EliminationAlgorithm

    SymAMD(; dense_row=10.0, dense_col=10.0, aggressive=1.0)

The column approximate minimum degree algorithm.

```julia-repl
julia> using CliqueTrees

julia> import AMD as AMDLib

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

julia> alg = SymAMD(; dense_row=5.0, dense_col=5.0, aggressive=2.0)
SymAMD:
    dense_row: 5.0
    dense_col: 5.0
    aggressive: 2.0


julia> treewidth(graph, alg)
2
```

### Parameters

  - `dense_row`: dense row parameter
  - `dense_column`: dense column parameter
  - `aggressive`: aggressive absorption

### References

  - Davis, Timothy A., et al. "A column approximate minimum degree ordering algorithm." *ACM Transactions on Mathematical Software* (TOMS) 30.3 (2004): 353-376.
"""
@kwdef struct SymAMD <: EliminationAlgorithm
    dense_row::Float64 = 10.0
    dense_col::Float64 = 10.0
    aggressive::Float64 = 1.0
end

"""
    METIS <: EliminationAlgorithm

    METIS(; ctype=-1, rtype=-1, nseps=-1, niter=-1, seed=-1,
            compress=-1, ccorder=-1, pfactor=-1, ufactor=-1)

The multilevel [nested dissection](https://en.wikipedia.org/wiki/Nested_dissection) algorithm implemented in METIS.

```julia-repl
julia> using CliqueTrees, Metis

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

julia> alg = METIS(ctype=Metis.METIS_CTYPE_RM)
METIS:
    ctype: 0
    rtype: -1
    nseps: -1
    niter: -1
    seed: -1
    compress: -1
    ccorder: -1
    pfactor: -1
    ufactor: -1


julia> treewidth(graph; alg)
3
```

### Parameters

  - `ctype`: matching scheme to be used during coarsening
  - `rtype`: algorithm used for refinement
  - `nseps`: number of different separators computed at each level of nested dissection
  - `niter`: number of iterations for refinement algorithm at each stage of the uncoarsening process
  - `seed`: random seed
  - `compress`: whether to combine vertices with identical adjacency lists
  - `ccorder`: whether to order connected components separately
  - `pfactor`: minimum degree of vertices that will be ordered last
  - `ufactor`: maximum allowed load imbalance partitions

### References

  - Karypis, George, and Vipin Kumar. "A fast and high quality multilevel scheme for partitioning irregular graphs." *SIAM Journal on Scientific Computing* 20.1 (1998): 359-392.
"""
@kwdef struct METIS <: EliminationAlgorithm
    ctype::Int = -1
    rtype::Int = -1
    nseps::Int = -1
    niter::Int = -1
    seed::Int = -1
    compress::Int = -1
    ccorder::Int = -1
    pfactor::Int = -1
    ufactor::Int = -1
end

"""
    Spectral <: EliminationAlgorithm

    Spectral(; tol=0.0)

The spectral ordering algorithm only works on connected graphs.
In order to use it, import the package [Laplacians](https://github.com/danspielman/Laplacians.jl).

```julia-repl
julia> using CliqueTrees, Laplacians

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

julia> alg = Spectral(; tol=0.001)
Spectral:
    tol: 0.001

julia> treewidth(graph; alg)
4
```

### Parameters

  - `tol`: tolerance for convergence

### References

  - Barnard, Stephen T., Alex Pothen, and Horst D. Simon. "A spectral algorithm for envelope reduction of sparse matrices." *Proceedings of the 1993 ACM/IEEE Conference on Supercomputing.* 1993.
"""
@kwdef struct Spectral <: EliminationAlgorithm
    tol::Float64 = 0.0
end

"""
    FlowCutter <: EliminationAlgorithm

    FlowCutter(; time=5, seed=0)

The FlowCutter algorithm.

```julia-repl
julia> using CliqueTrees, FlowCutterPACE17_jll

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

julia> alg = FlowCutter(; time=2, seed=1)
FlowCutter:
    time: 2
    seed: 1

julia> treewidth(graph; alg)
2
```

### Parameters

  - `time`: run time
  - `seed`: random seed

### References

  - Strasser, Ben. "Computing tree decompositions with flowcutter: PACE 2017 submission." arXiv preprint arXiv:1709.08949 (2017).
"""
@kwdef struct FlowCutter <: EliminationAlgorithm
    time::Int = 5
    seed::Int = 0
end

"""
    BT <: EliminationAlgorithm

    BT()

The Bouchitte-Todinca algorithm.

```julia-repl
julia> using CliqueTrees, TreeWidthSolver

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

julia> alg = BT()
BT

julia> treewidth(graph; alg)
2
```

### References

  - Korhonen, Tuukka, Jeremias Berg, and Matti Järvisalo. "Solving Graph Problems via Potential Maximal Cliques: An Experimental Evaluation of the Bouchitté-Todinca Algorithm." *Journal of Experimental Algorithmics (JEA)* 24 (2019): 1-19.
"""
struct BT <: EliminationAlgorithm end

"""
    SAT{H, L, U} <: EliminationAlgorithm

    SAT{H}(lb::WidthOrAlgorithn, ub::PermutationOrAlgorithm)

    SAT{H}()

Compute a minimum-treewidth permutation using a SAT solver.

```julia-repl
julia> using CliqueTrees, libpicosat_jll, PicoSAT_jll, CryptoMiniSat_jll, Lingeling_jll

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

julia> alg = SAT{libpicosat_jll}(MMW(), MF()) # picosat
SAT{libpicosat_jll, MMW, MF}:
    MMW
    MF

julia> alg = SAT{PicoSAT_jll}(MMW(), MF()) # picosat
SAT{PicoSAT_jll, MMW, MF}:
    MMW
    MF

julia> alg = SAT{CryptoMiniSat_jll}(MMW(), MF()) # cryptominisat
SAT{CryptoMiniSat_jll, MMW, MF}:
    MMW
    MF

julia> alg = SAT{Lingeling_jll}(MMW(), MF()) # lingeling
SAT{Lingeling_jll, MMW, MF}:
    MMW
    MF

julia> treewidth(graph; alg)
2
```

### Parameters

  - `lb`: lower bound algorithm
  - `ub`: upper bound algorithm

## References

  - Samer, Marko, and Helmut Veith. "Encoding treewidth into SAT." *Theory and Applications of Satisfiability Testing-SAT 2009: 12th International Conference, SAT 2009*, Swansea, UK, June 30-July 3, 2009. Proceedings 12. Springer Berlin Heidelberg, 2009.
  - Berg, Jeremias, and Matti Järvisalo. "SAT-based approaches to treewidth computation: An evaluation." *2014 IEEE 26th international conference on tools with artificial intelligence.* IEEE, 2014.
  - Bannach, Max, Sebastian Berndt, and Thorsten Ehlers. "Jdrasil: A modular library for computing tree decompositions." *16th International Symposium on Experimental Algorithms (SEA 2017)*. Schloss Dagstuhl–Leibniz-Zentrum fuer Informatik, 2017.
"""
struct SAT{H, L <: WidthOrAlgorithm, U <: PermutationOrAlgorithm} <: EliminationAlgorithm
    handle::Val{H}
    lb::L
    ub::U
end

function SAT{H}(lb::WidthOrAlgorithm, ub::PermutationOrAlgorithm) where {H}
    return SAT(Val(H), lb, ub)
end

function SAT{H}() where {H}
    return SAT{H}(DEFAULT_LOWER_BOUND_ALGORITHM, DEFAULT_ELIMINATION_ALGORITHM)
end

"""
    MinimalChordal{A} <: EliminationAlgorithm

    MinimalChordal(alg::PermutationOrAlgorithm)

    MinimalChordal()

Evaluate an elimination algorithm, and them improve its output using the MinimalChordal algorithm. The result is guaranteed to be minimal.

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

julia> alg1 = MCS()
MCS

julia> alg2 = MinimalChordal(MCS())
MinimalChordal{MCS}:
    MCS

julia> label1, tree1 = cliquetree(graph; alg=alg1);

julia> label2, tree2 = cliquetree(graph; alg=alg2);

julia> FilledGraph(tree1) # more edges
{8, 12} FilledGraph{Int64, Int64}

julia> FilledGraph(tree2) # fewer edges
{8, 11} FilledGraph{Int64, Int64}
```

### Parameters

  - `alg`: elimination algorithm

### References

  - Blair, Jean RS, Pinar Heggernes, and Jan Arne Telle. "A practical algorithm for making filled graphs minimal." *Theoretical Computer Science* 250.1-2 (2001): 125-141.
"""
struct MinimalChordal{A <: PermutationOrAlgorithm} <: EliminationAlgorithm
    alg::A
end

function MinimalChordal()
    return MinimalChordal(DEFAULT_ELIMINATION_ALGORITHM)
end

"""
    CompositeRotations{C, A} <: EliminationAlgorithm

    CompositeRotations(clique::AbstractVector, alg::EliminationAlgorithm)

    CompositeRotations(clique::AbstractVector)

Evaluate an eliminaton algorithm, ensuring that the given clique is at the end of the ordering.

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

julia> alg = CompositeRotations([2], MCS())
CompositeRotations{Vector{Int64}, MCS}:
    clique: [2]
    MCS

julia> order, index = permutation(graph; alg);

julia> order # 2 is the last vertex in the ordering
8-element Vector{Int64}:
 4
 5
 7
 8
 3
 6
 1
 2
```

### Parameters

  - `clique`: clique
  - `alg`: elimination algorithm

### References

  - Liu, Joseph WH. "Equivalent sparse matrix reordering by elimination tree rotations." *Siam Journal on Scientific and Statistical Computing* 9.3 (1988): 424-444.
"""
struct CompositeRotations{C <: AbstractVector, A <: PermutationOrAlgorithm} <:
    EliminationAlgorithm
    clique::C
    alg::A
end

function CompositeRotations(clique::AbstractVector)
    return CompositeRotations(clique, DEFAULT_ELIMINATION_ALGORITHM)
end

"""
    SafeRules{L, U} <: EliminationAlgorithm

    SafeRules(lb::WidthOrAlgorithm, ub::EliminationAlgororithm)

    SafeRules()

Preprocess a graph using safe reduction rules. The algorithm `lb` is used to compute a lower bound
to the treewidth; better lower bounds allow the algorithm to perform more reductions.

```julia-repl
julia> using CliqueTrees, TreeWidthSolver

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

julia> alg1 = BT()
BT

julia> alg2 = SafeRules(MMW(), BT())
SafeRules{MMW, BT}:
    MMW
    BT

julia> @time treewidth(graph; alg=alg1) # slower
  0.000177 seconds (1.41 k allocations: 90.031 KiB)
2

julia> @time treewidth(graph; alg=alg2) # faster
  0.000066 seconds (237 allocations: 16.875 KiB)
2
```

### Parameters

  - `lb`: lower bound algorithm
  - `ub`: elimination algorithm

### References

  - Bodlaender, Hans L., et al. "Pre-processing for triangulation of probabilistic networks." (2001).
  - Bodlaender, Hans L., Arie M.C.A. Koster, and Frank van den Eijkhof. "Preprocessing rules for triangulation of probabilistic networks." *Computational Intelligence* 21.3 (2005): 286-305.
  - van den Eijkhof, Frank, Hans L. Bodlaender, and Arie M.C.A. Koster. "Safe reduction rules for weighted treewidth." *Algorithmica* 47 (2007): 139-158. 
"""
struct SafeRules{L <: WidthOrAlgorithm, U <: EliminationAlgorithm} <: EliminationAlgorithm
    lb::L
    ub::U
end

function SafeRules(alg::PermutationOrAlgorithm)
    return SafeRules(DEFAULT_LOWER_BOUND_ALGORITHM, alg)
end

function SafeRules()
    return SafeRules(DEFAULT_ELIMINATION_ALGORITHM)
end

# deprecated
const RuleReduction{U} = SafeRules{MMW, U}
RuleReduction(alg) = SafeRules(MMW(), alg)

"""
    SafeSeparators{M, A} <: EliminationAlgorithm

    SafeSeparators(min::PermutationOrAlgorithm, alg::EliminationAlgorithm)

    SeparatorReducton()

Apple an elimination algorithm to the atoms of an almost-clique separator decomposition. The algorithm
`min` is used to compute the decomposition.

!!! warning
    The algorithm `min` must compute a *minimimal* ordering. I recommend wrapping a heuristic algorithm
    with `MinimalChordal`. 

### Parameters

  - `min`: minimal elimination algorithm
  - `alg`: elimination algorithm

### References

  - Bodlaender, Hans L., and Arie MCA Koster. "Safe separators for treewidth." *Discrete Mathematics* 306.3 (2006): 337-350.
  - Tamaki, Hisao. "A heuristic for listing almost-clique minimal separators of a graph." arXiv preprint arXiv:2108.07551 (2021).
"""
struct SafeSeparators{M <: PermutationOrAlgorithm, A <: EliminationAlgorithm} <: EliminationAlgorithm
    min::M
    alg::A
end

function SafeSeparators(min::PermutationOrAlgorithm)
    return SafeSeparators(min, DEFAULT_ELIMINATION_ALGORITHM)
end

function SafeSeparators()
    return SafeSeparators(MinimalChordal())
end

"""
    ConnectedComponents{A} <: EliminationAlgorithm

    ConnectedComponents(alg::PermutationOrAlgorithm)

    ConnectedComponents()

Apply an elimination algorithm to each connected component of a graph.

### Parameters

  - `alg`: elimination algorithm

"""
struct ConnectedComponents{A <: PermutationOrAlgorithm} <: EliminationAlgorithm
    alg::A
end

function ConnectedComponents()
    return ConnectedComponents(DEFAULT_ELIMINATION_ALGORITHM)
end

# deprecated
const ComponentReduction = ConnectedComponents

"""
    permutation([weights, ]graph;
        alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM)

Construct a fill-reducing permutation of the vertices of a simple graph.

```jldoctest
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

julia> order, index = permutation(graph);

julia> order
8-element Vector{Int64}:
 4
 1
 2
 8
 5
 3
 6
 7

julia> index == invperm(order)
true
```
"""
function permutation(graph; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM)
    return permutation(graph, alg)
end

function permutation(weights::AbstractVector, graph; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM)
    return permutation(weights, graph, alg)
end

# method ambiguity
function permutation(weights::AbstractVector, alg::EliminationAlgorithm)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::AbstractVector)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::BFS)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::MCS)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::LexBFS)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::RCMMD)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::RCMGL)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::LexM)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::MCSM)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::AMF)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::MF)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::MMD)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::AMD)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::SymAMD)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::METIS)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::Spectral)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::BT)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::SAT)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::MinimalChordal)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::CompositeRotations)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::SafeRules)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::SafeSeparators)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::ConnectedComponents)
    error()
end

function permutation(graph, alg::EliminationAlgorithm)
    throw(
        ArgumentError(
            "Algorithm $alg not implemented. You may need to load an additional package."
        ),
    )
end

function permutation(weights::AbstractVector, graph, alg::PermutationOrAlgorithm)
    return permutation(graph, alg)
end

function permutation(graph, alg::AbstractVector)
    return permutation(BipartiteGraph(graph), alg)
end

function permutation(graph::AbstractGraph{V}, alg::AbstractVector) where {V}
    order = Vector{V}(alg)
    return order, invperm(order)
end

function permutation(graph, alg::BFS)
    order = bfs(graph)
    return order, invperm(order)
end

function permutation(graph, alg::MCS)
    index, size = mcs(graph)
    return invperm(index), index
end

function permutation(graph, alg::LexBFS)
    index = lexbfs(graph)
    return invperm(index), index
end

function permutation(graph, alg::RCMMD)
    order = rcmmd(graph, alg.alg)
    return order, invperm(order)
end

function permutation(graph, alg::RCMGL)
    order = rcmgl(graph, alg.alg)
    return order, invperm(order)
end

function permutation(graph, alg::LexM)
    index = lexm(graph)
    return invperm(index), index
end

function permutation(graph, alg::MCSM)
    index = mcsm(graph)
    return invperm(index), index
end

function permutation(graph, alg::AMF)
    return amf(graph; speed = alg.speed)
end

function permutation(graph, alg::MF)
    order = mf(graph)
    return order, invperm(order)
end

function permutation(graph, alg::MMD)
    index = mmd(graph; delta = alg.delta)
    return invperm(index), index
end

function permutation(graph, alg::SAT{H}) where {H}
    order, index = permutation(graph, alg.ub)
    lb = lowerbound(graph, alg.lb)
    ub = treewidth(graph, order)

    if lb < ub
        order, width = sat(graph, H, lb, ub)
        index = invperm(order)
    end

    return order, index
end

function permutation(graph, alg::MinimalChordal)
    return minimalchordal(graph, permutation(graph, alg.alg)...)
end

function permutation(weights::AbstractVector, graph, alg::MinimalChordal)
    return minimalchordal(graph, permutation(weights, graph, alg = alg.alg)...)
end

function permutation(graph, alg::CompositeRotations)
    order, index = permutation(graph, alg.alg)
    upper = sympermute(graph, index, Forward)

    invpermute!(
        order, compositerotations(reverse(upper), etree(upper), view(index, alg.clique))
    )

    return order, invperm(order)
end

function permutation(weights::AbstractVector, graph, alg::CompositeRotations)
    order, index = permutation(weights, graph, alg.alg)
    upper = sympermute(graph, index, Forward)

    invpermute!(
        order, compositerotations(reverse(upper), etree(upper), view(index, alg.clique))
    )

    return order, invperm(order)
end

function permutation(graph, alg::SafeRules)
    kernel, stack, label, width = pr4(graph, lowerbound(graph, alg.lb))
    order, index = permutation(kernel, alg.ub)
    append!(stack, view(label, order))
    return stack, invperm(stack)
end

function permutation(weights::AbstractVector, graph, alg::SafeRules)
    kernel, stack, label, width = pr4(weights, graph, lowerbound(weights, graph, alg.lb))
    order, index = permutation(view(weights, label), kernel, alg.ub)
    append!(stack, view(label, order))
    return stack, invperm(stack)
end

function permutation(graph, alg::SafeSeparators)
    # construct almost-clique separator decomposition
    graph, label, tree = almosttree(graph, alg.min)

    # permute graph
    graph = Graph(sympermute(graph, invperm(label), Reverse))

    # compute ordering
    order = cliquedissect(graph, tree, alg.alg)

    for v in vertices(graph)
        order[v] = label[order[v]]
    end

    return order, invperm(order)
end

function permutation(weights::AbstractVector, graph, alg::SafeSeparators)
    # construct almost-clique separator decomposition
    graph, label, tree = almosttree(graph, alg.min)

    # permute graph
    weights = weights[label]
    graph = Graph(sympermute(graph, invperm(label), Reverse))

    # compute ordering
    order = cliquedissect(weights, graph, tree, alg.alg)

    for v in vertices(graph)
        order[v] = label[order[v]]
    end

    return order, invperm(order)
end

# TODO: multi-threading
function permutation(graph, alg::ConnectedComponents)
    components, subgraphs = connectedcomponents(graph)
    order = eltype(eltype(components))[]

    @inbounds for (component, subgraph) in zip(components, subgraphs)
        suborder, subindex = permutation(subgraph, alg.alg)
        append!(order, view(component, suborder))
    end

    return order, invperm(order)
end

function permutation(weights::AbstractVector, graph, alg::ConnectedComponents)
    components, subgraphs = connectedcomponents(graph)
    order = eltype(eltype(components))[]

    @inbounds for (component, subgraph) in zip(components, subgraphs)
        suborder, subindex = permutation(weights[component], subgraph, alg.alg)
        append!(order, view(component, suborder))
    end

    return order, invperm(order)
end

function bfs(graph)
    return bfs(BipartiteGraph(graph))
end

function bfs(graph::AbstractGraph{V}) where {V}
    tag = 0
    level = zeros(Int, nv(graph))
    order = Vector{V}(undef, nv(graph))

    # initialize queue
    queue = @view order[one(V):nv(graph)]

    @inbounds for root in vertices(graph)
        if iszero(level[root])
            _, queue, tag = bfs!(level, queue, graph, root, tag + 1)
        end
    end

    # reverse ordering
    return reverse!(order)
end

# Algorithmic Aspects of Vertex Elimination on Graphs
# Rose, Tarjan, and Lueker
# BFS
#
# Perform a breadth-first search of a simple graph.
# The complexity is O(m), where m = |E|.
function bfs!(
        level::AbstractVector{Int},
        queue::AbstractVector{V},
        graph::AbstractGraph{V},
        root::V,
        tag::Int,
    ) where {V}
    i = j = one(V)
    level[root] = tag
    queue[j] = root

    @inbounds while i <= j
        v = queue[i]
        l = level[v]

        for w in neighbors(graph, v)
            if level[w] < tag
                j += one(V)
                queue[j] = w
                level[w] = l + 1
            end
        end

        i += one(V)
    end

    n = convert(V, length(queue))
    @views return queue[one(V):j], queue[i:n], level[queue[j]]
end

"""
    mcs(graph[, clique])

Perform a maximum cardinality search, optionally specifying a clique to be ordered last.
Returns the inverse permutation.
"""
function mcs(graph)
    return mcs(BipartiteGraph(graph))
end

function mcs(graph, clique)
    return mcs(BipartiteGraph(graph), clique)
end

# Simple Linear-Time Algorithms to Test Chordality of BipartiteGraphs, Test Acyclicity of Hypergraphs, and Selectively Reduce Acyclic Hypergraphs
# Tarjan and Yannakakis
# Maximum Cardinality Search
#
# Construct a fill-reducing permutation of a graph.
# The complexity is O(m + n), where m = |E| and n = |V|.
function mcs(graph::AbstractGraph{V}, clique = oneto(zero(V))) where {V}
    j = one(V); n = nv(graph)
    size = ones(V, n)
    alpha = Vector{V}(undef, n)

    # construct bucket queue data structure
    head = zeros(V, n + one(V))
    prev = Vector{V}(undef, n)
    next = Vector{V}(undef, n)

    function set(i)
        return @inbounds DoublyLinkedList(view(head, i), prev, next)
    end

    @inbounds prepend!(set(j), vertices(graph))

    # run algorithm
    @inbounds for v in reverse(clique)
        delete!(set(j), v)
        alpha[v] = n
        size[v] = one(V) - j

        for w in neighbors(graph, v)
            if ispositive(size[w])
                delete!(set(size[w]), w)
                size[w] += one(V)
                pushfirst!(set(size[w]), w)
            end
        end

        j += one(V)

        while ispositive(j) && isempty(set(j))
            j -= one(V)
        end

        n -= one(V)
    end

    @inbounds while ispositive(n)
        v = popfirst!(set(j))
        alpha[v] = n
        size[v] = one(V) - j

        for w in neighbors(graph, v)
            if ispositive(size[w])
                delete!(set(size[w]), w)
                size[w] += one(V)
                pushfirst!(set(size[w]), w)
            end
        end

        j += one(V)

        while ispositive(j) && isempty(set(j))
            j -= one(V)
        end

        n -= one(V)
    end

    return alpha, size
end

function rcmmd(graph, alg::SortingAlgorithm = DEFAULT_UNSTABLE)
    return genrcm(graph, alg) do level, queue, graph, root, tag
        component, _, tag = bfs!(level, queue, graph, root, tag)

        root = argmin(component) do v
            eltypedegree(graph, v)
        end

        return root, tag
    end
end

function rcmgl(graph, alg::SortingAlgorithm = DEFAULT_UNSTABLE)
    return genrcm(fnroot!, graph, alg)
end

function genrcm(f::Function, graph, alg::SortingAlgorithm)
    return genrcm(f, BipartiteGraph(graph), alg)
end

function genrcm(
        f::Function, graph::Union{BipartiteGraph, AbstractSimpleGraph}, alg::SortingAlgorithm
    )
    return genrcm!(f, copy(graph), alg)
end

# Algorithms for Sparse Linear Systems
# Scott and Tuma
# Algorithm 8.3: CM and RCM algorithms for band and profile reduction
#
# Apply the reverse Cuthill-Mckee algorithm to each connected component of a graph.
# The complexity is O(m + n), where m = |E| and n = |V|.
function genrcm!(f::Function, graph::AbstractGraph{V}, alg::SortingAlgorithm) where {V}
    tag = 0
    level = zeros(Int, nv(graph))
    order = Vector{V}(undef, nv(graph))

    # sort neighbors
    scratch = Vector{V}(undef, Δout(graph))

    @inbounds for v in vertices(graph)
        sort!(neighbors(graph, v); alg, scratch, by = u -> eltypedegree(graph, u))
    end

    # initialize queue
    queue = @view order[one(V):nv(graph)]

    @inbounds for root in vertices(graph)
        if iszero(level[root])
            # find pseudo-peripheral vertex
            root, tag = f(level, queue, graph, root, tag + 1)

            # compute Cuthill-Mckee ordering
            _, queue, tag = bfs!(level, queue, graph, root, tag + 1)
        end
    end

    # reverse ordering
    return reverse!(order)
end

# Computer Solution of Sparse Linear Systems
# George, Liu, and Ng
# FNROOT (FiNd ROOT)
#
# Find a pseudo-peripheral vertex.
function fnroot!(
        level::AbstractVector{Int},
        queue::AbstractVector{V},
        graph::AbstractGraph{V},
        root::V,
        tag::Int,
    ) where {V}
    component, _, new = bfs!(level, queue, graph, root, tag)
    candidate = zero(V)

    @inbounds while root != candidate
        candidate = last(component)
        degree = eltypedegree(graph, candidate)
        eccentricity = level[candidate] - tag

        for v in Iterators.reverse(component)
            eccentricity + tag == level[v] || break

            d = eltypedegree(graph, v)

            if d < degree
                candidate, degree = v, d
            end
        end

        tag = new
        component, _, new = bfs!(level, queue, graph, candidate, tag + 1)

        if level[component[end]] <= eccentricity + tag
            root = candidate
        end
    end

    return root, new
end

function lexbfs(graph)
    return lexbfs(BipartiteGraph(graph))
end

# Algorithmic Aspects of Vertex Elimination on Graphs
# Rose, Tarjan, and Lueker
# LEX P
#
# Perform a lexicographic breadth-first search of a simple graph.
function lexbfs(graph::AbstractGraph{V}) where {V}
    I = promote_type(V, etype(graph))
    n = convert(I, max(ne(graph) * (2 - is_directed(graph)), nv(graph)) + 2)

    flag = Vector{I}(undef, n)
    head = Vector{I}(undef, n)
    next = Vector{I}(undef, n)
    back = Vector{I}(undef, n)

    alpha = Vector{V}(undef, nv(graph))
    cell = Vector{V}(undef, nv(graph))

    fixlist = Vector{I}(undef, Δout(graph))

    # (implicitly) assign label ∅ to all vertices
    head[one(I)] = two(I)
    back[two(I)] = one(I)
    head[two(I)] = back[one(I)] = next[one(I)] = flag[one(I)] = flag[two(I)] = zero(I)
    c = three(I)

    # c is the number of the first empty cell
    @inbounds for v in vertices(graph)
        head[c] = v
        cell[v] = next[c - one(I)] = c
        flag[c] = two(I)
        back[c] = c - one(I)
        c += one(I)
        alpha[v] = zero(V)
    end

    next[c - one(I)] = zero(I)

    @inbounds for i in reverse(oneto(nv(graph)))
        # skip empty sets
        while iszero(next[head[one(I)]])
            head[one(I)] = head[head[one(I)]]
            back[head[one(I)]] = one(I)
        end

        ##########
        # select #
        ##########

        # pick next vertex to number
        p = next[head[one(I)]]

        # delete cell of vertex from set
        next[head[one(I)]] = next[p]

        if !iszero(next[head[one(I)]])
            back[next[head[one(I)]]] = head[one(I)]
        end

        v = head[p]

        # assign v the number i
        alpha[v] = i
        f = zero(V)

        ###########
        # update2 #
        ###########

        for w in neighbors(graph, v)
            if iszero(alpha[w])
                # delete cell of w from set
                next[back[cell[w]]] = next[cell[w]]

                if !iszero(next[cell[w]])
                    back[next[cell[w]]] = back[cell[w]]
                end

                h = back[flag[cell[w]]]

                # if h is an old set then create a new set
                if iszero(flag[h])
                    head[c] = head[h]
                    head[h] = c
                    back[head[c]] = c
                    back[c] = h
                    flag[c] = one(I)
                    next[c] = zero(I)
                    f += one(V)
                    fixlist[f] = c
                    h = c
                    c += one(I)
                end

                # add cell of w to new set
                next[cell[w]] = next[h]

                if !iszero(next[h])
                    back[next[h]] = cell[w]
                end

                flag[cell[w]] = back[cell[w]] = h
                next[h] = cell[w]
            end
        end

        @views flag[fixlist[one(V):f]] .= zero(I)
    end

    return alpha
end

function lexm(graph)
    return lexm(BipartiteGraph(graph))
end

# Algorithmic Aspects of Vertex Elimination on Graphs
# Rose, Tarjan, and Lueker
# LEX M
#
# Perform a lexicographic breadth-first search of a simple graph.
# Returns a minimal ordering.
# The complexity is O(mn), where m = |E| and n = |V|.
function lexm(graph::AbstractGraph{V}) where {V}
    k = one(V); n = nv(graph)
    alpha = fill(-n - one(V), n)
    label = fill(twice(k), n)

    # construct stack data structure
    stack = Vector{V}(undef, n)

    # construct bucket queue data structures
    shead = zeros(V, twice(n) + two(V))
    sprev = Vector{V}(undef, n)
    snext = Vector{V}(undef, n)

    function set(j)
        @inbounds head = @view shead[j]
        return DoublyLinkedList(head, sprev, snext)
    end

    rhead = zeros(V, n)
    rprev = Vector{V}(undef, n)
    rnext = Vector{V}(undef, n)

    function reach(j)
        @inbounds head = @view rhead[j]
        return DoublyLinkedList(head, rprev, rnext)
    end

    @inbounds prepend!(set(twice(k)), vertices(graph))

    ########
    # loop #
    ########

    @inbounds while ispositive(n)
        ##########
        # select #
        ##########

        # assign v the number i
        v = popfirst!(set(twice(k)))
        alpha[v] = n

        for w in neighbors(graph, v)
            if isnegative(alpha[w])
                alpha[w] = -n
                pushfirst!(reach(half(label[w])), w)
                delete!(set(label[w]), w)
                label[w] += one(V)
                pushfirst!(set(label[w]), w)
            end
        end

        ##########
        # search #
        ##########

        for j in oneto(k)
            while !isempty(reach(j))
                w = popfirst!(reach(j))

                for z in neighbors(graph, w)
                    if alpha[z] < -n
                        alpha[z] = -n

                        if label[z] > twice(j)
                            pushfirst!(reach(half(label[z])), z)
                            delete!(set(label[z]), z)
                            label[z] += one(V)
                            pushfirst!(set(label[z]), z)
                        else
                            pushfirst!(reach(j), z)
                        end
                    end
                end
            end
        end

        ########
        # sort #
        ########

        m = twice(k) + one(V)
        k = zero(V)

        for j in oneto(m)
            w = shead[j]

            if ispositive(w)
                k += one(V)

                for v in set(j)
                    label[v] = twice(k)
                end

                stack[k] = w
                shead[j] = zero(V)
            end
        end

        for j in oneto(k)
            shead[twice(j)] = stack[j]
        end

        n -= one(V)
    end

    return alpha
end

function mcsm(graph)
    return mcsm(BipartiteGraph(graph))
end

function mcsm(graph, clique)
    return mcsm(BipartiteGraph(graph), clique)
end

# Maximum Cardinality Search for Computing Minimal Triangulations
# Berry, Blair, and Heggernes
# MCS-M
#
# Perform a maximum cardinality search of a simple graph.
# Returns a minimal ordering.
# The complexity is O(mn), where m = |E| and n = |V|.
function mcsm(graph::AbstractGraph{V}, clique = oneto(zero(V))) where {V}
    k = one(V); n = nv(graph)
    alpha = fill(-n - one(V), n)
    size = ones(V, n)

    # construct bucket queue data structures
    shead = zeros(V, n + one(V))
    sprev = Vector{V}(undef, n)
    snext = Vector{V}(undef, n)

    function set(j)
        @inbounds head = @view shead[j]
        return DoublyLinkedList(head, sprev, snext)
    end

    rhead = zeros(V, n)
    rprev = Vector{V}(undef, n)
    rnext = Vector{V}(undef, n)

    function reach(j)
        @inbounds head = @view rhead[j]
        return DoublyLinkedList(head, rprev, rnext)
    end

    @inbounds prepend!(set(k), vertices(graph))

    # run algorithm
    @inbounds for v in reverse(clique)
        delete!(set(k), v)
        alpha[v] = n

        for w in neighbors(graph, v)
            if isnegative(alpha[w])
                alpha[w] = -n
                pushfirst!(reach(size[w]), w)
                delete!(set(size[w]), w)
                size[w] += one(V)
                pushfirst!(set(size[w]), w)
            end
        end

        for j in oneto(k)
            while !isempty(reach(j))
                w = popfirst!(reach(j))

                for z in neighbors(graph, w)
                    if alpha[z] < -n
                        alpha[z] = -n

                        if size[z] > j
                            pushfirst!(reach(size[z]), z)
                            delete!(set(size[z]), z)
                            size[z] += one(V)
                            pushfirst!(set(size[z]), z)
                        else
                            pushfirst!(reach(j), z)
                        end
                    end
                end
            end
        end

        k += one(V)

        while ispositive(k) && isempty(set(k))
            k -= one(V)
        end

        n -= one(V)
    end

    @inbounds while ispositive(n)
        v = popfirst!(set(k))
        alpha[v] = n

        for w in neighbors(graph, v)
            if isnegative(alpha[w])
                alpha[w] = -n
                pushfirst!(reach(size[w]), w)
                delete!(set(size[w]), w)
                size[w] += one(V)
                pushfirst!(set(size[w]), w)
            end
        end

        for j in oneto(k)
            while !isempty(reach(j))
                w = popfirst!(reach(j))

                for z in neighbors(graph, w)
                    if alpha[z] < -n
                        alpha[z] = -n

                        if size[z] > j
                            pushfirst!(reach(size[z]), z)
                            delete!(set(size[z]), z)
                            size[z] += one(V)
                            pushfirst!(set(size[z]), z)
                        else
                            pushfirst!(reach(j), z)
                        end
                    end
                end
            end
        end

        k += one(V)

        while ispositive(k) && isempty(set(k))
            k -= one(V)
        end

        n -= one(V)
    end

    return alpha
end

"""
    mf(graph)

Compute the greedy minimum-fill ordering of a simple graph.
Returns the permutation and its inverse.
"""
function mf(graph, issorted::Val = Val(false))
    return mf(BipartiteGraph(graph), issorted)
end

function mf(graph::SparseMatrixCSC)
    return mf(BipartiteGraph(graph), Val(true))
end

function mf(graph::AbstractSimpleGraph)
    return mf(graph, Val(true))
end

function mf(graph::AbstractGraph{V}, ::Val{false}) where {V}
    degrees = Vector{V}(undef, nv(graph))
    scratch = Vector{V}(undef, Δout(graph))
    lists = Vector{Vector{V}}(undef, nv(graph))

    for v in vertices(graph)
        i = zero(V)
        list = Vector{V}(undef, eltypedegree(graph, v))

        for w in outneighbors(graph, v)
            if v != w
                i += one(V)
                list[i] = w
            end
        end

        degrees[v] = i
        lists[v] = sort!(resize!(list, i); scratch)
    end

    return mf!(nv(graph), degrees, lists)
end

function mf(graph::AbstractGraph{V}, ::Val{true}) where {V}
    degrees = Vector{V}(undef, nv(graph))
    lists = Vector{Vector{V}}(undef, nv(graph))

    for v in vertices(graph)
        i = zero(V)
        list = Vector{V}(undef, eltypedegree(graph, v))

        for w in outneighbors(graph, v)
            if v != w
                i += one(V)
                list[i] = w
            end
        end

        degrees[v] = i
        lists[v] = resize!(list, i)
    end

    return mf!(nv(graph), degrees, lists)
end

function mf!(
        nv::V, degrees::AbstractVector{V}, lists::AbstractVector{<:AbstractVector{V}}
    ) where {V}
    order = Vector{V}(undef, nv)
    label = zeros(Int, nv); tag = 0

    # construct stack data structure
    snum = zero(V) # size of stack
    stack = Vector{V}(undef, nv)

    # construct min-heap data structure
    heap = Heap{V, V}(nv)

    @inbounds for v in oneto(nv)
        count = zero(V)
        degree = degrees[v]
        list = lists[v]

        for j in oneto(degree)
            w = list[j]
            label[lists[w]] .= tag += 1

            for jj in (j + 1):degree
                ww = list[jj]

                if label[ww] != tag
                    count += one(V)
                end
            end
        end

        push!(heap, v => count)
    end

    hfall!(heap)

    # run algorithm
    i = one(V)

    @inbounds while i <= nv
        # select vertex from heap
        v = order[i] = argmin(heap)
        list = lists[v]
        degree = degrees[v]
        label[list] .= label[v] = tag += 1

        # append distinguishable neighbors to the stack
        snum = zero(V)
        ii = i + one(V)

        for w in list
            flag = false

            if degrees[w] == degree
                flag = true

                for x in lists[w]
                    if label[x] < tag
                        flag = false
                        break
                    end
                end
            end

            if flag
                order[ii] = w
                label[w] += 1
                ii += one(V)
            else
                snum += one(V)
                stack[snum] = w
            end
        end

        # remove vertex from graph
        for w in take(stack, snum)
            list = lists[w]
            degrees[w] -= one(V)
            deleteat!(list, searchsortedfirst(list, v))
        end

        # remove indistinguishable neighbors from graph
        if ii > i + one(V)
            tag += 1

            for w in take(stack, snum)
                list = lists[w]
                count = zero(V)

                for j in oneto(degrees[w])
                    x = list[j]

                    if label[x] == tag
                        count += one(V)
                    else
                        list[j - count] = x
                    end
                end

                resize!(list, degrees[w] -= count)
            end
        end

        # remove vertex and indistinguishable neighbors from heap
        for j in i:(ii - one(V))
            delete!(heap, order[j])
        end

        # update deficiencies
        if heap[order[i]] > zero(V)
            for j in oneto(snum)
                w = stack[j]
                label[lists[w]] .= tag += 1

                for jj in (j + one(V)):snum
                    ww = stack[jj]

                    if label[ww] != tag
                        count = zero(V)

                        for xx in lists[ww]
                            if label[xx] == tag
                                heap[xx] -= one(V)
                                hrise!(heap, xx)
                                count += one(V)
                            end
                        end

                        heap[w] += degrees[w] - count
                        heap[ww] += degrees[ww] - count
                        insert!(lists[w], searchsortedfirst(lists[w], ww), ww)
                        insert!(lists[ww], searchsortedfirst(lists[ww], w), w)
                        degrees[w] += one(V)
                        degrees[ww] += one(V)
                        label[ww] = tag
                    end
                end
            end
        end

        # update heap
        for w in take(stack, snum)
            heap[w] -= (ii - i) * (degrees[w] - snum + one(V))
            hrise!(heap, w)
            hfall!(heap, w)
        end

        i = ii
    end

    return order
end

function AMFLib.amf(graph; kwargs...)
    return amf(BipartiteGraph(graph); kwargs...)
end

function AMFLib.amf(graph::BipartiteGraph; kwargs...)
    return amf(pointers(graph), targets(graph); kwargs...)
end

function MMDLib.mmd(graph; kwargs...)
    return mmd(BipartiteGraph(graph); kwargs...)
end

function MMDLib.mmd(graph::BipartiteGraph; kwargs...)
    return mmd(pointers(graph), targets(graph); kwargs...)
end

function sat(graph, H::Module, lowerbound::Integer, upperbound::Integer)
    return sat(BipartiteGraph(graph), H, lowerbound, upperbound)
end

# Encoding Treewidth into SAT
# Samer and Veith
#
# SAT-Based Approaches to Treewidth Computation: An Evaluation
# Berg and Järvisalo
#
# Jdrasil: A Modular Library for Computing Tree Decompositions
# Bannach, Berndt, and Ehlers
function sat(graph::AbstractGraph{V}, H::Module, lowerbound::Integer, upperbound::Integer) where {V}
    @argcheck !isnegative(lowerbound) && lowerbound <= upperbound <= nv(graph)

    # compute a maximal clique
    clique = maximalclique(graph, H)

    # compute true twins
    truesets, truelist = twins(graph, Val(true))

    # compute false twins
    falsesets, falselist = twins(graph, Val(false))

    # run solver
    order, width = open(Solver{H}) do solver
        n = Int32(nv(graph))
        ord = Matrix{Int32}(undef, n, n)
        arc = Matrix{Int32}(undef, n, n)

        # define ord and arc variables
        for i in oneto(n), j in oneto(n)
            if i < j
                resize!(solver, length(solver) + one(Int32))
                ord[i, j] = length(solver)
            elseif i > j
                ord[i, j] = -ord[j, i]
            else
                ord[i, j] = zero(Int32)
            end

            resize!(solver, length(solver) + one(Int32))
            arc[i, j] = length(solver)
        end

        # base encoding
        for i in oneto(n), j in oneto(n), k in oneto(n)
            if i != j && j != k && k != i
                # ord(i, j) ∧ ord(j, k) → ord(i, k)
                clause!(solver, -ord[i, j], -ord[j, k], ord[i, k])

                # arc(k, i) ∧ arc(k, j) → arc(i, j) ∨ arc(j, i)
                clause!(solver, -arc[k, i], -arc[k, j], arc[i, j], arc[j, i])
            end
        end

        for i in oneto(n), j in neighbors(graph, i)
            if i < j
                # arc(i, j) ∨ arc(j, i)
                clause!(solver, arc[i, j], arc[j, i])
            end
        end

        for i in oneto(n)
            # ¬arc(i, i)
            clause!(solver, -arc[i, i])

            for j in oneto(n)
                if i != j
                    # ord(i, j) → ¬arc(j, i)
                    clause!(solver, -ord[i, j], -arc[j, i])

                    # arc(i, j) → ¬arc(j, i)
                    clause!(solver, -arc[i, j], -arc[j, i])
                end
            end
        end

        # encode maximal clique
        label = zeros(Int, n); tag = 1

        for j in clique
            label[j] = tag

            for i in oneto(n)
                if label[i] < tag
                    # ord(i, j)
                    clause!(solver, ord[i, j])
                end
            end
        end

        # encode twins
        for (sets, list) in ((truesets, truelist), (falsesets, falselist))
            for p in list
                set = sets(p); tag += 1

                for k in set
                    if iszero(label[k])
                        label[k] = tag
                    else
                        delete!(set, k)
                    end
                end

                for k in set
                    for j in set
                        j == k && break

                        # ord(j, k)
                        clause!(solver, ord[j, k])

                        for i in oneto(n)
                            if label[i] < tag
                                # ord(i, k) → ord(i, j)
                                clause!(solver, -ord[i, k], ord[i, j])
                            end
                        end
                    end
                end
            end
        end

        # encode sorting network
        for i in oneto(n)
            row = @view arc[i, :]
            sortingnetwork!(solver, row)
        end

        # initialize cache
        cache = Matrix{Bool}(undef, n, n)

        # initialize assumption
        count = upperbound

        for i in oneto(n)
            # Σ { arc(i, j) : j } <= count
            solver[arc[i, n - count]] = -one(Int32)
        end

        state = solve!(solver)

        if state != :sat
            error("no solutions found")
        end

        # decrement count until unsatisfiable
        while state == :sat && lowerbound <= count
            # update cache
            for i in oneto(n), j in oneto(n)
                cache[i, j] = i < j ? !isnegative(solver[ord[i, j]]) :
                    i > j ? !cache[j, i] :
                    false
            end

            # update assumption
            count -= one(Int32)

            for i in oneto(n)
                # Σ { arc(i, j) : j } <= count
                solver[arc[i, n - count]] = -one(Int32)
            end

            state = solve!(solver)
        end

        if state != :sat
            count += one(Int32)
        end

        return sort(vertices(graph); lt = (i, j) -> cache[i, j]), count
    end

    return order, width
end

# compute a maximal clique
function maximalclique(graph::AbstractGraph, H::Module)
    clique = open(Solver{H}, nv(graph)) do solver
        n = length(solver)
        variables = collect(oneto(n))
        label = zeros(Int32, n); tag = zero(Int32)

        # base encoding
        for j in oneto(n)
            label[j] = tag += one(Int32)

            for i in neighbors(graph, j)
                label[i] = tag
            end

            for i in oneto(j)
                if label[i] < tag
                    # ¬i ∨ ¬j
                    clause!(solver, -i, -j)
                end
            end
        end

        # encode sorting network
        sortingnetwork!(solver, variables)

        # initialize stack
        num = zero(Int32)
        stack = Vector{Int32}(undef, n)

        # initialize assumption
        count = zero(Int32)

        # Σ { variables(i) : i } > count
        solver[variables[n - count]] = one(Int32)
        state = solve!(solver)

        if state != :sat
            error("no solutions found")
        end

        # increment count until unsatisfiable
        while state == :sat && count < n - one(Int32)
            # update stack
            num = zero(Int32)

            for i in oneto(n)
                if !isnegative(solver[i])
                    num += one(Int32)
                    stack[num] = i
                end
            end

            # update assumption
            count += one(Int32)

            # Σ { variables(i) : i } > count
            solver[variables[n - count]] = one(Int32)
            state = solve!(solver)
        end

        return resize!(stack, num)
    end

    return clique
end

# partition a graph into (false) twins
function twins(graph::AbstractGraph{V}, ::Val{T}) where {V, T}
    n = nv(graph)

    # bucket queue data structure
    head = zeros(V, n)
    prev = Vector{V}(undef, n)
    next = Vector{V}(undef, n)

    function set(i)
        h = @view head[i]
        return DoublyLinkedList(h, prev, next)
    end

    # linked list data structures
    list = DoublyLinkedList{V}(n)

    # stack data structures
    pnum = wnum = zero(V)
    pstack = Vector{V}(undef, n)
    wstack = Vector{V}(undef, n)

    tag = zero(V)
    label = zeros(V, n)

    # run algorithm
    for i in oneto(n)
        pnum += one(V)
        pstack[pnum] = i
    end

    i = pstack[pnum]
    pnum -= one(V)
    pushfirst!(list, i)
    prepend!(set(i), vertices(graph))

    for v in vertices(graph)
        tag += one(V)

        if T
            label[v] = tag
        end

        for w in neighbors(graph, v)
            label[w] = tag
        end

        # refine partition
        wnum = zero(V)

        for i in list
            wnum += one(V)
            wstack[wnum] = i
        end

        while ispositive(wnum)
            i = wstack[wnum]
            wnum -= one(V)
            incount = outcount = zero(V)

            for v in set(i)
                if label[v] == tag
                    incount += one(V)
                else
                    outcount += one(V)
                end
            end

            if ispositive(incount) && ispositive(outcount)
                j = pstack[pnum]

                for v in set(i)
                    if (label[v] == tag) == (incount <= outcount)
                        delete!(set(i), v)
                        pushfirst!(set(j), v)
                    end
                end

                if !isempty(set(j))
                    pnum -= one(V)
                    wnum += one(V)
                    wstack[wnum] = j
                    pushfirst!(list, j)
                end

                if isempty(set(i))
                    pnum += one(V)
                    pstack[pnum] = i
                    delete!(list, i)
                end
            end
        end
    end

    return set, list
end

function minimalchordal(graph, order::AbstractVector, index::AbstractVector = invperm(order))
    return minimalchordal(BipartiteGraph(graph), order, index)
end

# A Practical Algorithm for Making Filled Graphs Minimal
# Barry, Heggernes, and Telle
# MinimalChordal
function minimalchordal(graph::AbstractGraph{V}, order::AbstractVector, index::AbstractVector) where {V}
    M = Graph(graph)
    F = Vector{Vector{Tuple{V, V}}}(undef, nv(graph))

    for (i, v) in enumerate(order)
        F[i] = Tuple{V, V}[]
        list = neighbors(M, v)
        degree = eltypedegree(M, v)

        for j in oneto(degree)
            w = list[j]

            if i < index[w]
                for jj in (j + one(V)):degree
                    ww = list[jj]

                    if i < index[ww] && !has_edge(M, w, ww)
                        add_edge!(M, w, ww)
                        push!(F[i], (w, ww))
                    end
                end
            end
        end
    end

    Candidate = Set{Tuple{V, V}}()
    Incident = V[]

    for i in reverse(oneto(nv(graph)))
        for (u, v) in F[i]
            flag = true

            for x in neighbors(M, u)
                if index[x] > i && has_edge(M, x, v) && !has_edge(M, x, order[i])
                    flag = false
                    break
                end
            end

            if flag
                push!(Candidate, (u, v))
                push!(Incident, u, v)
            end
        end

        if !isempty(Candidate)
            unique!(sort!(Incident))

            n = length(Incident)
            W = Graph{V}(n)

            for j in oneto(n)
                v = Incident[j]

                for jj in (j + 1):n
                    vv = Incident[jj]

                    if (v, vv) ∉ Candidate
                        add_edge!(W, j, jj)
                    end
                end
            end

            worder, windex = permutation(W; alg = MCSM())

            for (j, v) in enumerate(worder)
                list = neighbors(W, v)
                degree = eltypedegree(W, v)

                for k in oneto(degree)
                    w = list[k]

                    if j < windex[w]
                        for kk in (k + 1):degree
                            ww = list[kk]

                            if j < windex[ww] && !has_edge(W, w, ww)
                                add_edge!(W, w, ww)
                                delete!(Candidate, (Incident[w], Incident[ww]))
                            end
                        end
                    end
                end
            end

            for (u, v) in Candidate
                rem_edge!(M, u, v)
            end

            empty!(Candidate)
            empty!(Incident)
        end
    end

    return permutation(M; alg = MCS())
end

# Pre-processing for Triangulation of Probabilistic Networks
# Bodlaender, Koster, Eijkhof, and van der Gaag
#
# Preprocessing Rules for Triangulation of Probabilistic Networks
# Bodlaender, Koster, Eijkhof, and van der Gaag
#
# PR-3 (Islet, Twig, Series, Triangle, Buddy, and Cube)
function pr3(graph, width::Integer)
    return pr3(BipartiteGraph(graph), width)
end

function pr3(graph::AbstractGraph{V}, width::Integer) where {V}
    kernel = Graph(graph)
    return kernel, pr3!(kernel, convert(V, width))...
end

function pr3!(graph::Graph{V}, width::V) where {V}
    n = nv(graph)

    for v in vertices(graph)
        @inbounds rem_edge!(graph, v, v)
    end

    # bucket queue
    head = zeros(V, max(n, four(V)))
    prev = Vector{V}(undef, n)
    next = Vector{V}(undef, n)

    function set(i)
        @inbounds h = @view head[i + one(i)]
        return DoublyLinkedList(h, prev, next)
    end

    for v in vertices(graph)
        @inbounds pushfirst!(set(eltypedegree(graph, v)), v)
    end

    # treewidth lower bound
    width = max(width, zero(V))

    # stack of eliminated vertices
    lo = -one(V)
    hi = zero(V)
    stack = Vector{V}(undef, n)

    # apply rules until exhaustion
    while lo < hi
        lo = hi

        # islet rule
        v = head[1]

        while ispositive(v)
            # v
            hi += one(V); stack[hi] = v
            n = next[v]; delete!(set(0), v)
            v = n
        end

        if lo == hi
            # twig rule
            v = head[2]

            while ispositive(v)
                # w
                # |
                # v
                w = only(neighbors(graph, v))
                hi += one(V); stack[hi] = v

                delete!(set(eltypedegree(graph, w)), w)

                rem_edge!(graph, v, w)

                pushfirst!(set(eltypedegree(graph, w)), w)

                n = next[v]; delete!(set(1), v)
                v = n
            end

            if lo == hi
                width = max(width, two(V))

                # series rule
                v = head[3]

                while ispositive(v)
                    # w
                    # |
                    # v  ---  ww
                    w, ww = neighbors(graph, v)
                    hi += one(V); stack[hi] = v

                    delete!(set(eltypedegree(graph, w)), w)
                    delete!(set(eltypedegree(graph, ww)), ww)

                    rem_edge!(graph, v, w)
                    rem_edge!(graph, v, ww)

                    add_edge!(graph, w, ww)

                    pushfirst!(set(eltypedegree(graph, w)), w)
                    pushfirst!(set(eltypedegree(graph, ww)), ww)

                    n = next[v]; delete!(set(2), v)
                    v = n
                end

                if lo == hi
                    width = max(width, three(V))

                    # triangle rule
                    v = head[4]

                    while ispositive(v)
                        w = ww = www = zero(V)
                        x, xx, xxx = neighbors(graph, v)

                        if has_edge(graph, x, xx)
                            w, ww, www = x, xx, xxx
                        elseif has_edge(graph, x, xxx)
                            w, ww, www = x, xxx, xx
                        elseif has_edge(graph, xx, xxx)
                            w, ww, www = xx, xxx, x
                        end

                        if ispositive(w)
                            # w  ---  ww
                            # |   /
                            # v  ---  www
                            hi += one(V); stack[hi] = v

                            delete!(set(eltypedegree(graph, w)), w)
                            delete!(set(eltypedegree(graph, ww)), ww)
                            delete!(set(eltypedegree(graph, www)), www)

                            rem_edge!(graph, v, w)
                            rem_edge!(graph, v, ww)
                            rem_edge!(graph, v, www)

                            add_edge!(graph, w, www)
                            add_edge!(graph, ww, www)

                            pushfirst!(set(eltypedegree(graph, w)), w)
                            pushfirst!(set(eltypedegree(graph, ww)), ww)
                            pushfirst!(set(eltypedegree(graph, www)), www)

                            n = next[v]; delete!(set(3), v)
                        else
                            n = next[v]
                        end

                        v = n
                    end

                    if lo == hi
                        # buddy rule
                        v = head[4]

                        while ispositive(v)
                            w, ww, www = neighbors(graph, v)
                            vv = n = next[v]

                            while ispositive(vv)
                                x, xx, xxx = neighbors(graph, vv)

                                if (w, ww, www) == (x, xx, xxx)
                                    # w  -----------  vv
                                    # |           /   |
                                    # |       ww      |
                                    # |   /           |
                                    # v  -----------  www
                                    hi += one(V); stack[hi] = vv
                                    hi += one(V); stack[hi] = v

                                    delete!(set(eltypedegree(graph, w)), w)
                                    delete!(set(eltypedegree(graph, ww)), ww)
                                    delete!(set(eltypedegree(graph, www)), www)

                                    rem_edge!(graph, v, w)
                                    rem_edge!(graph, v, ww)
                                    rem_edge!(graph, v, www)
                                    rem_edge!(graph, vv, w)
                                    rem_edge!(graph, vv, ww)
                                    rem_edge!(graph, vv, www)

                                    add_edge!(graph, w, ww)
                                    add_edge!(graph, w, www)
                                    add_edge!(graph, ww, www)

                                    pushfirst!(set(eltypedegree(graph, w)), w)
                                    pushfirst!(set(eltypedegree(graph, ww)), ww)
                                    pushfirst!(set(eltypedegree(graph, www)), www)

                                    nn = zero(V); delete!(set(3), vv)
                                    n = next[v]; delete!(set(3), v)
                                else
                                    nn = next[vv]
                                end

                                vv = nn
                            end

                            v = n
                        end

                        if lo == hi
                            # cube rule
                            v = head[4]

                            while ispositive(v)
                                vv, vvv, vvvv = neighbors(graph, v)

                                if isthree(eltypedegree(graph, vv)) && isthree(eltypedegree(graph, vvv)) && isthree(eltypedegree(graph, vvvv))
                                    w = ww = www = zero(V)
                                    x, xx, xxx = neighbors(graph, vv)
                                    y, yy, yyy = neighbors(graph, vvv)
                                    z, zz, zzz = neighbors(graph, vvvv)

                                    if v == x
                                        x, xx = xx, xxx
                                    elseif v == xx
                                        x, xx = x, xxx
                                    end

                                    if v == y
                                        y, yy = yy, yyy
                                    elseif v == yy
                                        y, yy = y, yyy
                                    end

                                    if v == z
                                        z, zz = zz, zzz
                                    elseif v == zz
                                        z, zz = z, zzz
                                    end

                                    if x == y != vvvv && yy == zz != vv && z == xx != vvv
                                        w, ww, www = x, yy, z
                                    elseif x == y != vvvv && yy == z != vv && zz == xx != vvv
                                        w, ww, www = x, yy, zz
                                    elseif x == yy != vvvv && y == z != vv && zz == xx != vvv
                                        w, ww, www = x, y, zz
                                    elseif xx == yy != vvvv && y == z != vv && zz == x != vvv
                                        w, ww, www = xx, y, zz
                                    elseif xx == y != vvvv && yy == zz != vv && z == x != vvv
                                        w, ww, www = xx, yy, z
                                    elseif xx == yy != vvvv && y == zz != vv && z == x != vvv
                                        w, ww, www = xx, y, z
                                    end

                                    if ispositive(w)
                                        #         ww
                                        #     /       \
                                        # vvv             vvvv
                                        # |   \       /   |
                                        # |       v       |
                                        # |       |       |
                                        # w       |       www
                                        #     \   |   /
                                        #         vv
                                        hi += one(V); stack[hi] = vvvv
                                        hi += one(V); stack[hi] = vvv
                                        hi += one(V); stack[hi] = vv
                                        hi += one(V); stack[hi] = v

                                        delete!(set(eltypedegree(graph, w)), w)
                                        delete!(set(eltypedegree(graph, ww)), ww)
                                        delete!(set(eltypedegree(graph, www)), www)

                                        rem_edge!(graph, vv, v)
                                        rem_edge!(graph, vv, x)
                                        rem_edge!(graph, vv, xx)
                                        rem_edge!(graph, vvv, v)
                                        rem_edge!(graph, vvv, y)
                                        rem_edge!(graph, vvv, yy)
                                        rem_edge!(graph, vvvv, v)
                                        rem_edge!(graph, vvvv, z)
                                        rem_edge!(graph, vvvv, zz)

                                        add_edge!(graph, w, ww)
                                        add_edge!(graph, w, www)
                                        add_edge!(graph, ww, www)

                                        pushfirst!(set(eltypedegree(graph, w)), w)
                                        pushfirst!(set(eltypedegree(graph, ww)), ww)
                                        pushfirst!(set(eltypedegree(graph, www)), www)

                                        delete!(set(3), vvvv)
                                        delete!(set(3), vvv)
                                        delete!(set(3), vv)
                                        n = next[v]; delete!(set(3), v)
                                    else
                                        n = next[v]
                                    end
                                else
                                    n = next[v]
                                end

                                v = n
                            end

                            if lo == hi
                                width = max(width, four(V))
                            end
                        end
                    end
                end
            end
        end
    end

    label = rem_vertices!(graph, resize!(stack, hi); keep_order = true)
    return stack, label, width
end

# Pre-processing for Triangulation of Probabilistic Networks
# Bodlaender, Koster, Eijkhof, and van der Gaag
#
# Preprocessing Rules for Triangulation of Probabilistic Networks
# Bodlaender, Koster, Eijkhof, and van der Gaag
#
#  PR-4 (PR-3 + Simplicial + Almost Simplicial)
function pr4(graph, width::Integer)
    return pr4(BipartiteGraph(graph), width)
end

function pr4(graph::AbstractGraph{V}, width::Integer) where {V}
    kernel = Graph(graph)
    return kernel, pr4!(kernel, convert(V, width))...
end

function pr4!(graph::Graph{V}, width::V) where {V}
    # apply PR-3
    _stack, _label, width = pr3!(graph, width)

    # apply simplicial and almost simplicial rules
    n = nv(graph)
    marker = zeros(Int, n); tag = 0

    # heap
    heap = Heap{V, V}(n)

    for v in vertices(graph)
        list = neighbors(graph, v)
        degeneracy = zero(V)

        for i in oneto(eltypedegree(graph, v))
            w = list[i]
            marker[neighbors(graph, w)] .= tag += 1

            for ii in oneto(i - one(V))
                ww = list[ii]

                if marker[ww] < tag
                    degeneracy += one(V)
                end
            end
        end

        push!(heap, v => degeneracy)
    end

    hfall!(heap)

    # stack of eliminated vertices
    lo = -one(V)
    hi = zero(V)
    stack = Vector{V}(undef, n)

    while lo < hi
        lo = hi

        # simplicial
        while !isempty(heap) && iszero(minimum(heap))
            v = argmin(heap)
            list = neighbors(graph, v)
            degree = eltypedegree(graph, v)

            hi += one(V); stack[hi] = v; width = max(width, degree)
            delete!(heap, v)

            while !isempty(list)
                w = last(list)
                heap[w] -= (eltypedegree(graph, w) - degree)
                hrise!(heap, w)
                hfall!(heap, w)
                rem_edge!(graph, v, w)
            end
        end

        # almost-simplicial
        for w in keys(heap)
            marker[neighbors(graph, w)] .= tag += 1

            for v in neighbors(graph, w)
                list = neighbors(graph, v)
                degree = eltypedegree(graph, v)

                if degree <= width
                    count = zero(V)

                    for ww in neighbors(graph, v)
                        if w != ww && marker[ww] < tag
                            count += one(V)
                        end
                    end

                    if heap[v] == count
                        hi += one(V); stack[hi] = v
                        delete!(heap, v)

                        for ww in list
                            if w != ww && marker[ww] < tag
                                count = one(V)

                                for vv in neighbors(graph, ww)
                                    if v != vv && marker[vv] == tag
                                        heap[vv] -= one(V)
                                        hrise!(heap, vv)
                                        count += one(V)
                                    end
                                end

                                heap[w] += (eltypedegree(graph, w) - count)
                                heap[ww] += (eltypedegree(graph, ww) - count)
                                add_edge!(graph, w, ww); marker[ww] = tag
                            end
                        end

                        while !isempty(list)
                            ww = last(list)
                            heap[ww] -= (eltypedegree(graph, ww) - degree)
                            hrise!(heap, ww)
                            hfall!(heap, ww)
                            rem_edge!(graph, v, ww)
                        end

                        break
                    end
                end
            end
        end
    end

    append!(_stack, view(_label, resize!(stack, hi)))
    keepat!(_label, rem_vertices!(graph, stack; keep_order = true))
    return _stack, _label, width
end


# Safe Reduction Rules for Weighted Treewidth
# Eijkhof, Bodlaender, and Koster
# PR-3 (Islet, Twig, Series, Triangle, Buddy, and Cube)
function pr3(weights::AbstractVector, graph, width::Number)
    return pr3(weights, BipartiteGraph(graph), width)
end

function pr3(weights::AbstractVector{W}, graph::AbstractGraph, width::Number) where {W}
    kernel = Graph(graph)
    return kernel, pr3!(weights, kernel, convert(W, width))...
end

function pr3!(weights::AbstractVector{W}, graph::Graph{V}, width::W) where {W, V}
    n = nv(graph)
    tol = tolerance(weights)

    @inbounds for v in vertices(graph)
        rem_edge!(graph, v, v)
    end

    # neighbor weights
    nws = Vector{W}(undef, n)

    # bucket queue
    head = zeros(V, max(n, four(V)))
    prev = Vector{V}(undef, n)
    next = Vector{V}(undef, n)

    function set(i)
        @inbounds h = @view head[i + one(i)]
        return DoublyLinkedList(h, prev, next)
    end

    @inbounds for v in vertices(graph)
        nw = weights[v]

        for w in neighbors(graph, v)
            nw += weights[w]
        end

        nws[v] = nw
        pushfirst!(set(eltypedegree(graph, v)), v)
    end

    # stack of eliminated vertices
    lo = -one(V)
    hi = zero(V)
    stack = Vector{V}(undef, n)

    # apply rules until exhaustion
    while lo < hi
        lo = hi

        # islet rule
        v = head[1]

        while ispositive(v)
            # v
            hi += one(V); stack[hi] = v; width = max(width, nws[v])
            n = next[v]; delete!(set(0), v)
            v = n
        end

        if lo == hi
            # twig rule
            v = head[2]

            while ispositive(v)
                # w
                # |
                # v
                w = only(neighbors(graph, v))
                hi += one(V); stack[hi] = v; width = max(width, nws[v])

                delete!(set(eltypedegree(graph, w)), w)

                rem_edge!(graph, v, w); nws[w] -= weights[v]

                pushfirst!(set(eltypedegree(graph, w)), w)

                n = next[v]; delete!(set(1), v)
                v = n
            end

            if lo == hi
                # series rule
                v = head[3]

                while ispositive(v)
                    w = ww = zero(V)

                    if nws[v] < width + tol
                        x, xx = neighbors(graph, v)

                        if weights[v] + tol > min(weights[x], weights[xx])
                            w, ww = x, xx
                        end
                    end

                    if ispositive(w)
                        # w
                        # |
                        # v  ---  ww
                        hi += one(V); stack[hi] = v

                        delete!(set(eltypedegree(graph, w)), w)
                        delete!(set(eltypedegree(graph, ww)), ww)

                        rem_edge!(graph, v, w); nws[w] -= weights[v]
                        rem_edge!(graph, v, ww); nws[ww] -= weights[v]

                        if add_edge!(graph, w, ww)
                            nws[w] += weights[ww]
                            nws[ww] += weights[w]
                        end

                        pushfirst!(set(eltypedegree(graph, w)), w)
                        pushfirst!(set(eltypedegree(graph, ww)), ww)

                        n = next[v]; delete!(set(2), v)
                    else
                        n = next[v]
                    end

                    v = n
                end

                if lo == hi
                    # triangle rule
                    v = head[4]

                    while ispositive(v)
                        w = ww = www = zero(V)

                        if nws[v] < width + tol
                            x, xx, xxx = neighbors(graph, v)

                            if has_edge(graph, x, xx) && weights[v] + tol > weights[xxx]
                                w, ww, www = x, xx, xxx
                            elseif has_edge(graph, x, xxx) && weights[v] + tol > weights[xx]
                                w, ww, www = x, xxx, xx
                            elseif has_edge(graph, xx, xxx) && weights[v] + tol > weights[x]
                                w, ww, www = xx, xxx, x
                            end
                        end

                        if ispositive(w)
                            # w  ---  ww
                            # |   /
                            # v  ---  www
                            hi += one(V); stack[hi] = v

                            delete!(set(eltypedegree(graph, w)), w)
                            delete!(set(eltypedegree(graph, ww)), ww)
                            delete!(set(eltypedegree(graph, www)), www)

                            rem_edge!(graph, v, w); nws[w] -= weights[v]
                            rem_edge!(graph, v, ww); nws[ww] -= weights[v]
                            rem_edge!(graph, v, www); nws[www] -= weights[v]

                            if add_edge!(graph, w, www)
                                nws[w] += weights[ww]
                                nws[ww] += weights[w]
                            end

                            if add_edge!(graph, ww, www)
                                nws[ww] += weights[www]
                                nws[www] += weights[ww]
                            end

                            pushfirst!(set(eltypedegree(graph, w)), w)
                            pushfirst!(set(eltypedegree(graph, ww)), ww)
                            pushfirst!(set(eltypedegree(graph, www)), www)

                            n = next[v]; delete!(set(3), v)
                        else
                            n = next[v]
                        end

                        v = n
                    end

                    if lo == hi
                        # buddy rule
                        v = head[4]

                        while ispositive(v)
                            vv = n = next[v]

                            if nws[v] < width + tol
                                w, ww, www = neighbors(graph, v)

                                # sort the weights
                                p = weights[w]
                                pp = weights[ww]
                                ppp = weights[www]

                                if p - tol >= pp
                                    p, pp = pp, p
                                end

                                if pp - tol >= ppp
                                    pp, ppp = ppp, pp
                                end

                                if p - tol >= pp
                                    p, pp = pp, p
                                end

                                while ispositive(vv)
                                    x, xx, xxx = neighbors(graph, vv)

                                    # sort the weights
                                    q = weights[v]
                                    qq = weights[vv]

                                    if q - tol >= qq
                                        q, qq = qq, q
                                    end

                                    if nws[vv] < width + tol && p < q + tol && pp < qq + tol && (w, ww, www) == (x, xx, xxx)
                                        # w  -----------  vv
                                        # |           /   |
                                        # |       ww      |
                                        # |   /           |
                                        # v  -----------  www
                                        hi += one(V); stack[hi] = vv
                                        hi += one(V); stack[hi] = v

                                        delete!(set(eltypedegree(graph, w)), w)
                                        delete!(set(eltypedegree(graph, ww)), ww)
                                        delete!(set(eltypedegree(graph, www)), www)

                                        rem_edge!(graph, v, w); nws[w] -= weights[v]
                                        rem_edge!(graph, v, ww); nws[ww] -= weights[v]
                                        rem_edge!(graph, v, www); nws[www] -= weights[v]
                                        rem_edge!(graph, vv, w); nws[w] -= weights[vv]
                                        rem_edge!(graph, vv, ww); nws[ww] -= weights[vv]
                                        rem_edge!(graph, vv, www); nws[www] -= weights[vv]

                                        if add_edge!(graph, w, ww)
                                            nws[w] += weights[ww]
                                            nws[ww] += weights[w]
                                        end

                                        if add_edge!(graph, w, www)
                                            nws[w] += weights[www]
                                            nws[www] += weights[w]
                                        end

                                        if add_edge!(graph, ww, www)
                                            nws[ww] += weights[www]
                                            nws[www] += weights[ww]
                                        end

                                        pushfirst!(set(eltypedegree(graph, w)), w)
                                        pushfirst!(set(eltypedegree(graph, ww)), ww)
                                        pushfirst!(set(eltypedegree(graph, www)), www)

                                        nn = zero(V); delete!(set(3), vv)
                                        n = next[v]; delete!(set(3), v)
                                    else
                                        nn = next[vv]
                                    end

                                    vv = nn
                                end
                            end

                            v = n
                        end

                        if lo == hi
                            # cube rule
                            v = head[4]

                            while ispositive(v)
                                vv, vvv, vvvv = neighbors(graph, v)

                                if isthree(eltypedegree(graph, vv)) && isthree(eltypedegree(graph, vvv)) && isthree(eltypedegree(graph, vvvv)) && max(nws[vv], nws[vvv], nws[vvvv]) < width + tol
                                    w = ww = www = zero(V)
                                    x, xx, xxx = neighbors(graph, vv)
                                    y, yy, yyy = neighbors(graph, vvv)
                                    z, zz, zzz = neighbors(graph, vvvv)

                                    if v == x
                                        x, xx = xx, xxx
                                    elseif v == xx
                                        x, xx = x, xxx
                                    end

                                    if v == y
                                        y, yy = yy, yyy
                                    elseif v == yy
                                        y, yy = y, yyy
                                    end

                                    if v == z
                                        z, zz = zz, zzz
                                    elseif v == zz
                                        z, zz = z, zzz
                                    end

                                    if x == y != vvvv && yy == zz != vv && z == xx != vvv
                                        w, ww, www = x, yy, z
                                    elseif x == y != vvvv && yy == z != vv && zz == xx != vvv
                                        w, ww, www = x, yy, zz
                                    elseif x == yy != vvvv && y == z != vv && zz == xx != vvv
                                        w, ww, www = x, y, zz
                                    elseif xx == yy != vvvv && y == z != vv && zz == x != vvv
                                        w, ww, www = xx, y, zz
                                    elseif xx == y != vvvv && yy == zz != vv && z == x != vvv
                                        w, ww, www = xx, yy, z
                                    elseif xx == yy != vvvv && y == zz != vv && z == x != vvv
                                        w, ww, www = xx, y, z
                                    end

                                    if ispositive(w) && (
                                            (weights[vv] + tol > weights[www] && weights[vvv] + tol > weights[w] && weights[vvvv] + tol > weights[ww]) ||
                                                (weights[vv] + tol > weights[w] && weights[vvv] + tol > weights[ww] && weights[vvvv] + tol > weights[www])
                                        )
                                        #         ww
                                        #     /       \
                                        # vvv             vvvv
                                        # |   \       /   |
                                        # |       v       |
                                        # |       |       |
                                        # w       |       www
                                        #     \   |   /
                                        #         vv
                                        hi += one(V); stack[hi] = vvvv
                                        hi += one(V); stack[hi] = vvv
                                        hi += one(V); stack[hi] = vv
                                        hi += one(V); stack[hi] = v

                                        delete!(set(eltypedegree(graph, w)), w)
                                        delete!(set(eltypedegree(graph, ww)), ww)
                                        delete!(set(eltypedegree(graph, www)), www)

                                        rem_edge!(graph, vv, v); nws[vv] -= weights[v]
                                        rem_edge!(graph, vv, x); nws[vv] -= weights[x]
                                        rem_edge!(graph, vv, xx); nws[vv] -= weights[xx]
                                        rem_edge!(graph, vvv, v); nws[vvv] -= weights[v]
                                        rem_edge!(graph, vvv, y); nws[vvv] -= weights[y]
                                        rem_edge!(graph, vvv, yy); nws[vvv] -= weights[yy]
                                        rem_edge!(graph, vvvv, v); nws[vvvv] -= weights[v]
                                        rem_edge!(graph, vvvv, z); nws[vvvv] -= weights[z]
                                        rem_edge!(graph, vvvv, zz); nws[vvvv] -= weights[zz]

                                        if add_edge!(graph, w, ww)
                                            nws[w] += weights[ww]
                                            nws[ww] += weights[w]
                                        end

                                        if add_edge!(graph, w, www)
                                            nws[w] += weights[www]
                                            nws[www] += weights[w]
                                        end

                                        if add_edge!(graph, ww, www)
                                            nws[ww] += weights[www]
                                            nws[www] += weights[ww]
                                        end

                                        pushfirst!(set(eltypedegree(graph, w)), w)
                                        pushfirst!(set(eltypedegree(graph, ww)), ww)
                                        pushfirst!(set(eltypedegree(graph, www)), www)

                                        delete!(set(3), vvvv)
                                        delete!(set(3), vvv)
                                        delete!(set(3), vv)
                                        n = next[v]; delete!(set(3), v)
                                    else
                                        n = next[v]
                                    end
                                else
                                    n = next[v]
                                end

                                v = n
                            end
                        end
                    end
                end
            end
        end
    end

    label = rem_vertices!(graph, resize!(stack, hi); keep_order = true)
    return stack, label, width
end

# Safe Reduction Rules for Weighted Treewidth
# Eijkhof, Bodlaender, and Koster
# PR-4 (PR-3 + Simplicial + Almost Simplicial)
function pr4(weights::AbstractVector, graph, width::Number)
    return pr4(weights, BipartiteGraph(graph), width)
end

function pr4(weights::AbstractVector{W}, graph::AbstractGraph, width::Number) where {W}
    kernel = Graph(graph)
    return kernel, pr4!(weights, kernel, convert(W, width))...
end

function pr4!(weights::AbstractVector{W}, graph::Graph{V}, width::W) where {W, V}
    # apply PR-3
    _stack, _label, width = pr3!(weights, graph, width)

    # apply simplicial and almost simplicial rules
    n = nv(graph)
    tol = tolerance(weights)
    marker = zeros(Int, n); tag = 0

    # neighbor weights
    nws = Vector{W}(undef, n)

    # heap
    heap = Heap{V, V}(n)

    for v in vertices(graph)
        list = neighbors(graph, v)

        nw = weights[_label[v]]
        degeneracy = zero(V)

        for i in oneto(eltypedegree(graph, v))
            w = list[i]
            nw += weights[_label[w]]
            marker[neighbors(graph, w)] .= tag += 1

            for ii in oneto(i - one(V))
                ww = list[ii]

                if marker[ww] < tag
                    degeneracy += one(V)
                end
            end
        end

        nws[v] = nw
        push!(heap, v => degeneracy)
    end

    hfall!(heap)

    # stack of eliminated vertices
    lo = -one(V)
    hi = zero(V)
    stack = Vector{V}(undef, n)

    while lo < hi
        lo = hi

        # simplicial
        while !isempty(heap) && iszero(minimum(heap))
            v = argmin(heap)
            list = neighbors(graph, v)
            degree = eltypedegree(graph, v)

            hi += one(V); stack[hi] = v; width = max(width, nws[v])
            delete!(heap, v)

            while !isempty(list)
                w = last(list)
                nws[w] -= weights[_label[v]]
                heap[w] -= (eltypedegree(graph, w) - degree)
                hrise!(heap, w)
                hfall!(heap, w)
                rem_edge!(graph, v, w)
            end
        end

        # almost-simplicial
        for w in keys(heap)
            marker[neighbors(graph, w)] .= tag += 1

            for v in neighbors(graph, w)
                list = neighbors(graph, v)
                degree = eltypedegree(graph, v)

                if nws[v] < width + tol && weights[_label[w]] < weights[_label[v]] + tol
                    count = zero(V)

                    for ww in neighbors(graph, v)
                        if w != ww && marker[ww] < tag
                            count += one(V)
                        end
                    end

                    if heap[v] == count
                        hi += one(V); stack[hi] = v
                        delete!(heap, v)

                        for ww in list
                            if w != ww && marker[ww] < tag
                                count = one(V)

                                for vv in neighbors(graph, ww)
                                    if v != vv && marker[vv] == tag
                                        heap[vv] -= one(V)
                                        hrise!(heap, vv)
                                        count += one(V)
                                    end
                                end

                                nws[w] += weights[_label[ww]]
                                nws[ww] += weights[_label[w]]
                                heap[w] += (eltypedegree(graph, w) - count)
                                heap[ww] += (eltypedegree(graph, ww) - count)
                                add_edge!(graph, w, ww); marker[ww] = tag
                            end
                        end

                        while !isempty(list)
                            ww = last(list)
                            nws[ww] -= weights[_label[v]]
                            heap[ww] -= (eltypedegree(graph, ww) - degree)
                            hrise!(heap, ww)
                            hfall!(heap, ww)
                            rem_edge!(graph, v, ww)
                        end

                        break
                    end
                end
            end
        end
    end

    append!(_stack, view(_label, resize!(stack, hi)))
    keepat!(_label, rem_vertices!(graph, stack; keep_order = true))
    return _stack, _label, width
end

function connectedcomponents(graph)
    return connectedcomponents(BipartiteGraph(graph))
end

function connectedcomponents(graph::AbstractGraph{V}) where {V}
    E = etype(graph)
    n = nv(graph)
    m = is_directed(graph) ? ne(graph) : twice(ne(graph))
    components = SubArray{V, 1, Vector{V}, Tuple{UnitRange{V}}, true}[]
    subgraphs = BipartiteGraph{V, E, SubArray{E, 1, Vector{E}, Tuple{UnitRange{V}}, true}, SubArray{V, 1, Vector{V}, Tuple{UnitRange{E}}, true}}[]
    projection = Vector{V}(undef, n)

    # initialize work arrays
    level = zeros(Int, n); tag = 0
    queue = @view Vector{V}(undef, n)[one(V):n]
    nfree = @view Vector{E}(undef, twice(n))[one(V):twice(n)]
    mfree = @view Vector{V}(undef, m)[one(E):m]

    @inbounds for root in vertices(graph)
        if iszero(level[root])
            component, queue, tag = bfs!(level, queue, graph, root, tag + 1)
            nstop = one(V)
            mstop = zero(E)

            for v in component
                projection[v] = nstop
                nstop += one(V)
                mstop += convert(E, eltypedegree(graph, v))
            end

            subgraph = BipartiteGraph{V, E}(
                nstop - one(V),
                view(nfree, one(V):nstop),
                view(mfree, one(E):mstop),
            )

            p = pointers(subgraph)[begin] = one(E)

            for (i, v) in enumerate(component)
                pp = pointers(subgraph)[i + 1] = p + eltypedegree(graph, v)
                targets(subgraph)[p:(pp - one(E))] = @view projection[neighbors(graph, v)]
                p = pp
            end

            push!(components, component)
            push!(subgraphs, subgraph)
            nfree = @view nfree[(nstop + one(V)):twice(n)]
            mfree = @view mfree[(mstop + one(E)):m]
        end
    end

    return components, subgraphs
end

function Base.show(io::IO, ::MIME"text/plain", alg::A) where A <: EliminationAlgorithm
    indent = get(io, :indent, 0)
    println(io, " "^indent * "$A")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::RCMMD{A}) where {A}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "RCMMD{$A}:")

    for line in eachsplit(strip(repr(alg.alg)), "\n")
        println(io, " "^indent * "    $line")
    end

    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::RCMGL{A}) where {A}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "RCMGL{$A}:")

    for line in eachsplit(strip(repr(alg.alg)), "\n")
        println(io, " "^indent * "    $line")
    end

    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::MMD)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "MMD:")
    println(io, " "^indent * "    delta: $(alg.delta)")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::AMF)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "AMF:")
    println(io, " "^indent * "    speed: $(alg.speed)")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::AMD)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "AMD:")
    println(io, " "^indent * "    dense: $(alg.dense)")
    println(io, " "^indent * "    aggressive: $(alg.aggressive)")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::SymAMD)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "SymAMD:")
    println(io, " "^indent * "    dense_row: $(alg.dense_row)")
    println(io, " "^indent * "    dense_col: $(alg.dense_col)")
    println(io, " "^indent * "    aggressive: $(alg.aggressive)")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::METIS)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "METIS:")
    println(io, " "^indent * "    ctype: $(alg.ctype)")
    println(io, " "^indent * "    rtype: $(alg.rtype)")
    println(io, " "^indent * "    nseps: $(alg.nseps)")
    println(io, " "^indent * "    niter: $(alg.niter)")
    println(io, " "^indent * "    seed: $(alg.seed)")
    println(io, " "^indent * "    compress: $(alg.compress)")
    println(io, " "^indent * "    ccorder: $(alg.ccorder)")
    println(io, " "^indent * "    pfactor: $(alg.pfactor)")
    println(io, " "^indent * "    ufactor: $(alg.ufactor)")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::Spectral)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "Spectral:")
    println(io, " "^indent * "    tol: $(alg.tol)")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::FlowCutter)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "FlowCutter:")
    println(io, " "^indent * "    time: $(alg.time)")
    println(io, " "^indent * "    seed: $(alg.seed)")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::SAT{H, L, U}) where {H, L, U}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "SAT{$H, $L, $U}:")
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.lb)
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.ub)
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::MinimalChordal{A}) where {A}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "MinimalChordal{$A}:")
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.alg)
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::CompositeRotations{C, A}) where {C, A}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "CompositeRotations{$C, $A}:")
    println(io, " "^indent * "    clique: $(alg.clique)")
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.alg)
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::SafeRules{L, U}) where {L, U}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "SafeRules{$L, $U}:")
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.lb)
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.ub)
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::SafeSeparators{M, A}) where {M, A}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "SafeSeparators{$M, $A}:")
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.min)
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.alg)
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::ConnectedComponents{A}) where {A}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "ConnectedComponents{$A}:")
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.alg)
    return
end

"""
    DEFAULT_ELIMINATION_ALGORITHM = MMD()

The default algorithm.
"""
const DEFAULT_ELIMINATION_ALGORITHM = MMD()

"""
    RCM = RCMGL

The default variant of the reverse Cuthill-Mckee algorithm.
"""
const RCM = RCMGL
