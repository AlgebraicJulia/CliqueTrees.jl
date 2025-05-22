"""
    EliminationAlgorithm

A *graph elimination algorithm* computes a total ordering of the vertices of a graph. 
The algorithms implemented in CliqueTrees.jl can be divided into five categories.

  - triangulation recognition algorithms
  - bandwidth reduction algorithms
  - greedy algorithms
  - nested dissection algorithms
  - exact treewidth algorithms

# Triangulation Recognition Algorithms

| type             | name                                         | time     | space    | package |
|:-----------------|:-------------------------------------------- |:-------- |:-------- | :------ |
| [`MCS`](@ref)    | maximum cardinality search                   | O(m + n) | O(n)     |         |
| [`LexBFS`](@ref) | lexicographic breadth-first search           | O(m + n) | O(m + n) |         |
| [`MCSM`](@ref)   | maximum cardinality search (minimal)         | O(mn)    | O(n)     |         |
| [`LexM`](@ref)   | lexicographic breadth-first search (minimal) | O(mn)    | O(n)     |         |

These algorithms will compute perfect orderings when applied to chordal graphs.

# Bandwidth and Envelope Minimization Algorithms

| type            | name                                   | time     | space    | package |
|:----------------|:---------------------------------------|:-------- |:-------- | :------ |
| [`RCMMD`](@ref) | reverse Cuthill-Mckee (minimum degree) | O(m + n) | O(m + n) |         | 
| [`RCMGL`](@ref) | reverse Cuthill-Mckee (George-Liu)     | O(m + n) | O(m + n) |         |

These algorithms try to minimize the *bandwidth* and *envelope* of the ordered graph.

# Greedy Algorithms

| type             | name                              | time   | space    | package                                                   |
|:-----------------|:----------------------------------|:-------|:-------- | :-------------------------------------------------------- |
| [`MMD`](@ref)    | multiple minimum degree           | O(mn²) | O(m + n) |                                                           |
| [`MF`](@ref)     | minimum fill                      | O(mn²) |          |                                                           |
| [`AMD`](@ref)    | approximate minimum degree        | O(mn)  | O(m + n) | [AMD.jl](https://github.com/JuliaSmoothOptimizers/AMD.jl) |
| [`SymAMD`](@ref) | column approximate minimum degree | O(mn)  | O(m + n) | [AMD.jl](https://github.com/JuliaSmoothOptimizers/AMD.jl) |
| [`AMF`](@ref)    | approximate minimum fill          | O(mn)  | O(m + n) |                                                           |

These algorithms simulate the graph elimination process, greedily eliminating
vertices that minimize a cost function. They are faster then the nested dissection
algorithms, but have worse results.

# Nested Dissection Algorithms

| type             | name              | time   | space | package                                             |
|:-----------------|:------------------|:-------|:----- | :-------------------------------------------------- |
| [`METIS`](@ref)  | nested dissection |        |       | [Metis.jl](https://github.com/JuliaSparse/Metis.jl) |
| [`ND`](@ref)     | nested dissection |        |       |                                                     |

These algorithms recursively partition a graph, then call a greedy algorithm on the leaves.
These are slower than the greedy algorithms, but have better results.

# Exact Treewidth Algorithms

| type          | name              | time  | space | package                                                                 |
|:--------------|:------------------|:------|:------| :---------------------------------------------------------------------- |
| [`BT`](@ref)  | Bouchitte-Todinca |       |       | [TreeWidthSolver.jl](https://github.com/ArrogantGao/TreeWidthSolver.jl) |
| [`SAT`](@ref) | SAT encoding      |       |       |                                                                         |

The orderings computed by these algorithms induce minimum-width tree decompositions.

!!! warning
    This is an NP-hard problem.
"""
abstract type EliminationAlgorithm end

"""
    PermutationOrAlgorithm = Union{
    AbstractVector,
    Tuple{AbstractVector, AbstractVector},
    EliminationAlgorithm,
}

Either a permutation or an algorithm.
"""
const PermutationOrAlgorithm = Union{
    AbstractVector,
    Tuple{AbstractVector, AbstractVector},
    EliminationAlgorithm,
}

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
    ND{A, D} <: EliminationAlgorithm

    ND(alg::EliminationAlgorithm, dis::DissectionAlgorithm; limit=200, level=6)

    ND(alg::EliminationAlgorithm; limit=200, level=6)

    ND(; limit=200, level=6)

The [nested dissection algorithm](https://en.wikipedia.org/wiki/Nested_dissection).

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

julia> alg = ND(MF(), METISND(); limit=0, level=2)
ND{MF, METISND}:
    MF
    METISND:
        seed: -1
        ufactor: -1
    limit: 0
    level: 2

julia> treewidth(graph; alg)
2
```

### Parameters

  - `alg`: elimination algorithm
  - `dis`: dissection algorithm
  - `limit`: smallest subgraph
  - `level`: maximum depth
"""
struct ND{S, A <: EliminationAlgorithm, D <: DissectionAlgorithm} <: EliminationAlgorithm
    alg::A
    dis::D
    limit::Int
    level::Int
end

function ND{S}(alg::A = DEFAULT_ELIMINATION_ALGORITHM, dis::D = DEFAULT_DISSECTION_ALGORITHM; limit::Int = 200, level::Int = 6) where {S, A <: EliminationAlgorithm, D <: DissectionAlgorithm}
    return ND{S, A, D}(alg, dis, limit, level)
end

function ND(alg::EliminationAlgorithm = DEFAULT_ELIMINATION_ALGORITHM, dis::DissectionAlgorithm = DEFAULT_DISSECTION_ALGORITHM; limit::Int = 200, level::Int = 6)
    return ND{1}(alg, dis; limit, level)
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
    SAT{H, A} <: EliminationAlgorithm

    SAT{H}(alg::PermutationOrAlgorithm)

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

julia> alg = SAT{libpicosat_jll}(MF()) # picosat
SAT{libpicosat_jll, MF}:
    MF

julia> alg = SAT{PicoSAT_jll}(MF()) # picosat
SAT{PicoSAT_jll, MF}:
    MF

julia> alg = SAT{CryptoMiniSat_jll}(MF()) # cryptominisat
SAT{CryptoMiniSat_jll, MF}:
    MF

julia> alg = SAT{Lingeling_jll}(MMW(), MF()) # lingeling
SAT{Lingeling_jll, MF}:
    MF

julia> treewidth(graph; alg)
2
```

### Parameters

  - `alg`: elimination algorithm

## References

  - Samer, Marko, and Helmut Veith. "Encoding treewidth into SAT." *Theory and Applications of Satisfiability Testing-SAT 2009: 12th International Conference, SAT 2009*, Swansea, UK, June 30-July 3, 2009. Proceedings 12. Springer Berlin Heidelberg, 2009.
  - Berg, Jeremias, and Matti Järvisalo. "SAT-based approaches to treewidth computation: An evaluation." *2014 IEEE 26th international conference on tools with artificial intelligence.* IEEE, 2014.
  - Bannach, Max, Sebastian Berndt, and Thorsten Ehlers. "Jdrasil: A modular library for computing tree decompositions." *16th International Symposium on Experimental Algorithms (SEA 2017)*. Schloss Dagstuhl–Leibniz-Zentrum fuer Informatik, 2017.
"""
struct SAT{H, A <: PermutationOrAlgorithm} <: EliminationAlgorithm
    alg::A
end

function SAT{H}(alg::A) where {H, A <: PermutationOrAlgorithm}
    return SAT{H, A}(alg)
end

function SAT{H}() where {H}
    return SAT{H}(DEFAULT_ELIMINATION_ALGORITHM)
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
    SafeRules{A, L, U} <: EliminationAlgorithm

    SafeRules(alg::EliminationAlgorithm, lb::WidthOrAlgorithm, ub::EliminationAlgororithm)

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

julia> alg2 = SafeRules(BT(), MMW(), MF())
SafeRules{BT, MMW, MF}:
    BT
    MMW
    MF

julia> @time treewidth(graph; alg=alg1) # slow
  0.000177 seconds (1.41 k allocations: 90.031 KiB)
2

julia> @time treewidth(graph; alg=alg2) # fast
  0.000044 seconds (282 allocations: 15.969 KiB)
2
```

### Parameters

  - `alg`: elimination algorithm
  - `lb`: lower bound algorithm (used to lower bound the treiwidth)
  - `ub`: elimination algorithm (used to upper bound the treewidth)

### References

  - Bodlaender, Hans L., et al. "Pre-processing for triangulation of probabilistic networks." (2001).
  - Bodlaender, Hans L., Arie M.C.A. Koster, and Frank van den Eijkhof. "Preprocessing rules for triangulation of probabilistic networks." *Computational Intelligence* 21.3 (2005): 286-305.
  - van den Eijkhof, Frank, Hans L. Bodlaender, and Arie M.C.A. Koster. "Safe reduction rules for weighted treewidth." *Algorithmica* 47 (2007): 139-158. 
"""
struct SafeRules{A <: EliminationAlgorithm, L <: WidthOrAlgorithm, U <: EliminationAlgorithm} <: EliminationAlgorithm
    alg::A
    lb::L
    ub::U
end

function SafeRules(alg::PermutationOrAlgorithm, lb::WidthOrAlgorithm)
    return SafeRules(alg, lb, DEFAULT_ELIMINATION_ALGORITHM)
end

function SafeRules(alg::PermutationOrAlgorithm)
    return SafeRules(alg, DEFAULT_LOWER_BOUND_ALGORITHM)
end

function SafeRules()
    return SafeRules(DEFAULT_ELIMINATION_ALGORITHM)
end

# deprecated
const RuleReduction{A} = SafeRules{A, MMW{3}, MF}
RuleReduction(alg) = SafeRules(alg, MMW(), MF())

"""
    SafeSeparators{M, A} <: EliminationAlgorithm

    SafeSeparators(alg::EliminationAlgorithm, min::PermutationOrAlgorithm)

    SafeSeparators(alg::EliminationAlgorithm)

    SafeSeparators()

Apple an elimination algorithm to the atoms of an almost-clique separator decomposition. The algorithm
`min` is used to compute the decomposition.

!!! warning
    The algorithm `min` must compute a *minimimal* ordering. This property is guaranteed by the following
    algorithms:
      - [`MCSM`](@ref)
      - [`LexM`](@ref)
      - [`MinimalChordal`](@ref)

### Parameters

  - `alg`: elimination algorithm
  - `min`: minimal elimination algorithm

### References

  - Bodlaender, Hans L., and Arie MCA Koster. "Safe separators for treewidth." *Discrete Mathematics* 306.3 (2006): 337-350.
  - Tamaki, Hisao. "A heuristic for listing almost-clique minimal separators of a graph." arXiv preprint arXiv:2108.07551 (2021).
"""
struct SafeSeparators{A <: EliminationAlgorithm, M <: PermutationOrAlgorithm} <: EliminationAlgorithm
    alg::A
    min::M
end

function SafeSeparators(alg::EliminationAlgorithm)
    return SafeSeparators(alg, MinimalChordal())
end

function SafeSeparators()
    return SafeSeparators(DEFAULT_ELIMINATION_ALGORITHM)
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

struct Best{S, A <: Tuple} <: EliminationAlgorithm
    algs::A
end

const BestWidth = Best{1}
const BestFill = Best{2}

function Best{S}(algs::A) where {S, A <: Tuple}
    return Best{S, A}(algs)
end

function Best{S}(algs::PermutationOrAlgorithm...) where {S}
    return Best{S}(algs)
end

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

function permutation(graph, order::AbstractVector)
    return permutation(BipartiteGraph(graph), order)
end

function permutation(graph, (order, index)::Tuple{AbstractVector, AbstractVector})
    return permutation(BipartiteGraph(graph), (order, index))
end

function permutation(graph::AbstractGraph{V}, order::AbstractVector) where {V}
    order = Vector{V}(order)
    return order, invperm(order)
end

function permutation(graph::AbstractGraph{V}, (order, index)::Tuple{AbstractVector, AbstractVector}) where {V}
    order = Vector{V}(order)
    index = Vector{V}(index)
    return order, index
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

function permutation(weights::AbstractVector, graph, alg::MF)
    order = mf(weights, graph)
    return order, invperm(order)
end

function permutation(graph, alg::MMD)
    index = mmd(graph; delta = alg.delta)
    return invperm(index), index
end

function permutation(graph, alg::ND)
    return dissect(graph, alg)
end

function permutation(weights::AbstractVector, graph, alg::ND)
    return dissect(weights, graph, alg)
end

function permutation(graph, alg::SAT{H}) where {H}
    order, width = sat(graph, treewidth(graph, alg.alg), Val(H))
    return order, invperm(order)
end

function permutation(weights::AbstractVector, graph, alg::SAT{H}) where {H}
    order, width = sat(weights, graph, treewidth(weights, graph, alg.alg), Val(H))
    return order, invperm(order)
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
    width = lowerbound(graph, alg.lb)
    weights, graph, stack, index, width = saferules(graph, width)
    order, _ = permutation(weights, graph, alg.ub)

    if width < treewidth(weights, graph, order)
        order, _ = permutation(weights, graph, alg.alg)
    end

    for v in order
        append!(stack, neighbors(index, v))
    end

    return stack, invperm(stack)
end

function permutation(weights::AbstractVector, graph, alg::SafeRules)
    width = lowerbound(weights, graph, alg.lb)
    weights, graph, stack, index, width = saferules(weights, graph, width)
    order, _ = permutation(weights, graph, alg.ub)

    if width < treewidth(weights, graph, order)
        order, _ = permutation(weights, graph, alg.alg)
    end

    for v in order
        append!(stack, neighbors(index, v))
    end

    return stack, invperm(stack)
end

function permutation(graph, alg::SafeSeparators)
    # construct almost-clique separator decomposition
    graph, label, tree = safetree(graph, alg.min)

    # permute graph
    V = eltype(graph)
    graph = Graph{V}(permute!(sparse(graph), label, label))

    # compute ordering
    order = cliquedissect(graph, tree, alg.alg)

    for v in vertices(graph)
        order[v] = label[order[v]]
    end

    return order, invperm(order)
end

function permutation(weights::AbstractVector, graph, alg::SafeSeparators)
    # construct almost-clique separator decomposition
    graph, label, tree = safetree(graph, alg.min)

    # permute graph
    weights = weights[label]
    V = eltype(graph)
    graph = Graph{V}(permute!(sparse(graph), label, label))

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

function permutation(graph, alg::BestWidth)
    first, rest... = alg.algs
    return bestwidth(graph, first, rest)
end

function permutation(graph, alg::BestFill)
    first, rest... = alg.algs
    return bestfill(graph, first, rest)
end

function permutation(weights::AbstractVector, graph, alg::BestWidth)
    first, rest... = alg.algs
    return bestwidth(weights, graph, first, rest)
end

function permutation(weights::AbstractVector, graph, alg::BestFill)
    first, rest... = alg.algs
    return bestfill(weights, graph, first, rest)
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

function mf(graph)
    return mf(BipartiteGraph(graph))
end

function mf(graph::AbstractGraph{V}) where {V}
    weights = ones(V, nv(graph))
    return mf(weights, graph)
end

function mf(weights::AbstractVector, graph)
    return mf(weights, BipartiteGraph(graph))
end

function mf(weights::AbstractVector, graph::AbstractGraph)
    return mf!(weights, Graph(graph))
end

# Probabilistic Graphical Models: Principles and Techniques
# Koller and Friedman
# Algorithm 9.4 Greedy search for constructing an elimination ordering
# (Weighted Min-Fill)
function mf!(weights::AbstractVector{W}, graph::Graph{V}) where {W, V}
    n = nv(graph)
    order = Vector{V}(undef, n)
    label = zeros(Int, n); tag = 0

    # remove self edges
    @inbounds for v in vertices(graph)
        rem_edge!(graph, v, v)
    end

    # compute neighbor weights
    nweights = Vector{W}(undef, n)

    @inbounds for v in vertices(graph)
        nweight = weights[v]

        for w in neighbors(graph, v)
            nweight += weights[w]
        end

        nweights[v] = nweight
    end

    # construct stack data structure
    snum = zero(V) # size of stack
    stack = Vector{V}(undef, n)

    # construct min-heap data structure
    heap = Heap{V, W}(n)

    @inbounds for v in oneto(n)
        cost = zero(W)

        for w in neighbors(graph, v)
            label[neighbors(graph, w)] .= tag += 1
            wcost = zero(W)

            for ww in neighbors(graph, v)
                w == ww && break

                if label[ww] < tag
                    wcost += weights[ww]
                end
            end

            cost += wcost * weights[w]
        end

        push!(heap, v => cost)
    end

    hfall!(heap)

    # run algorithm
    i = one(V)

    @inbounds while i <= n
        # select vertex from heap
        order[i] = v = argmin(heap)
        label[neighbors(graph, v)] .= tag += 1
        degree = eltypedegree(graph, v)
        weight = weights[v]
        nweight = nweights[v]

        # append distinguishable neighbors to the stack
        snum = zero(V)
        sweight = zero(W)
        ii = i + one(V)

        for w in neighbors(graph, v)
            flag = false

            if eltypedegree(graph, w) == degree
                flag = true

                for vv in neighbors(graph, w)
                    if vv != v && label[vv] < tag
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
                sweight += weights[w]
                stack[snum] = w
            end
        end

        # remove vertex from graph
        graph.ne -= snum

        for j in oneto(snum)
            w = stack[j]
            list = neighbors(graph, w)
            nweights[w] -= weight
            deleteat!(list, searchsortedfirst(list, v))
        end

        # remove indistinguishable neighbors from graph
        if ii > i + one(V)
            tag += 1

            for j in oneto(snum)
                w = stack[j]
                list = neighbors(graph, w)
                count = zero(V)
                cost = weights[w]

                for x in neighbors(graph, w)
                    if label[x] < tag
                        count += one(V)
                        cost += weights[x]
                        list[count] = x
                    end
                end

                graph.ne -= (eltypedegree(graph, w) - count)
                nweights[w] = cost
                resize!(list, count)
            end
        end

        # remove vertex and indistinguishable neighbors from heap
        graph.ne -= half((ii - i) * (ii - i - one(V)))

        for j in i:(ii - one(V))
            w = order[j]
            delete!(heap, w)
            empty!(neighbors(graph, w))
            nweights[w] = weights[w]
        end

        # update deficiencies
        if ispositive(heap[v])
            for j in oneto(snum)
                w = stack[j]
                wweight = weights[w]
                label[neighbors(graph, w)] .= tag += 1

                for jj in (j + one(V)):snum
                    ww = stack[jj]
                    wwweight = weights[ww]

                    if label[ww] < tag
                        cost = wweight + wwweight

                        for xx in neighbors(graph, ww)
                            if label[xx] == tag
                                heap[xx] -= wweight * wwweight
                                hrise!(heap, xx)
                                cost += weights[xx]
                            end
                        end

                        nweights[w] += wwweight
                        nweights[ww] += wweight
                        heap[w] += wwweight * (nweights[w] - cost)
                        heap[ww] += wweight * (nweights[ww] - cost)
                        add_edge!(graph, w, ww)
                        label[ww] = tag
                    end
                end
            end
        end

        # update heap
        for j in oneto(snum)
            w = stack[j]
            heap[w] -= (nweight - sweight) * (nweights[w] - sweight)
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

# Algorithms for Sparse Linear Systems
# Scott and Tuma
# Algorithm 8.6: Nested Dissection Algorithm
function dissect(graph, alg::ND)
    return dissect(BipartiteGraph(graph), alg)
end

function dissect(graph::AbstractGraph{V}, alg::ND) where {V}
    weights = ones(V, nv(graph))
    return dissect(weights, graph, alg)
end

function dissect(graph::AbstractGraph{V}, alg::ND{<:Any, <:EliminationAlgorithm, METISND}) where {V}
    weights = ones(Int32, nv(graph))
    return dissect(weights, graph, alg)
end

# Algorithms for Sparse Linear Systems
# Scott and Tuma
# Algorithm 8.6: Nested Dissection Algorithm
function dissect(weights::AbstractVector, graph, alg::ND)
    return dissect(weights, BipartiteGraph(graph), alg)
end

function dissect(weights::AbstractVector, graph::AbstractGraph{V}, alg::ND) where {V}
    simple = simplegraph(graph)
    return dissect(weights, simple, vertices(simple), vertices(simple), zero(V), alg)
end

function dissect(weights::AbstractVector, graph::AbstractGraph{V}, alg::ND{<:Any, <:EliminationAlgorithm, METISND}) where {V}
    simple = simplegraph(Int32, Int32, graph)
    order::Vector{V}, index::Vector{V} = dissect(weights, simple, vertices(simple), vertices(simple), zero(Int32), alg)
    return order, index
end

function dissect(weights::AbstractVector{W}, graph::BipartiteGraph{V, E}, label::AbstractVector{V}, project::AbstractVector{V}, level::V, alg::ND{S}) where {W, V, E, S}
    n = nv(graph); m = ne(graph); nn = n + one(V)
    parents = Int[]; leaves = Bool[]; separators = View{V, Int}[]

    nodes = Tuple{
        BipartiteGraph{V, E, View{E, Int}, View{V, Int}}, # graph
        View{W, Int},                                     # weights
        View{V, Int},                                     # label
        View{V, Int},                                     # project
        V,                                                # level
    }[]

    wwork = sizehint!(W[], n * alg.level)
    vwork = sizehint!(V[], n * alg.level + n * alg.level + m * alg.level)
    ework = sizehint!(E[], n * alg.level + alg.level)

    node = (
        copy!(BipartiteGraph(n, allocate!(ework, nn), allocate!(vwork, m)), graph),
        copy!(allocate!(wwork, n), weights),
        copy!(allocate!(vwork, n), label),
        copy!(allocate!(vwork, n), project),
        level,
    )

    push!(parents, 0); push!(nodes, node)

    @inbounds for (i, (graph, weights, label, project, level)) in enumerate(nodes)
        n = nv(graph)

        if n <= alg.limit || level >= alg.level
            push!(leaves, true); push!(separators, allocate!(vwork, zero(V)))
        else
            # V = W ∪ B
            part = separator(weights, graph, alg.dis)
            n0 = zero(V); m0 = zero(E); project0 = allocate!(vwork, n)
            n1 = zero(V); m1 = zero(E); project1 = allocate!(vwork, n)
            n2 = zero(V)

            for v in vertices(graph)
                vv = part[v]

                if iszero(vv)    # v ∈ W - B
                    project0[v] = n0 += one(V); m0 += convert(E, eltypedegree(graph, v))
                elseif isone(vv) # v ∈ B - W
                    project1[v] = n1 += one(V); m1 += convert(E, eltypedegree(graph, v))
                else             # v ∈ W ∩ B
                    project0[v] = n0 += one(V); m0 += convert(E, twice(n2))
                    project1[v] = n1 += one(V); m1 += convert(E, twice(n2))
                    n2 += one(V)

                    for w in outneighbors(graph, v)
                        ww = part[w]

                        if iszero(ww)    # w ∈ W - B
                            m0 += one(E)
                        elseif isone(ww) # w ∈ B - W
                            m1 += one(E)
                        end
                    end
                end
            end

            level += one(V)

            while (iszero(n0) || iszero(n1)) && level < alg.level
                # V = W ∪ B
                part = separator(weights, graph, alg.dis)
                n0 = zero(V); m0 = zero(E)
                n1 = zero(V); m1 = zero(E)
                n2 = zero(V)

                for v in vertices(graph)
                    vv = part[v]

                    if iszero(vv)    # v ∈ W - B
                        project0[v] = n0 += one(V); m0 += convert(E, eltypedegree(graph, v))
                    elseif isone(vv) # v ∈ B - W
                        project1[v] = n1 += one(V); m1 += convert(E, eltypedegree(graph, v))
                    else             # v ∈ W ∩ B
                        project0[v] = n0 += one(V); m0 += convert(E, twice(n2))
                        project1[v] = n1 += one(V); m1 += convert(E, twice(n2))
                        n2 += one(V)

                        for w in outneighbors(graph, v)
                            ww = part[w]

                            if iszero(ww)    # w ∈ W - B
                                m0 += one(E)
                            elseif isone(ww) # w ∈ B - W
                                m1 += one(E)
                            end
                        end
                    end
                end

                level += one(V)
            end

            if level > alg.level
                push!(leaves, true); push!(separators, allocate!(vwork, zero(V)))
            else
                t0 = zero(V); label0 = allocate!(vwork, n0)
                t1 = zero(V); label1 = allocate!(vwork, n1)
                t2 = zero(V); label2 = allocate!(vwork, n2)

                for v in vertices(graph)
                    vv = part[v]

                    if iszero(vv)    # v ∈ W - B
                        t0 += one(V); label0[t0] = v
                    elseif isone(vv) # v ∈ B - W
                        t1 += one(V); label1[t1] = v
                    else             # v ∈ W ∩ B
                        t0 += one(V); label0[t0] = v
                        t1 += one(V); label1[t1] = v
                        t2 += one(V); label2[t2] = v
                    end
                end

                nn0 = n0 + one(V); graph0 = BipartiteGraph(n0, allocate!(ework, nn0), allocate!(vwork, m0))
                nn1 = n1 + one(V); graph1 = BipartiteGraph(n1, allocate!(ework, nn1), allocate!(vwork, m1))
                t0 = one(V); pointers(graph0)[t0] = p0 = one(E)
                t1 = one(V); pointers(graph1)[t1] = p1 = one(E)
                weights0 = allocate!(wwork, n0)
                weights1 = allocate!(wwork, n1)

                for v in vertices(graph)
                    vv = part[v]
                    wt = weights[v]

                    if iszero(vv)    # v ∈ W - B
                        for w in neighbors(graph, v) # w ∈ W
                            targets(graph0)[p0] = project0[w]; p0 += one(E)
                        end

                        weights0[t0] = wt; t0 += one(V); pointers(graph0)[t0] = p0
                    elseif isone(vv) # v ∈ B - W
                        for w in neighbors(graph, v) # w ∈ B
                            targets(graph1)[p1] = project1[w]; p1 += one(E)
                        end

                        weights1[t1] = wt; t1 += one(V); pointers(graph1)[t1] = p1
                    else             # v ∈ W ∩ B
                        for w in neighbors(graph, v)
                            ww = part[w]

                            if iszero(ww)            # w ∈ W - B
                                targets(graph0)[p0] = project0[w]; p0 += one(E)
                            elseif isone(ww)         # w ∈ B - W
                                targets(graph1)[p1] = project1[w]; p1 += one(E)
                            end
                        end

                        for w in label2              # w ∈ W ∩ B
                            v == w && continue
                            targets(graph0)[p0] = project0[w]; p0 += one(E)
                            targets(graph1)[p1] = project1[w]; p1 += one(E)
                        end

                        weights0[t0] = wt; t0 += one(V); pointers(graph0)[t0] = p0
                        weights1[t1] = wt; t1 += one(V); pointers(graph1)[t1] = p1
                    end
                end

                push!(parents, i, i), push!(leaves, false); push!(separators, label2)
                push!(nodes, (graph0, weights0, label0, project0, level))
                push!(nodes, (graph1, weights1, label1, project1, level))
            end
        end
    end

    ntree = Tree(parents)
    nindex = postorder(ntree)
    invpermute!(leaves, nindex)
    invpermute!(separators, nindex)
    invpermute!(nodes, nindex)
    return dissectdp(leaves, separators, nodes, alg)
end

function dissectdp(
        leaves::Vector{Bool},
        separators::Vector{View{V, Int}},
        nodes::Vector{
            Tuple{
                BipartiteGraph{V, E, View{E, Int}, View{V, Int}}, # graph
                View{W, Int},                                     # weights
                View{V, Int},                                     # label
                View{V, Int},                                     # project
                V,                                                # level
            },
        },
        alg::ND{S},
    ) where {W, V, E, S}

    graph = nodes[begin][begin]
    n = nv(graph); m = ne(graph); nn = n + one(V)
    tree = Tree{V}(n)
    vwork1 = Vector{V}(undef, n)
    vwork2 = Vector{V}(undef, half(m))
    vwork3 = Vector{V}(undef, half(m))
    ework1 = Vector{E}(undef, nn)
    ework2 = Vector{E}(undef, nn)
    ework3 = Vector{E}(undef, nn)
    stack = Tuple{Int, Vector{V}, Vector{V}}[]

    if isone(S)
        wwork1 = Vector{W}(undef, n)
    end

    @inbounds for (i, (leaf, order2, (graph, weights, label, project, level))) in enumerate(zip(leaves, separators, nodes))
        if leaf
            order, index = permutation(weights, graph, alg.alg)
        else
            n = nv(graph)
            n2 = convert(V, length(order2))
            t = zero(V); order = Vector{V}(undef, n); index = Vector{V}(undef, n)

            for (i0, order0, index0) in (pop!(stack), pop!(stack))
                #      i
                #    ↗   ↖
                # i0       i1
                graph0, weights0, label0, project0, level0 = nodes[i0]
                n0 = nv(graph0); m0 = ne(graph0); nn0 = n0 + one(V)

                # permute graph
                count0 = resize!(ework1, nn0)
                upper0 = BipartiteGraph(n0, resize!(ework2, nn0), resize!(vwork2, half(m0)))
                lower0 = BipartiteGraph(n0, resize!(ework3, nn0), resize!(vwork3, half(m0)))
                sympermute!(count0, upper0, graph0, index0, Forward)
                reverse!(count0, lower0, upper0)

                # construct elimination tree
                tree0 = resize!(tree, n0)
                etree!(tree0, resize!(vwork1, n0), upper0)

                # construct clique
                t0 = zero(V); clique0 = resize!(vwork1, n2)

                for v in order2
                    t0 += one(V); clique0[t0] = index0[project0[v]]
                end

                # move clique to end of ordering
                invpermute!(order0, compositerotations(lower0, tree0, clique0))

                # append to parent ordering
                for t0 in oneto(n0 - n2)
                    v = label0[order0[t0]]
                    t += one(V); order[t] = v; index[v] = t
                end
            end

            # append separator to ordering
            for v in order2
                t += one(V); order[t] = v; index[v] = t
            end

            # run inner algorithm and choose the best ordering
            f = resize!(vwork2, n); findex = resize!(vwork3, n)

            if isone(S)
                counts = resize!(wwork1, n)
                order, index = bestwidth!(f, findex, counts, weights, graph, (order, index), (alg.alg,))
            elseif istwo(S)
                order, index = bestfill!(f, findex, weights, graph, (order, index), (alg.alg,))
            end
        end

        push!(stack, (i, order, index))
    end

    i, order, index = only(stack)
    return order, index
end

function allocate!(work::Vector, n::Integer)
    m::Int = n
    resize!(work, length(work) + m)
    return @view work[(end - m + 1):end]
end

function sat(graph, upperbound::Integer, ::Val{H}) where {H}
    return sat(BipartiteGraph(graph), upperbound, Val(H))
end

function sat(graph::AbstractGraph, upperbound::I, ::Val{H}) where {I <: Integer, H}
    weights = ones(Int32, nv(graph))
    order, width = sat(weights, graph, upperbound + one(I), Val(H))
    return order, width - one(Int32)
end

function sat(weights::AbstractVector, graph, upperbound::Integer, ::Val{H}) where {H}
    return sat(weights, BipartiteGraph(graph), upperbound, Val(H))
end

# Encoding Treewidth into SAT
# Samer and Veith
#
# SAT-Based Approaches to Treewidth Computation: An Evaluation
# Berg and Järvisalo
#
# Jdrasil: A Modular Library for Computing Tree Decompositions
# Bannach, Berndt, and Ehlers
function sat(weights::AbstractVector, graph::AbstractGraph{V}, upperbound::Integer, ::Val{H}) where {V, H}
    @argcheck !isnegative(upperbound)
    n = Int32(nv(graph))

    # compute a maximal clique
    clique = maximalclique(weights, graph, Val(H))

    # compute false twins
    partition = twins(graph, Val(false))

    # run solver
    matrix, width = open(Solver{H}) do solver
        # compute total weight
        m = zero(Int32)

        for i in oneto(n)
            m += Int32(weights[i])
        end

        ord = Matrix{Int32}(undef, n, n)
        arc = Matrix{Int32}(undef, n, m)

        # define ord and arc variables
        for i in oneto(n)
            jj = n

            for j in oneto(n)
                # ord variables
                if i < j
                    resize!(solver, length(solver) + one(Int32))
                    ord[i, j] = length(solver)
                elseif i > j
                    ord[i, j] = -ord[j, i]
                else
                    ord[i, j] = zero(Int32)
                end

                # arc variables
                resize!(solver, length(solver) + one(Int32))
                arc[i, j] = length(solver)

                # duplicate arc variables
                weight = Int32(weights[j])

                for _ in oneto(weight - one(Int32))
                    jj += one(Int32)
                    arc[i, jj] = length(solver)
                end
            end
        end

        # base encoding
        for i in oneto(n), j in oneto(n), k in oneto(n)
            if i != j && j != k && k != i
                # ord(i, j) ∧ ord(j, k) → ord(i, k)
                clause!(solver, -ord[i, j], -ord[j, k], ord[i, k])

                # arc(i, j) ∧ arc(i, k) → arc(j, k) ∨ arc(k, j)
                clause!(solver, -arc[i, j], -arc[i, k], arc[j, k], arc[k, j])
            end
        end

        for i in oneto(n), j in neighbors(graph, i)
            if i < j
                # arc(i, j) ∨ arc(j, i)
                clause!(solver, arc[i, j], arc[j, i])
            end
        end

        for i in oneto(n)
            # arc(i, i)
            clause!(solver, arc[i, i])

            for j in oneto(n)
                if i != j
                    # ord(i, j) → -arc(j, i)
                    clause!(solver, -ord[i, j], -arc[j, i])

                    if i < j
                        # ord(i, j) → ¬ord(j, i)
                        clause!(solver, -ord[i, j], -ord[j, i])
                    end
                end
            end
        end

        # encode maximal clique
        label = zeros(Bool, n)
        label[clique] .= true

        for j in clique, i in oneto(n)
            if !label[i]
                # ord(i, j)
                clause!(solver, ord[i, j])
            elseif i < j
                # arc(i, j)
                clause!(solver, arc[i, j])
            end
        end

        # encode false twins
        for p in vertices(partition)
            for k in neighbors(partition, p), j in neighbors(partition, p)
                if j < k && !label[j] && !label[k]
                    # ord(j, k)
                    clause!(solver, ord[j, k])
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
        count = min(Int32(upperbound), m - one(Int32))

        for i in oneto(n)
            # Σ { arc(i, j) : j } <= count
            solver[arc[i, m - count]] = -one(Int32)
        end

        state = solve!(solver)

        # decrement count until unsatisfiable
        while state == :sat && !isnegative(count)
            # update cache
            for i in oneto(n), j in oneto(n)
                if i < j
                    cache[i, j] = ispositive(solver[ord[i, j]])
                elseif i > j
                    cache[i, j] = !cache[j, i]
                else
                    cache[i, j] = false
                end
            end

            # update assumption
            count -= one(Int32)

            for i in oneto(n)
                # Σ { arc(i, j) : j } <= count
                solver[arc[i, m - count]] = -one(Int32)
            end

            state = solve!(solver)
        end

        if state != :sat
            count += one(Int32)
        end

        return cache, count
    end

    order = Vector{V}(undef, n)
    sortperm!(order, oneto(n); lt = (i, j) -> matrix[i, j])
    return order, width
end

# compute a maximal clique
function maximalclique(weights::AbstractVector, graph::AbstractGraph, ::Val{H}) where {H}
    clique = open(Solver{H}, nv(graph)) do solver
        n = length(solver)
        label = zeros(Int32, n); tag = zero(Int32)

        # compute total weight
        m = zero(Int32)

        for i in oneto(n)
            m += Int32(weights[i])
        end

        # define variables
        var = Vector{Int32}(undef, m)
        ii = n

        for i in oneto(n)
            var[i] = i

            # duplicate variables
            weight = Int32(weights[i])

            for _ in oneto(weight - one(Int32))
                ii += one(Int32)
                var[ii] = i
            end
        end

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
        sortingnetwork!(solver, var)

        # initialize stack
        num = zero(Int32)
        stack = Vector{Int32}(undef, n)

        # initialize assumption
        count = zero(Int32)

        # Σ { var(i) : i } > count
        solver[var[m - count]] = one(Int32)
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

            # Σ { var(i) : i } > count
            solver[var[m - count]] = one(Int32)
            state = solve!(solver)
        end

        return resize!(stack, num)
    end

    return clique
end

# A Synthetis on Partition Refinement: A Useful Routine
# for Strings, Graphs, Boolean Matrices, and Automata
# Habib, Paul, and Viennot
# Algorithm 2: Computing twins
function twins(graph::AbstractGraph{V}, ::Val{T}) where {V, T}
    n = nv(graph)

    # bucket queue data structure
    head = zeros(V, n)
    prev = Vector{V}(undef, n)
    next = Vector{V}(undef, n)

    function set(i)
        @inbounds h = @view head[i]
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
    @inbounds for i in oneto(n)
        pnum += one(V)
        pstack[pnum] = i
    end

    if ispositive(n)
        i = pstack[pnum]
        pnum -= one(V)
        pushfirst!(list, i)
        prepend!(set(i), vertices(graph))
    end

    @inbounds for v in vertices(graph)
        tag += one(V)

        if T
            label[v] = tag
        end

        for w in neighbors(graph, v)
            if v != w
                label[w] = tag
            end
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

    # represent twin partition as a bipartite graph
    partition = BipartiteGraph{V, V}(n, n - pnum, n)
    pointers(partition)[begin] = p = j = one(V)

    @inbounds for i in list
        for v in set(i)
            targets(partition)[p] = v
            p += one(V)
        end

        j += one(V)
        pointers(partition)[j] = p
    end

    return partition
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
    tol = tolerance(W)

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
    tol = tolerance(W)
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

function compress(graph, type::Val)
    return compress(BipartiteGraph(graph), type)
end

function compress(graph::AbstractGraph{V}, type::Val) where {V}
    weights = ones(V, nv(graph))
    return compress(weights, graph, type)
end

function compress(weights::AbstractVector, graph, type::Val)
    return compress(weights, BipartiteGraph(graph), type)
end

# Engineering Data Reduction for Nested Dissection
# Ost, Schulz, Strash
# Reduction 2 (Indistinguishable node reduction)
# Reduction 3 (Twin Reduction)
function compress(weights::AbstractVector{W}, graph::AbstractGraph{V}, type::Val) where {W, V}
    n = nv(graph)
    m = ne(graph)

    if !is_directed(graph)
        m = twice(m)
    end

    marker = zeros(V, n); tag = zero(V)

    # compute twins
    partition = twins(graph, type)

    # compute weights and projection
    cn = nv(partition)
    project = Vector{V}(undef, n)
    cweights = Vector{W}(undef, cn)

    for i in vertices(partition)
        weight = zero(W)

        for v in neighbors(partition, i)
            project[v] = i
            weight += weights[v]
        end

        cweights[i] = weight
    end

    # compress graph
    E = etype(graph)
    cgraph = BipartiteGraph{V, E}(cn, cn, m)
    pointers(cgraph)[begin] = p = one(E)

    for i in vertices(partition)
        marker[i] = tag += one(V)

        @assert !isempty(neighbors(partition, i))

        for w in neighbors(graph, first(neighbors(partition, i)))
            k = project[w]

            if marker[k] < tag
                marker[k] = tag
                targets(cgraph)[p] = k
                p += one(E)
            end
        end

        pointers(cgraph)[i + one(V)] = p
    end

    resize!(targets(cgraph), p - one(E))
    return cweights, cgraph, partition
end

function saferules(graph, width::Integer)
    return saferules(BipartiteGraph(graph), width)
end

function saferules(graph::AbstractGraph{V}, width::Integer) where {V}
    weights = ones(V, nv(graph))
    return saferules(weights, graph, V(width) + one(V))
end

function saferules(weights::AbstractVector{<:Number}, graph, width::Number)
    return saferules(weights, BipartiteGraph(graph), width)
end

function saferules(weights::AbstractVector{W}, graph::AbstractGraph{V}, width::Number) where {W <: Number, V}
    return saferules(weights, graph, W(width))
end

function saferules(weights::AbstractVector{W}, graph::AbstractGraph{V}, width::W) where {W <: Number, V}
    # initialize hi
    hi = n = nv(graph)

    # apply rules
    rgraph, stack, inject, width = pr4(weights, graph, width)

    # compress graph
    weights, graph, project = compress(view(weights, inject), rgraph, Val(true))

    # initialize lo
    lo = nv(project); m = ne(project)

    # initialize index
    index = BipartiteGraph{V, V}(n, lo, m)
    pointers(index)[begin] = p = one(V)

    for v in vertices(project)
        for w in neighbors(project, v)
            targets(index)[p] = inject[w]
            p += one(V)
        end

        pointers(index)[v + one(V)] = p
    end

    # repeat until exhaustion
    while lo < hi
        # update hi
        hi = lo

        # apply rules
        rgraph, newstack, inject, width = pr4(weights, graph, width)

        # update stack
        for w in newstack, x in neighbors(index, w)
            push!(stack, x); m -= one(V)
        end

        # compress graph
        weights, graph, project = compress(view(weights, inject), rgraph, Val(true))

        # update lo
        lo = nv(project)

        # update index
        newindex = BipartiteGraph{V, V}(n, lo, m)
        pointers(newindex)[begin] = p = one(V)

        for v in vertices(project)
            for w in neighbors(project, v), x in neighbors(index, inject[w])
                targets(newindex)[p] = x
                p += one(V)
            end

            pointers(newindex)[v + one(V)] = p
        end

        index = newindex
    end

    return weights, graph, stack, index, width
end


function connectedcomponents(graph)
    return connectedcomponents(BipartiteGraph(graph))
end

function connectedcomponents(graph::AbstractGraph{V}) where {V}
    E = etype(graph)
    n = nv(graph)
    m = is_directed(graph) ? ne(graph) : twice(ne(graph))
    components = View{V, V}[]
    subgraphs = BipartiteGraph{V, E, View{E, V}, View{V, E}}[]
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

function bestwidth(graph, alg::PermutationOrAlgorithm, algs::Tuple)
    return bestwidth(BipartiteGraph(graph), alg, algs)
end

function bestwidth(graph::AbstractGraph{V}, alg::PermutationOrAlgorithm, algs::Tuple) where {V}
    n = nv(graph)
    f = Vector{V}(undef, n)
    findex = Vector{V}(undef, n)
    counts = Vector{V}(undef, n)
    return bestwidth!(f, findex, counts, graph, alg, algs)
end

function bestwidth!(
        f::Vector{V},
        findex::Vector{V},
        counts::Vector{V},
        graph::AbstractGraph{V},
        alg::PermutationOrAlgorithm,
        algs::Tuple
    ) where {V}

    minorder, minindex = permutation(graph, alg)
    minwidth = treewidth!(f, findex, counts, graph, minorder, minindex)

    for alg in algs
        order, index = permutation(graph, alg)
        width = treewidth!(f, findex, counts, graph, order, index)

        if width < minwidth
            minorder, minindex, minwidth = order, index, width
        end
    end

    return minorder, minindex
end

function bestwidth(weights::AbstractVector, graph, alg::PermutationOrAlgorithm, algs::Tuple)
    return bestwidth(weights, BipartiteGraph(graph), alg, algs)
end

function bestwidth(weights::AbstractVector{W}, graph::AbstractGraph{V}, alg::PermutationOrAlgorithm, algs::Tuple) where {W, V}
    n = nv(graph)
    f = Vector{V}(undef, n)
    findex = Vector{V}(undef, n)
    counts = Vector{W}(undef, n)
    return bestwidth!(f, findex, counts, weights, graph, alg, algs)
end

function bestwidth!(
        f::Vector{V},
        findex::Vector{V},
        counts::Vector{W},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        alg::PermutationOrAlgorithm,
        algs::Tuple
    ) where {W, V}

    minorder, minindex = permutation(weights, graph, alg)
    minwidth = treewidth!(f, findex, counts, weights, graph, minorder, minindex)

    for alg in algs
        order, index = permutation(weights, graph, alg)
        width = treewidth!(f, findex, counts, weights, graph, order, index)

        if width < minwidth
            minorder, minindex, minwidth = order, index, width
        end
    end

    return minorder, minindex
end

function bestfill(graph, alg::PermutationOrAlgorithm, algs::Tuple)
    return bestfill(BipartiteGraph(graph), alg, algs)
end

function bestfill(graph::AbstractGraph{V}, alg::PermutationOrAlgorithm, algs::Tuple) where {V}
    n = nv(graph)
    f = Vector{V}(undef, n)
    findex = Vector{V}(undef, n)
    return bestfill!(f, findex, graph, alg, algs)
end

function bestfill!(
        f::Vector{V},
        findex::Vector{V},
        graph::AbstractGraph{V},
        alg::PermutationOrAlgorithm,
        algs::Tuple
    ) where {V}

    minorder, minindex = permutation(graph, alg)
    minfill = treefill!(f, findex, graph, minorder, minindex)

    for alg in algs
        order, index = permutation(graph, alg)
        fill = treefill!(f, findex, graph, order, index)

        if fill < minfill
            minorder, minindex, minfill = order, index, fill
        end
    end

    return minorder, minindex
end

function bestfill(weights::AbstractVector, graph, alg::PermutationOrAlgorithm, algs::Tuple)
    return bestfill(weights, BipartiteGraph(graph), alg, algs)
end

function bestfill(weights::AbstractVector{W}, graph::AbstractGraph{V}, alg::PermutationOrAlgorithm, algs::Tuple) where {W, V}
    n = nv(graph)
    f = Vector{V}(undef, n)
    findex = Vector{V}(undef, n)
    return bestfill!(f, findex, weights, graph, alg, algs)
end

function bestfill!(
        f::Vector{V},
        findex::Vector{V},
        weights::AbstractVector{W},
        graph::AbstractGraph{V},
        alg::PermutationOrAlgorithm,
        algs::Tuple
    ) where {W, V}

    minorder, minindex = permutation(weights, graph, alg)
    minfill = treefill!(f, findex, weights, graph, minorder, minindex)

    for alg in algs
        order, index = permutation(weights, graph, alg)
        fill = treefill!(f, findex, weights, graph, order, index)

        if fill < minfill
            minorder, minindex, minfill = order, index, fill
        end
    end

    return minorder, minindex
end

function Base.show(io::IO, ::MIME"text/plain", alg::A) where {A <: EliminationAlgorithm}
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

function Base.show(io::IO, ::MIME"text/plain", alg::ND{S, A, D}) where {S, A, D}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "ND{$S, $A, $D}:")
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.alg)
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.dis)
    println(io, " "^indent * "    limit: $(alg.limit)")
    println(io, " "^indent * "    level: $(alg.level)")
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

function Base.show(io::IO, ::MIME"text/plain", alg::SAT{H, A}) where {H, A}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "SAT{$H, $A}:")
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.alg)
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

function Base.show(io::IO, ::MIME"text/plain", alg::SafeRules{A, L, U}) where {A, L, U}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "SafeRules{$A, $L, $U}:")
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.alg)
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.lb)
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.ub)
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::SafeSeparators{A, M}) where {A, M}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "SafeSeparators{$A, $M}:")
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.alg)
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.min)
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::ConnectedComponents{A}) where {A}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "ConnectedComponents{$A}:")
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.alg)
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::BestWidth{A}) where {A}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "BestWidth{$A}:")

    for alg in alg.algs
        show(IOContext(io, :indent => indent + 4), "text/plain", alg)
    end

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
