"""
    EliminationAlgorithm

A *graph elimination algorithm* computes a total ordering of the vertices of a graph. 
The algorithms implemented in CliqueTrees.jl can be divided into five categories.

  - triangulation recognition algorithms
  - bandwidth minimization algorithms
  - local algorithms
  - global algorithms
  - exact treewidth algorithms

# Triangulation Recognition Algorithms

| type             | name                                         | time     | space    | package |
|:-----------------|:-------------------------------------------- |:-------- |:-------- | :------ |
| [`MCS`](@ref)    | maximum cardinality search                   | O(m + n) | O(n)     |         |
| [`LexBFS`](@ref) | lexicographic breadth-first search           | O(m + n) | O(m + n) |         |
| [`MCSM`](@ref)   | maximum cardinality search (minimal)         | O(mn)    | O(n)     |         |
| [`LexM`](@ref)   | lexicographic breadth-first search (minimal) | O(mn)    | O(n)     |         |

These algorithms will compute perfect orderings when applied to chordal graphs.

# Bandwidth Minimization Algorithms

| type            | name                                   | time     | space    | package |
|:----------------|:---------------------------------------|:-------- |:-------- | :------ |
| [`RCMMD`](@ref) | reverse Cuthill-Mckee (minimum degree) | O(m + n) | O(m + n) |         | 
| [`RCMGL`](@ref) | reverse Cuthill-Mckee (George-Liu)     | O(m + n) | O(m + n) |         |

These algorithms try to minimize the *bandwidth* and *envelope* of the ordered graph.

# Local Algorithms

| type             | name                              | time   | space    | package                                                   |
|:-----------------|:----------------------------------|:-------|:-------- | :-------------------------------------------------------- |
| [`MMD`](@ref)    | multiple minimum degree           | O(mn²) | O(m + n) |                                                           |
| [`MD`](@ref)     | minimum degree                    | O(mnΔ) | O(m + n) |                                                           |
| [`MF`](@ref)     | minimum fill                      | O(mn²) |          |                                                           |
| [`AMD`](@ref)    | approximate minimum degree        | O(mn)  | O(m + n) | [AMD.jl](https://github.com/JuliaSmoothOptimizers/AMD.jl) |
| [`SymAMD`](@ref) | column approximate minimum degree | O(mn)  | O(m + n) | [AMD.jl](https://github.com/JuliaSmoothOptimizers/AMD.jl) |
| [`AMF`](@ref)    | approximate minimum fill          | O(mn)  | O(m + n) |                                                           |

These algorithms simulate the graph elimination process, greedily eliminating
vertices that minimize a cost function. They are faster then the global
algorithms, but have worse results.

# Global Algorithms

| type             | name              | time   | space | package                                             |
|:-----------------|:------------------|:-------|:----- | :-------------------------------------------------- |
| [`METIS`](@ref)  | nested dissection |        |       | [Metis.jl](https://github.com/JuliaSparse/Metis.jl) |
| [`SCOTCH`](@ref) | nested dissection |        |       | [Scotch.jl](https://github.com/Keluaa/Scotch.jl)    |
| [`ND`](@ref)     | nested dissection |        |       |                                                     |

These algorithms recursively partition a graph, then call a local algorithm on the leaves.
These are slower than the local algorithms, but have better results.

# Anytime Treewidth Heuristics

| type           | name                        | time  | space | package |
|:---------------|:----------------------------|:------|:------|:------- |
| [`HBT`](@ref)  | heuristic Bouchitte-Todinca |       |       |         |

These algorithms improve a tree decomposition for as long as they are allowed to run.

# Exact Treewidth Algorithms

| type          | name                  | time  | space | package                                                                 |
|:--------------|:----------------------|:------|:------| :---------------------------------------------------------------------- |
| [`BT`](@ref)  | Bouchitte-Todinca     |       |       | [TreeWidthSolver.jl](https://github.com/ArrogantGao/TreeWidthSolver.jl) |
| [`RTW`](@ref) | contraction-recursive |       |       |                                                                         |

The orderings computed by these algorithms induce minimum-width tree decompositions.

!!! warning
    Exact treewidth is an NP-hard problem.
"""
abstract type EliminationAlgorithm end

"""
    MinimalAlgorithm <: EliminationAlgorithm
"""
abstract type MinimalAlgorithm <: EliminationAlgorithm end

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
"""
struct BFS <: EliminationAlgorithm end

"""
    MCS <: EliminationAlgorithm

    MCS()

The maximum cardinality search algorithm.

### References

  - Tarjan, Robert E., and Mihalis Yannakakis. "Simple linear-time algorithms to test chordality of graphs, test acyclicity of hypergraphs, and selectively reduce acyclic hypergraphs." *SIAM Journal on Computing* 13.3 (1984): 566-579.
"""
struct MCS <: EliminationAlgorithm end

"""
    LexBFS <: EliminationAlgorithm

    LexBFS()

The [lexicographic breadth-first-search algorithm](https://en.wikipedia.org/wiki/Lexicographic_breadth-first_search).

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

### References

  - Rose, Donald J., R. Endre Tarjan, and George S. Lueker. "Algorithmic aspects of vertex elimination on graphs." *SIAM Journal on Computing* 5.2 (1976): 266-283.
"""
struct LexM <: MinimalAlgorithm end

"""
    MCSM <: EliminationAlgorithm

    MCSM()

A minimal variant of the maximal cardinality search algorithm.

### References

  - Berry, Anne, et al. "Maximum cardinality search for computing minimal triangulations of graphs." *Algorithmica* 39 (2004): 287-298.
"""
struct MCSM <: MinimalAlgorithm end

"""
    AMF <: EliminationAlgorithm

    AMF(; exact=false)

The approximate minimum fill algorithm.

### Parameters

  - `exact`: compute the initial score of each vertex from its exact fill-in, rather than its degree

### References

  - Rothberg, Edward, and Stanley C. Eisenstat. "Node selection strategies for bottom-up sparse matrix ordering." SIAM Journal on Matrix Analysis and Applications 19.3 (1998): 682-695.
"""
@kwdef struct AMF <: EliminationAlgorithm
    exact::Bool = false
end

"""
    MF <: EliminationAlgorithm

    MF(; strategy=0)

The greedy minimum fill algorithm.

### Parameters

  - `strategy`: node-selection score (`0` for minimum fill, `1` for minimum average fill)

### References

  - Ng, Esmond G., and Barry W. Peyton. "Fast implementation of the minimum local fill ordering heuristic." *CSC14: The Sixth SIAM Workshop on Combinatorial Scientific Computing.* 2014.
  - Otsuka, Hiromu, et al. "Experimental evaluation of greedy treewidth heuristics on huge graphs." *IPSJ SIG Technical Report* 2018-AL-166-12 (2018). (In Japanese.)
  - Tamaki, Hisao. "A heuristic for listing almost-clique minimal separators of a graph." *arXiv preprint* arXiv:2108.07551 (2021).
"""
@kwdef struct MF <: EliminationAlgorithm
    strategy::Int = 0
end

"""
    MMD <: EliminationAlgorithm

    MMD(; delta=0)

The [multiple minimum degree algorithm](https://en.wikipedia.org/wiki/Minimum_degree_algorithm).

### Parameters

  - `delta`: tolerance for multiple elimination

### References

  - Liu, Joseph WH. "Modification of the minimum-degree algorithm by multiple elimination." *ACM Transactions on Mathematical Software (TOMS)* 11.2 (1985): 141-153.
"""
@kwdef struct MMD <: EliminationAlgorithm
    delta::Int = 0
end

"""
    MD <: EliminationAlgorithm

    MD(; mass=false, external=false)

The [minimum degree algorithm](https://en.wikipedia.org/wiki/Minimum_degree_algorithm).

### Parameters

  - `mass`: mass elimination
  - `external`: minimum external degree

### References

  - Cummings, Robert, Matthew Fahrbach, and Animesh Fatehpuria. "A fast minimum degree algorithm and matching lower bound." *Proceedings of the 2021 ACM-SIAM Symposium on Discrete Algorithms (SODA)*. 2021.
"""
@kwdef struct MD <: EliminationAlgorithm
    mass::Bool = false
    external::Bool = false
end

"""
    AMD <: EliminationAlgorithm

    AMD(; dense=10.0, aggressive=1.0)

The approximate minimum degree algorithm.

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
    SCOTCH <: EliminationAlgorithm

    SCOTCH(; strategy=:default, level=0, imbalance=0.0)

The [SCOTCH](https://gitlab.inria.fr/scotch/scotch) nested dissection ordering.
This algorithm requires the package [Scotch.jl](https://github.com/Keluaa/Scotch.jl).

### Parameters

  - `strategy`: ordering strategy (`:default`, `:balance`, `:quality`, `:safety`, or `:speed`)
  - `level`: maximum number of dissection levels (`0` for no limit)
  - `imbalance`: maximum load imbalance ratio

### References

  - Pellegrini, François, and Jean Roman. "Scotch: A software package for static mapping by dual recursive bipartitioning of process and architecture graphs." *International Conference on High-Performance Computing and Networking*. Springer, 1996.
"""
@kwdef struct SCOTCH <: EliminationAlgorithm
    strategy::Symbol = :default
    level::Int = 0
    imbalance::Float64 = 0.0
end

"""
    ND{S, A, D} <: EliminationAlgorithm

    ND{S}(alg::EliminationAlgorithm, dis::DissectionAlgorithm;
        width = 120,
        level = 6,
        imbalance = 130,
        scale = 1,
    )

The [nested dissection algorithm](https://en.wikipedia.org/wiki/Nested_dissection).
The algorithm `dis` is used to compute vertex separators, and the algorithm `alg` is called on the of the separator tree.
The type parameter `S` controls the behavior of the algorithm: if `S` is equal to `1` or `2`, then `alg` is additionally called
on the branches of the separator tree. At each branch, the ordering computed by `alg` is compared to the ordering computed
by the nested dissection algorithm, and the worse of the two is discarded.

  - `1`: minimize width (slow)
  - `2`: minimize fill (slow)
  - `3`: no strategy (fast)

CliqueTrees currently has two vertex separator algorithms, both of which require loading an external package.

| type                | name                               | package                                              |
|:--------------------|:-----------------------------------|:-----------------------------------------------------|
| [`METISND`](@ref)   | multilevel vertex separation       | [Metis.jl](https://github.com/JuliaSparse/Metis.jl)  |
| [`KaHyParND`](@ref) | multilevel hypergraph partitioning | [KayHyPar.jl](https://github.com/kahypar/KaHyPar.jl) |

The algorithm `KaHyParND` computes a vertex separator indirectly, by partitioning a quasi-clique-cover of the original graph.
The parameters `width` and `level` control the recursion depth of the algorithm, and the parameter `imbalance` controls the
maximum imbalance of the vertex separator.

### Parameters

  - `S`: strategy
    - `1`: minimize width (slow)
    - `2`: minimize fill (slow)
    - `3`: no strategy (fast)
  - `alg`: elimination algorithm
  - `dis`: separation algorithm
  - `width`: minimum width
  - `level`: maximum level
  - `imbalance`: separator imbalance
  - `scale`: weight scale; the separator algorithms take integer weights, so
    floating-point weights are multiplied by `scale` and truncated
"""
struct ND{S, A <: EliminationAlgorithm, D <: DissectionAlgorithm} <: EliminationAlgorithm
    alg::A
    dis::D
    width::Int
    level::Int
    imbalance::Int
    scale::Int
end

function ND{S}(
        alg::A = DEFAULT_ELIMINATION_ALGORITHM, dis::D = DEFAULT_DISSECTION_ALGORITHM;
        width::Integer = 120,
        level::Integer = 6,
        imbalance::Integer = 130,
        scale::Integer = 1,
    ) where {S, A, D}
    @assert ispositive(width)
    @assert ispositive(level)
    @assert ispositive(imbalance)
    @assert ispositive(scale)
    return ND{S, A, D}(alg, dis, width, level, imbalance, scale)
end

function ND(args...; kwargs...)
    return ND{1}(args...; kwargs...)
end

struct NDS{S, A <: EliminationAlgorithm, D <: DissectionAlgorithm} <: EliminationAlgorithm
    alg::A
    dis::D
    width::Int
    level::Int
    imbalances::StepRange{Int, Int}
    scale::Int
end

function NDS{S}(
        alg::A = DEFAULT_ELIMINATION_ALGORITHM, dis::D = DEFAULT_DISSECTION_ALGORITHM;
        width::Integer = 120,
        level::Integer = 6,
        imbalances::AbstractRange = 130:130,
        scale::Integer = 1,
    ) where {S, A, D}
    @assert ispositive(width)
    @assert ispositive(level)
    @assert ispositive(first(imbalances))
    @assert ispositive(scale)
    return NDS{S, A, D}(alg, dis, width, level, imbalances, scale)
end

function NDS(args...; kwargs...)
    return NDS{1}(args...; kwargs...)
end

"""
    FlowCutter <: EliminationAlgorithm

    FlowCutter(; time=5, seed=0)

The FlowCutter algorithm.

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

### References

  - Korhonen, Tuukka, Jeremias Berg, and Matti Järvisalo. "Solving Graph Problems via Potential Maximal Cliques: An Experimental Evaluation of the Bouchitté-Todinca Algorithm." *Journal of Experimental Algorithmics (JEA)* 24 (2019): 1-19.
"""
struct BT <: EliminationAlgorithm end

"""
    PIDBT{A} <: EliminationAlgorithm

    PIDBT(lb::WidthOrAlgorithm; target=0)

    PIDBT(; lb=MMW(), target=0)

An exact algorithm for treewidth, by dynamic programming over minimal separators. It tries
the widths k from a lower bound up, each with the minimal separators of weight less than k
(Tamaki's separator-based algorithm).

### Parameters

  - `lb`: lower bound algorithm or lower bound
  - `target`: the algorithm stops at the first tree decomposition of width at most `target`; if
    there is none, the tree decomposition is optimal

Widths are those of [`treewidth`](@ref) with vertex weights: the largest weight of a bag. With
unit weights, that is the treewidth plus one.

### References

  - Tamaki, Hisao. "Computing treewidth via exact and heuristic lists of minimal separators." *17th International Symposium on Experimental Algorithms (SEA 2019)*. 2019.
"""
struct PIDBT{A <: WidthOrAlgorithm} <: EliminationAlgorithm
    lb::A
    target::Int
end

function PIDBT(lb::WidthOrAlgorithm; target::Integer = 0)
    return PIDBT(lb, convert(Int, target))
end

function PIDBT(; lb::WidthOrAlgorithm = DEFAULT_LOWER_BOUND_ALGORITHM, target::Integer = 0)
    return PIDBT(lb, convert(Int, target))
end

"""
    HBT{A, L, I} <: EliminationAlgorithm

    HBT(; time=10.0, seed=1, alg=MF(; strategy=1), lb=MMW(), kwargs...)

The heuristic Bouchitte-Todinca algorithm: an anytime heuristic for small-width tree decompositions.

A solution is a set of potential maximal cliques. Its value is the smallest width of a tree
decomposition whose bags all belong to the set, computed by the Bouchitte-Todinca dynamic program.
Solutions are improved by adding potential maximal cliques taken from triangulations of local
graphs, which are computed exactly when they are small (by the safe reduction rules and safe
separators, then dynamic programming over minimal separators, as in [`PIDBT`](@ref)) and greedily
with `alg` otherwise. Two strategies are used:

  - *diversification* (Tamaki 2019) picks one of the largest bags of the current decomposition and a random
    subtree around it, and re-triangulates the corresponding local graph while forcing that bag
    apart;
  - *merging* (Tamaki 2022) combines the solution with an independently generated solution of no
    greater width. It is used when diversification stalls.

Progress is measured by refined widths: a decomposition improves if it has a smaller largest bag,
or as large a largest bag but fewer of them. The algorithm runs until `time` seconds have elapsed
or the width reaches the lower bound computed by `lb`. Vertex weights need not be integers.

Given a batch of graphs (see [`batch_permutation`](@ref)), as by [`SafeSeparators`](@ref) or
[`ConnectedComponents`](@ref), the algorithm spends one time budget on the batch: on its widest graph,
until it is no longer the widest. Graphs with at most `base` vertices are triangulated exactly when
they are first the widest.

### Parameters

  - `time`: time budget in seconds (for a whole batch)
  - `seed`: random seed
  - `alg`: elimination algorithm used for greedy triangulations (made minimal with [`MinimalChordal`](@ref));
    the default is minimum average fill, as in the reference implementation
  - `lb`: lower bound algorithm or lower bound (used as a stopping criterion)
  - `init`: an elimination algorithm or ordering, or `nothing`; its minimal triangulation seeds the first
    solution, along with the greedy ones (a strong one, such as [`NDS`](@ref), is often a much better start).
    An ordering is one of a single graph, so it is not used for a batch
  - `ninit`: number of randomized greedy triangulations used to seed each solution (at least one without `init`)
  - `base`: local graphs with at most this many vertices are triangulated exactly
  - `margin`: local graphs around a bag with b vertices are also triangulated exactly if they have
    at most b + `margin` vertices
  - `pexact`: probability that a diversified region is drawn from just above the size of its bag
    (at most `margin` more vertices), where exact triangulation is fast
  - `ntry`: number of local graphs triangulated per merge
  - `refined`: measure progress by refined widths
  - `merge`: use merging
  - `diversify`: use diversification
  - `dsize`: maximum size of a diversified region (other sizes are drawn log-uniformly)
  - `nsep`: number of separators tried per diversification
  - `near`: diversification works around a bag of the largest weight k with probability 1/2,
    around a bag in the next band below k (as wide as the smallest vertex weight) with probability
    1/4, and so on, down `near` bands; with unit weights, the bands are the weights k - 1, k - 2, ...
  - `patience`: number of steps without progress before merging
  - `xtime`, `xmin`, `xstep`: time limit in seconds for each exact triangulation of a local graph (the greedy algorithm
    is used if it runs out). The limit starts at `xmin`, and doubles every `xstep` steps without a decrease of the
    width, up to `xtime`: most exact triangulations take milliseconds, and the slow ones are worth waiting for only
    once the fast ones have stopped making progress
  - `threads`: number of tasks: the solution is diversified by `threads - 1` workers (on Julia threads)
    as well as by the main task
  - `verbose`: print progress

### Example

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

julia> treewidth(graph; alg=HBT(; time=1.0))
2
```

### References

  - Tamaki, Hisao. "A heuristic use of dynamic programming to upperbound treewidth." arXiv preprint arXiv:1909.07647 (2019).
  - Tamaki, Hisao. "Heuristic computation of exact treewidth." *20th International Symposium on Experimental Algorithms (SEA 2022)*. 2022.
"""
@kwdef struct HBT{A <: EliminationAlgorithm, L <: WidthOrAlgorithm, I} <: EliminationAlgorithm
    time::Float64 = 10.0
    seed::Int = 1
    alg::A = MF(; strategy = 1)
    lb::L = DEFAULT_LOWER_BOUND_ALGORITHM
    init::I = nothing
    ninit::Int = 10
    base::Int = 60
    ntry::Int = 50
    refined::Bool = true
    merge::Bool = true
    diversify::Bool = true
    dsize::Int = typemax(Int)
    nsep::Int = 4
    patience::Int = 100
    xtime::Float64 = 1.0
    xmin::Float64 = 0.05
    xstep::Int = 8
    threads::Int = Threads.nthreads()
    near::Int = 3
    margin::Int = 40
    pexact::Float64 = 0.5
    verbose::Bool = false
end

"""
    RTW{A, H, L} <: EliminationAlgorithm

    RTW(; time=Inf, alg=MF(; strategy=1), hbt=HBT(; time=2.0), ticks=100, lb=MMW(), verbose=false)

The contraction-recursive algorithm: an exact algorithm for treewidth that certifies its answer.

Starting from a heuristic decomposition, the algorithm asks, again and again, whether the graph has a
lighter decomposition. It answers by recursing on contractions of the graph by its edges. A contraction
with no decomposition of the given width shows that the graph has none; otherwise, the decompositions
of the contraction are mapped back onto the graph, where they help to build a lighter one. Only when
every edge has been tried is a question settled by an exact search ([`PIDBT`](@ref)), on a contraction.

Before the recursion, the graph gets that exact search for a tenth of the time (at most ten seconds).
It solves many graphs at once; otherwise, the widths it rules out are a lower bound, and the recursion
stops when it meets it.

The upper bounds come from two heuristic searches. The first, [`HBT`](@ref), runs on the graph for its
`time` to give the first decomposition, and for longer while it keeps improving, up to half the time; it
then goes on in the background, with a quarter of the time, and the recursion moves on whenever it finds
a lighter decomposition. The second, Tamaki's heuristic positive-instance driven search,
runs on the contractions whose children did not give a light enough decomposition, for `ticks` steps
for each child.

A contracted vertex weighs as little as the lightest of its vertices, so that no contraction is wider
than the graph. Vertex weights need not be integers. The algorithm is anytime: when `time` runs out, it
returns the lightest decomposition it has found.

### Parameters

  - `time`: time limit in seconds (the pieces of [`SafeSeparators`](@ref) or [`ConnectedComponents`](@ref) share it)
  - `alg`: elimination algorithm for the greedy decomposition that seeds the search (made minimal with [`MinimalChordal`](@ref))
  - `hbt`: parameters of the first heuristic search ([`HBT`](@ref)), or `nothing` for none
  - `ticks`: steps of heuristic search for each child of a contraction
  - `lb`: lower bound algorithm or lower bound (the search stops when the width reaches it)
  - `verbose`: print progress

### Example

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

julia> treewidth(graph; alg=RTW(; time=1.0))
2
```

### References

  - Tamaki, Hisao. "A contraction-recursive algorithm for treewidth." *18th International Symposium on Parameterized and Exact Computation (IPEC 2023)*. 2023.
"""
@kwdef struct RTW{A <: PermutationOrAlgorithm, H <: Union{Nothing, HBT}, L <: WidthOrAlgorithm} <: EliminationAlgorithm
    time::Float64 = Inf
    alg::A = MF(; strategy = 1)
    hbt::H = HBT(; time = 2.0)
    ticks::Int = 100
    lb::L = DEFAULT_LOWER_BOUND_ALGORITHM
    verbose::Bool = false
end

"""
    MinimalChordal{A} <: EliminationAlgorithm

    MinimalChordal(alg::PermutationOrAlgorithm)

    MinimalChordal()

Evaluate an elimination algorithm, and them improve its output using the MinimalChordal algorithm. The result is guaranteed to be minimal.

### Parameters

  - `alg`: elimination algorithm

### References

  - Heggernes, Pinar, and Barry W. Peyton. "Fast computation of minimal fill inside a given elimination ordering." *SIAM journal on matrix analysis and applications* 30.4 (2009): 1424-1444.
"""
struct MinimalChordal{A <: PermutationOrAlgorithm} <: MinimalAlgorithm
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
    Compression{A} <: EliminationAlgorithm

    Compression(alg::EliminationAlgorithm; tau = 1.0)

Preprocess a graph by identifying indistinguishable vertices.
The algorithm `alg` is run on the compressed graph.

### Parameters

  - `alg`: elimination algorithm
  - `tau`: threshold parameter for graph compression


### References

  - Ashcraft, Cleve. "Compressed graphs and the minimum degree algorithm." *SIAM Journal on Scientific Computing* 16.6 (1995): 1404-1411.

"""
struct Compression{A <: EliminationAlgorithm} <: EliminationAlgorithm
    alg::A
    tau::Float64
end

function Compression(alg::EliminationAlgorithm; tau::Float64 = 1.0)
    return Compression(alg, tau)
end

function Compression(; kwargs...)
    return Compression(DEFAULT_ELIMINATION_ALGORITHM; kwargs...)
end

"""
    SafeRules{A, L, U} <: EliminationAlgorithm

    SafeRules(alg::EliminationAlgorithm, lb::WidthOrAlgorithm)

    SafeRules()

Preprocess a graph using safe reduction rules. The algorithm `lb` is used to compute a lower bound
to the treewidth; better lower bounds allow the algorithm to perform more reductions.

### Parameters

  - `alg`: elimination algorithm
  - `lb`: lower bound algorithm (used to lower bound the treiwidth)
  - `ub`: elimination algorithm (used to upper bound the treewidth)
  - `rules`: reduction level (`3`, `4`, or `5`)

### References

  - Bodlaender, Hans L., et al. "Pre-processing for triangulation of probabilistic networks." (2001).
  - Bodlaender, Hans L., Arie M.C.A. Koster, and Frank van den Eijkhof. "Preprocessing rules for triangulation of probabilistic networks." *Computational Intelligence* 21.3 (2005): 286-305.
  - van den Eijkhof, Frank, Hans L. Bodlaender, and Arie M.C.A. Koster. "Safe reduction rules for weighted treewidth." *Algorithmica* 47 (2007): 139-158. 
  - Medema, Michel, and Alexander Lazovik. "A safeness condition for minimal separators based on vertex connectivity." *Discrete Mathematics* 348 (2025): 114524.
"""
struct SafeRules{A <: EliminationAlgorithm, L <: WidthOrAlgorithm, U <: EliminationAlgorithm} <: EliminationAlgorithm
    alg::A
    lb::L
    ub::U
    rules::Int
    minfrac::Float64
end

function SafeRules(alg::EliminationAlgorithm, lb::WidthOrAlgorithm, ub::EliminationAlgorithm, rules::Integer)
    return SafeRules(alg, lb, ub, rules, 1.0e-3)
end

function SafeRules(alg::EliminationAlgorithm, lb::WidthOrAlgorithm, ub::EliminationAlgorithm; rules::Integer = 5, minfrac::Real = 1.0e-3)
    @assert !isnegative(minfrac)
    return SafeRules(alg, lb, ub, convert(Int, rules), convert(Float64, minfrac))
end

function SafeRules(alg::EliminationAlgorithm, lb::WidthOrAlgorithm; kwargs...)
    return SafeRules(alg, lb, DEFAULT_ELIMINATION_ALGORITHM; kwargs...)
end

function SafeRules(alg::EliminationAlgorithm; kwargs...)
    return SafeRules(alg, DEFAULT_LOWER_BOUND_ALGORITHM; kwargs...)
end

function SafeRules(; kwargs...)
    return SafeRules(DEFAULT_ELIMINATION_ALGORITHM; kwargs...)
end

struct SimplicialRule{A <: EliminationAlgorithm} <: EliminationAlgorithm
    alg::A
end

function SimplicialRule()
    return SimplicialRule(DEFAULT_ELIMINATION_ALGORITHM)
end

"""
    SafeSeparators{A, M} <: EliminationAlgorithm

    SafeSeparators(alg::EliminationAlgorithm, min::MinimalAlgorithm; maxmissing=0)

    SafeSeparators(alg::EliminationAlgorithm, mins::Tuple; maxmissing=0)

Apply an elimination algorithm to the atoms of a safe separator decomposition. The decomposition is
computed from minimal triangulations: the algorithm `min` (or each algorithm in `mins`, in turn) computes
one, and those of its minimal separators that are safe are filled. The atoms are decomposed again, recursively.

A minimal separator ``S`` is used if it is

  - a clique;
  - an *almost clique* (``S - v`` is a clique for some *center* ``v``) with two full components whose
    vertices are all at least as heavy as ``v``;
  - any separator with at most `maxmissing` missing edges that passes one of two tests, slower than the
    others. They are off by default (`maxmissing = 0`). Values between 25 and 100 find almost every
    separator they can find; much larger values cost a lot of time and find few more.
      - the connectivity test of Medema and Lazovik: for a clique ``K \\subseteq S`` and every component
        ``C`` of ``G - S``, the ends of every missing edge of ``S`` are joined by a flow of value
        ``w(S - K)`` in ``G - C - K``, with vertex capacities ``w``.
      - the minor test of Bodlaender and Koster: for every component ``C`` of ``G - S``, contracting
        parts of ``G - C`` turns ``N(C)`` into a clique. It is a greedy search.

With several algorithms in `mins`, more separators are found.

!!! warning
    The algorithm `min` must compute a *minimimal* ordering. This property is guaranteed by the following
    algorithms:
      - [`MCSM`](@ref)
      - [`LexM`](@ref)
      - [`MinimalChordal`](@ref)

### Parameters

  - `alg`: elimination algorithm
  - `min`: minimal elimination algorithm (or a tuple `mins` of them)
  - `maxmissing`: the largest number of missing edges in a separator given to the connectivity and minor tests (`0`: neither)

### References

  - Bodlaender, Hans L., and Arie MCA Koster. "Safe separators for treewidth." *Discrete Mathematics* 306.3 (2006): 337-350.
  - Medema, Michel, and Alexander Lazovik. "A safeness condition for minimal separators based on vertex connectivity." *Discrete Mathematics* 348 (2025): 114524.
  - Tamaki, Hisao. "A heuristic for listing almost-clique minimal separators of a graph." arXiv preprint arXiv:2108.07551 (2021).
"""
struct SafeSeparators{A <: EliminationAlgorithm, M <: Tuple} <: EliminationAlgorithm
    alg::A
    min::M
    maxmissing::Int

    function SafeSeparators{A, M}(alg::A, min::M, maxmissing::Integer) where {A <: EliminationAlgorithm, M <: Tuple}
        @assert !isnegative(maxmissing)
        return new{A, M}(alg, min, maxmissing)
    end
end

function SafeSeparators(alg::A, mins::M, maxmissing::Integer) where {A <: EliminationAlgorithm, M <: Tuple}
    return SafeSeparators{A, M}(alg, mins, maxmissing)
end

function SafeSeparators(alg::EliminationAlgorithm, mins::Tuple; maxmissing::Integer = 0)
    return SafeSeparators(alg, mins, maxmissing)
end

function SafeSeparators(alg::EliminationAlgorithm=DEFAULT_ELIMINATION_ALGORITHM, min::MinimalAlgorithm=DEFAULT_MINIMAL_ALGORITHM; maxmissing::Integer = 0)
    return SafeSeparators(alg, (min,), maxmissing)
end

"""
    ConnectedComponents{N, A} <: EliminationAlgorithm

    ConnectedComponents{N}(alg::PermutationOrAlgorithm)

    ConnectedComponents(alg::PermutationOrAlgorithm)

    ConnectedComponents()

Apply an elimination algorithm to the `N`-connected components of a graph, for `N = 1` (the
default), `2`, or `3`. Each `N` splits the components of `N - 1` further.

  - `N = 1`: the connected components.
  - `N = 2`: the blocks (biconnected components), which meet at cut vertices.
  - `N = 3`: the triconnected components of each block (its Tutte decomposition), which meet at
    separation pairs ``\\{a, b\\}``; the edge ``ab`` is added. A separation pair is used if
    ``a`` and ``b`` are adjacent, or if, for every component ``C`` of ``G - \\{a, b\\}``, the
    vertex-capacitated flow from ``a`` to ``b`` in ``G - C`` is at least ``\\min(w(a), w(b))``
    (Medema and Lazovik, Theorem 16). With unit weights, every separation pair is used; with other
    weights, the components may be coarser.

The separators are safe, so the treewidth of a graph is the largest treewidth of its components.
The decomposition takes linear time (with weights and `N = 3`, plus the flow tests).

### Parameters

  - `N`: connectivity (`1`, `2`, or `3`)
  - `alg`: elimination algorithm

### References

  - Hopcroft, John, and Robert Tarjan. "Algorithm 447: efficient algorithms for graph manipulation." *Communications of the ACM* 16.6 (1973): 372-378.
  - Bourneuf, Romain, and Tim Planken. "A Structural Linear-Time Algorithm for Computing the Tutte Decomposition." arXiv preprint arXiv:2508.06212 (2025).
  - Medema, Michel, and Alexander Lazovik. "A safeness condition for minimal separators based on vertex connectivity." *Discrete Mathematics* 348 (2025): 114524.
"""
struct ConnectedComponents{N, A <: PermutationOrAlgorithm} <: EliminationAlgorithm
    alg::A

    function ConnectedComponents{N, A}(alg::A) where {N, A <: PermutationOrAlgorithm}
        @assert N isa Int && 1 <= N <= 3
        return new{N, A}(alg)
    end
end

function ConnectedComponents{N}(alg::A) where {N, A <: PermutationOrAlgorithm}
    return ConnectedComponents{N, A}(alg)
end

function ConnectedComponents{N}() where {N}
    return ConnectedComponents{N}(DEFAULT_ELIMINATION_ALGORITHM)
end

function ConnectedComponents(alg::PermutationOrAlgorithm)
    return ConnectedComponents{1}(alg)
end

function ConnectedComponents()
    return ConnectedComponents{1}()
end

struct Best{S, A <: NTuple{<:Any, PermutationOrAlgorithm}} <: EliminationAlgorithm
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

struct Log{A <: PermutationOrAlgorithm} <: EliminationAlgorithm
    alg::A
    num::Scalar{Int}
end

function Log{A}(alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM) where {A <: PermutationOrAlgorithm}
    return Log{A}(alg, zeros(Int))
end

function Log(alg::A=DEFAULT_ELIMINATION_ALGORITHM) where {A <: PermutationOrAlgorithm}
    return Log{A}(alg)
end

"""
    permutation([weights, ]graph;
        alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM)

Construct a fill-reducing permutation of the vertices of a simple graph.

```julia
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
function permutation(graph; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM, verbose::Bool = false)
    return permutation(graph, alg; verbose)
end

function permutation(graph, alg::PermutationOrAlgorithm; verbose::Bool = false)
    return permutation(BipartiteGraph(graph), alg; verbose)
end

function permutation(graph::AbstractGraph{V}, alg::PermutationOrAlgorithm; verbose::Bool = false) where {V}
    n = nv(graph); weights = Ones{V}(n)
    return permutation(weights, graph, alg; verbose)
end

function permutation(weights::AbstractVector, graph; alg::PermutationOrAlgorithm = DEFAULT_ELIMINATION_ALGORITHM, verbose::Bool = false)
    return permutation(weights, graph, alg; verbose)
end

function permutation(weights::AbstractVector, graph, alg::PermutationOrAlgorithm; verbose::Bool = false)
    return permutation(weights, BipartiteGraph(graph), alg; verbose)
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::A; verbose::Bool = false) where A <: EliminationAlgorithm
    name = package(A)
    message = "`import $name` to use algorithm $A."
    throw(ArgumentError(message))
end

function permutation(weights::AbstractVector, graph::AbstractGraph{V}, order::AbstractVector; verbose::Bool = false) where {V}
    order = Vector{V}(order)
    return order, invperm(order)
end

function permutation(weights::AbstractVector, graph::AbstractGraph{V}, (order, index)::Tuple{AbstractVector, AbstractVector}; verbose::Bool = false) where {V}
    order = Vector{V}(order)
    index = Vector{V}(index)
    return order, index
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::BFS; verbose::Bool = false)
    order = bfs(graph)
    return order, invperm(order)
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::MCS; verbose::Bool = false)
    index, size = mcs(graph)
    return invperm(index), index
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::LexBFS; verbose::Bool = false)
    index = lexbfs(graph)
    return invperm(index), index
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::RCMMD; verbose::Bool = false)
    order = rcmmd(graph, alg.alg)
    return order, invperm(order)
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::RCMGL; verbose::Bool = false)
    order = rcmgl(graph, alg.alg)
    return order, invperm(order)
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::LexM; verbose::Bool = false)
    index = lexm(graph)
    return invperm(index), index
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::MCSM; verbose::Bool = false)
    index = mcsm(graph)
    return invperm(index), index
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::AMF; verbose::Bool = false)
    return amf(weights, graph; exact = alg.exact)
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::MF; verbose::Bool = false)
    return convert.(Vector, mlf(alg.strategy, weights, graph))
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::MMD; verbose::Bool = false)
    index = mmd(weights, graph; delta = alg.delta)
    return invperm(index), index
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::MD; verbose::Bool = false)
    index = md(weights, graph; mass = alg.mass, external = alg.external)
    return invperm(index), index
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::NDS{S}; verbose::Bool = false) where {S}
    return dissectsearch(weights, graph, alg.alg, alg.dis, alg.width, alg.level, alg.imbalances, alg.scale, Val(S))
end

function permutation(weights::AbstractVector{W}, graph::AbstractGraph{V}, alg::PIDBT; verbose::Bool = false) where {W, V}
    lo = max(convert(W, lowerbound(weights, graph, alg.lb)), convert(W, alg.target))
    perm, invp, _, _ = septw(weights, graph, lo, typemax(W))
    return convert(Vector{V}, perm), convert(Vector{V}, invp)
end

# Small graphs are solved exactly if that takes at most half the time; the
# others by the anytime search (hbtnew). The graph is assumed connected; a
# disconnected one is split by ConnectedComponents.
function permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::HBT; verbose::Bool = false) where {V}
    t0 = time(); n = nv(graph)
    iszero(n) && return V[], V[]

    if n <= alg.base
        bgraph, wgt = hbt_graph(weights, graph)
        lb = convert(eltype(wgt), lowerbound(wgt, bgraph, alg.lb))
        order, status = hbt_safe(wgt, bgraph, lb, lb, typemax(eltype(wgt)), t0 + alg.time / 2)

        if status == BT_FOUND
            order = convert(Vector{V}, order)
            return order, invperm(order)
        end
    end

    hbt = hbt_new(weights, graph, alg)
    order, invp, _ = hbtrun!(hbt; mt = alg.time - (time() - t0), nt = alg.threads)
    return convert(Vector{V}, order), convert(Vector{V}, invp)
end

function permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::RTW; verbose::Bool = false) where {V}
    order = only(rtw_batch([weights], [graph], alg, time() + alg.time))
    return order, invperm(order)
end

function batch_permutation(weights::AbstractVector{<:AbstractVector}, graphs::AbstractVector{<:AbstractGraph}, alg::RTW; verbose::Bool = false)
    return rtw_batch(weights, graphs, alg, time() + alg.time)
end

# A batch of graphs (see batch_permutation) is one problem whose width is
# that of its widest graph: RTW spends one time budget, to the deadline
# dl, on all of it. The graphs are taken by their greedy widths, widest
# first, each with the target of the largest lower bound so far: a lower
# bound of any graph, or the width of a graph shown optimal. A graph no
# wider than the batch then stops at its first decomposition that meets
# the target. The recursion works on connected graphs: a disconnected one
# is split by ConnectedComponents, with the time that is left.
#
# Each graph is first given an exact search (rtw_probe, as septw) for a
# short time (the share probe of the time left, at most RTW_PROBE_MAX
# seconds; none if probe is zero): it solves many graphs at once that
# the recursion takes longer to prove, and when it runs out of time, the
# widths it ruled out raise the lower bound, and so the target of the
# recursion, which then need not prove its last width optimal.
function rtw_batch(weights::AbstractVector{<:AbstractVector}, graphs::AbstractVector{<:AbstractGraph{V}}, alg::RTW, dl::Float64, probe::Real = RTW_PROBE) where {V}
    ngraph = length(graphs); orders = Vector{Vector{V}}(undef, ngraph)
    iszero(ngraph) && return orders
    items = map(i -> hbt_graph(weights[i], graphs[i]), oneto(ngraph))
    W = mapreduce(item -> eltype(last(item)), promote_type, items)
    lowers = W[convert(W, lowerbound(wgt, bgraph, alg.lb)) for (bgraph, wgt) in items]
    greedy = W[treewidth(wgt, bgraph; alg = alg.alg) for (bgraph, wgt) in items]
    lb = maximum(lowers); nt = isnothing(alg.hbt) ? 1 : alg.hbt.threads
    par = (; alg = alg.alg, hbt = alg.hbt, ticks = alg.ticks, alg.verbose)

    for i in sortperm(greedy; rev = true)
        bgraph, wgt = items[i]

        if rtw_connected(bgraph)
            if ispositive(probe)
                now = time(); pdl = min(dl, now + min(probe * max(zero(dl), dl - now), RTW_PROBE_MAX))
                perm, _, k, status = rtw_probe(wgt, bgraph, lb, pdl, nt)
                alg.verbose && println("rtw: exact search $(status == BT_FOUND ? "found width" : "lower bound") $(hbt_str(k)) in $(round(time() - now; digits = 2)) s")

                if status == BT_FOUND
                    # k is the width of the graph, or at most lb
                    orders[i] = convert(Vector{V}, perm); lb = max(lb, k)
                    continue
                end

                # every ordering of the graph has width at least k
                status == BT_TIMEOUT && (lb = max(lb, k))
            end

            perm, _, k, status, _ = rtw(wgt, bgraph, lb, typemax(W), dl, nt, par)
            orders[i] = convert(Vector{V}, perm)

            # k is the width of the graph
            status == BT_FOUND && k > lb && (lb = k)
        else
            left = RTW(; time = max(zero(dl), dl - time()), alg.alg, alg.hbt, alg.ticks, lb = lb, alg.verbose)
            orders[i] = convert(Vector{V}, first(permutation(weights[i], graphs[i], ConnectedComponents(left))))
        end
    end

    return orders
end

"""
    batch_permutation(weights, graphs, alg::EliminationAlgorithm)

Construct a fill-reducing permutation of each graph in a batch, with the vertex weights
`weights[i]` of `graphs[i]`. Returns a vector of permutations: the `i`th is one of `graphs[i]`.

The graphs of a batch are independent parts of one problem, such as the components of a graph
([`ConnectedComponents`](@ref)) or the leaves of a safe separator decomposition
([`SafeSeparators`](@ref)). By default, each graph is ordered by `alg` in turn, but an algorithm
can order a batch as a whole: [`HBT`](@ref) spends one time budget on it.

```julia-repl
julia> using CliqueTrees, Graphs

julia> graphs = [path_graph(3), cycle_graph(4)];

julia> weights = [ones(3), ones(4)];

julia> orders = batch_permutation(weights, graphs, MMD());

julia> map(length, orders)
2-element Vector{Int64}:
 3
 4
```
"""
function batch_permutation(weights::AbstractVector{<:AbstractVector}, graphs::AbstractVector{<:AbstractGraph}, alg::EliminationAlgorithm; verbose::Bool = false)
    # TODO: order the graphs in parallel
    return map(i -> first(permutation(weights[i], graphs[i], alg; verbose)), eachindex(graphs))
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::MinimalChordal; verbose::Bool = false)
    return mcs_etree(weights, graph, alg.alg)
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::CompositeRotations; verbose::Bool = false)
    return compositerotations(graph, alg.clique, alg.alg)
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::Compression; verbose::Bool = false)
    order = compress(weights, graph, alg.alg, alg.tau; verbose)
    return order, invperm(order)
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::SafeRules; verbose::Bool = false)
    order = saferules(weights, graph, alg.alg, alg.lb, alg.rules; verbose, minfrac = alg.minfrac)
    return order, invperm(order)
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::SimplicialRule; verbose::Bool = false)
    order = simplicialrule(weights, graph, alg.alg; verbose)
    return order, invperm(order)
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::SafeSeparators; verbose::Bool = false)
    return safeseparators(weights, graph, alg.alg, alg.min, alg.maxmissing; verbose)
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::ConnectedComponents{N}; verbose::Bool = false) where {N}
    return connectedcomponents(weights, graph, alg.alg, Val(N); verbose)
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::BestWidth; verbose::Bool = false)
    return bestwidth(weights, graph, alg.algs; verbose)
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::BestFill; verbose::Bool = false)
    return bestfill(weights, graph, alg.algs; verbose)
end

function permutation(weights::AbstractVector, graph::AbstractGraph, alg::Log; verbose::Bool = false)
    num = alg.num[] += 1

    if isone(num)
        print(  "|   #   |       |V|       |       |E|       |")
        print("\n| ----- | --------------- | --------------- |")
    end

    i = rpad(num,       5)
    n = rpad(nv(graph), 15)
    m = rpad(de(graph), 15)
    print("\n| $i | $n | $m |")

    return permutation(weights, graph, alg.alg; verbose)
end

# Algorithmic Aspects of Vertex Elimination on Graphs
# Rose, Tarjan, and Lueker
# BFS
#
# Perform a breadth-first search of a simple graph.
# The complexity is O(m), where m = |E|.
function bfs(graph::AbstractGraph{V}) where {V}
    n = nv(graph)
    order = Vector{V}(undef, n)
    marker = FVector{Bool}(undef, n)

    @inbounds for node in vertices(graph)
        marker[node] = false 
    end

    hi = n

    @inbounds for root in vertices(graph)
        if !marker[root]
            lo = hi; order[lo] = root; marker[root] = true

            while lo <= hi
                node = order[hi]; hi -= one(V)

                for nbr in neighbors(graph, node)
                    if !marker[nbr]
                        lo -= one(V); order[lo] = nbr; marker[nbr] = true 
                    end
                end
            end
        end
    end

    return order
end

# Simple Linear-Time Algorithms to Test Chordality of Graphs, Test Acyclicity of Hypergraphs, and Selectively Reduce Acyclic Hypergraphs
# Tarjan and Yannakakis
# Maximum Cardinality Search
#
# Construct a fill-reducing permutation of a graph.
# The complexity is O(m + n), where m = |E| and n = |V|.
function mcs(graph::AbstractGraph{V}, clique::AbstractVector{V} = oneto(zero(V))) where {V}
    j = one(V); n = nv(graph)
    size = FVector{V}(undef, n)
    alpha = Vector{V}(undef, n)

    @inbounds for v in vertices(graph)
        size[v] = one(V)
    end

    # construct bucket queue data structure
    head = FVector{V}(undef, n + one(V))
    prev = FVector{V}(undef, n)
    next = FVector{V}(undef, n)

    function set(i)
        @inbounds h = view(head, i)
        return DoublyLinkedList(h, prev, next)
    end

    @inbounds for i in oneto(n + one(V))
        empty!(set(i))
    end

    @inbounds prepend!(set(j), vertices(graph))

    # run algorithm
    @inbounds for v in Iterators.reverse(clique)
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

function rootls_impl!(
        xls::AbstractVector{V},
        ls::AbstractVector{V},
        graph::AbstractGraph{V},
        marker::AbstractVector{V},
        root::V,
        tag::V,
    ) where {V}
    @assert nv(graph) < length(xls)
    @assert nv(graph) <= length(ls)
    @assert nv(graph) <= length(marker)

    # initialization...
    n = nv(graph)
    nlvl = lvlend = zero(V)
    @inbounds marker[root] = tag
    @inbounds ccsize = one(V); ls[ccsize] = root

    # compute the current level width;
    # if it is nonzero, generate the next level
    @inbounds while lvlend < ccsize
        # `lbegin` is the pointer to the beginning of the current
        # level, and `lvlend` points to the end of this level
        lbegin = lvlend + one(V)
        lvlend = ccsize
        nlvl += one(V); xls[nlvl] = lbegin

        # generate the next leel by finding all the masked
        # neighbors of nodes in the current level
        for i in lbegin:lvlend
            node = ls[i]

            for nbr in neighbors(graph, node)
                if marker[nbr] < tag
                    ccsize += one(V); ls[ccsize] = nbr
                    marker[nbr] = tag
                end
            end
        end
    end

    @inbounds xls[nlvl + one(V)] = ccsize + one(V)
    return BipartiteGraph(n, nlvl, ccsize, xls, ls)
end

function fnroot_impl!(
        xls::AbstractVector{V},
        ls::AbstractVector{V},
        graph::AbstractGraph{V},
        marker::AbstractVector{V},
        root::V,
    ) where {V}
    @assert nv(graph) < length(xls)
    @assert nv(graph) <= length(ls)
    @assert nv(graph) <= length(marker)

    # initialize `tag`
    tag = one(V)

    # determine the level structure rooted at `root`
    level = rootls_impl!(xls, ls, graph, marker, root, tag)
    nlvl = nv(level); ccsize = ne(level)
    pnlvl = one(V)

    # increment `tag`
    tag += one(V)

    @inbounds while pnlvl < nlvl < ccsize
        pnlvl = nlvl

        # pick a node with minimum degree from the last level
        mindeg = ccsize

        for node in neighbors(level, nlvl)
            ndeg = eltypedegree(graph, node)

            if ndeg < mindeg
                root, mindeg = node, ndeg
            end
        end

        # and generate its rooted level structure
        level = rootls_impl!(xls, ls, graph, marker, root, tag)
        nlvl = nv(level)

        # increment `tag`
        tag += one(V)
    end

    return level
end

# Algorithms for Sparse Linear Systems
# Scott and Tuma
# Algorithm 8.3: CM and RCM algorithms for band and profile reduction
#
# Apply the reverse Cuthill-Mckee algorithm to each connected component of a graph.
# The complexity is O(m + n), where m = |E| and n = |V|.
function rcmgl(graph::AbstractGraph, alg::SortingAlgorithm)
    return rcmgl!(simplegraph(graph), alg)
end

function rcmgl!(graph::BipartiteGraph{V}, alg::SortingAlgorithm) where {V}
    n = nv(graph); nn = n + one(V)
    order = Vector{V}(undef, n)
    xls = FVector{V}(undef, nn)
    ls = FVector{V}(undef, n)
    marker = FVector{V}(undef, n)

    @inbounds for v in vertices(graph)
        marker[v] = zero(V)
        sort!(neighbors(graph, v); alg, scratch = order, by = w -> eltypedegree(graph, w))
    end

    resize!(order, n)
    hi = n

    @inbounds for root in vertices(graph)
        if iszero(marker[root])
            level = fnroot_impl!(xls, ls, graph, marker, root)

            for i in vertices(level), v in neighbors(level, i)
                order[hi] = v; hi -= one(V)
            end
        end
    end

    return order
end

function rcmmd(graph::AbstractGraph, alg::SortingAlgorithm)
    return rcmmd!(simplegraph(graph), alg)
end

function rcmmd!(graph::BipartiteGraph{V}, alg::SortingAlgorithm) where {V}
    n = nv(graph)
    order = Vector{V}(undef, n)
    marker = FVector{V}(undef, n)

    @inbounds for v in vertices(graph)
        marker[v] = zero(V)
        sort!(neighbors(graph, v); alg, scratch = order, by = w -> eltypedegree(graph, w))
    end

    resize!(order, n)
    hi = n

    @inbounds for root in vertices(graph)
        if iszero(marker[root])
            nhi = hi
            mindeg = eltypedegree(graph, root)
            lo = hi; order[lo] = root; marker[root] = one(V)

            while lo <= hi
                node = order[hi]; hi -= one(V)
                ndeg = eltypedegree(graph, node)

                if ndeg < mindeg
                    root, mindeg = node, ndeg
                end

                for nbr in neighbors(graph, node)
                    if iszero(marker[nbr])
                        lo -= one(V); order[lo] = nbr; marker[nbr] = one(V) 
                    end
                end
            end

            hi = nhi
            lo = hi; order[lo] = root; marker[root] = two(V)

            while lo <= hi
                node = order[hi]; hi -= one(V)

                for nbr in neighbors(graph, node)
                    if isone(marker[nbr])
                        lo -= one(V); order[lo] = nbr; marker[nbr] = two(V) 
                    end
                end
            end
        end
    end

    return order
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

# Maximum Cardinality Search for Computing Minimal Triangulations
# Berry, Blair, and Heggernes
# MCS-M
#
# Perform a maximum cardinality search of a simple graph.
# Returns a minimal ordering.
# The complexity is O(mn), where m = |E| and n = |V|.
function mcsm(graph::AbstractGraph{V}, clique::AbstractVector{V} = oneto(zero(V))) where {V}
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

function AMFLib.amf(weights::AbstractVector, graph::AbstractGraph; kwargs...)
    simple = simplegraph(graph)
    return amf(nv(simple), weights, pointers(simple), targets(simple); kwargs...)
end

function mlf(weights::AbstractVector, graph::AbstractGraph; kwargs...)
    return mlf(MinFill, weights, graph; kwargs...)
end

function mlf(strategy::Integer, weights::AbstractVector, graph::AbstractGraph; kwargs...)
    return mlf(MLFScore(strategy), weights, graph; kwargs...)
end

function mlf(score::MLFScore, weights::AbstractVector{W}, graph::AbstractGraph; kwargs...) where {W <: Union{Int8, Int16, Int32}}
    intweights = FVector{Int}(undef, nv(graph))

    @inbounds for v in vertices(graph)
        intweights[v] = convert(Int, weights[v])
    end

    return mlf(score, intweights, graph; kwargs...)
end

function mlf(score::MLFScore, weights::AbstractVector, graph::AbstractGraph{V}; kwargs...) where {V}
    E = etype(graph)
    n = nv(graph)
    m = de(graph)
    ptr = FVector{E}(undef, n + 1)
    tgt = FVector{V}(undef, m + 2n)
    simple = simplegraph!(ptr, tgt, graph)
    return mlf!(score, nv(simple), weights, pointers(simple), targets(simple); kwargs...)
end

function MMDLib.mmd(weights::AbstractVector, graph::AbstractGraph; kwargs...)
    simple = simplegraph(graph)
    return mmd(nv(simple), weights, pointers(simple), targets(simple); kwargs...)
end

function MDLib.md(weights::AbstractVector, graph::AbstractGraph; kwargs...)
    simple = simplegraph(graph)
    return md(nv(simple), weights, pointers(simple), targets(simple); kwargs...)
end

function dissectsearch(weights::AbstractVector, graph::AbstractGraph, alg::EliminationAlgorithm, dis::DissectionAlgorithm, width::Integer, level::Integer, imbalances::AbstractRange, scale::Integer, ::Val{S}) where {S}
    minscore = minpair = nothing

    for imbalance in imbalances
        curalg = ND{S}(alg, dis; width, level, imbalance, scale)
        curpair = permutation(weights, graph, curalg)

        if isone(S)
            curscore = (treewidth(weights, graph, curpair), treefill(weights, graph, curpair))
        else
            curscore = (treefill(weights, graph, curpair), treewidth(weights, graph, curpair))
        end

        if isnothing(minscore) || curscore < minscore
            minscore, minpair = curscore, curpair
        end
    end

    return minpair
end

function twins(graph::AbstractGraph{V}, ::Val{S}) where {V, S}
    n = nov(graph); nn = n + one(V)
    new = FScalar{V}(undef)
    var = FVector{V}(undef, nn)
    svar = FVector{V}(undef, n)
    flag = FVector{V}(undef, n)
    size = FVector{V}(undef, n)
    partition = twins_impl!(new, var, svar, flag, size, graph, Val(S))
    return svar, partition
end

# Exploiting Zeros on the Diagonal in the Direct Solution
# of Indefinite Sparse Symmetric Linear Systems
# Duff and Reid
# 2.5 Recognition of Supervariables
function twins_impl!(
        new::AbstractScalar{V},
        var::AbstractVector{V},
        svar::AbstractVector{V},
        flag::AbstractVector{V},
        size::AbstractVector{V},
        graph::AbstractGraph{V},
        ::Val{S},
    ) where {V, S}
    @assert S === :T || S === :F || S === :N
    @assert S === :N || nov(graph) == nv(graph)
    @assert nov(graph) < length(var)
    @assert nov(graph) <= length(svar)
    @assert nov(graph) <= length(flag)
    @assert nov(graph) <= length(size)

    n = nov(graph); m = zero(V)

    @inbounds for i in oneto(n)
        flag[i] = zero(V)
        size[i] = zero(V)
    end

    new[] = zero(V); free = SinglyLinkedList(new, var)
    @inbounds prepend!(free, oneto(n))

    if ispositive(n)
        s = popfirst!(free); m += one(V); size[s] = n

        @inbounds for i in oneto(n)
            svar[i] = s
        end
    end

    @inbounds for j in vertices(graph)
        for i in neighbors(graph, j)
            if S === :N || i != j
                m = twins_refine!(free, var, svar, flag, size, m, i, j)
            end
        end

        if S === :T
            m = twins_refine!(free, var, svar, flag, size, m, j, j)
        end
    end

    # number the classes in order of their labels, and list the items of
    # each class in increasing order (a counting sort)
    partition = BipartiteGraph(n, m, n, var, flag)
    @inbounds var[begin] = p = one(V); t = zero(V)

    @inbounds for s in oneto(n)
        if ispositive(size[s])
            t += one(V); var[t + one(V)] = p; p += size[s]
            size[s] = t
        end
    end

    # now var[t + 1] is the first position of class t, and size[s]
    # is the class of label s
    @inbounds for i in oneto(n)
        t = svar[i] = size[svar[i]]
        flag[var[t + one(V)]] = i
        var[t + one(V)] += one(V)
    end

    # now var[t] is the first position of class t
    return partition
end

# Item `i` lies in column `j`: move it out of its class, `s`, into the class
# of the first item of `s` that lies in column `j`. Returns the number of
# classes.
@propagate_inbounds function twins_refine!(
        free::SinglyLinkedList{V},
        var::AbstractVector{V},
        svar::AbstractVector{V},
        flag::AbstractVector{V},
        size::AbstractVector{V},
        m::V,
        i::V,
        j::V,
    ) where {V}
    # `s` is the old supervariable of `i`
    s = svar[i]

    # first occurrence of `s` for column `j`
    if flag[s] < j
        flag[s] = j

        if size[s] > one(V)
            size[s] -= one(V)
            var[s] = i
            ns = svar[i] = popfirst!(free); m += one(V)
            size[ns] += one(V)
        end
    # second or later occurrence of `s` for column `j`
    else
        # `k` is the first variable of `s` encountered in column `j`.
        size[s] -= one(V)
        k = var[s]
        ns = svar[i] = svar[k]
        size[ns] += one(V)

        if iszero(size[s])
            pushfirst!(free, s); m -= one(V)
        end
    end

    return m
end

# Algorithms for Sparse Linear Systems
# Scott & Tuma
#
# Algorithm 3.8: Find approximately indistinguishable vertex sets
# in an undirected graph
#
# Finding Exact and Approximate Block Structures for ILU Preconditioning
# Yousef Saad
#
# Algorithm 2.2: Cosine-based compression
function twins(graph::AbstractGraph{V}, ::Val{S}, tau::Number) where {V, S}
    @assert S === :T || S === :F

    if isone(tau)
        return twins(graph, Val(S))
    else
        n = nv(graph); nn = n + one(V)
        head = FScalar{V}(undef)
        prev = FVector{V}(undef, nn)
        next = FVector{V}(undef, n)
        adjmap = FVector{V}(undef, n)
        cosine = FVector{V}(undef, n)
        degree = FVector{V}(undef, n)
        partition = twins_impl!(head, prev, next, adjmap, cosine, degree, graph, Val(S), tau)
        return adjmap, partition
    end
end

function twins_impl!(
        head::AbstractScalar{V},
        prev::AbstractVector{V},
        next::AbstractVector{V},
        adjmap::AbstractVector{V},
        cosine::AbstractVector{V},
        degree::AbstractVector{V},
        graph::AbstractGraph{V},
        ::Val{S},
        tau::T,
    ) where {V, S, T}
    @assert nv(graph) < length(prev)
    @assert nv(graph) <= length(next)
    @assert nv(graph) <= length(adjmap)
    @assert nv(graph) <= length(cosine)
    @assert nv(graph) <= length(degree)
    @assert S === :T || S === :F
    @assert zero(T) < tau <= one(T)
    n = nv(graph); nb = zero(V)
    list = DoublyLinkedList(head, prev, next)

    @inbounds for i in vertices(graph)
        adjmap[i] = zero(V)
        cosine[i] = zero(V)

        if S === :T
            degree[i] = one(V)
        else
            degree[i] = zero(V)
        end

        for j in neighbors(graph, i)
            i == j && continue
            degree[i] += one(V)
        end
    end

    @inbounds head[] = zero(V)

    @inbounds for i in vertices(graph)
        if iszero(adjmap[i])
            # start a new set
            adjmap[i] = nb += one(V)

            # for each entry j in row i...
            for j in neighbors(graph, i)
                i == j && continue
                    
                for k in neighbors(graph, j)
                    j == k && continue
                        
                    # both rows i and k have an entry
                    # in column j
                    if i < k
                        # k has not yet been added to a part
                        if iszero(adjmap[k])
                            # increase partial dot product
                            iszero(cosine[k]) && pushfirst!(list, k)
                            cosine[k] += one(V)
                        end
                    end
                end

                if S === :T && i < j
                    # k has not yet been added to a part
                    if iszero(adjmap[j])
                        # increase partial dot product
                        iszero(cosine[j]) && pushfirst!(list, j)
                        cosine[j] += one(V)
                    end
                end
            end

            if S === :T
                for k in neighbors(graph, i)
                    # both rows i and k have an entry
                    # in column j
                    if i < k
                        # k has not yet been added to a part
                        if iszero(adjmap[k])
                            # increase partial dot product
                            iszero(cosine[k]) && pushfirst!(list, k)
                            cosine[k] += one(V)
                        end
                    end
                end
            end

            for k in list
                cos = convert(T, cosine[k])
                nzi = convert(T, degree[i])
                nzk = convert(T, degree[k])

                # test similarity of row patterns
                if cos * cos >= tau * tau * nzi * nzk
                    adjmap[k] = nb
                end

                delete!(list, k)
                cosine[k] = zero(V)
            end
        end
    end

    pointer = prev; target = next
    
    @inbounds for j in oneto(nb)
        jj = j + one(V)
        pointer[jj] = zero(V)
    end

    @inbounds for i in vertices(graph)
        j = adjmap[i]

        if j < nb
            jj = j + two(V)
            pointer[jj] += one(V)
        end
    end

    @inbounds pointer[begin] = p = one(V)

    @inbounds for j in oneto(nb)
        jj = j + one(V)
        pointer[jj] = p += pointer[jj]
    end

    @inbounds for i in vertices(graph)
        j = adjmap[i]; jj = j + one(V)
        target[pointer[jj]] = i
        pointer[jj] += one(V)
    end
        
    partition = BipartiteGraph(n, nb, n, pointer, target)
    return partition  
end

# Engineering Data Reduction for Nested Dissection
# Ost, Schulz, Strash
# Reduction 2 (Indistinguishable node reduction)
# Reduction 3 (Twin Reduction)
function compress(graph::AbstractGraph{V}, ::Val{S}) where {V, S}
    E = etype(graph); n = nv(graph); m = de(graph); nn = n + one(V)
    new = FScalar{V}(undef)
    var = FVector{V}(undef, nn)
    svar = FVector{V}(undef, n)
    flag = FVector{V}(undef, n)
    size = FVector{V}(undef, n)
    outptr = FVector{E}(undef, nn)
    outtgt = FVector{V}(undef, m)
    return compress_impl!(new, var, svar, flag, size, outptr, outtgt, graph, Val(S))
end

function compress_impl!(
        new::AbstractScalar{V},
        var::AbstractVector{V},
        svar::AbstractVector{V},
        flag::AbstractVector{V},
        size::AbstractVector{V},
        outptr::AbstractVector{E},
        outtgt::AbstractVector{V},
        graph::AbstractGraph{V},
        ::Val{S},
    ) where {V, E, S}
    @assert S === :T || S === :F
    @assert nv(graph) < length(outptr)
    @assert ne(graph) <= length(outtgt)

    partition = twins_impl!(new, var, svar, flag, size, graph, Val(S))
    project = svar; marker = size; tag = zero(V)

    @inbounds for v in vertices(graph)
        marker[v] = zero(V)
    end

    outptr[begin] = p = one(E)

    @inbounds for i in vertices(partition)
        ii = i + one(V)
        v = first(neighbors(partition, i))

        for w in neighbors(graph, v)
            j = project[w]

            if i != j && marker[j] < i
                marker[j] = i; outtgt[p] = j; p += one(E)
            end
        end

        outptr[ii] = p
    end

    outn = nv(partition); outm = p - one(E)
    outgraph = BipartiteGraph(outn, outn, outm, outptr, outtgt)
    return outgraph, partition
end

function compress(graph::AbstractGraph{V}, ::Val{S}, tau::Number) where {V, S}
    if isone(tau)
        return compress(graph, Val(S))
    else
        E = etype(graph); n = nv(graph); m = de(graph); nn = n + one(V)
        head = FScalar{V}(undef)
        prev = FVector{V}(undef, nn)
        next = FVector{V}(undef, n)
        adjmap = FVector{V}(undef, n)
        cosine = FVector{V}(undef, n)
        degree = FVector{V}(undef, n)
        outptr = FVector{E}(undef, nn)
        outtgt = FVector{V}(undef, m)
        return compress_impl!(head, prev, next, adjmap, cosine, degree, outptr, outtgt, graph, Val(S), tau)
    end
end

function compress_impl!(
        head::AbstractScalar{V},
        prev::AbstractVector{V},
        next::AbstractVector{V},
        adjmap::AbstractVector{V},
        cosine::AbstractVector{V},
        degree::AbstractVector{V},
        outptr::AbstractVector{E},
        outtgt::AbstractVector{V},
        graph::AbstractGraph{V},
        ::Val{S},
        tau::Number,
    ) where {V, E, S}
    @assert nv(graph) < length(outptr)
    @assert ne(graph) <= length(outtgt)

    partition = twins_impl!(head, prev, next, adjmap, cosine, degree, graph, Val(S), tau)
    project = adjmap; marker = cosine; tag = zero(V)

    @inbounds for v in vertices(graph)
        marker[v] = zero(V)
    end

    outptr[begin] = p = one(E)

    @inbounds for i in vertices(partition)
        ii = i + one(V)

        for v in neighbors(partition, i), w in neighbors(graph, v)
            j = project[w]

            if i != j && marker[j] < i
                marker[j] = i; outtgt[p] = j; p += one(E)
            end
        end

        outptr[ii] = p
    end

    outn = nv(partition); outm = p - one(E)
    outgraph = BipartiteGraph(outn, outn, outm, outptr, outtgt)
    return outgraph, partition
end

function compress(weights::AbstractVector{W}, graph::AbstractGraph{V}, alg::EliminationAlgorithm, tau::Number; verbose::Bool = false) where {W, V}
    order = Vector{V}(undef, nv(graph))
    cmpgraph, project = compress(graph, Val(:T), tau)
    cmpweights = compressweights(weights, project)
    cmporder, cmpindex = permutation(cmpweights, cmpgraph, alg; verbose)
    i = zero(V)

    @inbounds for vcmp in cmporder, v in neighbors(project, vcmp)
        i += one(V); order[i] = v
    end

    return order
end

function compressweights(weights::AbstractVector{W}, project::AbstractGraph) where {W}
    cmpweights = FVector{W}(undef, nv(project))
    compressweights_impl!(cmpweights, weights, project)
    return cmpweights
end

function compressweights_impl!(cmpweights::AbstractVector, weights::AbstractVector{W}, project::AbstractGraph) where {W}
    @assert nv(project) <= length(cmpweights)
    @assert nov(project) <= length(weights)

    @inbounds for vcmp in vertices(project)
        wgt = zero(W)

        for v in neighbors(project, vcmp)
            wgt += weights[v]
        end

        cmpweights[vcmp] = wgt
    end

    return
end

function saferules(weights::AbstractVector, graph::AbstractGraph{V}, alg::EliminationAlgorithm, lb::WidthOrAlgorithm, rules::Int; verbose::Bool = false, minfrac::Real = 0.0) where {V}
    width = lowerbound(weights, graph, lb)
    innerweights, innergraph, result = compressreduce(rules, weights, graph, width; verbose, minfrac)
    innerorder, innerindex = permutation(innerweights, innergraph, alg; verbose)
    return rule_pull(result, innerorder)
end

function simplicialrule(weights::AbstractVector{W}, graph::AbstractGraph{V}, alg::EliminationAlgorithm; verbose::Bool = false) where {W, V}
    width = zero(W)
    innerweights, innergraph, result = compressreduce(sr, weights, graph, width; verbose)
    innerorder, innerindex = permutation(innerweights, innergraph, alg; verbose)
    return rule_pull(result, innerorder)
end

function connectedcomponents(graph::AbstractGraph{V}) where {V}
    E = etype(graph); n = nv(graph); m = de(graph)

    cmp = Vector{Tuple{View{V, V}, BipartiteGraph{V, E, View{E, V}, View{V, E}}}}(undef, n)
    ptr = Vector{E}(undef, twice(n) + one(V))
    tgt = Vector{V}(undef, m)
    prj = Vector{V}(undef, n)
    inj = Vector{V}(undef, n)
    
    cmpend = one(V)
    injbeg = one(V)
    ptrbeg = one(V)
    tgtbeg = one(E)

    for v in vertices(graph)
        prj[v] = zero(V)
    end

    for v in vertices(graph)
        if iszero(prj[v])
            injcur = injbeg
            injend = injbeg
            prj[v] = injend; inj[injend] = v; injend += one(V)

            tgtend = tgtbeg
            ptrend = ptrbeg
            ptr[ptrend] = one(E); ptrend += one(V)

            while injcur < injend
                v = inj[injcur]; injcur += one(V)

                for w in neighbors(graph, v)
                    prjnbr = prj[w]

                    if iszero(prjnbr)
                        prj[w] = prjnbr = injend; inj[injend] = w; injend += one(V)
                    end

                    tgt[tgtend] = prjnbr - injbeg + one(V); tgtend += one(E)
                end 

                ptr[ptrend] = tgtend - tgtbeg + one(E); ptrend += one(V)
            end

            cmp[cmpend] = (
                view(inj, injbeg:injend - one(V)),
                BipartiteGraph(
                    injend - injbeg,
                    injend - injbeg,
                    tgtend - tgtbeg,
                    view(ptr, ptrbeg:ptrend - one(V)),
                    view(tgt, tgtbeg:tgtend - one(E)),
                )
            )

            cmpend += one(V)
            injbeg = injend
            ptrbeg = ptrend
            tgtbeg = tgtend
        end
    end

    return cmp, cmpend - one(V)
end

function connectedcomponents(weights::AbstractVector{W}, graph::AbstractGraph{V}, alg::EliminationAlgorithm; verbose::Bool = false) where {W, V}
    n = nv(graph); j = zero(V)
    order = Vector{V}(undef, n)
    index = Vector{V}(undef, n)
    components, m = connectedcomponents(graph)

    labels = [first(components[i]) for i in oneto(m)]
    subgraphs = [last(components[i]) for i in oneto(m)]
    subweights = Vector{Vector{W}}(undef, m)

    for i in oneto(m)
        label = labels[i]; subweight = subweights[i] = Vector{W}(undef, length(label))

        for v in eachindex(label)
            subweight[v] = weights[label[v]]
        end
    end

    suborders = batch_permutation(subweights, subgraphs, alg; verbose)

    for i in oneto(m), v in suborders[i]
        vv = labels[i][v]
        j += one(V); order[j] = vv; index[vv] = j
    end

    return order, index
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

function Base.show(io::IO, ::MIME"text/plain", alg::AMF)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "AMF:")
    println(io, " "^indent * "    exact: $(alg.exact)")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::MMD)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "MMD:")
    println(io, " "^indent * "    delta: $(alg.delta)")
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
    println(io, " "^indent * "    width: $(alg.width)")
    println(io, " "^indent * "    level: $(alg.level)")
    println(io, " "^indent * "    imbalance: $(alg.imbalance)")
    println(io, " "^indent * "    scale: $(alg.scale)")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::FlowCutter)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "FlowCutter:")
    println(io, " "^indent * "    time: $(alg.time)")
    println(io, " "^indent * "    seed: $(alg.seed)")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::A) where {A <: PIDBT}
    indent = get(io, :indent, 0)
    println(io, " "^indent * string(A))
    println(io, " "^indent * "    target: $(alg.target)")
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.lb)
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::HBT{A, L, I}) where {A, L, I}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "HBT{$A, $L, $I}:")

    for name in (:time, :seed, :ninit, :base, :ntry, :refined, :merge, :diversify, :dsize, :nsep, :near, :patience, :xtime, :xmin, :xstep, :margin, :pexact, :threads)
        println(io, " "^indent * "    $name: $(getfield(alg, name))")
    end

    show(IOContext(io, :indent => indent + 4), "text/plain", alg.alg)
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.lb)
    alg.init isa EliminationAlgorithm && show(IOContext(io, :indent => indent + 4), "text/plain", alg.init)
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::RTW{A, H, L}) where {A, H, L}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "RTW{$A, $H, $L}:")
    println(io, " "^indent * "    time: $(alg.time)")
    println(io, " "^indent * "    ticks: $(alg.ticks)")
    alg.alg isa EliminationAlgorithm && show(IOContext(io, :indent => indent + 4), "text/plain", alg.alg)
    isnothing(alg.hbt) || show(IOContext(io, :indent => indent + 4), "text/plain", alg.hbt)
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.lb)
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

function Base.show(io::IO, ::MIME"text/plain", alg::Compression{A}) where {A}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "Compression{$A}:")
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.alg)
    println(io, " "^indent * "    tau: $(alg.tau)")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::SimplicialRule{A}) where {A}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "SimplicialRule{$A}:")
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.alg)
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::SafeRules{A, L, U}) where {A, L, U}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "SafeRules{$A, $L, $U}:")
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.alg)
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.lb)
    println(io, " "^indent * "    rules: $(alg.rules)")
    println(io, " "^indent * "    minfrac: $(alg.minfrac)")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::SafeSeparators{A, M}) where {A, M}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "SafeSeparators{$A, $M}:")
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.alg)

    for min in alg.min
        show(IOContext(io, :indent => indent + 4), "text/plain", min)
    end

    println(io, " "^indent * "    maxmissing: $(alg.maxmissing)")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::ConnectedComponents{N, A}) where {N, A}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "ConnectedComponents{$N, $A}:")
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

function Base.show(io::IO, ::MIME"text/plain", alg::A) where {A <: Log}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "$A:")
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.alg)
    return
end

"""
    DEFAULT_ELIMINATION_ALGORITHM = AMF()

The default algorithm.
"""
const DEFAULT_ELIMINATION_ALGORITHM = AMF()

"""
    DEFAULT_MINIMAL_ALGORITHM = MinimalChordal(AMF())
"""
const DEFAULT_MINIMAL_ALGORITHM = MinimalChordal(DEFAULT_ELIMINATION_ALGORITHM)

"""
    RCM = RCMGL

The default variant of the reverse Cuthill-Mckee algorithm.
"""
const RCM = RCMGL
