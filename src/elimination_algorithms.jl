"""
    EliminationAlgorithm

A *graph elimination algorithm* computes a permutation of the vertices of a graph, which induces a chordal completion
of the graph. The algorithms below generally seek to minimize the *fill* (number of edges) or *width* (largest clique)
of the completed graph.

| type                         | name                                         | time     | space    |
|:---------------------------- |:-------------------------------------------- |:-------- |:-------- |
| [`BFS`](@ref)                | breadth-first search                         | O(m + n) | O(n)     |
| [`MCS`](@ref)                | maximum cardinality search                   | O(m + n) | O(n)     |
| [`LexBFS`](@ref)             | lexicographic breadth-first search           | O(m + n) | O(m + n) |
| [`RCMMD`](@ref)              | reverse Cuthill-Mckee (minimum degree)       | O(m + n) | O(m + n) |
| [`RCMGL`](@ref)              | reverse Cuthill-Mckee (George-Liu)           | O(m + n) | O(m + n) |
| [`MCSM`](@ref)               | maximum cardinality search (minimal)         | O(mn)    | O(n)     |
| [`LexM`](@ref)               | lexicographic breadth-first search (minimal) | O(mn)    | O(n)     |
| [`AMF`](@ref)                | approximate minimum fill                     | O(mn)    | O(m + n) |
| [`MF`](@ref)                 | minimum fill                                 | O(mn²)   |          |
| [`MMD`](@ref)                | multiple minimum degree                      | O(mn²)   | O(m + n) |
| [`MinimalChordal`](@ref)     | MinimalChordal                               |          |          |
| [`CompositeRotations`](@ref) | elimination tree rotation                    | O(m + n) | O(m + n) |
| [`RuleReduction`](@ref)      | treewith-safe rule-based reduction           |          |          |    
| [`ComponentReduction`](@ref) | connected component reduction                |          |          |

The following additional algorithms are implemented as package extensions and require loading an additional package.

| type               | name                              | time  | space    | package                                                                             |
|:------------------ |:--------------------------------- |:----- |:-------- |:----------------------------------------------------------------------------------- |
| [`AMD`](@ref)      | approximate minimum degree        | O(mn) | O(m + n) | [AMD.jl](https://github.com/JuliaSmoothOptimizers/AMD.jl)                           |
| [`SymAMD`](@ref)   | column approximate minimum degree | O(mn) | O(m + n) | [AMD.jl](https://github.com/JuliaSmoothOptimizers/AMD.jl)                           |
| [`METIS`](@ref)    | multilevel nested dissection      |       |          | [Metis.jl](https://github.com/JuliaSparse/Metis.jl)                                 |
| [`Spectral`](@ref) | spectral ordering                 |       |          | [Laplacians.jl](https://github.com/danspielman/Laplacians.jl)                       |
| [`BT`](@ref)       | Bouchitte-Todinca                 |       |          | [TreeWidthSolver.jl](https://github.com/ArrogantGao/TreeWidthSolver.jl)             |
| [`SAT`](@ref)      | SAT encoding (picosat)            |       |          | [PicoSAT_jll.jl](https://github.com/JuliaBinaryWrappers/PicoSAT_jll.jl)             |
| [`SAT`](@ref)      | SAT encoding (cryptominisat)      |       |          | [CryptoMiniSat_jll.jl](https://github.com/JuliaBinaryWrappers/CryptoMiniSat_jll.jl) |

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

These algorithm minimizes the treewidth of the completed graph. Beware! This is an NP-hard problem. I recommend wrapping exact
treewidth algorithms with preprocessors like [`RuleReduction`](@ref) or [`ComponentReduction`](@ref). 

# Meta Algorithms

  - [`MinimalChordal`](@ref)
  - [`CompositeRotations`](@ref)
  - [`RuleReduction`](@ref)
  - [`ComponentReduction`](@ref)

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
struct LexM <: EliminationAlgorithm end

"""
    MCSM <: EliminationAlgorithm

    MCSM()

A minimal variant of the maximal cardinality search algorithm.

### References

  - Berry, Anne, et al. "Maximum cardinality search for computing minimal triangulations of graphs." *Algorithmica* 39 (2004): 287-298.
"""
struct MCSM <: EliminationAlgorithm end

"""
    AMF <: EliminationAlgorithm

    AMF(; speed=1)

The approximate minimum fill algorithm.

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

### References

  - Tinney, William F., and John W. Walker. "Direct solutions of sparse network equations by optimally ordered triangular factorization." *Proceedings of the IEEE* 55.11 (1967): 1801-1809.
"""
struct MF <: EliminationAlgorithm end

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
    Spectral <: EliminationAlgorithm

    Spectral(; tol=0.0)

The spectral ordering algorithm only works on connected graphs.
In order to use it, import the package [Laplacians](https://github.com/danspielman/Laplacians.jl).

### Parameters

  - `tol`: tolerance for convergence

### References

  - Barnard, Stephen T., Alex Pothen, and Horst D. Simon. "A spectral algorithm for envelope reduction of sparse matrices." *Proceedings of the 1993 ACM/IEEE Conference on Supercomputing.* 1993.
"""
@kwdef struct Spectral <: EliminationAlgorithm
    tol::Float64 = 0.0
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
    SAT <: EliminationAlgorithm

    SAT{Handle}(lb::LowerBoundAlgorithn, ub::PermutationOrAlgorithm)

    SAT{Handle}()

Compute a minimum-treewidth permutation using a SAT solver.

### Parameters

  - `Handle`: solver module (either `PicoSAT_jll` or `CryptoMiniSat_jll`)
  - `lb`: lower bound algorithm
  - `ub`: upper bound algorithm

## References

  - Samer, Marko, and Helmut Veith. "Encoding treewidth into SAT." *Theory and Applications of Satisfiability Testing-SAT 2009: 12th International Conference, SAT 2009*, Swansea, UK, June 30-July 3, 2009. Proceedings 12. Springer Berlin Heidelberg, 2009.
  - Berg, Jeremias, and Matti Järvisalo. "SAT-based approaches to treewidth computation: An evaluation." *2014 IEEE 26th international conference on tools with artificial intelligence.* IEEE, 2014.
  - Bannach, Max, Sebastian Berndt, and Thorsten Ehlers. "Jdrasil: A modular library for computing tree decompositions." *16th International Symposium on Experimental Algorithms (SEA 2017)*. Schloss Dagstuhl–Leibniz-Zentrum fuer Informatik, 2017.
"""
struct SAT{Handle, LB <: LowerBoundAlgorithm, UB <: PermutationOrAlgorithm} <: EliminationAlgorithm
    handle::Val{Handle}
    lb::LB
    ub::UB
end

function SAT{Handle}(lb::LowerBoundAlgorithm, ub::PermutationOrAlgorithm) where {Handle}
    return SAT(Val(Handle), lb, ub)
end

function SAT{Handle}() where {Handle}
    return SAT{Handle}(DEFAULT_LOWER_BOUND_ALGORITHM, DEFAULT_ELIMINATION_ALGORITHM)
end

"""
    MinimalChordal{A} <: EliminationAlgorithm

    MinimalChordal(alg::PermutationOrAlgorithm)

    MinimalChordal()

Evaluate an elimination algorithm, and them improve its output using the MinimalChordal algorithm. The result is guaranteed to be minimal.

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
    RuleReduction{A} <: EliminationAlgorithm

    RuleReduction(alg::PermutationOrAlgororithm)

    RuleReduction()

Preprocess a graph using safe reduction rules.

### Parameters

  - `alg`: elimination algorithm

### References

  - Bodlaender, Hans L., Arie M.C.A. Koster, and Frank van den Eijkhof. "Preprocessing rules for triangulation of probabilistic networks." *Computational Intelligence* 21.3 (2005): 286-305.
  - van den Eijkhof, Frank, Hans L. Bodlaender, and Arie M.C.A. Koster. "Safe reduction rules for weighted treewidth." *Algorithmica* 47 (2007): 139-158. 
"""
struct RuleReduction{A <: PermutationOrAlgorithm} <: EliminationAlgorithm
    alg::A
end

function RuleReduction()
    return RuleReduction(DEFAULT_ELIMINATION_ALGORITHM)
end

"""
    ComponentReduction{A} <: EliminationAlgorithm

    ComponentReduction(alg::PermutationOrAlgorithm)

    ComponentReduction()

Apply an elimination algorithm to each connected component of a graph.

### Parameters

  - `alg`: elimination algorithm

"""
struct ComponentReduction{A <: PermutationOrAlgorithm} <: EliminationAlgorithm
    alg::A
end

function ComponentReduction()
    return ComponentReduction(DEFAULT_ELIMINATION_ALGORITHM)
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
function permutation(weights::AbstractVector, alg::RuleReduction)
    error()
end

# method ambiguity
function permutation(weights::AbstractVector, alg::ComponentReduction)
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

function permutation(graph, alg::SAT{Handle}) where {Handle}
    order, index = permutation(graph, alg.ub)
    lb = lowerbound(graph, alg.lb)
    ub = treewidth(graph, order)

    if lb < ub
        order, width = sat(graph, Handle, lb, ub)
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

function permutation(graph, alg::RuleReduction)
    stack, label, kernel = rulereduction(graph)
    order, index = permutation(kernel, alg.alg)
    append!(stack, view(label, order))
    return stack, invperm(stack)
end

function permutation(weights::AbstractVector, graph, alg::RuleReduction)
    stack, label, kernel = rulereduction(weights, graph)
    order, index = permutation(view(weights, label), kernel, alg.alg)
    append!(stack, view(label, order))
    return stack, invperm(stack)
end

# TODO: multi-threading
function permutation(graph, alg::ComponentReduction)
    components, subgraphs = componentreduction(graph)
    order = eltype(eltype(components))[]

    @inbounds for (component, subgraph) in zip(components, subgraphs)
        suborder, subindex = permutation(subgraph, alg.alg)
        append!(order, view(component, suborder))
    end

    return order, invperm(order)
end

function permutation(weights::AbstractVector, graph, alg::ComponentReduction)
    components, subgraphs = componentreduction(graph)
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
            outdegree(graph, v)
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
        sort!(neighbors(graph, v); alg, scratch, by = u -> outdegree(graph, u))
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
        i = convert(V, length(component))
        candidate = component[i]
        degree = outdegree(graph, candidate)
        eccentricity = level[candidate] - tag

        while i > firstindex(component) && eccentricity + tag == level[component[i]]
            v = component[i]
            n = outdegree(graph, v)

            if degree > n
                candidate, degree = v, n
            end

            i -= one(V)
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
        list = Vector{V}(undef, outdegree(graph, v))

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
        list = Vector{V}(undef, outdegree(graph, v))

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
    order = zeros(V, nv)
    index = zeros(V, nv)
    label = zeros(Int, nv)
    tag = 0

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

function sat(graph, Handle::Module, lowerbound::Integer, upperbound::Integer)
    return sat(BipartiteGraph(graph), Handle, lowerbound, upperbound)
end

# Encoding Treewidth into SAT
# Samer and Veith
#
# SAT-Based Approaches to Treewidth Computation: An Evaluation
# Berg and Järvisalo
#
# Jdrasil: A Modular Library for Computing Tree Decompositions
# Bannach, Berndt, and Ehlers
function sat(graph::AbstractGraph{V}, Handle::Module, lowerbound::Integer, upperbound::Integer) where {V}
    @argcheck !isnegative(lowerbound) && lowerbound <= upperbound <= nv(graph)

    # compute a maximal clique
    clique = maximalclique(graph, Handle)

    # compute true twins
    trueset, truelist = twins(graph, Val(true))

    # compute false twins
    falseset, falselist = twins(graph, Val(false))

    # choose partition with the fewest sets
    if count(_ -> true, truelist) < count(_ -> true, falselist)
        set, list = trueset, truelist
    else
        set, list = falseset, falselist
    end

    # run solver
    order, width = open(Solver{Handle}) do solver
        n = Int32(nv(graph))
        ord = Matrix{Int32}(undef, n, n)
        arc = Matrix{Int32}(undef, n, n)

        # define ord and arc variables
        for i in oneto(n), j in oneto(n)
            ord[i, j] = i < j ? variable!(solver) :
                i > j ? -ord[j, i] :
                zero(Int32)

            arc[i, j] = variable!(solver)
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
        label = zeros(Bool, n)

        for j in clique
            label[j] = true

            for i in oneto(n)
                if !label[i]
                    # ord(i, j)
                    clause!(solver, ord[i, j])
                end
            end
        end

        # encode twins
        for p in list, i in set(p), j in drop(rest(set(p), i), 1)
            if !label[i] && !label[j]
                # ord(i, j)
                clause!(solver, ord[i, j])
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
            assume!(solver, -arc[i, n - count])
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
                assume!(solver, -arc[i, n - count])
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
function maximalclique(graph::AbstractGraph, Handle::Module)
    clique = open(Solver{Handle}, nv(graph)) do solver
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
        state = solve!(assume!(solver, variables[n - count]))

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
            state = solve!(assume!(solver, variables[n - count]))
        end

        return resize!(stack, num)
    end

    return clique
end

# encode a sorting network
function sortingnetwork!(solver::Solver, var::AbstractVector)
    n = length(var)

    mergesort(n) do i, j
        # min ↔ var(i) ∧ var(j)
        min = variable!(solver)
        clause!(solver, var[i], -min)
        clause!(solver, var[j], -min)
        clause!(solver, -var[i], -var[j], min)

        # max ↔ var(i) ∨ var(j)
        max = variable!(solver)
        clause!(solver, -var[i], max)
        clause!(solver, -var[j], max)
        clause!(solver, var[i], var[j], -max)

        var[i] = min
        var[j] = max
    end

    return solver
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

# Batcher's Odd-Even Merge Sort
# https://gist.github.com/stbuehler/883635
function mergesort(f::Function, n::Integer)
    mergesort(f, oneto(n))
    return
end

function mergesort(f::Function, slice::AbstractRange)
    if length(slice) <= 2
        sorttwo(f, slice)
    else
        lhs, rhs = halves(slice)
        mergesort(f, lhs)
        mergesort(f, rhs)
        oddevenmerge(f, slice)
    end

    return
end

function is2pot(n::Integer)
    return ispositive(n) && iszero(n & (n - 1))
end

function is2pot(slice::AbstractRange)
    return is2pot(length(slice))
end

function odd(slice::AbstractRange)
    return (first(slice) + step(slice)):twice(step(slice)):last(slice)
end

function even(slice::AbstractRange)
    return first(slice):twice(step(slice)):last(slice)
end

function halves(slice::AbstractRange)
    if length(slice) <= 1
        lhs = slice
        rhs = 1:1:0
    else
        if is2pot(slice)
            mid = first(slice) + half(length(slice)) * step(slice)
        else
            len = 2

            while len < length(slice)
                len = twice(len)
            end

            mid = first(slice) + half(len) * step(slice)
        end

        lhs = first(slice):step(slice):(mid - 1)
        rhs = mid:step(slice):last(slice)
    end

    return lhs, rhs
end

function sorttwo(f::Function, slice::AbstractRange)
    if istwo(length(slice))
        f(slice[1], slice[2])
    end

    return
end

function oddevenmerge(f::Function, slice::AbstractRange)
    if length(slice) <= 2
        sorttwo(f, slice)
    else
        oddevenmerge(f, odd(slice))
        oddevenmerge(f, even(slice))

        for i in 2:2:(length(slice) - 1)
            f(slice[i], slice[i + 1])
        end
    end

    return
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
        degree = outdegree(M, v)

        for j in oneto(degree)
            w = list[j]

            if i < index[w]
                for jj in (j + 1):degree
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
                degree = outdegree(W, v)

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

# Preprocessing Rules for Triangulation of Probabilistic Networks
# Bodlaender, Koster, Ejkhof, and van der Gaag
# PR-3
function rulereduction(graph)
    return rulereduction(BipartiteGraph(graph))
end

function rulereduction(graph::AbstractGraph)
    return rulereduction!(Graph(graph))
end

function rulereduction!(graph::Graph{V}) where {V}
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
        @inbounds pushfirst!(set(outdegree(graph, v)), v)
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

                delete!(set(outdegree(graph, w)), w)

                rem_edge!(graph, v, w)

                pushfirst!(set(outdegree(graph, w)), w)

                n = next[v]; delete!(set(1), v)
                v = n
            end

            if lo == hi
                # series rule
                v = head[3]

                while ispositive(v)
                    # w
                    # |
                    # v  ---  ww
                    w, ww = neighbors(graph, v)
                    hi += one(V); stack[hi] = v

                    delete!(set(outdegree(graph, w)), w)
                    delete!(set(outdegree(graph, ww)), ww)

                    rem_edge!(graph, v, w)
                    rem_edge!(graph, v, ww)

                    add_edge!(graph, w, ww)

                    pushfirst!(set(outdegree(graph, w)), w)
                    pushfirst!(set(outdegree(graph, ww)), ww)

                    n = next[v]; delete!(set(2), v)
                    v = n
                end

                if lo == hi
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

                            delete!(set(outdegree(graph, w)), w)
                            delete!(set(outdegree(graph, ww)), ww)
                            delete!(set(outdegree(graph, www)), www)

                            rem_edge!(graph, v, w)
                            rem_edge!(graph, v, ww)
                            rem_edge!(graph, v, www)

                            add_edge!(graph, w, www)
                            add_edge!(graph, ww, www)

                            pushfirst!(set(outdegree(graph, w)), w)
                            pushfirst!(set(outdegree(graph, ww)), ww)
                            pushfirst!(set(outdegree(graph, www)), www)

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

                                    delete!(set(outdegree(graph, w)), w)
                                    delete!(set(outdegree(graph, ww)), ww)
                                    delete!(set(outdegree(graph, www)), www)

                                    rem_edge!(graph, v, w)
                                    rem_edge!(graph, v, ww)
                                    rem_edge!(graph, v, www)
                                    rem_edge!(graph, vv, w)
                                    rem_edge!(graph, vv, ww)
                                    rem_edge!(graph, vv, www)

                                    add_edge!(graph, w, ww)
                                    add_edge!(graph, w, www)
                                    add_edge!(graph, ww, www)

                                    pushfirst!(set(outdegree(graph, w)), w)
                                    pushfirst!(set(outdegree(graph, ww)), ww)
                                    pushfirst!(set(outdegree(graph, www)), www)

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

                                if isthree(outdegree(graph, vv)) && isthree(outdegree(graph, vvv)) && isthree(outdegree(graph, vvvv))
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

                                        delete!(set(outdegree(graph, w)), w)
                                        delete!(set(outdegree(graph, ww)), ww)
                                        delete!(set(outdegree(graph, www)), www)

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

                                        pushfirst!(set(outdegree(graph, w)), w)
                                        pushfirst!(set(outdegree(graph, ww)), ww)
                                        pushfirst!(set(outdegree(graph, www)), www)

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

    resize!(stack, hi)
    return stack, rem_vertices!(graph, stack), graph
end

# Safe Reduction Rules for Weighted Treewidth
# Eijkhof, Bodlaender, and Koster
function rulereduction(weights::AbstractVector, graph)
    return rulereduction(weights, BipartiteGraph(graph))
end

function rulereduction(weights::AbstractVector, graph::AbstractGraph)
    return rulereduction!(weights, Graph(graph))
end

function rulereduction!(weights::AbstractVector{W}, graph::Graph{V}) where {W, V}
    n = nv(graph)

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
        pushfirst!(set(outdegree(graph, v)), v)
    end

    # lower bound
    width = lowerbound(weights, graph)

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

                delete!(set(outdegree(graph, w)), w)

                rem_edge!(graph, v, w); nws[w] -= weights[v]

                pushfirst!(set(outdegree(graph, w)), w)

                n = next[v]; delete!(set(1), v)
                v = n
            end

            if lo == hi
                # series rule
                v = head[3]

                while ispositive(v)
                    w = ww = zero(V)

                    if nws[v] <= width
                        x, xx = neighbors(graph, v)

                        if weights[v] >= min(weights[x], weights[xx])
                            w, ww = x, xx
                        end
                    end

                    if ispositive(w)
                        # w
                        # |
                        # v  ---  ww
                        hi += one(V); stack[hi] = v

                        delete!(set(outdegree(graph, w)), w)
                        delete!(set(outdegree(graph, ww)), ww)

                        rem_edge!(graph, v, w); nws[w] -= weights[v]
                        rem_edge!(graph, v, ww); nws[ww] -= weights[v]

                        if add_edge!(graph, w, ww)
                            nws[w] += weights[ww]
                            nws[ww] += weights[w]
                        end

                        pushfirst!(set(outdegree(graph, w)), w)
                        pushfirst!(set(outdegree(graph, ww)), ww)

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

                        if nws[v] <= width
                            x, xx, xxx = neighbors(graph, v)

                            if has_edge(graph, x, xx) && weights[v] >= weights[xxx]
                                w, ww, www = x, xx, xxx
                            elseif has_edge(graph, x, xxx) && weights[v] >= weights[xx]
                                w, ww, www = x, xxx, xx
                            elseif has_edge(graph, xx, xxx) && weights[v] >= weights[x]
                                w, ww, www = xx, xxx, x
                            end
                        end

                        if ispositive(w)
                            # w  ---  ww
                            # |   /
                            # v  ---  www
                            hi += one(V); stack[hi] = v

                            delete!(set(outdegree(graph, w)), w)
                            delete!(set(outdegree(graph, ww)), ww)
                            delete!(set(outdegree(graph, www)), www)

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

                            pushfirst!(set(outdegree(graph, w)), w)
                            pushfirst!(set(outdegree(graph, ww)), ww)
                            pushfirst!(set(outdegree(graph, www)), www)

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

                            if nws[v] <= width
                                w, ww, www = neighbors(graph, v)

                                # sort the weights
                                p = weights[w]
                                pp = weights[ww]
                                ppp = weights[www]

                                if p > pp
                                    p, pp = pp, p
                                end

                                if pp > ppp
                                    pp, ppp = ppp, pp
                                end

                                if p > pp
                                    p, pp = pp, p
                                end

                                while ispositive(vv)
                                    x, xx, xxx = neighbors(graph, vv)

                                    # sort the weights
                                    q = weights[v]
                                    qq = weights[vv]

                                    if q > qq
                                        q, qq = qq, q
                                    end

                                    if nws[vv] <= width && p <= q && pp <= qq && (w, ww, www) == (x, xx, xxx)
                                        # w  -----------  vv
                                        # |           /   |
                                        # |       ww      |
                                        # |   /           |
                                        # v  -----------  www
                                        hi += one(V); stack[hi] = vv
                                        hi += one(V); stack[hi] = v

                                        delete!(set(outdegree(graph, w)), w)
                                        delete!(set(outdegree(graph, ww)), ww)
                                        delete!(set(outdegree(graph, www)), www)

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

                                        pushfirst!(set(outdegree(graph, w)), w)
                                        pushfirst!(set(outdegree(graph, ww)), ww)
                                        pushfirst!(set(outdegree(graph, www)), www)

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

                                if isthree(outdegree(graph, vv)) && isthree(outdegree(graph, vvv)) && isthree(outdegree(graph, vvvv)) && max(nws[vv], nws[vvv], nws[vvvv]) <= width
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
                                            (weights[vv] >= weights[www] && weights[vvv] >= weights[w] && weights[vvvv] >= weights[ww]) ||
                                                (weights[vv] >= weights[w] && weights[vvv] >= weights[ww] && weights[vvvv] >= weights[www])
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

                                        delete!(set(outdegree(graph, w)), w)
                                        delete!(set(outdegree(graph, ww)), ww)
                                        delete!(set(outdegree(graph, www)), www)

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

                                        pushfirst!(set(outdegree(graph, w)), w)
                                        pushfirst!(set(outdegree(graph, ww)), ww)
                                        pushfirst!(set(outdegree(graph, www)), www)

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

    resize!(stack, hi)
    return stack, rem_vertices!(graph, stack), graph
end

function componentreduction(graph)
    return componentreduction(BipartiteGraph(graph))
end

function componentreduction(graph::AbstractGraph{V}) where {V}
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
                mstop += convert(E, outdegree(graph, v))
            end

            subgraph = BipartiteGraph{V, E}(
                nstop - one(V),
                view(nfree, one(V):nstop),
                view(mfree, one(E):mstop),
            )

            p = pointers(subgraph)[begin] = one(E)

            for (i, v) in enumerate(component)
                pp = pointers(subgraph)[i + 1] = p + outdegree(graph, v)
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

function Base.show(io::IO, ::MIME"text/plain", alg::MCS)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "MCS")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::LexBFS)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "LexBFS")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::RCMMD)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "RCMMD:")

    for line in eachsplit(strip(repr(alg.alg)), "\n")
        println(io, " "^indent * "    $line")
    end

    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::RCMGL)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "RCMGL:")

    for line in eachsplit(strip(repr(alg.alg)), "\n")
        println(io, " "^indent * "    $line")
    end

    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::MCSM)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "MCSM")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::LexM)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "LexM")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::MMD)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "MMD:")
    println(io, " "^indent * "    delta: $(alg.delta)")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::MF)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "MF")
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

function Base.show(io::IO, ::MIME"text/plain", alg::BT)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "BT")
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::SAT{Handle, LB, UB}) where {Handle, LB, UB}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "SAT{$Handle,$LB,$UB}:")
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
    println(io, " "^indent * "CompositeRotations{$C,$A}:")
    println(io, " "^indent * "    clique: $(alg.clique)")
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.alg)
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::RuleReduction{A}) where {A}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "RuleReduction{$A}:")
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.alg)
    return
end

function Base.show(io::IO, ::MIME"text/plain", alg::ComponentReduction{A}) where {A}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "ComponentReduction{$A}:")
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
