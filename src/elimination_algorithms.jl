"""
    EliminationAlgorithm

A graph elimination algorithm. The options are

| type             | name                                         | time     | space    |
|:---------------- |:-------------------------------------------- |:-------- |:-------- |
| [`BFS`](@ref)    | breadth-first search                         | O(m + n) | O(n)     |
| [`MCS`](@ref)    | maximum cardinality search                   | O(m + n) | O(n)     |
| [`LexBFS`](@ref) | lexicographic breadth-first search           | O(m + n) | O(m + n) |
| [`RCMMD`](@ref)  | reverse Cuthill-Mckee (minimum degree)       | O(m + n) | O(m + n) |
| [`RCMGL`](@ref)  | reverse Cuthill-Mckee (George-Liu)           | O(m + n) | O(m + n) |
| [`MCSM`](@ref)   | maximum cardinality search (minimal)         | O(mn)    | O(n)     |
| [`LexM`](@ref)   | lexicographic breadth-first search (minimal) | O(mn)    | O(n)     |
| [`AMF`](@ref)    | approximate minimum fill                     |          | O(m + n) |
| [`MF`](@ref)     | minimum fill                                 | O(mn²)   | O(m + n) |
| [`MMD`](@ref)    | multiple minimum degree                      | O(mn²)   | O(m + n) |

for a graph with m edges and n vertices. The following additional algorithms are implemented as package extensions and require loading an additional package.

| type               | name                              | time  | space    | package                                                                 |
|:------------------ |:--------------------------------- |:----- |:-------- |:----------------------------------------------------------------------- |
| [`AMD`](@ref)      | approximate minimum degree        | O(mn) | O(m + n) | [AMD.jl](https://github.com/JuliaSmoothOptimizers/AMD.jl)               |
| [`SymAMD`](@ref)   | column approximate minimum degree | O(mn) | O(m + n) | [AMD.jl](https://github.com/JuliaSmoothOptimizers/AMD.jl)               |
| [`METIS`](@ref)    | multilevel nested dissection      |       |          | [Metis.jl](https://github.com/JuliaSparse/Metis.jl)                     |
| [`Spectral`](@ref) | spectral ordering                 |       |          | [Laplacians.jl](https://github.com/danspielman/Laplacians.jl)           |
| [`BT`](@ref)       | Bouchitte-Todinca                 |       |          | [TreeWidthSolver.jl](https://github.com/ArrogantGao/TreeWidthSolver.jl) |

The algorithm [`Spectral`](@ref) only works on connected graphs.
"""
abstract type EliminationAlgorithm end

"""
    PermutationOrAlgorithm = Union{AbstractVector, EliminationAlgorithm}

Either a permutation or an algorithm.
"""
const PermutationOrAlgorithm = Union{AbstractVector,EliminationAlgorithm}

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

### Reference

Tarjan, Robert E., and Mihalis Yannakakis. "Simple linear-time algorithms to test chordality of graphs, test acyclicity of hypergraphs, and selectively reduce acyclic hypergraphs." *SIAM Journal on Computing* 13.3 (1984): 566-579.
"""
struct MCS <: EliminationAlgorithm end

"""
    LexBFS <: EliminationAlgorithm

    LexBFS()

The [lexicographic breadth-first-search algorithm](https://en.wikipedia.org/wiki/Lexicographic_breadth-first_search).

### Reference

Rose, Donald J., R. Endre Tarjan, and George S. Lueker. "Algorithmic aspects of vertex elimination on graphs." *SIAM Journal on Computing* 5.2 (1976): 266-283.
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

### Reference

Cuthill, Elizabeth, and James McKee. "Reducing the bandwidth of sparse symmetric matrices." *Proceedings of the 1969 24th National Conference.* 1969.
"""
struct RCMMD{A<:SortingAlgorithm} <: EliminationAlgorithm
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

### Reference

George, Alan, and Joseph WH Liu. "An implementation of a pseudoperipheral node finder." *ACM Transactions on Mathematical Software (TOMS)* 5.3 (1979): 284-295.
"""
struct RCMGL{A<:SortingAlgorithm} <: EliminationAlgorithm
    alg::A
end

function RCMGL()
    return RCMGL(DEFAULT_UNSTABLE)
end

"""
    LexM <: EliminationAlgorithm

    LexM()

A minimal variant of the [lexicographic breadth-first-search algorithm](https://en.wikipedia.org/wiki/Lexicographic_breadth-first_search).

### Reference

Rose, Donald J., R. Endre Tarjan, and George S. Lueker. "Algorithmic aspects of vertex elimination on graphs." *SIAM Journal on Computing* 5.2 (1976): 266-283.
"""
struct LexM <: EliminationAlgorithm end

"""
    MCSM <: EliminationAlgorithm

    MCSM()

A minimal variant of the maximal cardinality search algorithm.

### Reference

Berry, Anne, et al. "Maximum cardinality search for computing minimal triangulations of graphs." *Algorithmica* 39 (2004): 287-298.
"""
struct MCSM <: EliminationAlgorithm end

"""
    MinimalChordal{A} <: EliminationAlgorithm

    MinimalChordal(alg::PermutationOrAlgorithm)

    MinimalChordal()

The MinimalChordal algorithm.

### Parameters

  - `alg`: elimination algorithm

### Reference

Blair, Jean RS, Pinar Heggernes, and Jan Arne Telle. "A practical algorithm for making filled graphs minimal." *Theoretical Computer Science* 250.1-2 (2001): 125-141.
"""
struct MinimalChordal{A<:PermutationOrAlgorithm} <: EliminationAlgorithm
    alg::A
end

function MinimalChordal()
    return MinimalChordal(DEFAULT_ELIMINATION_ALGORITHM)
end

struct CompositeRotations{C<:AbstractVector,A<:PermutationOrAlgorithm} <:
       EliminationAlgorithm
    clique::C
    alg::A
end

function CompositeRotations(clique::AbstractVector)
    return CompositeRotations(clique, DEFAULT_ELIMINATION_ALGORITHM)
end

function CompositeRotations()
    return CompositeRotations(oneto(0))
end

"""
    AMF <: EliminationAlgorithm

    AMF(; speed=1)

The approximate minimum fill algorithm.

### Parameters

  - `speed`: fill approximation strategy (1, 2, or, 3)

### Reference

Rothberg, Edward, and Stanley C. Eisenstat. "Node selection strategies for bottom-up sparse matrix ordering." SIAM Journal on Matrix Analysis and Applications 19.3 (1998): 682-695.
"""
@kwdef struct AMF <: EliminationAlgorithm
    speed::Int = 1
end

"""
    MF <: EliminationAlgorithm

    MF()

The greedy minimum fill algorithm.

### Reference

Tinney, William F., and John W. Walker. "Direct solutions of sparse network equations by optimally ordered triangular factorization." *Proceedings of the IEEE* 55.11 (1967): 1801-1809.
"""
struct MF <: EliminationAlgorithm end

"""
    MMD <: EliminationAlgorithm

    MMD(; delta=0)

The [multiple minimum degree algorithm](https://en.wikipedia.org/wiki/Minimum_degree_algorithm).

### Parameters

  - `delta`: tolerance for multiple elimination

### Reference

Liu, Joseph WH. "Modification of the minimum-degree algorithm by multiple elimination." *ACM Transactions on Mathematical Software (TOMS)* 11.2 (1985): 141-153.
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

### Reference

Amestoy, Patrick R., Timothy A. Davis, and Iain S. Duff. "An approximate minimum degree ordering algorithm." *SIAM Journal on Matrix Analysis and Applications* 17.4 (1996): 886-905.
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

### Reference

Davis, Timothy A., et al. "A column approximate minimum degree ordering algorithm." *ACM Transactions on Mathematical Software* (TOMS) 30.3 (2004): 353-376.
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

### Reference

Karypis, George, and Vipin Kumar. "A fast and high quality multilevel scheme for partitioning irregular graphs." *SIAM Journal on Scientific Computing* 20.1 (1998): 359-392.
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

### Reference

Barnard, Stephen T., Alex Pothen, and Horst D. Simon. "A spectral algorithm for envelope reduction of sparse matrices." *Proceedings of the 1993 ACM/IEEE Conference on Supercomputing.* 1993.
"""
@kwdef struct Spectral <: EliminationAlgorithm
    tol::Float64 = 0.0
end

"""
    BT <: EliminationAlgorithm

    BT()

The Bouchitte-Todinca algorithm.

### Reference

Korhonen, Tuukka, Jeremias Berg, and Matti Järvisalo. "Solving Graph Problems via Potential Maximal Cliques: An Experimental Evaluation of the Bouchitté-Todinca Algorithm." *Journal of Experimental Algorithmics (JEA)* 24 (2019): 1-19.
"""
struct BT <: EliminationAlgorithm end

"""
    permutation(graph;
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
function permutation(graph; alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM)
    return permutation(graph, alg)
end

function permutation(graph, alg::EliminationAlgorithm)
    throw(
        ArgumentError(
            "Algorithm $alg not implemented. You may need to load an additional package."
        ),
    )
end

function permutation(graph, alg::AbstractVector)
    return permutation(BipartiteGraph(graph), alg)
end

function permutation(graph::AbstractGraph{V}, alg::AbstractVector) where {V}
    order = Vector{V}(alg)
    return order, invperm(order)
end

function permutation(graph::SparseMatrixCSC{<:Any,I}, alg::AbstractVector) where {I}
    order::Vector{I} = alg
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

function permutation(graph, alg::MinimalChordal)
    return minimalchordal(graph, alg.alg)
end

function permutation(graph, alg::CompositeRotations)
    order = compositerotations(graph, alg.clique, alg.alg)
    return order, invperm(order)
end

function permutation(graph, alg::AMF)
    return amf(graph; speed=alg.speed)
end

function permutation(graph, alg::MF)
    return mf(graph)
end

function permutation(graph, alg::MMD)
    index = mmd(graph; delta=alg.delta)
    return invperm(index), index
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
    mcs(graph[, clique::AbstractVector])

Perform a maximum cardinality search, optionally specifying a clique to be ordered last.
Returns the inverse permutation.
"""
function mcs(graph, clique::AbstractVector=oneto(0))
    return mcs(BipartiteGraph(graph), clique)
end

# Simple Linear-Time Algorithms to Test Chordality of BipartiteGraphs, Test Acyclicity of Hypergraphs, and Selectively Reduce Acyclic Hypergraphs
# Tarjan and Yannakakis
# Maximum Cardinality Search
#
# Construct a fill-reducing permutation of a graph.
# The complexity is O(m + n), where m = |E| and n = |V|.
function mcs(graph::AbstractGraph{V}, clique::AbstractVector) where {V}
    # construct disjoint sets data structure
    head = zeros(V, nv(graph) + one(V))
    prev = Vector{V}(undef, nv(graph) + one(V))
    next = Vector{V}(undef, nv(graph) + one(V))

    function set(i)
        @inbounds DoublyLinkedList(view(head, i), prev, next)
    end

    # run algorithm
    alpha = Vector{V}(undef, nv(graph))
    size = ones(V, nv(graph))
    prepend!(set(one(V)), vertices(graph))

    j = one(V)
    k = convert(V, lastindex(clique))

    @inbounds for i in reverse(oneto(nv(graph)))
        v = zero(V)

        if k in eachindex(clique)
            v = convert(V, clique[k])
            k -= one(V)
            delete!(set(j), v)
        else
            v = popfirst!(set(j))
        end

        alpha[v] = i
        size[v] = one(V) - size[v]

        for w in neighbors(graph, v)
            if size[w] >= one(V)
                delete!(set(size[w]), w)
                size[w] += one(V)
                pushfirst!(set(size[w]), w)
            end
        end

        j += one(V)

        while j >= one(V) && isempty(set(j))
            j -= one(V)
        end
    end

    return alpha, size
end

function rcmmd(graph, alg::SortingAlgorithm=DEFAULT_UNSTABLE)
    genrcm(graph, alg) do level, queue, graph, root, tag
        component, _, tag = bfs!(level, queue, graph, root, tag)

        root = argmin(component) do v
            outdegree(graph, v)
        end

        return root, tag
    end
end

function rcmgl(graph, alg::SortingAlgorithm=DEFAULT_UNSTABLE)
    return genrcm(fnroot!, graph, alg)
end

function genrcm(f::Function, graph, alg::SortingAlgorithm)
    return genrcm(f, BipartiteGraph(graph), alg)
end

function genrcm(
    f::Function, graph::Union{BipartiteGraph,AbstractSimpleGraph}, alg::SortingAlgorithm
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
        sort!(neighbors(graph, v); alg, scratch, by=u -> outdegree(graph, u))
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
    unnumbered = Vector{V}(undef, nv(graph))
    alpha = Vector{V}(undef, nv(graph))
    scratch = Vector{V}(undef, nv(graph))
    isreached = Vector{Bool}(undef, nv(graph))
    label = Vector{Int}(undef, nv(graph))

    # construct disjoint sets data structure
    head = Vector{V}(undef, nv(graph))
    prev = Vector{V}(undef, nv(graph))
    next = Vector{V}(undef, nv(graph))

    function reach(j)
        @inbounds return DoublyLinkedList(view(head, j), prev, next)
    end

    # run algorithm
    unnumbered .= vertices(graph)
    alpha .= zero(V)
    label .= 2
    p = 0
    n = 0
    k = one(V)

    ########
    # loop #
    ########

    @inbounds for i in reverse(oneto(nv(graph)))
        ##########
        # select #
        ##########

        # assign v the number i
        v = pop!(unnumbered)
        isreached[v] = true
        alpha[v] = i

        for j in oneto(k)
            empty!(reach(j))
        end

        isreached[unnumbered] .= false

        for w in neighbors(graph, v)
            if iszero(alpha[w])
                pushfirst!(reach(div(label[w], 2)), w)
                isreached[w] = true
                label[w] += 1
            end
        end

        ##########
        # search #
        ##########

        for j in oneto(k)
            while !isempty(reach(j))
                w = popfirst!(reach(j))

                for z in neighbors(graph, w)
                    if !isreached[z]
                        isreached[z] = true

                        if label[z] > 2j
                            pushfirst!(reach(div(label[z], 2)), z)
                            label[z] += 1
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

        sort!(unnumbered; scratch, by=w -> label[w])

        p = 0
        k = zero(V)

        for w in unnumbered
            n = label[w]

            if p < n
                k += one(V)
                p = n
            end

            label[w] = 2k
        end
    end

    return alpha
end

function mcsm(graph)
    return mcsm(BipartiteGraph(graph))
end

# Maximum Cardinality Search for Computing Minimal Triangulations
# Berry, Blair, and Heggernes
# MCS-M
#
# Perform a maximum cardinality search of a simple graph.
# Returns a minimal ordering.
# The complexity is O(mn), where m = |E| and n = |V|.
function mcsm(graph::AbstractGraph{V}) where {V}
    alpha = Vector{V}(undef, nv(graph))
    isreached = Vector{Bool}(undef, nv(graph))
    label = Vector{V}(undef, nv(graph))

    # construct disjoint sets data structure
    head = Vector{V}(undef, nv(graph))
    prev = Vector{V}(undef, nv(graph))
    next = Vector{V}(undef, nv(graph))

    function reach(j)
        @inbounds return DoublyLinkedList(view(head, j), prev, next)
    end

    # run algorithm
    alpha .= zero(V)
    label .= one(V)

    @inbounds for i in reverse(oneto(nv(graph)))
        v = k = zero(V)

        for w in vertices(graph)
            if iszero(alpha[w])
                isreached[w] = false

                if label[w] > k
                    v, k = w, label[w]
                end
            end
        end

        isreached[v] = true
        alpha[v] = i

        for j in oneto(k)
            empty!(reach(j))
        end

        for w in neighbors(graph, v)
            if iszero(alpha[w])
                isreached[w] = true
                pushfirst!(reach(label[w]), w)
                label[w] += one(V)
            end
        end

        for j in oneto(k)
            while !isempty(reach(j))
                w = popfirst!(reach(j))

                for z in neighbors(graph, w)
                    if !isreached[z]
                        isreached[z] = true

                        if label[z] > j
                            pushfirst!(reach(label[z]), z)
                            label[z] += one(V)
                        else
                            pushfirst!(reach(j), z)
                        end
                    end
                end
            end
        end
    end

    return alpha
end

"""
    mf(graph)

Compute the greedy minimum-fill ordering of a simple graph.
Returns the permutation and its inverse.
"""
function mf(graph, issorted::Val=Val(false))
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
    hnum = nv # size of heap
    hkey = Vector{V}(undef, nv)
    heap = Vector{V}(undef, nv)
    hinv = Vector{V}(undef, nv)

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

        hkey[v] = count
        heap[v] = hinv[v] = v
    end

    @inbounds for v in reverse(oneto(nv))
        hfall!(hnum, hkey, hinv, heap, v)
    end

    # run algorithm
    i = one(V)

    @inbounds while i <= nv
        # select vertex from heap
        v = order[i] = first(heap)
        list = lists[v]
        degree = degrees[v]
        index[list] .= index[v] = i

        # append distinguishable neighbors to the stack
        snum = zero(V)
        ii = i + one(V)

        for w in list
            if degrees[w] == degree
                flag = true

                for x in lists[w]
                    if index[x] < i
                        flag = false
                        break
                    end
                end

                if flag
                    order[ii] = w
                    index[w] = ii
                    ii += one(V)
                else
                    snum += one(V)
                    stack[snum] = w
                end
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
            for w in take(stack, snum)
                list = lists[w]
                count = zero(V)

                for j in oneto(degrees[w])
                    x = list[j]

                    if index[x] > i
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
            k = hinv[order[j]]
            key = hkey[heap[k]]
            heap[k] = heap[hnum]
            hinv[heap[k]] = k
            hnum -= one(V)

            if key < hkey[heap[k]]
                hfall!(hnum, hkey, hinv, heap, k)
            else
                hrise!(hkey, hinv, heap, k)
            end
        end

        # update deficiencies
        if hkey[order[i]] > zero(V)
            for j in oneto(snum)
                w = stack[j]
                label[lists[w]] .= tag += 1

                for jj in (j + one(V)):snum
                    ww = stack[jj]

                    if label[ww] != tag
                        count = zero(V)

                        for xx in lists[ww]
                            if label[xx] == tag
                                hkey[xx] -= one(V)
                                hrise!(hkey, hinv, heap, hinv[xx])
                                count += one(V)
                            end
                        end

                        hkey[w] += degrees[w] - count
                        hkey[ww] += degrees[ww] - count
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
            hkey[w] -= (ii - i) * (degrees[w] - snum + one(V))
            hrise!(hkey, hinv, heap, hinv[w])
            hfall!(hnum, hkey, hinv, heap, hinv[w])
        end

        i = ii
    end

    return order, index
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

function minimalchordal(graph, alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM)
    return minimalchordal(BipartiteGraph(graph), alg)
end

# A Practical Algorithm for Making Filled Graphs Minimal
# Barry, Heggernes, and Telle
# MinimalChordal
function minimalchordal(graph::AbstractGraph{V}, alg::PermutationOrAlgorithm) where {V}
    order, index = permutation(graph; alg)
    M = Graph(graph)
    F = Vector{Vector{Tuple{V,V}}}(undef, nv(graph))

    for (i, v) in enumerate(order)
        F[i] = Tuple{V,V}[]
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

    Candidate = Set{Tuple{V,V}}()
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

            worder, windex = permutation(W; alg=MCSM())

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

    return permutation(M; alg=MCS())
end

function compositerotations(
    graph,
    clique::AbstractVector=oneto(0),
    alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM,
)
    return compositerotations(BipartiteGraph(graph), clique, alg)
end

function compositerotations(
    graph::AbstractGraph, clique::AbstractVector, alg::PermutationOrAlgorithm
)
    order, index = permutation(graph; alg)
    upper = sympermute(graph, index, Forward)
    invpermute!(
        order, compositerotations(reverse(upper), etree(upper), view(index, clique))
    )
    return order
end

function Base.show(io::IO, ::MIME"text/plain", alg::MCS)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "MCS")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", alg::LexBFS)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "LexBFS")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", alg::RCMMD)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "RCMMD:")
    println(io, " "^indent * "    alg:")

    for line in eachsplit(strip(repr(alg.alg)), "\n")
        println(io, " "^indent * "        $line")
    end

    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", alg::RCMGL)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "RCMGL:")
    println(io, " "^indent * "    alg:")

    for line in eachsplit(strip(repr(alg.alg)), "\n")
        println(io, " "^indent * "        $line")
    end

    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", alg::MCSM)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "MCSM")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", alg::LexM)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "LexM")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", alg::MMD)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "MMD:")
    println(io, " "^indent * "    delta: $(alg.delta)")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", alg::MF)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "MF")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", alg::MinimalChordal{A}) where {A}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "MinimalChordal{$A}:")
    println(io, " "^indent * "    alg:")
    return show(IOContext(io, :indent => indent + 8), "text/plain", alg.alg)
end

function Base.show(io::IO, ::MIME"text/plain", alg::CompositeRotations{C,A}) where {C,A}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "CompositeRotations{$C,$A}:")
    println(io, " "^indent * "    clique: $(alg.clique)")
    println(io, " "^indent * "    alg:")
    return show(IOContext(io, :indent => indent + 8), "text/plain", alg.alg)
end

function Base.show(io::IO, ::MIME"text/plain", alg::AMD)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "AMD:")
    println(io, " "^indent * "    dense: $(alg.dense)")
    println(io, " "^indent * "    aggressive: $(alg.aggressive)")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", alg::SymAMD)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "SymAMD:")
    println(io, " "^indent * "    dense_row: $(alg.dense_row)")
    println(io, " "^indent * "    dense_col: $(alg.dense_col)")
    println(io, " "^indent * "    aggressive: $(alg.aggressive)")
    return nothing
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
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", alg::Spectral)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "Spectral:")
    println(io, " "^indent * "    tol: $(alg.tol)")
    return nothing
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
