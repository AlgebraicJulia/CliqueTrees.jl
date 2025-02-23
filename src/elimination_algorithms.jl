"""
    EliminationAlgorithm

A graph elimination algorithm. The options are

| type               | name                                         | time     | space    |
|:------------------ |:-------------------------------------------- |:-------- |:-------- |
| [`BFS`](@ref)      | breadth-first search                         | O(m + n) | O(n)     |
| [`MCS`](@ref)      | maximum cardinality search                   | O(m + n) | O(n)     |
| [`LexBFS`](@ref)   | lexicographic breadth-first search           | O(m + n) | O(m + n) |
| [`RCMMD`](@ref)    | reverse Cuthill-Mckee (minimum degree)       | O(m + n) | O(m + n) |
| [`RCMGL`](@ref)    | reverse Cuthill-Mckee (George-Liu)           | O(m + n) | O(m + n) |
| [`MCSM`](@ref)     | maximum cardinality search (minimal)         | O(mn)    | O(n)     |
| [`LexM`](@ref)     | lexicographic breadth-first search (minimal) | O(mn)    | O(n)     |
| [`AMD`](@ref)      | approximate minimum degree                   | O(mn)    | O(m + n) |
| [`SymAMD`](@ref)   | column approximate minimum degree            | O(mn)    | O(m + n) |
| [`MF`](@ref)       | minimum fill                                 | O(mn²)   | O(n)     |
| [`MMD`](@ref)      | multiple minimum degree                      | O(mn²)   | O(m + n) |
| [`METIS`](@ref)    | multilevel nested dissection                 |          |          |
| [`Spectral`](@ref) | spectral ordering                            |          |          |
| [`BT`](@ref)       | Bouchitte-Todinca                            |          |          |

for a graph with m edges and n vertices. The algorithm [`Spectral`](@ref) only works on connected graphs.
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
    RCMMD <: EliminationAlgorithm

The [reverse Cuthill-McKee algorithm](https://en.wikipedia.org/wiki/Cuthill%E2%80%93McKee_algorithm).
An initial vertex is selected using the minimum degree heuristic.

### Reference

Cuthill, Elizabeth, and James McKee. "Reducing the bandwidth of sparse symmetric matrices." *Proceedings of the 1969 24th National Conference.* 1969.
"""
struct RCMMD <: EliminationAlgorithm end

"""
    RCMGL <: EliminationAlgorithm

The [reverse Cuthill-McKee algorithm](https://en.wikipedia.org/wiki/Cuthill%E2%80%93McKee_algorithm).
An initial vertex is selected using George and Liu's variant of the GPS algorithm.

### Reference

George, Alan, and Joseph WH Liu. "An implementation of a pseudoperipheral node finder." *ACM Transactions on Mathematical Software (TOMS)* 5.3 (1979): 284-295.
"""
struct RCMGL <: EliminationAlgorithm end

"""
    LexM <: EliminationAlgorithm

A minimal variant of the [lexicographic breadth-first-search algorithm](https://en.wikipedia.org/wiki/Lexicographic_breadth-first_search).

### Reference

Rose, Donald J., R. Endre Tarjan, and George S. Lueker. "Algorithmic aspects of vertex elimination on graphs." *SIAM Journal on Computing* 5.2 (1976): 266-283.
"""
struct LexM <: EliminationAlgorithm end

"""
    MCSM <: EliminationAlgorithm

A minimal variant of the maximal cardinality search algorithm.

### Reference

Berry, Anne, et al. "Maximum cardinality search for computing minimal triangulations of graphs." *Algorithmica* 39 (2004): 287-298.
"""
struct MCSM <: EliminationAlgorithm end

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
    MF <: EliminationAlgorithm

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

```julia
julia> using CliqueTrees

julia> graph = [
           0 1 1 0 0 0 0 0
           1 0 1 0 0 1 0 0
           1 1 0 1 1 0 0 0
           0 0 1 0 1 0 0 0
           0 0 1 1 0 0 1 1
           0 1 0 0 0 0 1 0
           0 0 0 0 1 1 0 1
           0 0 0 0 1 0 1 0
       ];

julia> order, index = permutation(graph);

julia> order
8-element Vector{Int64}:
 4
 8
 7
 6
 5
 1
 3
 2

julia> index == invperm(order)
true
```
"""
function permutation(graph; alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM)
    return permutation(graph, alg)
end

function permutation(graph, alg::PermutationOrAlgorithm)
    return permutation(BipartiteGraph(graph), alg)
end

function permutation(graph::AbstractGraph{V}, alg::AbstractVector) where {V}
    order::Vector{V} = alg
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
    order = rcmmd(graph)
    return order, invperm(order)
end

function permutation(graph, alg::RCMGL)
    order = rcmgl(graph)
    return order, invperm(order)
end

function permutation(graph::BipartiteGraph{V}, alg::Union{AMD,SymAMD,METIS}) where {V}
    order::Vector{V}, index::Vector{V} = permutation(SparseMatrixCSC{Bool}(graph), alg)
    return order, index
end

function permutation(graph::BipartiteGraph{V}, alg::BT) where {V}
    order::Vector{V}, index::Vector{V} = permutation(Graph{Int}(graph), alg)
    return order, index
end

function permutation(matrix::SparseMatrixCSC{T,I}, alg::Union{AMD,SymAMD}) where {T,I}
    # convert matrix
    converted::SparseMatrixCSC{T,Int} = matrix

    # run algorithm
    order::Vector{I}, index::Vector{I} = permutation(converted, alg)
    return order, index
end

function permutation(graph, alg::LexM)
    index = lexm(graph)
    return invperm(index), index
end

function permutation(graph, alg::MCSM)
    index = mcsm(graph)
    return invperm(index), index
end

function permutation(
    matrix::SparseMatrixCSC{<:Any,I}, alg::AMD
) where {I<:Union{Int32,Int64}}
    # set parameters
    meta = AMDJL.Amd()
    meta.control[AMDJL.AMD_DENSE] = alg.dense
    meta.control[AMDJL.AMD_AGGRESSIVE] = alg.aggressive

    # run algorithm
    order::Vector{I} = AMDJL.amd(matrix, meta)
    return order, invperm(order)
end

function permutation(
    matrix::SparseMatrixCSC{<:Any,I}, alg::SymAMD
) where {I<:Union{Int32,Int64}}
    # set parameters
    meta = AMDJL.Colamd{I}()
    meta.knobs[AMDJL.COLAMD_DENSE_ROW] = alg.dense_row
    meta.knobs[AMDJL.COLAMD_DENSE_COL] = alg.dense_col
    meta.knobs[AMDJL.COLAMD_AGGRESSIVE] = alg.aggressive

    # run algorithm
    order::Vector{I} = AMDJL.symamd(matrix, meta)
    return order, invperm(order)
end

function permutation(graph, alg::MF)
    return mf(graph)
end

function permutation(graph, alg::MMD)
    index = mmd(graph; delta=alg.delta)
    return invperm(index), index
end

function permutation(matrix::SparseMatrixCSC{T,I}, alg::METIS) where {T,I}
    # set options
    options = Vector{Metis.idx_t}(undef, Metis.METIS_NOPTIONS)
    options .= -1
    options[Metis.METIS_OPTION_CTYPE + 1] = alg.ctype
    options[Metis.METIS_OPTION_RTYPE + 1] = alg.rtype
    options[Metis.METIS_OPTION_NSEPS + 1] = alg.nseps
    options[Metis.METIS_OPTION_NUMBERING + 1] = 1
    options[Metis.METIS_OPTION_NITER + 1] = alg.niter
    options[Metis.METIS_OPTION_SEED + 1] = alg.seed
    options[Metis.METIS_OPTION_COMPRESS + 1] = alg.compress
    options[Metis.METIS_OPTION_CCORDER + 1] = alg.ccorder
    options[Metis.METIS_OPTION_PFACTOR + 1] = alg.pfactor
    options[Metis.METIS_OPTION_UFACTOR + 1] = alg.ufactor

    # construct permutation
    graph = Metis.graph(matrix; check_hermitian=false)
    perm = Vector{Metis.idx_t}(undef, graph.nvtxs)
    iperm = Vector{Metis.idx_t}(undef, graph.nvtxs)
    Metis.@check Metis.METIS_NodeND(
        Ref{Metis.idx_t}(graph.nvtxs),
        graph.xadj,
        graph.adjncy,
        graph.vwgt,
        options,
        perm,
        iperm,
    )

    # restore types
    order::Vector{I} = perm
    index::Vector{I} = iperm
    return order, index
end

function permutation(graph::Graph{Int}, alg::BT)
    order::Vector{Int} = reverse!(
        reduce(vcat, TreeWidthSolver.elimination_order(graph); init=Int[])
    )
    return order, invperm(order)
end

"""
    bfs(graph)

Perform a [breadth-first search](https://en.wikipedia.org/wiki/Breadth-first_search) of a graph.
"""
function bfs(graph)
    return bfs(BipartiteGraph(graph))
end

function bfs(graph::AbstractGraph{V}) where {V}
    level = zeros(V, nv(graph))
    order = Vector{V}(undef, nv(graph))

    # initialize queue
    queue = @view order[one(V):nv(graph)]

    @inbounds for v in vertices(graph)
        if iszero(level[v])
            queue = last(bfs!(level, queue, graph, v))
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
    level::AbstractVector{V},
    queue::AbstractVector{V},
    graph::AbstractGraph{V},
    root::Integer,
) where {V}
    i = j = one(V)
    level[root] = one(V)
    queue[j] = root

    @inbounds while i <= j
        v = queue[i]
        l = level[v]

        for w in neighbors(graph, v)
            if iszero(level[w])
                j += one(V)
                queue[j] = w
                level[w] = l + one(V)
            end
        end

        i += one(V)
    end

    n = convert(V, length(queue))
    @views return queue[one(V):j], queue[i:n]
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

"""
    rcmmd(graph)

The [reverse Cuthill-Mckee algorithm](https://en.wikipedia.org/wiki/Cuthill%E2%80%93McKee_algorithm).
An initial vertex is selected using the mininum degree heuristic.
"""
function rcmmd(graph)
    genrcm(graph) do level, queue, graph, root
        component = first(bfs!(level, queue, graph, root))
        level[component] .= zero(eltype(level))

        argmin(component) do v
            outdegree(graph, v)
        end
    end
end

"""
    rcmgl(graph)

The [reverse Cuthill-Mckee algorithm](https://en.wikipedia.org/wiki/Cuthill%E2%80%93McKee_algorithm).
An initial vertex is selected using George and Liu's variant of the GPS algorithm.
This is the algorithm used by MATLAB.
"""
function rcmgl(graph)
    return genrcm(fnroot!, graph)
end

function genrcm(f::Function, graph)
    return genrcm(f, BipartiteGraph(graph))
end

function genrcm(f::Function, graph::Union{BipartiteGraph,AbstractSimpleGraph})
    return genrcm!(f, copy(graph))
end

# Algorithms for Sparse Linear Systems
# Scott and Tuma
# Algorithm 8.3: CM and RCM algorithms for band and profile reduction
#
# Apply the reverse Cuthill-Mckee algorithm to each connected component of a graph.
# The complexity is O(m + n), where m = |E| and n = |V|.
function genrcm!(f::Function, graph::AbstractGraph{V}) where {V}
    level = zeros(V, nv(graph))
    order = Vector{V}(undef, nv(graph))

    # sort neighbors
    scratch = Vector{V}(undef, Δout(graph))

    @inbounds for v in vertices(graph)
        sort!(neighbors(graph, v); by=u -> outdegree(graph, u), scratch)
    end

    # initialize queue
    queue = @view order[one(V):nv(graph)]

    @inbounds for v in vertices(graph)
        if iszero(level[v])
            # find pseudo-peripheral vertex
            root = f(level, queue, graph, v)

            # compute Cuthill-Mckee ordering
            queue = last(bfs!(level, queue, graph, root))
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
    level::AbstractVector{V}, queue::AbstractVector{V}, graph::AbstractGraph{V}, root::V
) where {V}
    component = first(bfs!(level, queue, graph, root))
    v = zero(V)

    @inbounds while root != v
        i = convert(V, length(component))
        v = component[i]
        eccentricity = level[v]

        while i > firstindex(component) && eccentricity == level[component[i]]
            w = component[i]

            if outdegree(graph, v) > outdegree(graph, w)
                v = w
            end

            i -= one(V)
        end

        level[component] .= zero(V)
        component = first(bfs!(level, queue, graph, v))

        if level[component[end]] <= eccentricity
            root = v
        end
    end

    level[component] .= zero(V)
    return root
end

"""
    lexbfs(graph)

Perform a [lexicographic breadth-first search](https://en.wikipedia.org/wiki/Lexicographic_breadth-first_search).
Returns the inverse permutation.
"""
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

"""
    lexm(graph)

A minimal variant of the [lexicographic breadth-first search algorithm](https://en.wikipedia.org/wiki/Lexicographic_breadth-first_search).
Returns the inverse permutation.
"""
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

"""
    mcsm(graph)

A minimal variant of the maximum cardinality search algorithm. Returns the inverse permutation.
"""
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
        v = first(heap)
        order[i] = v
        index[v] = i
        list = lists[v]
        index[list] .= i
        degree = degrees[v]

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

"""
    mmd(graph; delta::Integer=0)

Compute the multiple minimum degree ordering of a simple graph. Returns the inverse permutation.
"""
function mmd(graph; kwargs...)
    return mmd(BipartiteGraph(graph); kwargs...)
end

function mmd(graph::BipartiteGraph{V}; delta::Integer=0) where {V}
    return genmmd(nv(graph), pointers(graph), targets(graph), V(delta), typemax(V))
end

function Base.show(io::IO, ::MIME"text/plain", alg::AMD)
    println(io, "AMD:")
    println(io, "   dense: $(alg.dense)")
    println(io, "   aggressive: $(alg.aggressive)")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", alg::SymAMD)
    println(io, "SymAMD:")
    println(io, "   dense row: $(alg.dense_row)")
    println(io, "   dense col: $(alg.dense_col)")
    println(io, "   aggressive: $(alg.aggressive)")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", alg::Spectral)
    println(io, "Spectral:")
    println(io, "   tol: $(alg.tol)")
    return nothing
end

"""
    DEFAULT_ELIMINATION_ALGORITHM = AMD()

The default algorithm.
"""
const DEFAULT_ELIMINATION_ALGORITHM = AMD()

"""
    RCM = RCMGL

The default variant of the reverse Cuthill-Mckee algorithm.
"""
const RCM = RCMGL

"""
    rcm = rcmgl

The default variant of the reverse Cuthill-Mckee algorithm.
"""
const rcm = rcmgl
