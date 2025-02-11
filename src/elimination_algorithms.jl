"""
    EliminationAlgorithm

A graph elimination algorithm. The options are

| type               | name                               | complexity | exact |
|:------------------ |:---------------------------------- |:---------- |:----- |
| [`BFS`](@ref)      | breadth-first search               | O(m)       | false |
| [`MCS`](@ref)      | maximum cardinality search         | O(m + n)   | false |
| [`LexBFS`](@ref)   | lexicographic breadth-first search | O(m + n)   | false |
| [`RCM`](@ref)      | reverse Cuthill-Mckee              | O(mΔ)      | false |
| [`AAMD`](@ref)     | approximate minimum degree         | O(mn)      | false |
| [`SymAMD`](@ref)   | column approximate minimum degree  | O(mn)      | false |
| [`MMD`](@ref)      | multiple minimum degree            | O(mn²)     | false |
| [`NodeND`](@ref)   | nested dissection                  |            | false |
| [`Spectral`](@ref) | spectral ordering                  |            | false |
| [`BT`](@ref)       | Bouchitte-Todinca                  | O(2.6183ⁿ) | true  |

for a graph with m edges, n vertices, and maximum degree Δ. The algorithm [`Spectral`](@ref) only works on connected graphs.
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
"""
struct MCS <: EliminationAlgorithm end

"""
    LexBFS <: EliminationAlgorithm

    LexBFS()

The [lexicographic breadth-first-search algorithm](https://en.wikipedia.org/wiki/Lexicographic_breadth-first_search).
"""
struct LexBFS <: EliminationAlgorithm end

"""
    RCM <: EliminationAlgorithm

The [reverse Cuthill-McKee algorithm](https://en.wikipedia.org/wiki/Cuthill%E2%80%93McKee_algorithm).
"""
struct RCM <: EliminationAlgorithm end

"""
    AAMD <: EliminationAlgorithm

    AAMD(; dense=10.0, aggressive=1.0)

The approximate minimum degree algorithm.

  - `dense`: dense row parameter
  - `aggressive`: aggressive absorption
"""
@kwdef struct AAMD <: EliminationAlgorithm
    dense::Float64 = 10.0
    aggressive::Float64 = 1.0
end

"""
    SymAMD <: EliminationAlgorithm

    SymAMD(; dense_row=10.0, dense_col=10.0, aggressive=1.0)

The column approximate minimum degree algorithm.

  - `dense_row`: dense row parameter
  - `dense_column`: dense column parameter
  - `aggressive`: aggressive absorption
"""
@kwdef struct SymAMD <: EliminationAlgorithm
    dense_row::Float64 = 10.0
    dense_col::Float64 = 10.0
    aggressive::Float64 = 1.0
end

"""
    MMD <: EliminationAlgorithm

    MMD()

The [multiple minimum degree algorithm](https://en.wikipedia.org/wiki/Minimum_degree_algorithm).
"""
struct MMD <: EliminationAlgorithm end

"""
    NodeND <: EliminationAlgorithm

    NodeND()

The [nested dissection algorithm](https://en.wikipedia.org/wiki/Nested_dissection).
"""
struct NodeND <: EliminationAlgorithm end

"""
    Spectral <: EliminationAlgorithm

    Spectral(; tol=0.0)

The spectral ordering algorithm only works on connected graphs.
In order to use it, import the package [Laplacians](https://github.com/danspielman/Laplacians.jl).

  - `tol`: tolerance for convergence
"""
@kwdef struct Spectral <: EliminationAlgorithm
    tol::Float64 = 0.0
end

"""
    BT <: EliminationAlgorithm

    BT()

The Bouchitte-Todinca algorithm.
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

function permutation(graph, alg::RCM)
    order = rcm(graph)
    return order, invperm(order)
end

function permutation(graph::BipartiteGraph{V}, alg::Union{AAMD,SymAMD,NodeND}) where {V}
    order::Vector{V}, index::Vector{V} = permutation(SparseMatrixCSC{Bool}(graph), alg)
    return order, index
end

function permutation(graph::BipartiteGraph{V}, alg::BT) where {V}
    order::Vector{V}, index::Vector{V} = permutation(Graph{Int}(graph), alg)
    return order, index
end

function permutation(matrix::SparseMatrixCSC{T,I}, alg::Union{AAMD,SymAMD}) where {T,I}
    # convert matrix
    converted::SparseMatrixCSC{T,Int} = matrix

    # run algorithm
    order::Vector{I}, index::Vector{I} = permutation(converted, alg)
    return order, index
end

function permutation(
    matrix::SparseMatrixCSC{<:Any,I}, alg::AAMD
) where {I<:Union{Int32,Int64}}
    # set parameters
    meta = AMD.Amd()
    meta.control[AMD.AMD_DENSE] = alg.dense
    meta.control[AMD.AMD_AGGRESSIVE] = alg.aggressive

    # run algorithm
    order::Vector{I} = AMD.amd(matrix, meta)
    return order, invperm(order)
end

function permutation(
    matrix::SparseMatrixCSC{<:Any,I}, alg::SymAMD
) where {I<:Union{Int32,Int64}}
    # set parameters
    meta = AMD.Colamd{I}()
    meta.knobs[AMD.COLAMD_DENSE_ROW] = alg.dense_row
    meta.knobs[AMD.COLAMD_DENSE_COL] = alg.dense_col
    meta.knobs[AMD.COLAMD_AGGRESSIVE] = alg.aggressive

    # run algorithm
    order::Vector{I} = AMD.symamd(matrix, meta)
    return order, invperm(order)
end

function permutation(graph::BipartiteGraph{V,E}, alg::MMD) where {V,E}
    # convert graph
    I = promote_type(V, E)
    converted::BipartiteGraph{I,I,Vector{I},Vector{I}} = graph

    # run algorithm
    order::Vector{V}, index::Vector{V} = permutation(converted, alg)
    return order, index
end

function permutation(graph::BipartiteGraph{I,I,Vector{I},Vector{I}}, alg::MMD) where {I}
    order = Vector{I}(undef, nv(graph))
    index = Vector{I}(undef, nv(graph))
    SpkMmd._generalmmd(nv(graph), pointers(graph), targets(graph), order, index)
    return order, index
end

function permutation(matrix::SparseMatrixCSC{T,I}, alg::NodeND) where {T,I}
    graph = Metis.graph(matrix; check_hermitian=false)
    order::Vector{I}, index::Vector{I} = Metis.permutation(graph)
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
    label = fill(false, nv(graph))
    order = Vector{V}(undef, nv(graph))

    if !iszero(nv(graph))
        root = one(V)
        component, queue = bfs!(label, order, graph, root)

        for j in vertices(graph)
            if !label[j]
                component, queue = bfs!(label, queue, graph, j)
            end
        end
    end

    return reverse!(order)
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
    head = zeros(V, nv(graph) + 1)
    prev = Vector{V}(undef, nv(graph) + 1)
    next = Vector{V}(undef, nv(graph) + 1)

    function set(i)
        @inbounds DoublyLinkedList(view(head, i), prev, next)
    end

    # run algorithm
    alpha = Vector{V}(undef, nv(graph))
    size = Vector{V}(undef, nv(graph))

    @inbounds for v in reverse(vertices(graph))
        size[v] = 1
        pushfirst!(set(1), v)
    end

    j::V = 1
    k::V = lastindex(clique)

    @inbounds for i in reverse(vertices(graph))
        v::V = 0

        if k in eachindex(clique)
            v = clique[k]
            k -= 1
        else
            v = popfirst!(set(j))
        end

        alpha[v] = i
        size[v] = 1 - size[v]

        for w in neighbors(graph, v)
            if size[w] >= 1
                delete!(set(size[w]), w)
                size[w] += 1
                pushfirst!(set(size[w]), w)
            end
        end

        j += 1

        while j >= 1 && isempty(set(j))
            j -= 1
        end
    end

    return alpha, size
end

"""
    rcm(graph)

The [reverse Cuthill-Mckee algorithm](https://en.wikipedia.org/wiki/Cuthill%E2%80%93McKee_algorithm).
"""
function rcm(graph)
    return rcm(BipartiteGraph(graph))
end

function rcm(graph::Union{BipartiteGraph,Graph,DiGraph})
    return rcm!(copy(graph))
end

# Algorithms for Sparse Linear Systems
# Scott and Tuma
# Algorithm 8.3: CM and RCM algorithms for band and profile reduction
#
# Apply the reverse Cuthill-Mckee algorithm to each connected component of a graph.
# The complexity is O(mΔ), where m = |E|.
function rcm!(graph::AbstractGraph{V}) where {V}
    label = fill(false, nv(graph))
    order = Vector{V}(undef, nv(graph))

    # sort neighbors
    scratch = Vector{V}(undef, Δout(graph))

    for j in vertices(graph)
        sort!(neighbors(graph, j); by=i -> outdegree(graph, i), scratch)
    end

    if !iszero(nv(graph))
        # find pseudo-peripheral vertex
        root = argmin(vertices(graph)) do i
            outdegree(graph, i)
        end

        # compute Cuthill-Mckee ordering
        component, queue = bfs!(label, order, graph, root)

        for j in vertices(graph)
            if !label[j]
                # compute connected component
                component, queue = bfs!(label, queue, graph, j)

                # find pseudo-peripheral vertex
                root = argmin(component) do i
                    outdegree(graph, i)
                end

                # reset labels
                label[component] .= false

                # compute Cuthill-Mckee ordering
                bfs!(label, component, graph, root)
            end
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
@views function bfs!(
    label::AbstractVector{Bool},
    queue::AbstractVector{V},
    graph::AbstractGraph{V},
    root::Integer,
) where {V}
    i = j = firstindex(queue)
    label[root] = true
    queue[j] = root

    @inbounds while i <= j
        v = queue[i]

        for w in neighbors(graph, v)
            if !label[w]
                j += 1
                label[w] = true
                queue[j] = w
            end
        end

        i += 1
    end

    return queue[begin:j], queue[i:end]
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
    n = max(ne(graph) * (2 - is_directed(graph)), nv(graph)) + 2

    flag = Vector{I}(undef, n)
    head = Vector{I}(undef, n)
    next = Vector{I}(undef, n)
    back = Vector{I}(undef, n)

    alpha = Vector{V}(undef, nv(graph))
    cell = Vector{V}(undef, nv(graph))

    fixlist = I[]

    # (implicitly) assign label ∅ to all vertices
    head[1] = 2
    back[2] = 1
    head[2] = back[1] = next[1] = flag[1] = flag[2] = 0
    c::I = 3

    # c is the number of the first empty cell
    @inbounds for v in vertices(graph)
        head[c] = v
        cell[v] = next[c - 1] = c
        flag[c] = 2
        back[c] = c - 1
        c += 1
        alpha[v] = 0
    end

    next[c - 1] = 0

    @inbounds for i in reverse(vertices(graph))
        # skip empty sets
        while iszero(next[head[1]])
            head[1] = head[head[1]]
            back[head[1]] = 1
        end

        ##########
        # select #
        ##########

        # pick next vertex to number
        p = next[head[1]]

        # delete cell of vertex from set
        next[head[1]] = next[p]

        if !iszero(next[head[1]])
            back[next[head[1]]] = head[1]
        end

        v = head[p]

        # assign v the number i
        alpha[v] = i
        empty!(fixlist)

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
                    flag[c] = 1
                    next[c] = 0
                    push!(fixlist, c)
                    h = c
                    c += 1
                end

                # add cell of w to new set
                next[cell[w]] = next[h]

                if !iszero(next[h])
                    back[next[h]] = cell[w]
                end

                flag[cell[w]] = back[cell[w]] = h
                next[h] = cell[w]
            end

            flag[fixlist] .= 0
        end
    end

    return alpha
end

function Base.show(io::IO, ::MIME"text/plain", alg::AAMD)
    println(io, "AAMD:")
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
    DEFAULT_ELIMINATION_ALGORITHM = AAMD()

The default algorithm.
"""
const DEFAULT_ELIMINATION_ALGORITHM = AAMD()
