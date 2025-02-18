"""
    FilledGraph{V, E} <: AbstractGraph{V}

A filled graph.
"""
struct FilledGraph{V,E} <: AbstractGraph{V}
    tree::CliqueTree{V,E}
    index::Vector{V}
    ne::Int
end

function FilledGraph{V,E}(tree::CliqueTree) where {V,E}
    index = Vector{V}(undef, pointers(residuals(tree))[end] - 1)
    ne = 0

    for (i, clique) in enumerate(tree)
        index[residual(clique)] .= i
        ne += length(separator(clique)) * length(residual(clique))
        ne += (length(residual(clique)) - 1) * length(residual(clique)) ÷ 2
    end

    return FilledGraph(tree, index, ne)
end

function FilledGraph{V}(tree::CliqueTree{<:Any,E}) where {V,E}
    return FilledGraph{V,E}(tree)
end

function FilledGraph(tree::CliqueTree{V}) where {V}
    return FilledGraph{V}(tree)
end

function BipartiteGraph{V,E}(graph::FilledGraph) where {V,E}
    result = BipartiteGraph{V,E}(nv(graph), nv(graph), ne(graph))
    pointers(result)[begin] = j = 1

    @inbounds for clique in graph.tree
        for i in eachindex(residual(clique))
            pointers(result)[j + 1] = pointers(result)[j] + length(clique) - i
            neighbors(result, j)[begin:(length(residual(clique)) - i)] .= residual(clique)[(i + 1):end]
            neighbors(result, j)[(end - length(separator(clique)) + 1):end] .= separator(
                clique
            )
            j += 1
        end
    end

    return result
end

function Base.Matrix{T}(graph::FilledGraph) where {T}
    return Matrix(sparse(T, graph))
end

function Base.Matrix(graph::FilledGraph)
    return Matrix(sparse(graph))
end

# Construct a sparse symmetric matrix with a given sparsity graph.
# The row indices of the matrix are not necessarily sorted. You can sort them as follows
#    julia> matrix = SparseMatrixCSC{T, I}(graph)
#    julia> sorted = copy(transpose(copy(transpose(matrix))))
function SparseArrays.SparseMatrixCSC{T,I}(graph::FilledGraph) where {T,I}
    return SparseMatrixCSC{T}(BipartiteGraph{I,I}(graph))
end

# See above.
function SparseArrays.SparseMatrixCSC{T}(graph::FilledGraph) where {T}
    return SparseMatrixCSC{T,Int}(graph)
end

# See above.
function SparseArrays.SparseMatrixCSC(graph::FilledGraph)
    return SparseMatrixCSC{Bool}(graph)
end

# Construct the adjacency matrix of a graph.
function SparseArrays.sparse(T::Type, I::Type, graph::FilledGraph)
    matrix = SparseMatrixCSC{T,I}(graph)
    fill!(nonzeros(matrix), 1)
    return matrix
end

# See above.
function SparseArrays.sparse(T::Type, graph::FilledGraph)
    return sparse(T, Int, graph)
end

# See above.
function SparseArrays.sparse(graph::FilledGraph)
    return sparse(Bool, graph)
end

"""
    ischordal(graph)

Determine whether a simple graph is [chordal](https://en.wikipedia.org/wiki/Chordal_graph).
"""
function ischordal(graph)
    index, size = mcs(graph)
    return isperfect(graph, invperm(index), index)
end

"""
    isperfect(graph, order::AbstractVector[, index::AbstractVector])

Determine whether an fill-reducing permutation is perfect.
"""
function isperfect(graph, order::AbstractVector, index::AbstractVector=invperm(order))
    return isperfect(BipartiteGraph(graph), order, index)
end

# Simple Linear-Time Algorithms to Test Chordality of BipartiteGraphs, Test Acyclicity of Hypergraphs, and Selectively Reduce Acyclic Hypergraphs
# Tarjan and Yannakakis
# Test for Zero Fill-In.
#
# Determine whether a fill-reducing permutation is perfect.
# The complexity is O(m + n), where m = |E| and n = |V|.
function isperfect(
    graph::AbstractGraph{V}, order::AbstractVector{V}, index::AbstractVector{V}
) where {V}
    # validate arguments
    vertices(graph) != eachindex(index) &&
        throw(ArgumentError("vertices(graph) != eachindex(index)"))
    eachindex(order) != eachindex(index) &&
        throw(ArgumentError("eachindex(order) != eachindex(index)"))

    # run algorithm
    f = Vector{V}(undef, nv(graph))
    findex = Vector{V}(undef, nv(graph))

    for (i, w) in enumerate(order)
        f[w] = w
        findex[w] = i

        for v in neighbors(graph, w)
            if index[v] < i
                findex[v] = i

                if f[v] == v
                    f[v] = w
                end
            end
        end

        for v in neighbors(graph, w)
            if index[v] < i && findex[f[v]] < i
                return false
            end
        end
    end

    return true
end

function Base.show(io::IO, ::MIME"text/plain", graph::G) where {G<:FilledGraph}
    m = ne(graph)
    n = nv(graph)
    return println(io, "{$n, $m} $G")
end

############################
# Abstract Graph Interface #
############################

function Graphs.is_directed(::Type{<:FilledGraph})
    return true
end

function Graphs.nv(graph::FilledGraph{V}) where {V}
    n::V = pointers(residuals(graph.tree))[end] - 1
    return n
end

function Graphs.ne(graph::FilledGraph)
    return graph.ne
end

@propagate_inbounds function Graphs.has_edge(graph::FilledGraph, i::Integer, j::Integer)
    @boundscheck checkbounds(graph.index, i)
    return i < j && j in graph.tree[graph.index[i]]
end

function Graphs.vertices(graph::FilledGraph{V}) where {V}
    return oneto(nv(graph))
end

@propagate_inbounds function Graphs.outneighbors(
    graph::FilledGraph{V,E}, i::Integer
) where {V,E}
    @boundscheck checkbounds(graph.index, i)
    clique = graph.tree[graph.index[i]]
    return Clique{V,E}((i + 1):last(residual(clique)), separator(clique))
end

# slow
function Graphs.inneighbors(graph::FilledGraph, i::Integer)
    return Iterators.filter(oneto(i)) do j
        has_edge(graph, j, i)
    end
end

function Graphs.all_neighbors(graph::FilledGraph, i::Integer)
    return Iterators.flatten((inneighbors(graph, i), neighbors(graph, i)))
end

@propagate_inbounds function Graphs.outdegree(graph::FilledGraph{V}, i::Integer) where {V}
    @boundscheck checkbounds(graph.index, i)
    clique = graph.tree[graph.index[i]]
    n::V = last(residual(clique)) + length(separator(clique)) - i
    return n
end

@propagate_inbounds function Graphs.indegree(graph::FilledGraph{V}, i::Integer) where {V}
    @boundscheck checkbounds(graph.tree.count, i)
    n::V = graph.tree.count[i] - 1
    return n
end
