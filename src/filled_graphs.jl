"""
    FilledGraph{V, E} <: AbstractGraph{V}

A filled graph.
"""
struct FilledGraph{V, E} <: AbstractGraph{V}
    tree::CliqueTree{V, E}
    index::Vector{V}
    ne::Int
end

function FilledGraph{V, E}(tree::CliqueTree) where {V, E}
    index = Vector{V}(undef, pointers(residuals(tree))[end] - 1)
    ne = 0

    for (i, clique) in enumerate(tree)
        index[residual(clique)] .= i
        ne += length(separator(clique)) * length(residual(clique))
        ne += (length(residual(clique)) - 1) * length(residual(clique)) ÷ 2
    end

    return FilledGraph(tree, index, ne)
end

function FilledGraph{V}(tree::CliqueTree{<:Any, E}) where {V, E}
    return FilledGraph{V, E}(tree)
end

function FilledGraph(tree::CliqueTree{V}) where {V}
    return FilledGraph{V}(tree)
end

function BipartiteGraph{V, E}(graph::FilledGraph) where {V, E}
    result = BipartiteGraph{V, E}(nv(graph), nv(graph), ne(graph))
    pointers(result)[begin] = one(E)
    j = zero(V)

    @inbounds for clique in graph.tree
        m = convert(V, length(residual(clique)))
        n = convert(V, length(separator(clique)))

        for i in oneto(m)
            j += one(V)
            pointers(result)[j + one(V)] = pointers(result)[j] + m - i + n
            neighbors(result, j)[one(V):(m - i)] = residual(clique)[(i + one(V)):m]
            neighbors(result, j)[(m - i + one(V)):(m - i + n)] = separator(clique)
        end
    end

    return result
end

function Graphs.Graph{V}(graph::FilledGraph) where {V}
    n = nv(graph)
    count = Vector{V}(undef, n)
    fadjlist = Vector{Vector{V}}(undef, n)

    @inbounds for v in reverse(vertices(graph))
        count[v] = zero(V)

        for w in neighbors(graph, v)
            count[w] += one(V)
        end
    end

    @inbounds for v in vertices(graph)
        fadjlist[v] = Vector{V}(undef, count[v] + outdegree(graph, v))
        count[v] = zero(V)
    end

    @inbounds for v in vertices(graph)
        i = count[v]

        for w in neighbors(graph, v)
            count[w] += one(V)
            fadjlist[w][count[w]] = v

            i += one(V)
            fadjlist[v][i] = w
        end
    end

    return Graph{V}(ne(graph), fadjlist)
end

function Graphs.DiGraph{V}(graph::FilledGraph) where {V}
    n = nv(graph)
    count = Vector{V}(undef, n)
    fadjlist = Vector{Vector{V}}(undef, n)
    badjlist = Vector{Vector{V}}(undef, n)

    @inbounds for v in reverse(vertices(graph))
        count[v] = zero(V)

        for w in neighbors(graph, v)
            count[w] += one(V)
        end
    end

    @inbounds for v in vertices(graph)
        fadjlist[v] = Vector{V}(undef, outdegree(graph, v))
        badjlist[v] = Vector{V}(undef, count[v])
        count[v] = zero(V)
    end

    @inbounds for v in vertices(graph)
        i = zero(V)

        for w in neighbors(graph, v)
            count[w] += one(V)
            badjlist[w][count[w]] = v

            i += one(V)
            fadjlist[v][i] = w
        end
    end

    return DiGraph{V}(ne(graph), fadjlist, badjlist)
end

function Base.Matrix{T}(graph::FilledGraph) where {T}
    return Matrix(sparse(T, graph))
end

function Base.Matrix(graph::FilledGraph)
    return Matrix(sparse(graph))
end

# Construct a sparse symmetric matrix with a given sparsity graph.
function SparseArrays.SparseMatrixCSC{T, I}(graph::FilledGraph) where {T, I}
    return SparseMatrixCSC{T}(BipartiteGraph{I, I}(graph))
end

# See above.
function SparseArrays.SparseMatrixCSC{T}(graph::FilledGraph) where {T}
    return SparseMatrixCSC{T, Int}(graph)
end

# See above.
function SparseArrays.SparseMatrixCSC(graph::FilledGraph)
    return SparseMatrixCSC{Bool}(graph)
end

# Construct the adjacency matrix of a graph.
function SparseArrays.sparse(T::Type, I::Type, graph::FilledGraph)
    matrix = SparseMatrixCSC{T, I}(graph)
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

function Base.show(io::IO, ::MIME"text/plain", graph::G) where {G <: FilledGraph}
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
        graph::FilledGraph{V, E}, i::Integer
    ) where {V, E}
    @boundscheck checkbounds(graph.index, i)
    clique = graph.tree[graph.index[i]]
    return Clique{V, E}((i + 1):last(residual(clique)), separator(clique))
end

@propagate_inbounds function Graphs.outdegree(graph::FilledGraph{V}, i::Integer) where {V}
    @boundscheck checkbounds(graph.index, i)
    clique = graph.tree[graph.index[i]]
    n::V = last(residual(clique)) + length(separator(clique)) - i
    return n
end
