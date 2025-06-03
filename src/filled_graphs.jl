"""
    FilledGraph{V, E} <: AbstractGraph{V}

A filled graph.
"""
struct FilledGraph{V, E} <: AbstractGraph{V}
    tree::CliqueTree{V, E}
    index::Vector{V}
    ne::E
end

function FilledGraph{V, E}(tree::CliqueTree) where {V, E}
    n = ne(residuals(tree)); m = zero(E); i = zero(V)
    index = Vector{V}(undef, n)

    @inbounds for bag in tree
        res = residual(bag)
        sep = separator(bag)
        nres = convert(E, length(res))
        nsep = convert(E, length(sep))

        index[res] .= i += one(V)
        m += nsep * nres + half(nres * (nres - one(E)))
    end

    return FilledGraph{V, E}(tree, index, m)
end

function FilledGraph{V}(tree::CliqueTree{<:Any, E}) where {V, E}
    return FilledGraph{V, E}(tree)
end

function FilledGraph(tree::CliqueTree{V}) where {V}
    return FilledGraph{V}(tree)
end

function BipartiteGraph{V, E}(graph::FilledGraph) where {V, E}
    n = nv(graph); m = ne(graph)
    result = BipartiteGraph{V, E}(n, n, m)
    j = one(V); pointers(result)[j] = p = one(E)

    @inbounds for bag in graph.tree
        res = residual(bag)
        sep = separator(bag)
        nres = convert(V, length(res))
        nsep = convert(V, length(sep))

        for i in oneto(nres)
            src = i + one(V); num = nres - i
            copyto!(targets(result), p, res, src, num)

            p += convert(E, num); src = one(V); num = nsep
            copyto!(targets(result), p, sep, src, num)

            j += one(V); pointers(result)[j] = p += convert(E, num)
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

    m = Int(ne(graph))
    return Graph{V}(m, fadjlist)
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

    m = Int(ne(graph))
    return DiGraph{V}(m, fadjlist, badjlist)
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
function SparseArrays.SparseMatrixCSC{T}(graph::FilledGraph{V, E}) where {T, V, E}
    I = promote_type(V, E)
    return SparseMatrixCSC{T, I}(graph)
end

# See above.
function SparseArrays.SparseMatrixCSC(graph::FilledGraph)
    return SparseMatrixCSC{Bool}(graph)
end

# Construct the adjacency matrix of a graph.
function SparseArrays.sparse(::Type{T}, ::Type{I}, graph::FilledGraph) where {T, I}
    matrix = SparseMatrixCSC{T, I}(graph)
    fill!(nonzeros(matrix), one(T))
    return matrix
end

# See above.
function SparseArrays.sparse(::Type{T}, graph::FilledGraph{V, E}) where {T, V, E}
    I = promote_type(V, E)
    return sparse(T, I, graph)
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
    return ne(residuals(graph.tree))
end

function Graphs.ne(graph::FilledGraph)
    return graph.ne
end

@propagate_inbounds function Graphs.has_edge(graph::FilledGraph, i::Integer, j::Integer)
    @boundscheck checkbounds(graph.index, i)
    @inbounds bag = graph.tree[graph.index[i]]
    return i < j && j in bag
end

function Graphs.vertices(graph::FilledGraph{V}) where {V}
    return oneto(nv(graph))
end

@propagate_inbounds function Graphs.outneighbors(graph::FilledGraph{V, E}, i::Integer) where {V, E}
    ii = convert(V, i) + one(V)
    @boundscheck checkbounds(graph.index, i)
    @inbounds bag = graph.tree[graph.index[i]]
    res = residual(bag)
    sep = separator(bag)
    return Clique{V, E}(ii:last(res), sep)
end

@propagate_inbounds function Graphs.outdegree(graph::FilledGraph{V}, i::Integer) where {V}
    @boundscheck checkbounds(graph.index, i)
    @inbounds bag = graph.tree[graph.index[i]]
    res = residual(bag)
    sep = separator(bag)
    return convert(Int, last(res)) + length(sep) - convert(Int, i)
end
