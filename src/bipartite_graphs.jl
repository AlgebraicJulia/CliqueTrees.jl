"""
    BipartiteGraph{V, E, Ptr, Tgt} <: AbstractGraph{V}

A directed bipartite multigraph G = (U, V, E).
"""
struct BipartiteGraph{V <: Integer, E <: Integer, Ptr <: AbstractVector{E}, Tgt <: AbstractVector{V}} <: AbstractGraph{V}
    """
    U is equal to the set

        U := {1, 2, ..., nov}

    """
    nov::V

    """
    V is equal to the set

        V := {1, 2, ..., nv}

    """
    nv::V

    """
    E is equal to the set

        E := {1, 2, ..., ne}

    """
    ne::E

    """
    Each vertex v ∈ V is the source of the arcs

        {ptr[v], ptr[v] + 1, ..., ptr[v + 1] - 1} ⊆ E

    """
    ptr::Ptr

    """
    Each arc e ∈ E has target vertex

        tgt[e] ∈ V

    """
    tgt::Tgt

    function BipartiteGraph{V, E, Ptr, Tgt}(
            nov::Integer,
            nv::Integer,
            ne::Integer,
            ptr::AbstractVector,
            tgt::AbstractVector,
        ) where {V <: Integer, E <: Integer, Ptr <: AbstractVector{E}, Tgt <: AbstractVector{V}}
        @argcheck !isnegative(nov)
        @argcheck !isnegative(nv)
        @argcheck !isnegative(ne)
        @argcheck nv < length(ptr)
        @argcheck ne <= length(tgt)
        return new{V, E, Ptr, Tgt}(nov, nv, ne, ptr, tgt)
    end
end

function BipartiteGraph{V, E}(nov::Integer, nv::Integer, ne::Integer, ptr::AbstractVector, tgt::AbstractVector) where {V, E}
    graph = BipartiteGraph(
        convert(V, nov),
        convert(V, nv),
        convert(E, ne),
        convert(AbstractVector{E}, ptr),
        convert(AbstractVector{V}, tgt),
    )

    return graph
end

function BipartiteGraph{V, E}(nov::Integer, nv::Integer, ne::Integer) where {V, E}
    ptr = Vector{E}(undef, nv + 1)
    tgt = Vector{V}(undef, ne)
    return BipartiteGraph{V, E}(nov, nv, ne, ptr, tgt)
end

function BipartiteGraph{V, E, Ptr, Tgt}(graph::BipartiteGraph) where {V, E, Ptr, Tgt}
    return BipartiteGraph{V, E, Ptr, Tgt}(nov(graph), nv(graph), ne(graph), pointers(graph), targets(graph))
end

function BipartiteGraph{V, E}(graph::AbstractGraph) where {V, E}
    n = convert(V, nv(graph)); nn = n + one(V)
    m = convert(E, de(graph))
    ptr = Vector{E}(undef, nn)
    tgt = Vector{V}(undef, m)
    ptr[begin] = p = one(E)

    @inbounds for j in vertices(graph)
        jj = convert(V, j)

        for i in neighbors(graph, jj)
            ii = convert(V, i)
            tgt[p] = ii; p += one(E)
        end

        jj += one(V); ptr[jj] = p
    end

    return BipartiteGraph{V, E}(n, n, p - one(E), ptr, tgt)
end

function BipartiteGraph{V, E}(graph::BipartiteGraph) where {V, E}
    graph = BipartiteGraph{V, E}(
        nov(graph),
        nv(graph),
        ne(graph),
        pointers(graph),
        targets(graph),
    )

    return graph
end

function BipartiteGraph{V, E}(matrix::SparseMatrixCSC) where {V, E}
    graph = BipartiteGraph{V, E}(
        size(matrix)...,
        nnz(matrix),
        getcolptr(matrix),
        rowvals(matrix),
    )

    return graph
end

function BipartiteGraph{V, E}(matrix::AbstractMatrix) where {V, E}
    graph = BipartiteGraph{V, E}(sparse(matrix))
    return graph
end

function BipartiteGraph(
        nov::V,
        nv::V,
        ne::E,
        ptr::Ptr,
        tgt::Tgt,
    ) where {V <: Integer, E <: Integer, Ptr <: AbstractVector{E}, Tgt <: AbstractVector{V}}
    return BipartiteGraph{V, E, Ptr, Tgt}(nov, nv, ne, ptr, tgt)
end

function BipartiteGraph(graph::AbstractGraph{V}) where {V}
    E = etype(graph)
    return BipartiteGraph{V, E}(graph)
end

function BipartiteGraph(matrix::SparseMatrixCSC{<:Any, I}) where {I}
    return BipartiteGraph{I, I}(matrix)
end

function BipartiteGraph(matrix::AbstractMatrix)
    return BipartiteGraph(sparse(matrix))
end

function Base.Matrix{T}(graph::BipartiteGraph) where {T}
    return Matrix(sparse(T, graph))
end

function Base.Matrix(graph::BipartiteGraph)
    return Matrix(sparse(graph))
end

# Construct a sparse symmetric matrix with a given sparsity graph.
# The row indices of the matrix are not necessarily sorted. You can sort them as follows
#    julia> matrix = SparseMatrixCSC{T, I}(graph)
#    julia> sorted = copy(transpose(copy(transpose(matrix))))
function SparseArrays.SparseMatrixCSC{T, I}(graph::BipartiteGraph) where {T, I}
    colptr::Vector{I} = pointers(graph)
    rowval::Vector{I} = targets(graph)
    nzval = Vector{T}(undef, ne(graph))
    return SparseMatrixCSC{T, I}(nov(graph), nv(graph), colptr, rowval, nzval)
end

# See above.
function SparseArrays.SparseMatrixCSC{T}(graph::BipartiteGraph{V, E}) where {T, V, E}
    I = promote_type(V, E)
    return SparseMatrixCSC{T, I}(graph)
end

# See above.
function SparseArrays.SparseMatrixCSC(graph::BipartiteGraph)
    return SparseMatrixCSC{Bool}(graph)
end

# Construct the adjacency matrix of a graph.
function SparseArrays.sparse(::Type{T}, ::Type{I}, graph::BipartiteGraph) where {T, I}
    graph = reverse(reverse(graph))
    matrix = SparseMatrixCSC{T, I}(graph)
    fill!(nonzeros(matrix), 1)
    return matrix
end

# See above.
function SparseArrays.sparse(::Type{T}, graph::BipartiteGraph{V, E}) where {T, V, E}
    I = promote_type(V, E)
    return sparse(T, I, graph)
end

# See above.
function SparseArrays.sparse(graph::BipartiteGraph)
    return sparse(Bool, graph)
end

# Direct Methods for Sparse Linear Systems §2.11
# Davis
# cs_symperm
#
# Permute the vertices of a graph and orient the edges from lower to higher.
# The complexity is O(m), where m = |E|.
function sympermute(graph, index::AbstractVector, order::Ordering)
    return sympermute(BipartiteGraph(graph), index, order)
end

function sympermute(graph::AbstractGraph{V}, index::AbstractVector, order::Ordering) where {V}
    E = etype(graph); m = half(de(graph)); n = nv(graph); nn = n + one(V)
    count = Vector{E}(undef, n)
    ptr = Vector{E}(undef, nn)
    tgt = Vector{V}(undef, m)
    return sympermute!_impl!(count, ptr, tgt, graph, index, order)
end

function sympermute!(
        result::BipartiteGraph{V, E},
        graph::AbstractGraph,
        index::AbstractVector,
        order::Ordering,
    ) where {V, E}
    n = nv(graph)
    count = Vector{E}(undef, n)
    sympermute!_impl!(count, result, graph, index, order)
    return result
end

function sympermute!_impl!(
        count::AbstractVector{E},
        target::BipartiteGraph{V, E},
        source::AbstractGraph{V},
        index::AbstractVector{V},
        order::Ordering
    ) where {V, E}
    ptr = pointers(target)
    tgt = targets(target)

    sympermute!_impl!(count, ptr,
        tgt, source, index, order)

    return
end

function sympermute!_impl!(
        count::AbstractVector{E},
        ptr::AbstractVector{E},
        tgt::AbstractVector{V},
        source::AbstractGraph,
        index::AbstractVector,
        order::Ordering,
    ) where {V, E}
    @argcheck nv(source) < length(ptr)
    @argcheck nv(source) <= length(count)
    @argcheck nv(source) <= length(index)
    @argcheck half(de(source)) <= length(tgt)
    n = nv(source)

    @inbounds for i in oneto(n)
        count[i] = zero(E)
    end

    @inbounds for j in vertices(source)
        jj = index[j]

        for i in neighbors(source, j)
            if lt(order, i, j)
                ii = index[i]
                kk = jj

                if lt(order, kk, ii)
                    kk = ii
                end

                count[kk] += one(E)
            end
        end
    end

    ptr[begin] = p = one(E)

    @inbounds for i in oneto(n)
        ii = i + one(V); pp = p + count[i]
        count[i] = p; ptr[ii] = pp
        p = pp
    end

    @inbounds for j in vertices(source)
        jj = index[j]

        for i in neighbors(source, j)
            if lt(order, i, j)
                ii = index[i]
                kk = jj

                if lt(order, kk, ii)
                    ii, kk = kk, ii
                end

                tgt[count[kk]] = ii
                count[kk] += one(E)
            end
        end
    end

    m = p - one(E)
    return BipartiteGraph(n, n, m, ptr, tgt)
end

# Compute the transpose of a graph.
# The transposed graph has sorted adjacency lists.
function Base.reverse(graph::BipartiteGraph{V, E}) where {V, E}
    result = BipartiteGraph{V, E}(nv(graph), nov(graph), ne(graph))
    return reverse!(result, graph)
end

function Base.reverse!(result::BipartiteGraph{V, E}, graph::AbstractGraph{V}) where {V, E}
    n = nov(graph)
    count = Vector{E}(undef, n)
    reverse!_impl!(count, result, graph)
    return result
end

function reverse!_impl!(
        count::AbstractVector{E},
        target::BipartiteGraph{V, E},
        source::AbstractGraph{V},
    ) where {V, E}
    @argcheck nov(target) == nv(source)
    @argcheck nov(source) == nv(target)
    @argcheck nov(source) <= length(count)
    n = nv(target)

    @inbounds for i in oneto(n)
        count[i] = zero(E)
    end

    @inbounds for j in vertices(source), i in neighbors(source, j)
        count[i] += one(E)
    end

    pointers(target)[begin] = p = one(E)

    @inbounds for i in oneto(n)
        ii = i + one(V); pp = p + count[i]
        count[i] = p; pointers(target)[ii] = pp
        p = pp
    end

    @inbounds for j in vertices(source), i in neighbors(source, j)
        targets(target)[count[i]] = j
        count[i] += one(E)
    end

    return
end

function simplegraph(::Type{V}, ::Type{E}, graph) where {V, E}
    return simplegraph(V, E, BipartiteGraph(graph))
end

function simplegraph(::Type{V}, ::Type{E}, graph::AbstractGraph) where {V, E}
    n = convert(V, nv(graph)); nn = n + one(V)
    m = convert(E, de(graph))
    ptr = Vector{E}(undef, nn)
    tgt = Vector{V}(undef, m)
    ptr[begin] = p = one(E)

    @inbounds for j in vertices(graph)
        jj = convert(V, j)

        for i in neighbors(graph, jj)
            ii = convert(V, i)

            if ii != jj
                tgt[p] = ii; p += one(E)
            end
        end

        jj += one(V); ptr[jj] = p
    end

    return BipartiteGraph{V, E}(n, n, p - one(E), ptr, tgt)
end

function simplegraph(graph)
    return simplegraph(BipartiteGraph(graph))
end

function simplegraph(graph::AbstractGraph{V}) where {V}
    E = etype(graph)
    return simplegraph(V, E, graph)
end

function pointers(graph::BipartiteGraph)
    return graph.ptr
end

function targets(graph::BipartiteGraph)
    return graph.tgt
end

function Base.convert(::Type{BipartiteGraph{V, E, Ptr, Tgt}}, graph) where {V, E, Ptr, Tgt}
    return BipartiteGraph{V, E, Ptr, Tgt}(graph)
end

function Base.convert(::Type{BipartiteGraph{V, E, Ptr, Tgt}}, graph::BipartiteGraph{V, E, Ptr, Tgt}) where {V, E, Ptr, Tgt}
    return graph
end

function Base.show(io::IO, ::MIME"text/plain", graph::G) where {G <: BipartiteGraph}
    h = nov(graph)
    n = nv(graph)
    m = ne(graph)
    return println(io, "{$h, $n, $m} $G")
end

function Base.:(==)(left::BipartiteGraph, right::BipartiteGraph)
    if nov(left) != nov(right)
        return false
    end

    if nv(left) != nv(right)
        return false
    end

    if ne(left) != ne(right)
        return false
    end

    for v in vertices(left)
        if neighbors(left, v) != neighbors(right, v)
            return false
        end
    end

    return true
end

function Base.copy(graph::BipartiteGraph)
    return BipartiteGraph(
        nov(graph),
        nv(graph),
        ne(graph),
        copy(pointers(graph)),
        copy(targets(graph)),
    )
end

function Base.copy!(dst::BipartiteGraph, src::BipartiteGraph)
    @argcheck nov(dst) == nov(src)
    @argcheck nv(dst) == nv(src)
    @argcheck ne(dst) == ne(src)
    copyto!(pointers(dst), pointers(src))
    copyto!(targets(dst), targets(src))
    return dst
end

function Base.copy!(dst::BipartiteGraph{V, E, Ptr, OneTo{V}}, src::BipartiteGraph) where {V <: Integer, E <: Integer, Ptr <: AbstractVector{E}}
    @argcheck nov(dst) == nov(src)
    @argcheck nv(dst) == nv(src)
    @argcheck ne(dst) == ne(src)
    @argcheck targets(dst) == targets(src)
    copyto!(pointers(dst), pointers(src))
    return dst
end

#############################
# Bipartite Graph Interface #
#############################

function nov(graph::BipartiteGraph)
    return graph.nov
end

function nov(graph::AbstractGraph)
    return nv(graph)
end

function outvertices(graph::AbstractGraph)
    return oneto(nov(graph))
end

function etype(::G) where {G}
    return etype(G)
end

function etype(::Type)
    return Int
end

function etype(::Type{<:BipartiteGraph{<:Any, E}}) where {E}
    return E
end

############################
# Abstract Graph Interface #
############################

function Base.zero(::Type{BipartiteGraph{V, E, Ptr, Tgt}}) where {V, E, Ptr, Tgt}
    return BipartiteGraph{V, E, Ptr, Tgt}(
        zero(V),
        zero(V),
        zero(E),
        oneto(one(E)),
        oneto(zero(V)),
    )
end

function Graphs.is_directed(::Type{<:BipartiteGraph})
    return true
end

function Graphs.nv(graph::BipartiteGraph)
    return graph.nv
end

function Graphs.ne(graph::BipartiteGraph)
    return graph.ne
end

# slow
function Graphs.has_edge(graph::BipartiteGraph, i::Integer, j::Integer)
    return j in neighbors(graph, i)
end

function Graphs.vertices(graph::BipartiteGraph{V}) where {V}
    return oneto(nv(graph))
end

@propagate_inbounds function Graphs.outneighbors(graph::BipartiteGraph{<:Any, E}, i::I) where {E, I <: Integer}
    j = i + one(I); ptr = pointers(graph); tgt = targets(graph)
    @boundscheck checkbounds(vertices(graph), i)
    @inbounds strt = ptr[i]
    @inbounds stop = ptr[j]
    @inbounds neighbors = view(tgt, strt:stop - one(E))
    return neighbors
end

# slow
function Graphs.inneighbors(graph::BipartiteGraph, i::Integer)
    return Iterators.filter(vertices(graph)) do j
        has_edge(graph, j, i)
    end
end

@propagate_inbounds function Graphs.outdegree(graph::BipartiteGraph, i::I) where {I <: Integer}
    j = i + one(I); ptr = pointers(graph); tgt = targets(graph)
    @boundscheck checkbounds(vertices(graph), i)
    @inbounds strt = ptr[i]
    @inbounds stop = ptr[j]
    degree = convert(Int, stop - strt)
    return degree
end

# slow
function Graphs.indegree(graph::BipartiteGraph{V}, i::Integer) where {V}
    n::Int = sum(j -> has_edge(graph, j, i), vertices(graph))
    return n
end

function Graphs.Δout(graph::BipartiteGraph)
    return maximum(vertices(graph); init = 0) do i
        outdegree(graph, i)
    end
end
