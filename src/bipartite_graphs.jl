# A simple bipartite graph (U, V, E). If U = V, then you can think of this as a directed graph.
# This type implements the abstract graph interface.
struct BipartiteGraph{V <: Integer, E <: Integer, Ptr <: AbstractVector{E}, Tgt <: AbstractVector{V}} <:
    AbstractGraph{V}
    nov::V
    ptr::Ptr
    tgt::Tgt
end

function BipartiteGraph{V, E}(
        nov::Integer, ptr::AbstractVector, tgt::AbstractVector
    ) where {V, E}
    nov::V = nov
    ptr::AbstractVector{E} = ptr
    tgt::AbstractVector{V} = tgt
    return BipartiteGraph(nov, ptr, tgt)
end

function BipartiteGraph{V, E}(nov::Integer, nv::Integer, ne::Integer) where {V, E}
    tgt = Vector{V}(undef, ne)
    return BipartiteGraph{V, E}(nov, nv, tgt)
end

function BipartiteGraph{V, E}(nov::Integer, nv::Integer, tgt::AbstractVector) where {V, E}
    ptr = Vector{E}(undef, nv + 1)
    return BipartiteGraph{V, E}(nov, ptr, tgt)
end

function BipartiteGraph{V}(
        nov::Integer, ptr::AbstractVector{E}, tgt::AbstractVector
    ) where {V, E}
    return BipartiteGraph{V, E}(nov, ptr, tgt)
end

function BipartiteGraph{V}(nov::Integer, ptr::AbstractVector{E}, ne::Integer) where {V, E}
    tgt = Vector{V}(undef, ne)
    return BipartiteGraph{V, E}(nov, ptr, tgt)
end

function BipartiteGraph(nov::Integer, ptr::AbstractVector, tgt::AbstractVector{V}) where {V}
    return BipartiteGraph{V}(nov, ptr, tgt)
end

function BipartiteGraph{V, E, Ptr, Tgt}(graph::BipartiteGraph) where {V, E, Ptr, Tgt}
    return BipartiteGraph{V, E, Ptr, Tgt}(nov(graph), pointers(graph), targets(graph))
end

function BipartiteGraph{V, E}(graph::AbstractGraph) where {V, E}
    n = nv(graph); m = de(graph)
    result = BipartiteGraph{V, E}(n, n, m)
    pointers(result)[begin] = one(E)

    for j in vertices(graph)
        jj = j + one(V)
        pointers(result)[jj] = pointers(result)[j] + convert(E, eltypedegree(graph, j))
        copyto!(neighbors(result, j), neighbors(graph, j))
    end

    resize!(targets(result), last(pointers(result)) - one(E))
    return result
end

function BipartiteGraph{V, E}(graph::BipartiteGraph) where {V, E}
    return BipartiteGraph{V, E}(nov(graph), pointers(graph), targets(graph))
end

function BipartiteGraph{V, E}(matrix::SparseMatrixCSC) where {V, E}
    return BipartiteGraph{V, E}(size(matrix, 1), getcolptr(matrix), rowvals(matrix))
end

function BipartiteGraph{V, E}(matrix::AbstractMatrix) where {V, E}
    return BipartiteGraph{V, E}(sparse(matrix))
end

function BipartiteGraph{V}(graph::AbstractGraph) where {V}
    E = etype(graph)
    return BipartiteGraph{V, E}(graph)
end

function BipartiteGraph{V}(graph::BipartiteGraph) where {V}
    return BipartiteGraph{V}(nov(graph), pointers(graph), targets(graph))
end

function BipartiteGraph{V}(matrix::SparseMatrixCSC) where {V}
    return BipartiteGraph{V}(size(matrix, 1), getcolptr(matrix), rowvals(matrix))
end

function BipartiteGraph{V}(matrix::AbstractMatrix) where {V}
    return BipartiteGraph{V}(sparse(matrix))
end

function BipartiteGraph(graph::AbstractGraph{V}) where {V}
    return BipartiteGraph{V}(graph)
end

function BipartiteGraph(graph::BipartiteGraph)
    return BipartiteGraph(nov(graph), pointers(graph), targets(graph))
end

function BipartiteGraph(matrix::SparseMatrixCSC)
    return BipartiteGraph(size(matrix, 1), getcolptr(matrix), rowvals(matrix))
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
    # sort adjacency lists
    graph = reverse(reverse(graph))

    # set weights
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

function sympermute(graph::AbstractGraph, index::AbstractVector, order::Ordering)
    return sympermute(etype(graph), graph, index, order)
end

function sympermute(
        ::Type{E}, graph::AbstractGraph{V}, index::AbstractVector, order::Ordering
    ) where {V, E}
    # validate arguments
    @argcheck vertices(graph) == eachindex(index)

    # compute column counts
    total = zero(E)
    count = Vector{E}(undef, nv(graph) + one(V))
    count[one(V)] = one(E)
    count[two(V):(nv(graph) + one(V))] .= zero(E)

    @inbounds for j in vertices(graph)
        for i in neighbors(graph, j)
            if lt(order, i, j)
                u = index[i]
                v = index[j]

                if lt(order, v, u)
                    u, v = v, u
                end

                total += one(E)
                count[v + one(V)] += one(E)
            end
        end
    end

    # permute graph
    result = BipartiteGraph{V, E}(nv(graph), nv(graph), total)
    count .= cumsum!(pointers(result), count)

    @inbounds for j in vertices(graph)
        for i in neighbors(graph, j)
            if lt(order, i, j)
                u = index[i]
                v = index[j]

                if lt(order, v, u)
                    u, v = v, u
                end

                targets(result)[count[v]] = u
                count[v] += one(E)
            end
        end
    end

    return result
end

function sympermute!(
        result::BipartiteGraph{V, E},
        graph::AbstractGraph,
        index::AbstractVector,
        order::Ordering,
    ) where {V, E}
    n = nv(graph); nn = n + one(V)
    count = Vector{E}(undef, nn)
    sympermute!_impl!(count, result, graph, index, order)
    return result
end

function sympermute!_impl!(
        count::AbstractVector{E},
        target::BipartiteGraph{V, E},
        source::AbstractGraph,
        index::AbstractVector,
        order::Ordering,
    ) where {V, E}
    @argcheck nv(target) == nv(source)
    @argcheck nv(target) <= length(count)
    @argcheck nv(target) <= length(index)
    n = nv(target); nn = n + one(V)
    fill!(view(count, oneto(nn)), zero(E))

    @inbounds for j in vertices(source)
        jj = index[j]

        for i in neighbors(source, j)
            if lt(order, i, j)
                ii = index[i]
                kk = jj

                if lt(order, kk, ii)
                    kk = ii
                end

                count[kk + one(V)] += one(E)
            end
        end
    end

    count[begin] = one(E)
    cumsum!(pointers(target), count)
    copyto!(count, one(V), pointers(target), one(V), nn)

    @inbounds for j in vertices(source)
        jj = index[j]

        for i in neighbors(source, j)
            if lt(order, i, j)
                ii = index[i]
                kk = jj

                if lt(order, kk, ii)
                    ii, kk = kk, ii
                end

                targets(target)[count[kk]] = ii
                count[kk] += one(E)
            end
        end
    end

    return
end

# Compute the transpose of a graph.
# The transposed graph has sorted adjacency lists.
function Base.reverse(graph::BipartiteGraph{V, E}) where {V, E}
    result = BipartiteGraph{V, E}(nv(graph), nov(graph), ne(graph))
    return reverse!(result, graph)
end

function Base.reverse!(result::BipartiteGraph{V, E}, graph::AbstractGraph{V}) where {V, E}
    n = nov(graph); nn = n + one(V)
    count = Vector{E}(undef, nn)
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
    n = nv(target); nn = n + one(V)
    fill!(view(count, oneto(nn)), zero(E))

    @inbounds for j in vertices(source), i in neighbors(source, j)
        ii = i + one(V); count[ii] += one(E)
    end

    count[begin] = one(E)
    cumsum!(pointers(target), count)
    copyto!(count, one(V), pointers(target), one(V), nn)

    @inbounds for j in vertices(source), i in neighbors(source, j)
        targets(target)[count[i]] = j
        count[i] += one(E)
    end

    return
end

function simplegraph(::Type{V}, ::Type{E}, graph) where {V, E}
    return simplegraph(V, E, BipartiteGraph(graph))
end

function simplegraph(::Type{V}, ::Type{E}, graph::AbstractGraph{VV}) where {V, E, VV}
    m = de(graph); n = nv(graph)
    simple = BipartiteGraph{V, E}(n, n, m)
    pointers(simple)[begin] = p = one(E)

    @inbounds for v in vertices(graph)
        for w in neighbors(graph, v)
            if v != w
                targets(simple)[p] = w
                p += one(E)
            end
        end

        vv = v + one(VV)
        pointers(simple)[vv] = p
    end

    pp = p - one(E)
    resize!(targets(simple), pp)
    return simple
end

function simplegraph(::Type{V}, graph) where {V}
    return simplegraph(V, BipartiteGraph(graph))
end

function simplegraph(::Type{V}, graph::AbstractGraph) where {V}
    return simplegraph(V, etype(graph), graph)
end

function simplegraph(graph)
    return simplegraph(BipartiteGraph(graph))
end

function simplegraph(graph::AbstractGraph{V}) where {V}
    return simplegraph(V, graph)
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

function Base.convert(
        ::Type{BipartiteGraph{V, E, Ptr, Tgt}}, graph::BipartiteGraph{V, E, Ptr, Tgt}
    ) where {V, E, Ptr, Tgt}
    return graph
end

function Base.show(io::IO, ::MIME"text/plain", graph::G) where {G <: BipartiteGraph}
    m = ne(graph)
    n = nv(graph)
    return println(io, "{$n, $m} $G")
end

function Base.:(==)(left::BipartiteGraph, right::BipartiteGraph)
    return nov(left) == nov(right) &&
        pointers(left) == pointers(right) &&
        targets(left) == targets(right)
end

function Base.copy(graph::BipartiteGraph)
    return BipartiteGraph(nov(graph), copy(pointers(graph)), copy(targets(graph)))
end

function Base.copy!(dst::BipartiteGraph, src::BipartiteGraph)
    @argcheck nov(dst) == nov(src)
    copy!(pointers(dst), pointers(src))
    copy!(targets(dst), targets(src))
    return dst
end

function Base.copy!(dst::BipartiteGraph{V, E, Ptr, OneTo{V}}, src::BipartiteGraph) where {V <: Integer, E <: Integer, Ptr <: AbstractVector{E}}
    @argcheck nov(dst) == nov(src)
    @argcheck targets(dst) == targets(src)
    copy!(pointers(dst), pointers(src))
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
    return BipartiteGraph{V, E, Ptr, Tgt}(0, oneto(1), oneto(0))
end

function Graphs.is_directed(::Type{<:BipartiteGraph})
    return true
end

function Graphs.nv(graph::BipartiteGraph{V}) where {V}
    n::V = length(pointers(graph)) - 1
    return n
end

function Graphs.ne(graph::BipartiteGraph{<:Any, E}) where {E}
    m::E = length(targets(graph))
    return m
end

# slow
function Graphs.has_edge(graph::BipartiteGraph, i::Integer, j::Integer)
    return j in neighbors(graph, i)
end

function Graphs.vertices(graph::BipartiteGraph{V}) where {V}
    return oneto(nv(graph))
end

@propagate_inbounds function Graphs.outneighbors(graph::BipartiteGraph{<:Any, E}, i::I) where {E, I <: Integer}
    ii = i + one(I)
    @boundscheck checkbounds(pointers(graph), i)
    @boundscheck checkbounds(pointers(graph), ii)
    @inbounds p = pointers(graph)[i]
    @inbounds pp = pointers(graph)[ii]
    return @view targets(graph)[p:(pp - one(E))]
end

# slow
function Graphs.inneighbors(graph::BipartiteGraph, i::Integer)
    return Iterators.filter(vertices(graph)) do j
        has_edge(graph, j, i)
    end
end

@propagate_inbounds function Graphs.outdegree(graph::BipartiteGraph, i::I) where {I <: Integer}
    ii = i + one(I)
    @boundscheck checkbounds(pointers(graph), i)
    @boundscheck checkbounds(pointers(graph), ii)
    @inbounds p = pointers(graph)[i]
    @inbounds pp = pointers(graph)[ii]
    n::Int = pp - p
    return n
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
