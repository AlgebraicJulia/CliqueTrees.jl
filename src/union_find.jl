struct UnionFind{I <: Integer, Rank <: AbstractVector{I}, Parent <: AbstractVector{I}, Stack <: AbstractVector{I}} <: AbstractVector{I}
    n::I
    rank::Rank
    parent::Parent
    stack::Stack
end

function UnionFind{I}(n::Integer) where {I <: Integer}
    rank = FVector{I}(undef, n)
    parent = FVector{I}(undef, n)
    stack = FVector{I}(undef, n)
    return UnionFind(convert(I, n), rank, parent, stack)
end

@propagate_inbounds function Base.union!(uf::UnionFind{I}, i::I, j::I) where {I <: Integer}
    @boundscheck checkbounds(uf.parent, i)
    @boundscheck checkbounds(uf.parent, j)
    @inbounds ii = uf.rank[i]
    @inbounds jj = uf.rank[j]

    if ii < jj
        i, j = j, i
    elseif ii == jj
        @inbounds uf.rank[i] += one(I)
    end

    @inbounds uf.parent[j] = i
    return i
end

############################
# Abstract Array Interface #
############################

function Base.IndexStyle(::Type{<:UnionFind})
    return IndexLinear()
end

function Base.size(uf::UnionFind)
    return (uf.n,)
end

@propagate_inbounds function Base.getindex(uf::UnionFind{I}, i::I) where {I <: Integer}
    @boundscheck checkbounds(oneto(uf.n), i)
    @inbounds ii = uf.parent[i]
    n = zero(I)

    @inbounds while ispositive(ii)
        n += one(I); uf.stack[n] = i
        i = ii; ii = uf.parent[i]
    end

    ii = i; nn = n

    @inbounds for n in oneto(nn)
        i = uf.stack[n]
        uf.parent[i] = ii
    end

    return ii
end
