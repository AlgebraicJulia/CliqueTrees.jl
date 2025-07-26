struct UnionFind{
        I <: Integer,
        Rank <: AbstractVector{I},
        Parent <: AbstractVector{I},
        Stack <: AbstractVector{I},
    }
    rank::Rank
    parent::Parent
    stack::Stack
end

function UnionFind{I}(n::Integer) where {I}
    rank = FVector{I}(undef, n)
    parent = FVector{I}(undef, n)
    stack = FVector{I}(undef, n)
    return UnionFind(rank, parent, stack)
end

@propagate_inbounds function find!(uf::UnionFind{I}, i::I) where {I}
    @boundscheck checkbounds(uf.parent, i)
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

@propagate_inbounds function rootunion!(uf::UnionFind{I}, i::I, j::I) where {I}
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
