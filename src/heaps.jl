struct Heap{I, K}
    num::Scalar{I}
    key::Vector{K}
    inv::Vector{I}
    heap::Vector{I}
end

function Heap{I, K}(n::Integer) where {I, K}
    num = fill(zero(I))
    key = Vector{K}(undef, n)
    inv = Vector{I}(undef, n)
    heap = Vector{I}(undef, n)
    return Heap{I, K}(num, key, inv, heap)
end

@propagate_inbounds function hrise!(heap::Heap, v::Integer)
    @boundscheck checkbounds(heap.inv, v)
    @inbounds _hrise!(heap, heap.inv[v])
    return
end

@propagate_inbounds function hfall!(heap::Heap, v::Integer)
    @boundscheck checkbounds(heap.inv, v)
    @inbounds _hfall!(heap, heap.inv[v])
    return
end

function hfall!(heap::Heap)
    @inbounds for i in reverse(oneto(heap.num[]))
        _hfall!(heap, i)
    end

    return
end

@propagate_inbounds function _hrise!(heap::Heap{I}, i::I) where {I}
    @boundscheck checkbounds(heap.heap, i)
    j = i ÷ two(I)

    @inbounds while j > zero(I)
        if heap.key[heap.heap[j]] > heap.key[heap.heap[i]]
            swap!(heap.heap, i, j)
            swap!(heap.inv, heap.heap[i], heap.heap[j])
            i = j
            j = i ÷ two(I)
        else
            break
        end
    end

    return
end

@propagate_inbounds function _hfall!(heap::Heap{I}, i::I) where {I}
    @boundscheck checkbounds(heap.heap, i)
    j = i * two(I)

    @inbounds while j <= heap.num[]
        if j < heap.num[] && heap.key[heap.heap[j + one(I)]] < heap.key[heap.heap[j]]
            j += one(I)
        end

        if heap.key[heap.heap[i]] > heap.key[heap.heap[j]]
            swap!(heap.heap, i, j)
            swap!(heap.inv, heap.heap[i], heap.heap[j])
            i = j
            j = i * two(I)
        else
            break
        end
    end

    return
end

@propagate_inbounds function Base.push!(heap::Heap{I}, (v, k)::Pair) where {I}
    @boundscheck checkbounds(heap.key, v)
    @inbounds heap.key[v] = k
    @inbounds i = heap.inv[v] = heap.num[] += one(I)
    @inbounds heap.heap[i] = v
    return heap
end

@propagate_inbounds function Base.delete!(heap::Heap{I}, v::I) where {I}
    @boundscheck checkbounds(heap.key, v)
    @inbounds i = heap.inv[v]
    @inbounds k = heap.key[heap.heap[i]]
    @inbounds vv = heap.heap[i] = heap.heap[heap.num[]]
    @inbounds heap.inv[vv] = i
    @inbounds kk = heap.key[vv]
    heap.num[] -= one(I)

    if k < kk
        @inbounds _hfall!(heap, i)
    else
        @inbounds _hrise!(heap, i)
    end

    return heap
end

function Base.argmin(heap::Heap)
    @argcheck !isempty(heap)
    return first(heap.heap)
end

function Base.minimum(heap::Heap)
    v = argmin(heap)
    @inbounds k = heap[v]
    return k
end

function Base.length(heap::Heap)
    return heap.num[]
end

function Base.isempty(heap::Heap)
    return iszero(length(heap))
end

@propagate_inbounds function Base.getindex(heap::Heap, i::Integer)
    @boundscheck checkbounds(heap.key, i)
    @inbounds return heap.key[i]
end

@propagate_inbounds function Base.setindex!(heap::Heap, v, i::Integer)
    @boundscheck checkbounds(heap.key, i)
    @inbounds heap.key[i] = v
    return heap
end
