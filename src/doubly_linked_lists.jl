struct DoublyLinkedList{I, Head <: AbstractScalar{I}, Prev <: AbstractVector{I}, Next <: AbstractVector{I}} <: AbstractLinkedList{I}
    head::Head
    prev::Prev
    next::Next
end

function DoublyLinkedList{I}(n::Integer) where {I}
    head = fill(zero(I))
    prev = Vector{I}(undef, n)
    next = Vector{I}(undef, n)
    return DoublyLinkedList(head, prev, next)
end

@propagate_inbounds function Base.pushfirst!(list::DoublyLinkedList, i::Integer)
    @boundscheck checkbounds(list.prev, i)
    @inbounds n = list.next[i] = list.head[]
    @inbounds list.head[] = i

    if ispositive(n)
        @inbounds list.prev[n] = i
    end

    return list
end

@propagate_inbounds function Base.delete!(list::DoublyLinkedList, i::Integer)
    @boundscheck checkbounds(list.prev, i)
    @inbounds h = list.head[]
    @inbounds n = list.next[i]

    if i == h
        @inbounds list.head[] = n
    else
        @inbounds p = list.prev[i]
        @inbounds list.next[p] = n

        if ispositive(n)
            @inbounds list.prev[n] = p
        end
    end

    return list
end

@propagate_inbounds function Base.prepend!(list::DoublyLinkedList, vector::AbstractVector)
    if !isempty(vector)
        @inbounds i = vector[begin]
        @inbounds h = list.head[]
        @inbounds list.head[] = i
        @boundscheck checkbounds(list.prev, i)

        for j in @view vector[(begin + 1):end]
            @boundscheck checkbounds(list.prev, j)
            @inbounds list.prev[j] = i
            @inbounds list.next[i] = j
            i = j
        end

        @inbounds list.next[i] = h
    end

    return list
end
