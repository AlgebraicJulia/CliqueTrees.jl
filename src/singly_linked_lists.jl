struct SinglyLinkedList{I, Head <: AbstractScalar{I}, Next <: AbstractVector{I}} <: AbstractLinkedList{I}
    head::Head
    next::Next
end

function SinglyLinkedList{I}(n::Integer) where {I}
    head = FScalar{I}(undef)
    next = FVector{I}(undef, n)
    
    head[] = zero(I)
    return SinglyLinkedList(head, next)
end

@propagate_inbounds function Base.pushfirst!(list::SinglyLinkedList, i::Integer)
    @boundscheck checkbounds(list.next, i)
    @inbounds list.next[i] = list.head[]
    @inbounds list.head[] = i
    return list
end

@propagate_inbounds function Base.prepend!(list::SinglyLinkedList{I}, v::AbstractVector) where {I}
    if !isempty(v)
        @inbounds i = v[begin]
        @inbounds h = list.head[]
        @inbounds list.head[] = i

        for j in @view v[begin + 1:end]
            @boundscheck checkbounds(list.next, j)
            @inbounds list.next[i] = j
            i = j
        end

        @inbounds list.next[i] = h
    end

    return list
end


