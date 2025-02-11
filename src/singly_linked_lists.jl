# A doubly linked list of distinct natural numbers.
struct SinglyLinkedList{I,Init<:AbstractScalar{I},Next<:AbstractVector{I}} <:
       AbstractLinkedList{I}
    head::Init
    next::Next
end

@propagate_inbounds function Base.pushfirst!(list::SinglyLinkedList, i::Integer)
    @boundscheck checkbounds(list.next, i)
    @inbounds list.next[i] = list.head[]
    list.head[] = i
    return list
end

@propagate_inbounds function Base.popfirst!(list::SinglyLinkedList)
    i = list.head[]
    @boundscheck checkbounds(list.next, i)
    @inbounds list.head[] = list.next[i]
    return i
end
