# A doubly linked list of distinct natural numbers.
struct SinglyLinkedList{I, Init <: AbstractScalar{I}, Next <: AbstractVector{I}} <:
    AbstractLinkedList{I}
    head::Init
    next::Next
end

function SinglyLinkedList{I}(n::Integer) where {I}
    head = fill(zero(I))
    next = Vector{I}(undef, n)
    return SinglyLinkedList(head, next)
end

function SinglyLinkedList{I}(vector::AbstractVector) where {I}
    list = SinglyLinkedList{I}(length(vector))
    return prepend!(list, vector)
end

function SinglyLinkedList(vector::AbstractVector{I}) where {I}
    return SinglyLinkedList{I}(vector)
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

function Base.prepend!(list::SinglyLinkedList, vector::AbstractVector)
    @views list.next[vector[begin:(end - 1)]] = vector[(begin + 1):end]

    if !isempty(vector)
        list.next[vector[end]] = list.head[]
        list.head[] = vector[begin]
    end

    return list
end
