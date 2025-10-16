"""
    SinglyLinkedList{I, Head, Next} <: AbstractLinkedList{I}

    SinglyLinkedList{I}(n::Integer)

A singly linked list of distinct natural numbers. This type supports the [iteration interface](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-iteration).

```jldoctest
julia> using CliqueTrees

julia> list = SinglyLinkedList{Int}(10)
SinglyLinkedList{Int64, Array{Int64, 0}, Vector{Int64}}:

julia> pushfirst!(list, 4, 5, 6, 7, 8, 9)
SinglyLinkedList{Int64, Array{Int64, 0}, Vector{Int64}}:
 4
 5
 6
 7
 8
 ⋮

julia> collect(list)
6-element Vector{Int64}:
 4
 5
 6
 7
 8
 9
```
"""
struct SinglyLinkedList{I, Head <: AbstractScalar{I}, Next <: AbstractVector{I}} <: AbstractLinkedList{I}
    """
    If `list` is empty, then `list.head[]` is equal to `0`.
    Otherwise, `list.head[]` is the first element of `list`
    """
    head::Head

    """
    If `i` is not in `list`, then `list.next[i]` is undefined.
    If `i` is the last element of `list`, then `list.next[i]` is equal to `0`.
    Otherwise, `list.next[i]` is the element of `list` following `i`.
    """
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


