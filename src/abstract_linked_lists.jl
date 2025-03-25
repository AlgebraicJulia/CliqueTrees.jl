abstract type AbstractLinkedList{I <: Integer} end

function Base.popfirst!(list::AbstractLinkedList)
    @inbounds i = list.head[]
    @inbounds list.head[] = list.next[i]
    return i
end

function Base.empty!(list::AbstractLinkedList{I}) where {I}
    list.head[] = zero(I)
    return list
end

function Base.isempty(list::AbstractLinkedList)
    return iszero(list.head[])
end

function Base.show(io::IO, ::MIME"text/plain", list::AbstractLinkedList)
    printiterator(io, list)
    return
end

#######################
# Iteration Interface #
#######################

@propagate_inbounds function Base.iterate(list::AbstractLinkedList{I}, i::I = list.head[]) where {I}
    if ispositive(i)
        @boundscheck checkbounds(list.next, i)
        @inbounds return (i, list.next[i])
    end

    return
end

function Base.IteratorSize(::Type{<:AbstractLinkedList})
    return Base.SizeUnknown()
end

function Base.eltype(::Type{<:AbstractLinkedList{I}}) where {I}
    return I
end
