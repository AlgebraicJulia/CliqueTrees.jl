struct DAG{T}
    label::Vector{T}
    head::Vector{Int}
    indegree::Vector{Int}
    prev::Vector{Int}
    next::Vector{Int}
    target::Vector{Int}
    vhead::Scalar{Int}
    ehead::Scalar{Int}
    stack::Vector{Int}
end

function DAG{T}() where {T}
    return DAG{T}(T[], Int[], Int[], Int[], Int[], Int[], zeros(Int), zeros(Int), Int[])
end

function incident(dag::DAG, v::Int)
    return DoublyLinkedList(view(dag.head, v), dag.prev, dag.next)
end

function target(dag::DAG, p::Int)
    return dag.target[p]
end

function Base.getindex(dag::DAG, v::Int)
    return dag.label[v]
end

function indegree(dag::DAG, v::Int)
    return dag.indegree[v]
end

function add_vertex!(dag::DAG{T}, lbl::T) where {T}
    free = SinglyLinkedList(dag.vhead, dag.head)

    if isempty(free)
        push!(dag.label, lbl)
        push!(dag.head, 0)
        push!(dag.indegree, 0)
        v = length(dag.label)
    else
        v = popfirst!(free)
        dag.label[v] = lbl
        dag.head[v] = 0
        dag.indegree[v] = 0
    end

    return v
end

function add_edge!(dag::DAG, u::Int, v::Int)
    free = SinglyLinkedList(dag.ehead, dag.next)

    if isempty(free)
        push!(dag.target, v)
        push!(dag.prev, 0)
        push!(dag.next, 0)
        p = length(dag.target)
    else
        p = popfirst!(free)
        dag.target[p] = v
    end

    pushfirst!(incident(dag, u), p)
    dag.indegree[v] += 1
    return p
end

function rem_vertex!(dag::DAG, u::Int, ::Val{true} = Val(true))
    @assert iszero(indegree(dag, u))
    vfree = SinglyLinkedList(dag.vhead, dag.head)
    efree = SinglyLinkedList(dag.ehead, dag.next)
    stack = dag.stack

    push!(stack, u)

    while !isempty(stack)
        u = pop!(stack)

        for p in incident(dag, u)
            v = target(dag, p)
            n = dag.indegree[v] -= 1
            iszero(n) && push!(stack, v)
            pushfirst!(efree, p)
        end

        pushfirst!(vfree, u)
    end

    return dag
end

function rem_vertex!(dag::DAG, u::Int, ::Val{false})
    @assert iszero(indegree(dag, u))
    vfree = SinglyLinkedList(dag.vhead, dag.head)
    efree = SinglyLinkedList(dag.ehead, dag.next)

    for p in incident(dag, u)
        v = target(dag, p)
        dag.indegree[v] -= 1
        pushfirst!(efree, p)
    end

    pushfirst!(vfree, u)
    return dag
end

function rem_edge!(dag::DAG, u::Int, p::Int)
    free = SinglyLinkedList(dag.ehead, dag.next)
    v = target(dag, p)
    dag.indegree[v] -= 1
    delete!(incident(dag, u), p)
    pushfirst!(free, p)
    return dag
end
