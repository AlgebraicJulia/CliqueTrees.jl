# Sieve: a modified trie for storing and querying PTDURs by vertex sets.
# Flat struct-of-arrays with left-child right-sibling tree representation.
# Each node stores a stop value and a bitset (the edge label from parent).
# Leaves have stop < 0 (encoding -ptdur_root); inner nodes have stop ≥ 1.
# The start of an inner node's interval is parent.stop + 1 (or 1 for the root).
# Children of each node form a doubly linked list (head/prev/next).

# ==================== Type ====================

const MAX_CHILDREN_PER_NODE = 32

struct Sieve{PSet <: AbstractPackedSet}
    stop::Vector{Int}       # ≥ 1 for inner nodes (interval stop); negative for leaves (-ptdur_root)
    set::Vector{PSet}
    parent::Vector{Int}
    head::Vector{Int}       # first child; doubles as free-list next
    prev::Vector{Int}       # previous sibling
    next::Vector{Int}       # next sibling
    outdegree::Vector{Int}

    free::Scalar{Int}
    leaf::Dict{Int, Int}   # ptdur root index -> trie leaf node
end

# ==================== Constructor ====================

function Sieve{PSet}() where {PSet <: AbstractPackedSet}
    sieve = Sieve{PSet}(
        Int[], PSet[], Int[], Int[], Int[], Int[], Int[],
        zeros(Int), Dict{Int, Int}())

    add_vertex!(sieve, domain(PSet), PSet())
    return sieve
end

# ==================== Helpers ====================

function children(sieve::Sieve, v::Int)
    return DoublyLinkedList(view(sieve.head, v), sieve.prev, sieve.next)
end

function is_leaf(sieve::Sieve, v::Int)
    return sieve.stop[v] < 0
end

function outdegree(sieve::Sieve, v::Int)
    return sieve.outdegree[v]
end

function add_vertex!(sieve::Sieve{PSet}, stop::Int, V::PSet) where {PSet}
    list = SinglyLinkedList(sieve.free, sieve.head)

    if isempty(list)
        push!(sieve.stop, stop)
        push!(sieve.set, V)
        push!(sieve.parent, 0)
        push!(sieve.head, 0)
        push!(sieve.prev, 0)
        push!(sieve.next, 0)
        push!(sieve.outdegree, 0)
        v = length(sieve.stop)
    else
        v = popfirst!(list)
        sieve.stop[v] = stop
        sieve.set[v] = V
        sieve.parent[v] = 0
        sieve.head[v] = 0
        sieve.prev[v] = 0
        sieve.next[v] = 0
        sieve.outdegree[v] = 0
    end

    return v
end

# ==================== setindex! ====================

function Base.setindex!(sieve::Sieve{PSet}, V::PSet, i::Int) where {PSet}
    v = start = 1; isdone = false

    while !isdone
        I = packedset(PSet, start:sieve.stop[v])

        isfound = false

        for w in children(sieve, v)
            W = sieve.set[w]

            if V ∩ I == W ∩ I
                isfound = true

                if is_leaf(sieve, w)
                    isdone = true
                    delete!(sieve.leaf, -sieve.stop[w])
                    sieve.stop[w] = -i
                    sieve.leaf[i] = w
                else
                    start = sieve.stop[v] + 1
                    v = w
                end

                break
            end
        end

        if !isfound
            isdone = true

            if sieve.stop[v] < domain(PSet)
                w = add_vertex!(sieve, domain(PSet), V)
                add_edge_split!(sieve, v, w, start)
                start = sieve.stop[v] + 1
                v = w
            end

            w = sieve.leaf[i] = add_vertex!(sieve, -i, V)
            add_edge_split!(sieve, v, w, start)
        end
    end

    return sieve
end

# ==================== Query ====================

# Find all key-value pairs i => V such that
#
#   - V ∩ R = ∅
#   - w(S - V) ≤ margin
#

function init!(stack::Vector{Tuple{Int, Int, Int}})
    push!(empty!(stack), (1, 1, 0))
    return stack
end

function next!(sieve::Sieve{PSet}, stack::Vector{Tuple{Int, Int, Int}}, R::PSet, S::PSet, margin::Int, weights::Vector{Int}) where {PSet}
    result = 0

    while !isempty(stack)
        v, start, i = pop!(stack)

        if is_leaf(sieve, v)
            result = -sieve.stop[v]
            break
        else
            V = packedset(PSet, start:sieve.stop[v])

            for w in children(sieve, v)
                W = sieve.set[w]

                if isempty(R ∩ W ∩ V)
                    j = i + wt(weights, setdiff(S ∩ V, W))
                    j > margin || push!(stack, (w, sieve.stop[v] + 1, j))
                end
            end
        end
    end

    return result
end

# ==================== replace_key! ====================

function replace_key!(sieve::Sieve, old_root::Int, new_root::Int)
    node = sieve.leaf[old_root]
    delete!(sieve.leaf, old_root)
    sieve.stop[node] = -new_root
    sieve.leaf[new_root] = node
    return sieve
end

# ==================== delete! ====================

function Base.delete!(sieve::Sieve, i::Int)
    list = SinglyLinkedList(sieve.free, sieve.head)

    v = pop!(sieve.leaf, i)

    while !isone(v) && iszero(outdegree(sieve, v))
        u = sieve.parent[v]
        rem_edge!(sieve, u, v)
        pushfirst!(list, v)
        v = u
    end

    return sieve
end

# ==================== Internal: add_edge! ====================

function add_edge!(sieve::Sieve, u::Int, v::Int)
    sieve.parent[v] = u
    pushfirst!(children(sieve, u), v)
    sieve.outdegree[u] += 1
    return
end

function rem_edge!(sieve::Sieve, u::Int, v::Int)
    sieve.parent[v] = 0
    delete!(children(sieve, u), v)
    sieve.outdegree[u] -= 1
    return
end

function add_edge_split!(sieve::Sieve{PSet}, u::Int, v::Int, start::Int) where {PSet}
    add_edge!(sieve, u, v)
    outdegree(sieve, u) > MAX_CHILDREN_PER_NODE && split_vertex!(sieve, u, start)
    return
end

# ==================== Internal: split_vertex! ====================

function split_vertex!(sieve::Sieve{PSet}, v::Int, start::Int) where {PSet}
    oldstop = sieve.stop[v]
    newstop = sieve.stop[v] = split_index(start, oldstop)

    V = packedset(PSet, start:newstop)

    dict = Dict{PSet, Int}()

    for x in children(sieve, v)
        rem_edge!(sieve, v, x)

        X = sieve.set[x]

        w = get!(dict, X ∩ V) do
            w = add_vertex!(sieve, oldstop, X)
            add_edge!(sieve, v, w)
            return w
        end

        add_edge!(sieve, w, x)
    end

    return
end

# ==================== Internal: _get_split_vertex ====================

function split_index(start::Int, stop::Int)
    return start + (1 << floor(Int, log(stop - start))) - 1
end
