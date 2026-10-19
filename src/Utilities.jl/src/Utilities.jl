module Utilities

using Base: @propagate_inbounds, oneto
using Base.Iterators
using Base.Order
using FillArrays: Ones
using FixedSizeArrays
using Graphs

# arrays
export AbstractScalar, Scalar, FScalar, FVector, FMatrix, FArray

# arithmetic
export tolerance, twice, half, two, three, four, five, six, eight, ispositive, isnegative, istwo, isthree, isfour

# graphs
export eltypedegree, de

# printing
export MAX_ITEMS_PRINTED, printiterator

# work arrays
export Workspace, BUCKETS, withworkspace, workspace, reserve!, workvector, spawnwork

const AbstractScalar{T} = AbstractArray{T, 0}
const Scalar{T} = Array{T, 0}

const FArray{T} = FixedSizeArrayDefault{T}
const FMatrix{T} = FArray{T, 2}
const FVector{T} = FArray{T, 1}
const FScalar{T} = FArray{T, 0}

const MAX_ITEMS_PRINTED = 5

# `tol = tolerance(W)` should satisfy
#     v < w iff v ≤ w - tol
#     v ≤ w iif v < w + tol
# for all weights v and w.
function tolerance(::Type{W}) where {W <: AbstractFloat}
    return convert(W, 1.0e-5)
end

function tolerance(::Type{W}) where {W <: Integer}
    return one(W)
end

function twice(i::I) where {I}
    return i + i
end

function half(i::I) where {I <: Integer}
    return i ÷ two(I)
end

function half(i::W) where {W <: AbstractFloat}
    return i / two(W)
end

function eight(::Type{I}) where {I}
    return twice(four(I))
end

function eight(::I) where {I}
    return eight(I)
end

function six(::Type{I}) where {I}
    return twice(three(I))
end

function six(::I) where {I}
    return six(I)
end

function five(::Type{I}) where {I}
    return one(I) + four(I)
end

function five(::I) where {I}
    return five(I)
end

function four(::Type{I}) where {I}
    return one(I) + three(I)
end

function four(::I) where {I}
    return four(I)
end

function three(::Type{I}) where {I}
    return one(I) + two(I)
end

function three(::I) where {I}
    return three(I)
end

function two(::Type{I}) where {I}
    return one(I) + one(I)
end

function two(::I) where {I}
    return two(I)
end

@static if isdefined(Base, :ispositive)
    import Base.ispositive
else
    function ispositive(i::I) where {I}
        return i > zero(I)
    end
end

@static if isdefined(Base, :isnegative)
    import Base.isnegative
else
    function isnegative(i::I) where {I}
        return i < zero(I)
    end
end

function isfour(i::I) where {I}
    return i == four(I)
end

function isthree(i::I) where {I}
    return i == three(I)
end

function istwo(i::I) where {I}
    return i == two(I)
end

function eltypedegree(graph::AbstractGraph{V}, i::Integer) where {V}
    n::V = outdegree(graph, i)
    return n
end

function de(graph::AbstractGraph)
    m = ne(graph)

    if !is_directed(graph)
        m = twice(m)
    end

    return m
end

function printiterator(io::IO, iterator::T) where {T}
    return printiterator(io, iterator, Base.IteratorSize(T))
end

function printiterator(io::IO, iterator::T, ::Any) where {T}
    print(io, "$T:")
    printelements(io, iterator)
    return
end

function printiterator(io::IO, iterator::T, ::Base.HasLength) where {T}
    n = length(iterator)
    print(io, "$n-element $T:")
    printelements(io, iterator)
    return
end

function printelements(io::IO, iterator)
    count = 0

    for v in iterator
        count += 1

        if count <= MAX_ITEMS_PRINTED
            print(io, "\n $v")
        else
            print(io, "\n ⋮")
            break
        end
    end

    return
end

# A task-local cache of work arrays, installed by `withworkspace` for the
# duration of a call that runs many small orderings, such as a nested
# dissection. An ordering takes its arrays when it starts and is done with
# them when it returns, so the orderings run one after another by a task share
# the slots; an ordering must not run another ordering while it holds slots,
# and must not return its slots. The slots are
#
#   - 1: the bucket array of AMF and MMD (`BUCKETS`)
#   - 2, 3: arrays as large as the graph (AMF, MMD, MF, `bestwidth`)
#   - 4 to 12: AMF and MMD
#   - 21 to 43: MF
#   - 51 to 66: `bestwidth` and `bestfill`
#
# A task does not see the workspace of the task that spawned it; tasks spawned
# with `spawnwork` get workspaces of their own, from a pool shared by every
# workspace descended from the same root (see `withworkspace`).
struct Workspace
    slots::Vector{Any}
    hints::Vector{Int}
    lock::ReentrantLock
    idle::Vector{Workspace}
end

function Workspace()
    return Workspace(Any[], Int[], ReentrantLock(), Workspace[])
end

# a workspace sharing the pool of `root`
function Workspace(root::Workspace)
    return Workspace(Any[], Int[], root.lock, root.idle)
end

const WORKSPACE = :CliqueTreesWorkspace

const BUCKETS = 1

function withworkspace(f)
    storage = current_task().storage

    if !isnothing(storage) && haskey(storage::IdDict{Any, Any}, WORKSPACE)
        return f()
    end

    return task_local_storage(f, WORKSPACE, Workspace())
end

# the workspace of the current task, if any
function workspace()
    storage = current_task().storage
    isnothing(storage) && return nothing
    return get(storage::IdDict{Any, Any}, WORKSPACE, nothing)::Union{Nothing, Workspace}
end

# Expect arrays of length up to `n` from slot `i`. A request for at least half
# of that gets all of it, so that a slot grows at most once for a graph whose
# size is known in advance.
function reserve!(ws::Workspace, i::Int, n::Integer)
    hints = ws.hints

    while length(hints) < i
        push!(hints, 0)
    end

    hints[i] = n
    return
end

# the largest arrays in slots 1 to 3 for orderings of a graph; a hint that is
# off costs at most an allocation, so overflow is ignored
function reserve!(ws::Workspace, weights::AbstractVector, graph::AbstractGraph)
    n = nv(graph); m = de(graph); w = 0

    @inbounds for v in oneto(n)
        w += hintweight(weights[v])
    end

    reserve!(ws, BUCKETS, twice(w) + 2)
    reserve!(ws, 2, twice(m) + 4n + 1)
    reserve!(ws, 3, m)
    return
end

function hintweight(w::Integer)
    return w % Int
end

function hintweight(w::AbstractFloat)
    return isfinite(w) && abs(w) < 1.0e18 ? trunc(Int, w) : 0
end

function hintweight(w)
    return 0
end

# an uninitialized array of length `n`: slot `i` of the workspace, or a new
# array if there is no workspace; a slot grows by at least a factor of two,
# or to its hint (see `reserve!`)
function workvector(ws::Union{Nothing, Workspace}, ::Type{T}, i::Int, n::Integer) where {T}
    n = convert(Int, n)

    if isnothing(ws)
        mem = Memory{T}(undef, n)
    else
        slots = ws.slots

        while length(slots) < i
            push!(slots, nothing)
        end

        mem = slots[i]

        if !(mem isa Memory{T}) || length(mem) < n
            hint = i <= length(ws.hints) ? ws.hints[i] : 0

            if n <= hint <= twice(n)
                len = hint
            elseif mem isa Memory{T}
                len = max(n, twice(length(mem)))
            else
                len = n
            end

            mem = slots[i] = Memory{T}(undef, len)
        end
    end

    return WorkVector(mem::Memory{T}, n)
end

# The first `n` entries of a slot. A view of the slot would do, but the
# orderings index views of `Memory` about 10% slower than `FVector`s.
struct WorkVector{T} <: DenseVector{T}
    mem::Memory{T}
    n::Int
end

function Base.size(vector::WorkVector)
    return (vector.n,)
end

function Base.IndexStyle(::Type{<:WorkVector})
    return IndexLinear()
end

@propagate_inbounds function Base.getindex(vector::WorkVector, i::Int)
    @boundscheck checkbounds(vector, i)
    return @inbounds vector.mem[i]
end

@propagate_inbounds function Base.setindex!(vector::WorkVector, x, i::Int)
    @boundscheck checkbounds(vector, i)
    @inbounds vector.mem[i] = x
    return vector
end

function Base.unsafe_convert(::Type{Ptr{T}}, vector::WorkVector{T}) where {T}
    return Base.unsafe_convert(Ptr{T}, vector.mem)
end

function Base.elsize(::Type{WorkVector{T}}) where {T}
    return Base.elsize(Memory{T})
end

# Spawn a task that runs `f()`, an ordering of `graph`, with a workspace from
# the pool of the workspace of the current task, or with none if the current
# task has none. The task takes the workspace when it starts and puts it back
# when it is done, so a pool holds about as many workspaces as there were
# tasks running at once.
function spawnwork(f, weights::AbstractVector, graph::AbstractGraph)
    ws = workspace()
    isnothing(ws) && return Threads.@spawn f()

    return Threads.@spawn begin
        child = @lock ws.lock begin
            isempty(ws.idle) ? Workspace(ws) : pop!(ws.idle)
        end

        reserve!(child, weights, graph)

        try
            task_local_storage(f, WORKSPACE, child)
        finally
            @lock ws.lock push!(ws.idle, child)
        end
    end
end

end
