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
export Workspace, withworkspace, workspace, workvector, spawnwork, giveback!

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
#   - 1: the bucket array of AMF and MMD
#   - 2, 3: arrays as large as the graph (AMF, MMD, MF, `bestwidth`)
#   - 4 to 12: AMF and MMD
#   - 21 to 43: MF
#   - 51 to 66: `bestwidth` and `bestfill`
#
# A task does not see the workspace of the task that spawned it; tasks spawned
# with `spawnwork` get workspaces of their own, which are kept for later tasks.
struct Workspace
    slots::Vector{Any}
    idle::Vector{Workspace}
end

function Workspace()
    return Workspace(Any[], Workspace[])
end

const WORKSPACE = :CliqueTreesWorkspace

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

# an uninitialized array of length `n`: slot `i` of the workspace, or a new
# array if there is no workspace; a slot grows by at least a factor of two
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

        if !(mem isa Memory{T})
            mem = slots[i] = Memory{T}(undef, n)
        elseif length(mem) < n
            mem = slots[i] = Memory{T}(undef, max(n, twice(length(mem))))
        end
    end

    return view(mem::Memory{T}, oneto(n))
end

# Spawn a task that runs `f()`. If the current task has a workspace, then the
# new task gets one too, taken from the idle workspaces of the current one;
# the current task gives it back with `giveback!` once the new task is done.
function spawnwork(f)
    ws = workspace()

    if isnothing(ws)
        return Threads.@spawn(f()), nothing
    end

    child = isempty(ws.idle) ? Workspace() : pop!(ws.idle)
    return Threads.@spawn(task_local_storage(f, WORKSPACE, child)), child
end

function giveback!(child::Union{Nothing, Workspace})
    ws = workspace()

    if !isnothing(ws) && !isnothing(child)
        push!(ws.idle, child)
    end

    return
end

end
