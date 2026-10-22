#
# The rounds of the exact searches. A round of septw at the width k
# decides whether the graph has an ordering whose bags weigh at most k:
# sepenum lists the minimal separators lighter than k, and sepdp
# searches the blocks of these separators. Every exact search of the
# package runs its rounds through the backend SEPTW[]: septw (and so
# PIDBT, the exact steps of HBT, and the final decisions of RTW), and
# the probes of RTW (rtw_probe).
#
#   - CPUSeptw: every round on the CPU (the default);
#   - GPUSeptw: the rounds start on the CPU, and ask for the GPU when
#     the CPU is expected to take longer (see GPUSeptw). The GPU rounds
#     are in the extension CUDAExt; without it, or without a working
#     device, the rounds stay on the CPU.
#
# A round is run by bt_round!(esc, weights, graph, k, dl, nt), where
# esc = bt_esc(SEPTW[], weights, graph, dl) is made once for each
# search, before its first round. The result is (perm, next, status):
#
#   - BT_FOUND: perm is an ordering of width next <= k;
#   - BT_ABOVE: every ordering has width at least next > k (with the
#     separators lighter than k, sepdp bounds the widths below k + w,
#     where w is the smallest weight: next is at most k + w);
#   - BT_TIMEOUT: the round ran out of time.
#
abstract type SeptwBackend end

"""
    CPUSeptw <: SeptwBackend

The rounds of the exact searches on the CPU: `sepenum` and `sepdp`. This
is the default backend.
"""
struct CPUSeptw <: SeptwBackend end

"""
    GPUSeptw(; engine=:auto, budget=NaN, nmin=0, race=true, batch=true) <: SeptwBackend

The rounds of the exact searches on the CPU and on the GPU. The GPU rounds
need the package extension of CUDA.jl.

A round starts on the CPU. It asks for the GPU when the CPU time that it is
expected to take exceeds the time that the GPU is expected to take, both
learned from the rounds measured so far (by the size of the graph). The GPU
serves the rounds of all searches from a queue, the most promising first,
with the rounds of sepenum and sepdp of other searches whose graphs have
sets of the same size in the same launch (a batch); with `race`, the round
goes on on the CPU in the meantime, and the first to finish answers it.

  - `engine`: `:septw` runs sepenum and sepdp on the GPU (the same
    separators, blocks, and orderings as the CPU, up to 512 vertices);
    `:pid` answers the rounds by a positive-instance driven search (Tamaki
    2017), downward from a greedy ordering: the width and the status are
    those of the CPU, but not the ordering. A search of the PID engine
    answers the later rounds of the graph too. `:auto` asks, for each
    round, the engine with the least expected GPU time for the size of the
    graph (learned as for the CPU).
  - `budget`: if not NaN, a round asks for the GPU once the CPU has spent
    `budget` seconds on the graph, instead.
  - `nmin`: graphs with fewer vertices stay on the CPU.
  - `race`: if false, a round that asks for the GPU waits for it, instead.
  - `batch`: if false, the GPU serves one round at a time.

The worker of the queue needs an interactive thread besides the one of the
main task (julia -t n,2 from Julia 1.12, where the main task runs on an
interactive thread). Without one, a round that asks for the GPU runs on it
if it is free, and on the CPU if it is not.
A GPU round that fails (out of memory, a set too large for the bitsets of
the device code) is run on the CPU; the result is the same.
"""
@kwdef struct GPUSeptw <: SeptwBackend
    engine::Symbol = :auto
    budget::Float64 = NaN
    nmin::Int = 0
    race::Bool = true
    batch::Bool = true
end

const SEPTW = Ref{SeptwBackend}(CPUSeptw())

"""
    septwbackend()

The backend of the rounds of the exact searches.
"""
septwbackend() = SEPTW[]

"""
    septwbackend!(backend::SeptwBackend)

Set the backend of the rounds of the exact searches: [`CPUSeptw`](@ref) or
[`GPUSeptw`](@ref). Returns the old one.
"""
function septwbackend!(backend::SeptwBackend)
    old = SEPTW[]; SEPTW[] = backend
    return old
end

"""
    septwwarmup!(backend::SeptwBackend; nmax=512)

Prepare the backend for searches of graphs with up to `nmax` vertices: for
[`GPUSeptw`](@ref), compile the device code and measure the cost of a GPU
round, which a search would otherwise do when it first needs them. Returns
the backend.
"""
function septwwarmup! end

septwwarmup!(backend::SeptwBackend; nmax::Integer = 512) = backend

# The state of the rounds of one search: nothing, for rounds on the CPU.
# The extension of the GPU adds a method for GPUSeptw.
bt_esc(backend::SeptwBackend, weights::AbstractVector, graph::AbstractGraph, dl::Real) = nothing

# A round on the CPU.
function bt_round!(::Nothing, weights::AbstractVector{W}, graph::AbstractGraph, k::W, dl::Real, nt::Integer) where {W}
    sep, status = sepenum(weights, graph, k, dl, nt)
    status == BT_TIMEOUT && return sep_none(graph), k, BT_TIMEOUT
    perm, _, mink, status = sepdp(weights, graph, sep, k, dl, nt)
    status == BT_ABOVE && return perm, bt_next(weights, graph, k, mink), BT_ABOVE
    return perm, mink, status
end

sep_none(graph::AbstractGraph{V}) where {V} = FVector{V}(undef, 0)

# The next width after a failed round at k, where sepdp bounds the
# width by mink.
function bt_next(weights::AbstractVector{W}, graph::AbstractGraph, k::W, mink::W) where {W}
    return min(mink, k + minimum(view(weights, oneto(nv(graph)))))
end
