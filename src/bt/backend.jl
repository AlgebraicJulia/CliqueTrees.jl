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
#   - GPUSeptw: the rounds of an atom start on the CPU, and go to the
#     GPU when the CPU has spent its budget on the atom (see
#     GPUSeptw). The GPU rounds are in the extension CUDAExt; without
#     it, or without a working device, the rounds stay on the CPU.
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
    GPUSeptw(; engine=:septw, budget=NaN, delta=2.0, nmin=0) <: SeptwBackend

The rounds of an exact search start on the CPU and go to the GPU when the
CPU has spent `budget` seconds on the atom, or when the next round is
expected to end after that (judged by the growth of the last two rounds).
The rest of the rounds of the atom stay on the GPU. The GPU rounds need the
package extension of CUDA.jl.

  - `engine`: `:septw` runs sepenum and sepdp on the GPU (the same
    separators, blocks, and orderings as the CPU); `:pid` answers the
    rounds by a positive-instance driven search (Tamaki 2017), downward
    from a greedy ordering: the width and the status are those of the CPU,
    but not the ordering.
  - `budget`: the CPU seconds of an atom. If it is NaN, it is `delta` times
    the cost of a GPU round on a small graph, measured on the device at hand.
  - `nmin`: atoms with fewer vertices stay on the CPU.

The GPU is shared by the tasks of a search: a round that finds it busy runs
on the CPU. A GPU round that fails (out of memory, a set too large for the
bitsets of the device code) is run again on the CPU; the result is the same.
"""
@kwdef struct GPUSeptw <: SeptwBackend
    engine::Symbol = :septw
    budget::Float64 = NaN
    delta::Float64 = 2.0
    nmin::Int = 0
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
