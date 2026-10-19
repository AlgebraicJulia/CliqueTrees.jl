"""
    MTW{L} <: LowerBoundAlgorithm

    MTW(; lb=MMW(), budget=1000, time=10.0)

The minor-treewidth heuristic: the exact widths of a sequence of minors.

The edges of the graph are contracted one at a time, greedily, until the minor is dense: at most twice as many
vertices as its least degree. The edges come in the order of [`RTW`](@ref), which tends to keep the width. The
width of that minor, found exactly, is a lower bound. Then the contractions are undone, last first, and the bound
is raised while a minor has no decomposition of width at most it. A contracted vertex weighs as little as the
lightest of its vertices, so that no minor is wider than the graph.

The bound starts from `lb`, so that it is at least as large. Each exact search is cheap while the minor has few
minimal separators lighter than the bound, and the search stops at a minor with more than `budget` of them. The
greedy contractions take quadratic time on large graphs: pre-processing with [`SafeRules`](@ref) and
[`SafeSeparators`](@ref) leaves smaller ones.

### Parameters

  - `lb`: lower bound algorithm or lower bound to start from
  - `budget`: maximum number of minimal separators of a minor
  - `time`: time limit in seconds

### References

  - Tamaki, Hisao. "A contraction-recursive algorithm for treewidth." *18th International Symposium on Parameterized and Exact Computation (IPEC 2023)*. 2023.
"""
@kwdef struct MTW{L <: WidthOrAlgorithm} <: LowerBoundAlgorithm
    lb::L = MMW()
    budget::Int = 1000
    time::Float64 = 10.0
end

function lowerbound(weights::AbstractVector, graph::AbstractGraph, alg::MTW)
    return mtw(weights, graph, lowerbound(weights, graph, alg.lb), alg.budget, time() + alg.time)
end

function Base.show(io::IO, ::MIME"text/plain", alg::MTW{L}) where {L}
    indent = get(io, :indent, 0)
    println(io, " "^indent * "MTW{$L}:")
    println(io, " "^indent * "    budget: $(alg.budget)")
    println(io, " "^indent * "    time: $(alg.time)")
    show(IOContext(io, :indent => indent + 4), "text/plain", alg.lb)
    return nothing
end
