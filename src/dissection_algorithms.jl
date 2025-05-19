"""
    DissectionAlgorithm

An algorithm for computing a vertex separator of a graph.
"""
abstract type DissectionAlgorithm end

"""
    METISND <: DissectionAlgorithm

Compute a vertex separator using METIS.
"""
@kwdef struct METISND <: DissectionAlgorithm
    ufactor::Int = -1
    seed::Int = -1
end

# Find a pair of subsets W and B such that V = W ∪ B and
#     N[W - B] ⊆ W
#     N[B - W] ⊆ B
# Then the intersection W ∩ B is called a vertex separator.
# The function returns a vector `project` satisfying
#     project(i) = 0 iff i ∈ W - B
#     project(i) = 1 iff i ∈ B - W
#     project(i) = 2 iff i ∈ W ∩ B
function separator(weights::AbstractVector, graph::AbstractGraph, alg::DissectionAlgorithm)
    throw(
        ArgumentError(
            "Algorithm $alg not implemented. You may need to load an additional package."
        ),
    )
end

function Base.show(io::IO, ::MIME"text/plain", alg::METISND)
    indent = get(io, :indent, 0)
    println(io, " "^indent * "METISND:")
    println(io, " "^indent * "    seed: $(alg.seed)")
    println(io, " "^indent * "    ufactor: $(alg.ufactor)")
    return
end

"""
    DEFAULT_DISSECTION_ALGORITHM = METISND()

The default dissection algorithm.
"""
const DEFAULT_DISSECTION_ALGORITHM = METISND()
