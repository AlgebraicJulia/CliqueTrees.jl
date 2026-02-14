module PIDBTLib

using Base: oneto, setindex, OneTo, @propagate_inbounds
using Random
using Random: SamplerType
import Graphs

export pidbt

const AbstractScalar{T} = AbstractArray{T, 0}
const Scalar{T} = Array{T, 0}

# Packed set types
include("packed_sets/abstract_packed_set.jl")

# Linked list types
include("linked_lists/abstract_linked_lists.jl")

# DAG pool
include("dags.jl")

# Data structures
include("mutable_graph.jl")
include("immutable_graph.jl")
include("ptd.jl")
include("sieve.jl")
include("layered_sieve.jl")

# Algorithm components
include("heuristics.jl")
include("safe_separator.jl")

# Main algorithm
include("treewidth.jl")

# Public API
include("pidbt.jl")

end
