module IPASIR

using Base: oneto
using ..Utilities

export Solver, clause!, solve!, sortingnetwork!

include("solvers.jl")
include("sorting.jl")

end
