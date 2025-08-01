module CliqueTrees

using AbstractTrees
using ArgCheck
using Base: OneTo, oneto, @kwdef, @propagate_inbounds
using Base.Iterators
using Base.Order
using Base.Sort: DEFAULT_UNSTABLE, Algorithm as SortingAlgorithm
using FillArrays
using FixedSizeArrays
using Graphs
using Graphs: AbstractSimpleGraph, Coloring, SimpleEdge
using LinearAlgebra
using LinearAlgebra: ldiv
using SparseArrays
using SparseArrays: getcolptr

include("./Utilities.jl/src/Utilities.jl")
include("./IPASIR.jl/src/IPASIR.jl")
include("./AMFLib.jl/src/AMFLib.jl")
include("./MMDLib.jl/src/MMDLib.jl")

using .Utilities
using .IPASIR
using .AMFLib
using .MMDLib

const View{T, I} = SubArray{T, 1, Vector{T}, Tuple{UnitRange{I}}, true}

# Linked Lists
export SinglyLinkedList

# Graphs
export BipartiteGraph, BipartiteEdgeIter, pointers, targets

# Lower Bound Algorithms
export MMW, lowerbound

# Dissection Algorithms
export METISND, KaHyParND

# Elimination Algorithms
export BFS,
    MCS,
    LexBFS,
    RCMMD,
    RCMGL,
    RCM,
    LexM,
    MCSM,
    AMF,
    MF,
    MMD,
    AMD,
    SymAMD,
    METIS,
    ND,
    NDS,
    Spectral,
    FlowCutter,
    BT,
    SAT,
    MinimalChordal,
    CompositeRotations,
    SafeRules,
    SimplicialRule,
    SafeSeparators,
    ConnectedComponents,
    BestWidth,
    BestFill,
    permutation

# Trees
export Tree, eliminationtree, setrootindex!

# Supernode Types
export Nodal, Maximal, Fundamental

# Supernode Trees
export SupernodeTree, supernodetree, residuals

# Cliques
export Clique, separator, residual

# Clique Trees
export CliqueTree, cliquetree, treewidth, treefill, separators

# Abstract Trees
export rootindices, ancestorindices

# Filled Graphs
export FilledGraph, FilledEdgeIter, ischordal, isperfect

include("union_find.jl")
include("heaps.jl")
include("abstract_linked_lists.jl")
include("singly_linked_lists.jl")
include("doubly_linked_lists.jl")
include("bipartite_graphs.jl")
include("bipartite_edge_iter.jl")
include("lower_bound_algorithms.jl")
include("dissection_algorithms.jl")
include("elimination_algorithms.jl")
include("parent.jl")
include("trees.jl")
include("supernode_types.jl")
include("supernode_trees.jl")
include("cliques.jl")
include("clique_trees.jl")
include("abstract_trees.jl")
include("filled_graphs.jl")
include("filled_edge_iter.jl")
include("chordal_graphs.jl")
include("ambiguities.jl")
include("packages.jl")
include("pr3.jl")
include("pr4.jl")
include("chol_fact.jl")

end
