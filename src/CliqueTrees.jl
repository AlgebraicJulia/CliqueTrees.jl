module CliqueTrees

using AbstractTrees
using ArgCheck
using Base: OneTo, oneto, @kwdef, @propagate_inbounds
using Base.Order
using Base.Sort: DEFAULT_UNSTABLE, Algorithm as SortingAlgorithm
using Base.Iterators: take, takewhile
using DataStructures: IntDisjointSets, find_root!, root_union!
using Graphs
using Graphs: AbstractSimpleGraph, Coloring, SimpleEdge
using LinearAlgebra
using SparseArrays
using SparseArrays: getcolptr

include("./AMFLib.jl/src/AMFLib.jl")
include("./MMDLib.jl/src/MMDLib.jl")

using .AMFLib
using .MMDLib

const AbstractScalar{T} = AbstractArray{T, 0}
const Scalar{T} = Array{T, 0}
const MAX_ITEMS_PRINTED = 5

# Linked Lists
export SinglyLinkedList

# Graphs
export BipartiteGraph, BipartiteEdgeIter, pointers, targets

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
    MinimalChordal,
    CompositeRotations,
    AMD,
    SymAMD,
    METIS,
    Spectral,
    BT,
    permutation,
    mcs

# Trees
export Tree, eliminationtree, setrootindex!

# Supernode Types
export Nodal, Maximal, Fundamental

# Supernode Trees
export SupernodeTree, supernodetree, residuals

# Cliques
export Clique, separator, residual

# Clique Trees
export CliqueTree, cliquetree, treewidth, separators, relatives

# Abstract Trees
export firstchildindex, rootindices, ancestorindices

# Filled Graphs
export FilledGraph, FilledEdgeIter, ischordal, isperfect

include("abstract_linked_lists.jl")
include("singly_linked_lists.jl")
include("doubly_linked_lists.jl")
include("bipartite_graphs.jl")
include("bipartite_edge_iter.jl")
include("elimination_algorithms.jl")
include("trees.jl")
include("supernode_types.jl")
include("supernode_trees.jl")
include("cliques.jl")
include("clique_trees.jl")
include("abstract_trees.jl")
include("filled_graphs.jl")
include("filled_edge_iter.jl")
include("chordal_graphs.jl")
include("utils.jl")

end
