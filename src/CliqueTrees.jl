module CliqueTrees

using AbstractTrees
using AMD: AMD
using Base: OneTo, oneto, @propagate_inbounds
using Base.Order
using Base.Iterators: take, takewhile
using DataStructures: IntDisjointSets, find_root!, root_union!
using Graphs
using Graphs: SimpleEdge
using LinearAlgebra
using Metis: Metis
using SparseArrays
using SparseArrays: getcolptr
using Sparspak: SpkMmd
using TreeWidthSolver: TreeWidthSolver

const AbstractScalar{T} = AbstractArray{T,0}
const Scalar{T} = Array{T,0}
const MAX_ITEMS_PRINTED = 5

# Linked Lists
export SinglyLinkedList

# Graphs
export BipartiteGraph, BipartiteEdgeIter, pointers, targets

# Elimination Algorithms
export BFS,
    MCS,
    LexBFS,
    RCM,
    AAMD,
    SymAMD,
    MMD,
    NodeND,
    Spectral,
    BT,
    permutation,
    bfs,
    mcs,
    lexbfs,
    rcm

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

# Chordal Graphs
export eliminationgraph, eliminationgraph!, ischordal, isfilled, isperfect

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
include("chordal_graphs.jl")
include("utils.jl")

end
