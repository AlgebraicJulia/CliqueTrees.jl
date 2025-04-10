# Library Reference

```@meta
CurrentModule = CliqueTrees
```

## Lower Bound Algorithms
```@docs
LowerBoundAlgorithm
MMW
lowerbound
DEFAULT_LOWER_BOUND_ALGORITHM
```

## Elimination Algorithms

```@docs
EliminationAlgorithm
PermutationOrAlgorithm
DEFAULT_ELIMINATION_ALGORITHM
BFS
MCS
LexBFS
RCMMD
RCMGL
RCM
LexM
MCSM
AMD
SymAMD
AMF
MF
MMD
METIS
Spectral
FlowCutter
BT
SAT
MinimalChordal
CompositeRotations
SafeRules
SafeSeparators
ConnectedComponents
permutation
mcs
```

## Supernodes

```@docs
SupernodeType
DEFAULT_SUPERNODE_TYPE
Nodal
Maximal
Fundamental
```

## Linked Lists
```@docs
SinglyLinkedList
```

## Trees

```@docs
AbstractTree
rootindices
firstchildindex
ancestorindices
```

### Trees

```@docs
Tree
eliminationtree
```

### Supernode Trees

```@docs
SupernodeTree
supernodetree
```

### Clique Trees

```@docs
Clique
CliqueTree
cliquetree
treewidth
separator
residual
```

## Filled Graphs

```@docs
FilledGraph
ischordal
isperfect
```
