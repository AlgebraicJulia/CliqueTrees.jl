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

## Dissection Algorithms
```@docs
DissectionAlgorithm
METISND
KaHyParND
DEFAULT_DISSECTION_ALGORITHM
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
ND
Spectral
FlowCutter
BT
SAT
MinimalChordal
CompositeRotations
SafeRules
ConnectedComponents
permutation
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

## Matrix Factorization
```@docs
CholFact
CliqueTrees.cholesky
```
