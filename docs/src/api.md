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
MinimalChordal
CompositeRotations
Compression
SafeRules
SafeSeparators
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
Multifrontal.ChordalSymbolic
Multifrontal.ChordalCholesky
Multifrontal.ChordalLDLt
Multifrontal.ChordalTriangular
Multifrontal.Permutation
Multifrontal.AbstractRegularization
Multifrontal.DynamicRegularization
Multifrontal.GMW81
Multifrontal.SE99
Multifrontal.symbolic
Multifrontal.cholesky!
Multifrontal.ldlt!
Multifrontal.selinv!
Multifrontal.complete!
```
