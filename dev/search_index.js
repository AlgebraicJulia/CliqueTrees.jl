var documenterSearchIndex = {"docs":
[{"location":"api/#Library-Reference","page":"Library Reference","title":"Library Reference","text":"","category":"section"},{"location":"api/","page":"Library Reference","title":"Library Reference","text":"CurrentModule = CliqueTrees","category":"page"},{"location":"api/#Elimination-Algorithms","page":"Library Reference","title":"Elimination Algorithms","text":"","category":"section"},{"location":"api/","page":"Library Reference","title":"Library Reference","text":"EliminationAlgorithm\nPermutationOrAlgorithm\nDEFAULT_ELIMINATION_ALGORITHM\nBFS\nMCS\nLexBFS\nRCM\nAAMD\nSymAMD\nMMD\nNodeND\nSpectral\nBT\npermutation\nbfs\nmcs\nlexbfs\nrcm","category":"page"},{"location":"api/#CliqueTrees.EliminationAlgorithm","page":"Library Reference","title":"CliqueTrees.EliminationAlgorithm","text":"EliminationAlgorithm\n\nA graph elimination algorithm. The options are\n\ntype name complexity exact\nBFS breadth-first search O(m) false\nMCS maximum cardinality search O(m + n) false\nLexBFS lexicographic breadth-first search O(m + n) false\nRCM reverse Cuthill-Mckee O(mΔ) false\nAAMD approximate minimum degree O(mn) false\nSymAMD column approximate minimum degree O(mn) false\nMMD multiple minimum degree O(mn²) false\nNodeND nested dissection  false\nSpectral spectral ordering  false\nBT Bouchitte-Todinca O(2.6183ⁿ) true\n\nfor a graph with m edges, n vertices, and maximum degree Δ. The algorithm Spectral only works on connected graphs.\n\n\n\n\n\n","category":"type"},{"location":"api/#CliqueTrees.PermutationOrAlgorithm","page":"Library Reference","title":"CliqueTrees.PermutationOrAlgorithm","text":"PermutationOrAlgorithm = Union{AbstractVector, EliminationAlgorithm}\n\nEither a permutation or an algorithm.\n\n\n\n\n\n","category":"type"},{"location":"api/#CliqueTrees.DEFAULT_ELIMINATION_ALGORITHM","page":"Library Reference","title":"CliqueTrees.DEFAULT_ELIMINATION_ALGORITHM","text":"DEFAULT_ELIMINATION_ALGORITHM = AAMD()\n\nThe default algorithm.\n\n\n\n\n\n","category":"constant"},{"location":"api/#CliqueTrees.BFS","page":"Library Reference","title":"CliqueTrees.BFS","text":"BFS <: EliminationAlgorithm\n\nBFS()\n\nThe breadth-first search algorithm.\n\n\n\n\n\n","category":"type"},{"location":"api/#CliqueTrees.MCS","page":"Library Reference","title":"CliqueTrees.MCS","text":"MCS <: EliminationAlgorithm\n\nMCS()\n\nThe maximum cardinality search algorithm.\n\n\n\n\n\n","category":"type"},{"location":"api/#CliqueTrees.LexBFS","page":"Library Reference","title":"CliqueTrees.LexBFS","text":"LexBFS <: EliminationAlgorithm\n\nLexBFS()\n\nThe lexicographic breadth-first-search algorithm.\n\n\n\n\n\n","category":"type"},{"location":"api/#CliqueTrees.RCM","page":"Library Reference","title":"CliqueTrees.RCM","text":"RCM <: EliminationAlgorithm\n\nThe reverse Cuthill-McKee algorithm.\n\n\n\n\n\n","category":"type"},{"location":"api/#CliqueTrees.AAMD","page":"Library Reference","title":"CliqueTrees.AAMD","text":"AAMD <: EliminationAlgorithm\n\nAAMD(; dense=10.0, aggressive=1.0)\n\nThe approximate minimum degree algorithm.\n\ndense: dense row parameter\naggressive: aggressive absorption\n\n\n\n\n\n","category":"type"},{"location":"api/#CliqueTrees.SymAMD","page":"Library Reference","title":"CliqueTrees.SymAMD","text":"SymAMD <: EliminationAlgorithm\n\nSymAMD(; dense_row=10.0, dense_col=10.0, aggressive=1.0)\n\nThe column approximate minimum degree algorithm.\n\ndense_row: dense row parameter\ndense_column: dense column parameter\naggressive: aggressive absorption\n\n\n\n\n\n","category":"type"},{"location":"api/#CliqueTrees.MMD","page":"Library Reference","title":"CliqueTrees.MMD","text":"MMD <: EliminationAlgorithm\n\nMMD()\n\nThe multiple minimum degree algorithm.\n\n\n\n\n\n","category":"type"},{"location":"api/#CliqueTrees.NodeND","page":"Library Reference","title":"CliqueTrees.NodeND","text":"NodeND <: EliminationAlgorithm\n\nNodeND()\n\nThe nested dissection algorithm.\n\n\n\n\n\n","category":"type"},{"location":"api/#CliqueTrees.Spectral","page":"Library Reference","title":"CliqueTrees.Spectral","text":"Spectral <: EliminationAlgorithm\n\nSpectral(; tol=0.0)\n\nThe spectral ordering algorithm only works on connected graphs. In order to use it, import the package Laplacians.\n\ntol: tolerance for convergence\n\n\n\n\n\n","category":"type"},{"location":"api/#CliqueTrees.BT","page":"Library Reference","title":"CliqueTrees.BT","text":"BT <: EliminationAlgorithm\n\nBT()\n\nThe Bouchitte-Todinca algorithm.\n\n\n\n\n\n","category":"type"},{"location":"api/#CliqueTrees.permutation","page":"Library Reference","title":"CliqueTrees.permutation","text":"permutation(graph;\n    alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM)\n\nConstruct a fill-reducing permutation of the vertices of a simple graph.\n\njulia> using CliqueTrees\n\njulia> graph = [\n           0 1 1 0 0 0 0 0\n           1 0 1 0 0 1 0 0\n           1 1 0 1 1 0 0 0\n           0 0 1 0 1 0 0 0\n           0 0 1 1 0 0 1 1\n           0 1 0 0 0 0 1 0\n           0 0 0 0 1 1 0 1\n           0 0 0 0 1 0 1 0\n       ];\n\njulia> order, index = permutation(graph);\n\njulia> order\n8-element Vector{Int64}:\n 4\n 8\n 7\n 6\n 5\n 1\n 3\n 2\n\njulia> index == invperm(order)\ntrue\n\n\n\n\n\n","category":"function"},{"location":"api/#CliqueTrees.bfs","page":"Library Reference","title":"CliqueTrees.bfs","text":"bfs(graph)\n\nPerform a breadth-first search of a graph.\n\n\n\n\n\n","category":"function"},{"location":"api/#CliqueTrees.mcs","page":"Library Reference","title":"CliqueTrees.mcs","text":"mcs(graph[, clique::AbstractVector])\n\nPerform a maximum cardinality search, optionally specifying a clique to be ordered last. Returns the inverse permutation.\n\n\n\n\n\n","category":"function"},{"location":"api/#CliqueTrees.lexbfs","page":"Library Reference","title":"CliqueTrees.lexbfs","text":"lexbfs(graph)\n\nPerform a lexicographic breadth-first search. Returns the inverse permutation.\n\n\n\n\n\n","category":"function"},{"location":"api/#CliqueTrees.rcm","page":"Library Reference","title":"CliqueTrees.rcm","text":"rcm(graph)\n\nThe reverse Cuthill-Mckee algorithm.\n\n\n\n\n\n","category":"function"},{"location":"api/#Supernodes","page":"Library Reference","title":"Supernodes","text":"","category":"section"},{"location":"api/","page":"Library Reference","title":"Library Reference","text":"SupernodeType\nDEFAULT_SUPERNODE_TYPE\nNodal\nMaximal\nFundamental","category":"page"},{"location":"api/#CliqueTrees.SupernodeType","page":"Library Reference","title":"CliqueTrees.SupernodeType","text":"SupernodeType\n\nA type of supernode partition. The options are\n\ntype name\nNodal nodal supernode partition\nMaximal maximal supernode partition\nFundamental fundamental supernode partition\n\n\n\n\n\n","category":"type"},{"location":"api/#CliqueTrees.DEFAULT_SUPERNODE_TYPE","page":"Library Reference","title":"CliqueTrees.DEFAULT_SUPERNODE_TYPE","text":"DEFAULT_SUPERNODE_TYPE = Maximal()\n\nThe default supernode partition.\n\n\n\n\n\n","category":"constant"},{"location":"api/#CliqueTrees.Nodal","page":"Library Reference","title":"CliqueTrees.Nodal","text":"Nodal <: SupernodeType\n\nA nodal  supernode partition.\n\n\n\n\n\n","category":"type"},{"location":"api/#CliqueTrees.Maximal","page":"Library Reference","title":"CliqueTrees.Maximal","text":"Maximal <: SupernodeType\n\nA maximal supernode partition.\n\n\n\n\n\n","category":"type"},{"location":"api/#CliqueTrees.Fundamental","page":"Library Reference","title":"CliqueTrees.Fundamental","text":"Fundamental <: SupernodeType\n\nA fundamental supernode partition.\n\n\n\n\n\n","category":"type"},{"location":"api/#Chordal-Graphs","page":"Library Reference","title":"Chordal Graphs","text":"","category":"section"},{"location":"api/","page":"Library Reference","title":"Library Reference","text":"eliminationgraph\neliminationgraph!\nischordal\nisfilled\nisperfect","category":"page"},{"location":"api/#CliqueTrees.eliminationgraph","page":"Library Reference","title":"CliqueTrees.eliminationgraph","text":"eliminationgraph(tree::CliqueTree)\n\nConstruct the subtree graph of a clique tree. The result is stored in graph.\n\n\n\n\n\neliminationgraph(graph;\n    alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM,\n    snd::SupernodeType=DEFAULT_SUPERNODE_TYPE)\n\nConstruct the elimination graph of a simple graph.\n\njulia> using CliqueTrees, SparseArrays\n\njulia> graph = [\n           0 1 1 0 0 0 0 0\n           1 0 1 0 0 1 0 0\n           1 1 0 1 1 0 0 0\n           0 0 1 0 1 0 0 0\n           0 0 1 1 0 0 1 1\n           0 1 0 0 0 0 1 0\n           0 0 0 0 1 1 0 1\n           0 0 0 0 1 0 1 0\n       ];\n\njulia> label, filledgraph = eliminationgraph(graph);\n\njulia> sparse(filledgraph)\n8×8 SparseMatrixCSC{Bool, Int64} with 13 stored entries:\n ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅\n ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅\n ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅\n ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅\n ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅\n 1  1  1  1  ⋅  ⋅  ⋅  ⋅\n 1  ⋅  ⋅  ⋅  1  1  ⋅  ⋅\n ⋅  ⋅  ⋅  1  1  1  1  ⋅\n\njulia> isfilled(filledgraph)\ntrue\n\n\n\n\n\n","category":"function"},{"location":"api/#CliqueTrees.eliminationgraph!","page":"Library Reference","title":"CliqueTrees.eliminationgraph!","text":"eliminationgraph!(graph, tree::CliqueTree)\n\nSee eliminationgraph. The result is stored in graph.\n\n\n\n\n\n","category":"function"},{"location":"api/#CliqueTrees.ischordal","page":"Library Reference","title":"CliqueTrees.ischordal","text":"ischordal(graph)\n\nDetermine whether a simple graph is chordal.\n\n\n\n\n\n","category":"function"},{"location":"api/#CliqueTrees.isfilled","page":"Library Reference","title":"CliqueTrees.isfilled","text":"isfilled(graph)\n\nDetermine whether a directed graph is filled.\n\n\n\n\n\n","category":"function"},{"location":"api/#CliqueTrees.isperfect","page":"Library Reference","title":"CliqueTrees.isperfect","text":"isperfect(graph, order::AbstractVector[, index::AbstractVector])\n\nDetermine whether an fill-reducing permutation is perfect.\n\n\n\n\n\n","category":"function"},{"location":"api/#Abstract-Trees","page":"Library Reference","title":"Abstract Trees","text":"","category":"section"},{"location":"api/","page":"Library Reference","title":"Library Reference","text":"AbstractTree\nrootindices\nfirstchildindex\nancestorindices","category":"page"},{"location":"api/#CliqueTrees.AbstractTree","page":"Library Reference","title":"CliqueTrees.AbstractTree","text":"AbstractTree{V} = Union{Tree{V}, SupernodeTree{V}, CliqueTree{V}}\n\nA rooted forest. This type implements the indexed tree interface.\n\n\n\n\n\n","category":"type"},{"location":"api/#CliqueTrees.rootindices","page":"Library Reference","title":"CliqueTrees.rootindices","text":"rootindices(tree::AbstractTree)\n\nGet the roots of a rooted forest.\n\n\n\n\n\n","category":"function"},{"location":"api/#CliqueTrees.firstchildindex","page":"Library Reference","title":"CliqueTrees.firstchildindex","text":"firstchildindex(tree::AbstractTree, i::Integer)\n\nGet the first child of node i. Returns nothing if i is a leaf.\n\n\n\n\n\n","category":"function"},{"location":"api/#CliqueTrees.ancestorindices","page":"Library Reference","title":"CliqueTrees.ancestorindices","text":"ancestorindices(tree::AbstractTree, i::Integer)\n\nGet the proper ancestors of node i.\n\n\n\n\n\n","category":"function"},{"location":"api/#Trees","page":"Library Reference","title":"Trees","text":"","category":"section"},{"location":"api/","page":"Library Reference","title":"Library Reference","text":"Tree\neliminationtree","category":"page"},{"location":"api/#CliqueTrees.Tree","page":"Library Reference","title":"CliqueTrees.Tree","text":"Tree{V <: Signed} <: AbstractUnitRange{V}\n\nTree(tree::AbstractTree)\n\nTree{V}(tree::AbstractTree) where V\n\nA rooted forest with vertices of type V. This type implements the indexed tree interface.\n\n\n\n\n\n","category":"type"},{"location":"api/#CliqueTrees.eliminationtree","page":"Library Reference","title":"CliqueTrees.eliminationtree","text":"eliminationtree(graph;\n    alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM)\n\nConstruct a tree-depth decomposition of a simple graph.\n\njulia> using CliqueTrees\n\njulia> graph = [\n           0 1 1 0 0 0 0 0\n           1 0 1 0 0 1 0 0\n           1 1 0 1 1 0 0 0\n           0 0 1 0 1 0 0 0\n           0 0 1 1 0 0 1 1\n           0 1 0 0 0 0 1 0\n           0 0 0 0 1 1 0 1\n           0 0 0 0 1 0 1 0\n       ];\n\njulia> label, tree = eliminationtree(graph);\n\njulia> tree\n8-element Tree{Int64}:\n 8\n └─ 7\n    ├─ 5\n    │  ├─ 1\n    │  └─ 4\n    │     └─ 3\n    │        └─ 2\n    └─ 6\n\n\n\n\n\n","category":"function"},{"location":"api/#Supernode-Trees","page":"Library Reference","title":"Supernode Trees","text":"","category":"section"},{"location":"api/","page":"Library Reference","title":"Library Reference","text":"SupernodeTree\nsupernodetree","category":"page"},{"location":"api/#CliqueTrees.SupernodeTree","page":"Library Reference","title":"CliqueTrees.SupernodeTree","text":"SupernodeTree{V} <: AbstractVector{UnitRange{V}}\n\nA supernodal elimination tree with vertices of type V. This type implements the indexed tree interface.\n\n\n\n\n\n","category":"type"},{"location":"api/#CliqueTrees.supernodetree","page":"Library Reference","title":"CliqueTrees.supernodetree","text":"supernodetree(graph;\n    alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM,\n    snd::SupernodeType=DEFAULT_SUPERNODE_TYPE)\n\nConstruct a supernodal elimination tree.\n\n\n\n\n\n","category":"function"},{"location":"api/#Clique-Trees","page":"Library Reference","title":"Clique Trees","text":"","category":"section"},{"location":"api/","page":"Library Reference","title":"Library Reference","text":"Clique\nCliqueTree\ncliquetree\ntreewidth\nseparator\nresidual","category":"page"},{"location":"api/#CliqueTrees.Clique","page":"Library Reference","title":"CliqueTrees.Clique","text":"Clique{V, E} <: AbstractVector{V}\n\nA clique of a clique tree.\n\n\n\n\n\n","category":"type"},{"location":"api/#CliqueTrees.CliqueTree","page":"Library Reference","title":"CliqueTrees.CliqueTree","text":"CliqueTree{V, E} <: AbstractVector{Clique{V, E}}\n\nA clique tree with vertices of type V and edges of type E. This type implements the indexed tree interface.\n\n\n\n\n\n","category":"type"},{"location":"api/#CliqueTrees.cliquetree","page":"Library Reference","title":"CliqueTrees.cliquetree","text":"cliquetree(graph;\n    alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM,\n    snd::SupernodeType=DEFAULT_SUPERNODE_TYPE)\n\nConstruct a tree decomposition of a simple graph. The vertices of the graph are first ordered by a fill-reducing permutation computed by the algorithm alg. The size of the resulting decomposition is determined by the supernode partition snd.\n\njulia> using CliqueTrees\n\njulia> graph = [\n           0 1 1 0 0 0 0 0\n           1 0 1 0 0 1 0 0\n           1 1 0 1 1 0 0 0\n           0 0 1 0 1 0 0 0\n           0 0 1 1 0 0 1 1\n           0 1 0 0 0 0 1 0\n           0 0 0 0 1 1 0 1\n           0 0 0 0 1 0 1 0\n       ];\n\njulia> label, tree = cliquetree(graph);\n\njulia> tree\n6-element CliqueTree{Int64, Int64}:\n [6, 7, 8]\n ├─ [1, 6, 7]\n ├─ [4, 6, 8]\n │  └─ [3, 4, 6]\n │     └─ [2, 3, 6]\n └─ [5, 7, 8]\n\n\n\n\n\n","category":"function"},{"location":"api/#CliqueTrees.treewidth","page":"Library Reference","title":"CliqueTrees.treewidth","text":"treewidth(tree::CliqueTree)\n\nCompute the width of a clique tree.\n\n\n\n\n\ntreewidth(graph;\n    alg::PermutationOrAlgorithm=DEFAULT_ELIMINATION_ALGORITHM)\n\nCompute an upper bound to the tree width of a simple graph.\n\n\n\n\n\n","category":"function"},{"location":"api/#CliqueTrees.separator","page":"Library Reference","title":"CliqueTrees.separator","text":"separator(clique::Clique)\n\nGet the separator of a clique.\n\n\n\n\n\nseparator(tree::CliqueTree, i::Integer)\n\nGet the separator at node i.\n\n\n\n\n\n","category":"function"},{"location":"api/#CliqueTrees.residual","page":"Library Reference","title":"CliqueTrees.residual","text":"residual(clique::Clique)\n\nGet the residual of a clique.\n\n\n\n\n\nresidual(tree::CliqueTree, i::Integer)\n\nGet the residual at node i.\n\n\n\n\n\n","category":"function"},{"location":"#CliqueTrees.jl","page":"CliqueTrees.jl","title":"CliqueTrees.jl","text":"","category":"section"},{"location":"","page":"CliqueTrees.jl","title":"CliqueTrees.jl","text":"CliqueTrees.jl implements clique trees in Julia. You can use it to construct tree decompositions and chordal completions of graphs.","category":"page"},{"location":"#Installation","page":"CliqueTrees.jl","title":"Installation","text":"","category":"section"},{"location":"","page":"CliqueTrees.jl","title":"CliqueTrees.jl","text":"To install CliqueTrees.jl, enter the Pkg REPL by typing ] and run the following command.","category":"page"},{"location":"","page":"CliqueTrees.jl","title":"CliqueTrees.jl","text":"pkg> add CliqueTrees","category":"page"},{"location":"#Basic-Usage","page":"CliqueTrees.jl","title":"Basic Usage","text":"","category":"section"},{"location":"#Tree-Decompositions","page":"CliqueTrees.jl","title":"Tree Decompositions","text":"","category":"section"},{"location":"","page":"CliqueTrees.jl","title":"CliqueTrees.jl","text":"The function cliquetree computes tree decompositions.","category":"page"},{"location":"","page":"CliqueTrees.jl","title":"CliqueTrees.jl","text":"julia> using CliqueTrees\n\njulia> graph = [\n           0 1 1 0 0 0 0 0\n           1 0 1 0 0 1 0 0\n           1 1 0 1 1 0 0 0\n           0 0 1 0 1 0 0 0\n           0 0 1 1 0 0 1 1\n           0 1 0 0 0 0 1 0\n           0 0 0 0 1 1 0 1\n           0 0 0 0 1 0 1 0\n       ];\n\njulia> label, tree = cliquetree(graph);\n\njulia> tree\n6-element CliqueTree{Int64, Int64}:\n [6, 7, 8]\n ├─ [1, 6, 7]\n ├─ [4, 6, 8]\n │  └─ [3, 4, 6]\n │     └─ [2, 3, 6]\n └─ [5, 7, 8]","category":"page"},{"location":"","page":"CliqueTrees.jl","title":"CliqueTrees.jl","text":"The clique tree tree is a tree decomposition of the permuted graph graph[label, label]. A clique tree is a vector of cliques, so you can retrieve the clique at node 3 by typing tree[3].","category":"page"},{"location":"","page":"CliqueTrees.jl","title":"CliqueTrees.jl","text":"julia> tree[3]\n3-element Clique{Int64, Int64}:\n 3\n 4\n 6","category":"page"},{"location":"","page":"CliqueTrees.jl","title":"CliqueTrees.jl","text":"The width of a clique tree is computed by the function treewidth.","category":"page"},{"location":"","page":"CliqueTrees.jl","title":"CliqueTrees.jl","text":"julia> treewidth(tree)\n2","category":"page"},{"location":"#Chordal-Completions","page":"CliqueTrees.jl","title":"Chordal Completions","text":"","category":"section"},{"location":"","page":"CliqueTrees.jl","title":"CliqueTrees.jl","text":"The function eliminationgraph computes elimination graphs.","category":"page"},{"location":"","page":"CliqueTrees.jl","title":"CliqueTrees.jl","text":"julia> using CliqueTrees, LinearAlgebra, SparseArrays\n\njulia> graph = [\n           0 1 1 0 0 0 0 0\n           1 0 1 0 0 1 0 0\n           1 1 0 1 1 0 0 0\n           0 0 1 0 1 0 0 0\n           0 0 1 1 0 0 1 1\n           0 1 0 0 0 0 1 0\n           0 0 0 0 1 1 0 1\n           0 0 0 0 1 0 1 0\n       ];\n\njulia> label, filledgraph = eliminationgraph(graph);\n\njulia> sparse(filledgraph)\n8×8 SparseMatrixCSC{Bool, Int64} with 13 stored entries:\n ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅\n ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅\n ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅\n ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅\n ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅\n 1  1  1  1  ⋅  ⋅  ⋅  ⋅\n 1  ⋅  ⋅  ⋅  1  1  ⋅  ⋅\n ⋅  ⋅  ⋅  1  1  1  1  ⋅","category":"page"},{"location":"","page":"CliqueTrees.jl","title":"CliqueTrees.jl","text":"The graph filledgraph is ordered: its edges are directed from lower to higher vertices. The underlying undirected graph is a chordal completion of the permuted graph graph[label, label].","category":"page"},{"location":"","page":"CliqueTrees.jl","title":"CliqueTrees.jl","text":"julia> chordalgraph = Symmetric(sparse(filledgraph), :L)\n8×8 Symmetric{Bool, SparseMatrixCSC{Bool, Int64}}:\n ⋅  ⋅  ⋅  ⋅  ⋅  1  1  ⋅\n ⋅  ⋅  1  ⋅  ⋅  1  ⋅  ⋅\n ⋅  1  ⋅  1  ⋅  1  ⋅  ⋅\n ⋅  ⋅  1  ⋅  ⋅  1  ⋅  1\n ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  1\n 1  1  1  1  ⋅  ⋅  1  1\n 1  ⋅  ⋅  ⋅  1  1  ⋅  1\n ⋅  ⋅  ⋅  1  1  1  1  ⋅\n\njulia> ischordal(graph)\nfalse\n\njulia> ischordal(chordalgraph)\ntrue\n\njulia> all(graph[label, label] .<= chordalgraph)\ntrue","category":"page"},{"location":"#Graphs","page":"CliqueTrees.jl","title":"Graphs","text":"","category":"section"},{"location":"","page":"CliqueTrees.jl","title":"CliqueTrees.jl","text":"Users can input graphs as adjacency matrices. Additionally, CliqueTrees.jl supports the HasGraph type from Catlab.jl and the AbstractGraph type from Graphs.jl. Instances of the latter should implement the following subset of the abstract graph interface.","category":"page"},{"location":"","page":"CliqueTrees.jl","title":"CliqueTrees.jl","text":"is_directed\nne\nnv\noutneighbors\nvertices","category":"page"},{"location":"","page":"CliqueTrees.jl","title":"CliqueTrees.jl","text":"Weights and self-edges are always ignored.","category":"page"},{"location":"#References","page":"CliqueTrees.jl","title":"References","text":"","category":"section"},{"location":"","page":"CliqueTrees.jl","title":"CliqueTrees.jl","text":"CliqueTrees.jl was inspired by the book Chordal Graphs and Semidefinite Optimization by Vandenberghe and Andersen.","category":"page"}]
}
