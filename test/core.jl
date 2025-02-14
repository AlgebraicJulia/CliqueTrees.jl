using AbstractTrees
using Base.Order
using Catlab: Catlab
using CliqueTrees
using CliqueTrees: DoublyLinkedList, RCMGL, LexM, rcmgl, lexm, sympermute
using Graphs
using Graphs: SimpleEdge
using JET
using Laplacians: Laplacians
using LinearAlgebra
using SparseArrays
using Test

@testset "trees" begin
    @testset "interface" begin
        tree = Tree(Int8[2, 5, 4, 5, 0])
        @test rootindex(tree) === Int8(5)
        setrootindex!(tree, 1)
        @test rootindex(tree) === Int8(1)

        node = IndexNode(tree)
        @test ParentLinks(node) === StoredParents()
        @test SiblingLinks(node) === StoredSiblings()
        @test NodeType(node) === HasNodeType()
        @test nodetype(node) === typeof(node)
    end

    @testset "construction" begin
        graph = BipartiteGraph{Int8,Int16}(
            [
                0 1 1 0 0 0 0 0
                1 0 1 0 1 1 1 0
                1 1 0 1 1 0 0 0
                0 0 1 0 1 0 0 0
                0 1 1 1 0 0 1 1
                0 1 0 0 0 0 1 0
                0 1 0 0 1 1 0 1
                0 0 0 0 1 0 1 0
            ],
        )

        label, tree = eliminationtree(graph; alg=1:8)
        @test isequal(Tree(tree), tree)
        @test isequal(Tree{Int32}(tree), tree)

        label, tree = supernodetree(graph; alg=1:8)
        @test isequal(Tree(tree), tree.tree)
        @test isequal(Tree{Int32}(tree), tree.tree)

        label, tree = cliquetree(graph; alg=1:8)
        @test isequal(Tree(tree), tree.tree.tree)
        @test isequal(Tree{Int32}(tree), tree.tree.tree)
    end
end

@testset "bipartite graphs" begin
    graph = BipartiteGraph{Int8,Int16}(
        [
            0 1 1 0 0 0 0 0
            1 0 1 0 1 1 1 0
            1 1 0 1 1 0 0 0
            0 0 1 0 1 0 0 0
            0 1 1 1 0 0 1 1
            0 1 0 0 0 0 1 0
            0 1 0 0 1 1 0 1
            0 0 0 0 1 0 1 0
        ],
    )

    @testset "conversion" begin
        @test isa(
            convert(BipartiteGraph{Int32,Int64,Vector{Int64},Vector{Int32}}, graph),
            BipartiteGraph{Int32,Int64,Vector{Int64},Vector{Int32}},
        )
        @test convert(BipartiteGraph{Int8,Int16,Vector{Int16},Vector{Int8}}, graph) ===
            graph
    end

    @testset "construction" begin
        @test allequal((
            graph,
            BipartiteGraph(graph),
            BipartiteGraph{Int32}(graph),
            BipartiteGraph{Int32,Int64}(graph),
        ))

        @test allequal((
            graph,
            BipartiteGraph(Matrix(graph)),
            BipartiteGraph{Int8}(Matrix{Float64}(graph)),
            BipartiteGraph{Int8,Int16}(Matrix{Float64}(graph)),
        ))

        @test allequal((
            graph,
            BipartiteGraph(SparseMatrixCSC(graph)),
            BipartiteGraph{Int8}(SparseMatrixCSC{Float64}(graph)),
            BipartiteGraph{Int8,Int16}(SparseMatrixCSC{Float64,Int32}(graph)),
        ))

        @test allequal((
            graph,
            BipartiteGraph(sparse(graph)),
            BipartiteGraph{Int8}(sparse(Float64, graph)),
            BipartiteGraph{Int8,Int16}(sparse(Float64, Int32, graph)),
        ))

        @test allequal((
            graph,
            BipartiteGraph(Graph(graph)),
            BipartiteGraph{Int8}(Graph{Int64}(graph)),
            BipartiteGraph{Int8,Int16}(Graph{Int32}(graph)),
        ))

        @test allequal((
            graph,
            BipartiteGraph(DiGraph(graph)),
            BipartiteGraph{Int8}(DiGraph{Int64}(graph)),
            BipartiteGraph{Int8,Int16}(DiGraph{Int32}(graph)),
        ))

        @test allequal((
            graph,
            BipartiteGraph(Catlab.Graph(graph)),
            BipartiteGraph{Int8}(Catlab.Graph(graph)),
            BipartiteGraph{Int8,Int16}(Catlab.Graph(graph)),
        ))

        @test allequal((
            graph,
            BipartiteGraph(Catlab.SymmetricGraph(graph)),
            BipartiteGraph{Int8}(Catlab.SymmetricGraph(graph)),
            BipartiteGraph{Int8,Int16}(Catlab.SymmetricGraph(graph)),
        ))
    end

    @testset "interface" begin
        nullgraph = zero(BipartiteGraph{Int8,Int16,Vector{Int16},Vector{Int8}})
        @test nv(nullgraph) === zero(Int8)
        @test ne(nullgraph) === zero(Int16)

        @test nv(graph) === Int8(8)
        @test ne(graph) === Int16(26)
        @test eltype(graph) === Int8
        @test edgetype(graph) === SimpleEdge{Int8}

        @test has_edge(graph, 1, 2)
        @test !has_edge(graph, 1, 4)

        @test SimpleEdge(1, 2) ∈ edges(graph)
        @test SimpleEdge(1, 4) ∉ edges(graph)

        @test vertices(graph) == 1:8
        @test src.(edges(graph)) ==
            [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8]
        @test dst.(edges(graph)) ==
            [2, 3, 1, 3, 5, 6, 7, 1, 2, 4, 5, 3, 5, 2, 3, 4, 7, 8, 2, 7, 2, 5, 6, 8, 5, 7]

        @test outneighbors(graph, 1) == [2, 3]
        @test collect(inneighbors(graph, 1)) == [2, 3]
        @test all_neighbors(graph, 1) == [2, 3]

        @test outdegree(graph, 1) == 2
        @test indegree(graph, 1) == 2
        @test degree(graph, 1) == 4
    end
end

@testset "linked lists" begin
    @testset "singly linked list" begin
        list = SinglyLinkedList{Int}(3)
        @test isempty(list)
        @test collect(list) == []

        @test pushfirst!(list, 1) === list
        @test !isempty(list)
        @test collect(list) == [1]

        @test pushfirst!(list, 2) === list
        @test !isempty(list)
        @test collect(list) == [2, 1]

        @test pushfirst!(list, 3) === list
        @test !isempty(list)
        @test collect(list) == [3, 2, 1]

        @test popfirst!(list) == 3
        @test !isempty(list)
        @test collect(list) == [2, 1]

        @test popfirst!(list) == 2
        @test !isempty(list)
        @test collect(list) == [1]

        @test popfirst!(list) == 1
        @test isempty(list)
        @test collect(list) == []
    end

    @testset "doubly linked list" begin
        list = DoublyLinkedList{Int}(3)
        @test isempty(list)
        @test collect(list) == []

        @test pushfirst!(list, 1) === list
        @test !isempty(list)
        @test collect(list) == [1]

        @test pushfirst!(list, 2) === list
        @test !isempty(list)
        @test collect(list) == [2, 1]

        @test pushfirst!(list, 3) === list
        @test !isempty(list)
        @test collect(list) == [3, 2, 1]

        @test delete!(list, 2) === list
        @test !isempty(list)
        @test collect(list) == [3, 1]

        @test delete!(list, 3) === list
        @test !isempty(list)
        @test collect(list) == [1]

        @test popfirst!(list) == 1
        @test isempty(list)
        @test collect(list) == []
    end
end

@testset "representation" begin
    for A in (BFS, MCS, LexBFS, RCM, RCMGL, LexM, AAMD, SymAMD, MMD, NodeND, Spectral, BT)
        @test isa(repr("text/plain", A()), String)
    end

    for L in (SinglyLinkedList, DoublyLinkedList)
        @test isa(repr("text/plain", L([1, 2, 3, 4, 5, 6])), String)
    end

    graph = BipartiteGraph(
        [
            1 0 0 0 0 0
            0 1 0 0 0 0
            0 0 1 0 0 0
            0 0 0 1 0 0
            0 0 0 0 1 0
            0 0 0 0 0 1
        ]
    )

    @test isa(repr("text/plain", graph), String)
    @test isa(repr("text/plain", edges(graph)), String)
    label, tree = eliminationtree(graph)
    @test isa(repr("text/plain", tree), String)
    label, tree = supernodetree(graph)
    @test isa(repr("text/plain", tree), String)
    label, tree = cliquetree(graph)
    @test isa(repr("text/plain", tree), String)
end

@testset "null graph" begin
    graph = spzeros(0, 0)
    @test ischordal(graph)
    @test isfilled(graph)
    @test iszero(treewidth(graph))

    @test permutation(graph; alg=BFS()) == ([], [])
    @test permutation(graph; alg=MCS()) == ([], [])
    @test permutation(graph; alg=LexBFS()) == ([], [])
    @test permutation(graph; alg=RCM()) == ([], [])
    @test permutation(graph; alg=RCMGL()) == ([], [])
    @test permutation(graph; alg=LexM()) == ([], [])
    @test permutation(graph; alg=AAMD()) == ([], [])
    @test permutation(graph; alg=SymAMD()) == ([], [])
    @test permutation(graph; alg=MMD()) == ([], []) skip = true
    @test permutation(graph; alg=NodeND()) == ([], []) skip = true
    @test permutation(graph; alg=BT()) == ([], [])

    label, tree = cliquetree(graph; snd=Nodal())
    @test iszero(length(tree))
    @test isnothing(rootindex(tree))
    @test iszero(treewidth(tree))
    @test iszero(nv(tree))
    @test iszero(ne(tree))

    label, tree = cliquetree(graph; snd=Maximal())
    @test iszero(length(tree))
    @test isnothing(rootindex(tree))
    @test iszero(treewidth(tree))
    @test iszero(nv(tree))
    @test iszero(ne(tree))

    label, tree = cliquetree(graph; snd=Fundamental())
    @test iszero(length(tree))
    @test isnothing(rootindex(tree))
    @test iszero(treewidth(tree))
    @test iszero(nv(tree))
    @test iszero(ne(tree))
end

@testset "singleton graph" begin
    graph = spzeros(1, 1)
    @test ischordal(graph)
    @test isfilled(graph)
    @test iszero(treewidth(graph))

    @test permutation(graph; alg=BFS()) == ([1], [1])
    @test permutation(graph; alg=MCS()) == ([1], [1])
    @test permutation(graph; alg=LexBFS()) == ([1], [1])
    @test permutation(graph; alg=RCM()) == ([1], [1])
    @test permutation(graph; alg=RCMGL()) == ([1], [1])
    @test permutation(graph; alg=LexM()) == ([1], [1])
    @test permutation(graph; alg=AAMD()) == ([1], [1])
    @test permutation(graph; alg=SymAMD()) == ([1], [1])
    @test permutation(graph; alg=MMD()) == ([1], [1])
    @test permutation(graph; alg=NodeND()) == ([1], [1])
    @test permutation(graph; alg=Spectral()) == ([1], [1]) skip = true
    @test permutation(graph; alg=BT()) == ([1], [1])

    label, tree = cliquetree(graph; snd=Nodal())
    @test isone(length(tree))
    @test isone(rootindex(tree))
    @test iszero(treewidth(tree))
    @test isone(nv(tree))
    @test iszero(ne(tree))
    @test isnothing(parentindex(tree, 1))
    @test isempty(childindices(tree, 1))
    @test isempty(separator(tree, 1))
    @test isempty(neighbors(relatives(tree), 1))
    @test isone(only(residual(tree, 1)))
    @test isone(only(tree[1]))

    label, tree = cliquetree(graph; snd=Maximal())
    @test isone(length(tree))
    @test isone(length(tree))
    @test isone(rootindex(tree))
    @test iszero(treewidth(tree))
    @test isone(nv(tree))
    @test iszero(ne(tree))
    @test isnothing(parentindex(tree, 1))
    @test isempty(childindices(tree, 1))
    @test isempty(separator(tree, 1))
    @test isempty(neighbors(relatives(tree), 1))
    @test isone(only(residual(tree, 1)))
    @test isone(only(tree[1]))

    label, tree = cliquetree(graph; snd=Fundamental())
    @test isone(length(tree))
    @test isone(rootindex(tree))
    @test iszero(treewidth(tree))
    @test isone(nv(tree))
    @test iszero(ne(tree))
    @test isnothing(parentindex(tree, 1))
    @test isempty(childindices(tree, 1))
    @test isempty(separator(tree, 1))
    @test isempty(neighbors(relatives(tree), 1))
    @test isone(only(residual(tree, 1)))
    @test isone(only(tree[1]))
end

# Chordal Graphs and Semidefinite Optimization
# Vandenberghe and Andersen
@testset "vandenberghe and andersen" begin
    types = (
        (BipartiteGraph{Int8,Int16}, Int8, Int16),
        (Matrix{Float64}, Int, Int),
        (SparseMatrixCSC{Float64,Int16}, Int16, Int16),
        (Graph{Int8}, Int8, Int),
        (DiGraph{Int8}, Int8, Int),
        (Catlab.Graph, Int, Int),
        (Catlab.SymmetricGraph, Int, Int),
    )

    __graph = BipartiteGraph(
        [
            0 0 1 1 1 0 0 0 0 0 0 0 0 0 1 0 0
            0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0
            0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0
            0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0
            0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 1
            0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 1
            0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0
            1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1
            0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 0 0
        ],
    )

    # Figure 4.2
    __completion = BipartiteGraph(
        [
            0 0 1 1 1 0 0 0 0 0 0 0 0 0 1 0 0
            0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 1 0 1 1 0 0 0 0 0 0 0 0 0 1 0 0
            1 1 1 0 1 0 0 0 0 0 0 0 0 0 1 0 0
            1 0 1 1 0 0 0 0 1 0 0 0 0 0 1 1 0
            0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0
            0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0
            0 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0 0
            0 0 0 0 1 1 1 1 0 0 0 0 0 0 1 1 0
            0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 1
            0 0 0 0 0 0 0 0 0 1 0 0 1 1 0 0 1
            0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 1
            0 0 0 0 0 0 0 0 0 1 1 1 0 1 0 1 1
            0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 1 1
            1 0 1 1 1 0 1 1 1 0 0 0 0 0 0 1 1
            0 0 0 0 1 1 0 0 1 0 0 1 1 1 1 0 1
            0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0
        ],
    )

    for (G, V, E) in types
        @testset "$(nameof(G))" begin
            graph = G(__graph)
            completion = G(__completion)

            @testset "inference" begin
                @inferred ischordal(graph)
                @inferred isfilled(graph)
                @inferred bfs(graph)
                @inferred mcs(graph)
                @inferred mcs(graph, V[1, 3])
                @inferred lexbfs(graph)
                @inferred rcm(graph)
                @inferred rcmgl(graph)
                @inferred lexm(graph)
                @inferred treewidth(graph; alg=1:17)
                @inferred eliminationtree(graph; alg=1:17)
                @inferred supernodetree(graph; alg=1:17, snd=Nodal())
                @inferred supernodetree(graph; alg=1:17, snd=Maximal())
                @inferred supernodetree(graph; alg=1:17, snd=Fundamental())
                @inferred cliquetree(graph; alg=1:17, snd=Nodal())
                @inferred cliquetree(graph; alg=1:17, snd=Maximal())
                @inferred cliquetree(graph; alg=1:17, snd=Fundamental())
                @inferred eliminationgraph(graph; alg=1:17, snd=Nodal())
                @inferred eliminationgraph(graph; alg=1:17, snd=Maximal())
                @inferred eliminationgraph(graph; alg=1:17, snd=Fundamental())

                label, tree = cliquetree(graph; alg=1:17)
                @inferred treewidth(tree)
                @inferred eliminationgraph(tree)
            end

            @testset "JET" begin
                @test_call ischordal(graph)
                @test_call isfilled(graph)
                @test_call bfs(graph)
                @test_call mcs(graph)
                @test_call mcs(graph, V[1, 3])
                @test_call lexbfs(graph)
                @test_call rcm(graph)
                @test_call rcmgl(graph)
                @test_call lexm(graph)
                @test_call treewidth(graph; alg=1:17)
                @test_call eliminationtree(graph; alg=1:17)
                @test_call supernodetree(graph; alg=1:17, snd=Nodal())
                @test_call supernodetree(graph; alg=1:17, snd=Maximal())
                @test_call supernodetree(graph; alg=1:17, snd=Fundamental())
                @test_call cliquetree(graph; alg=1:17, snd=Nodal())
                @test_call cliquetree(graph; alg=1:17, snd=Maximal())
                @test_call cliquetree(graph; alg=1:17, snd=Fundamental())
                @test_call eliminationgraph(graph; alg=1:17, snd=Nodal())
                @test_call eliminationgraph(graph; alg=1:17, snd=Maximal())
                @test_call eliminationgraph(graph; alg=1:17, snd=Fundamental())

                label, tree = cliquetree(graph; alg=1:17)
                @test_call treewidth(tree)
                @test_call eliminationgraph(tree)
            end

            @testset "chordality" begin
                @test !ischordal(graph)
                @test !isfilled(graph)
                @test !isperfect(graph, permutation(graph, MCS())...)
                @test !isperfect(graph, permutation(graph, LexBFS())...)
                @test !isperfect(graph, permutation(graph, LexM())...)
                @test treewidth(graph; alg=1:17) === V(4)

                @test isfilled(completion)
                @test ischordal(completion)
                @test isperfect(completion, permutation(completion, MCS())...)
                @test isperfect(completion, permutation(completion, LexBFS())...)
                @test isperfect(completion, permutation(completion, LexM())...)
                @test treewidth(completion; alg=1:17) === V(4)

                label, _completion = eliminationgraph(graph; alg=1:17)
                @test _completion ==
                    reverse(sympermute(__completion, invperm(label), Forward))
            end

            @testset "permutations" begin
                for A in (
                    BFS,
                    MCS,
                    LexBFS,
                    RCM,
                    RCMGL,
                    LexM,
                    AAMD,
                    SymAMD,
                    MMD,
                    NodeND,
                    Spectral,
                    BT,
                )
                    order, index = permutation(graph; alg=A())
                    @test isa(order, Vector{V})
                    @test isa(index, Vector{V})
                    @test length(order) == 17
                    @test order[index] == 1:17
                end
            end

            @testset "clique trees" begin
                @testset "nodal" begin
                    # Figure 4.3
                    label, tree = cliquetree(graph; alg=1:17, snd=Nodal())
                    @test isa(label, Vector{V})
                    @test isa(tree, CliqueTree{V,E})
                    @test length(tree) == 17
                    @test rootindex(tree) === V(17)
                    @test treewidth(tree) === V(4)
                    @test nv(tree) === V(17)
                    @test ne(tree) === E(42)
                    @test eliminationgraph(tree) ==
                        reverse(sympermute(__completion, invperm(label), Forward))

                    @test map(i -> parentindex(tree, i), 1:17) ==
                        [2, 4, 4, 5, 16, 8, 8, 9, 10, 14, 14, 13, 14, 15, 16, 17, nothing]

                    @test map(i -> collect(childindices(tree, i)), 1:17) == [
                        [],
                        [1],
                        [],
                        [2, 3],
                        [4],
                        [],
                        [],
                        [6, 7],
                        [8],
                        [9],
                        [],
                        [],
                        [12],
                        [10, 11, 13],
                        [14],
                        [5, 15],
                        [16],
                    ]

                    @test map(i -> view(label, residual(tree, i)), 1:17) == [
                        [10], # j
                        [11], # k
                        [12], # l
                        [13], # m
                        [14], # n
                        [1],  # a
                        [2],  # b
                        [3],  # c
                        [4],  # d
                        [5],  # e
                        [6],  # f
                        [7],  # g
                        [8],  # h
                        [9],  # i
                        [15], # o
                        [16], # p
                        [17], # q
                    ]

                    @test map(i -> view(label, separator(tree, i)), 1:17) == [
                        [11, 13, 14, 17], # k m n q
                        [13, 14, 17],     # m n q
                        [13, 14, 16, 17], # m n p q
                        [14, 16, 17],     # n p q
                        [16, 17],         # p q
                        [3, 4, 5, 15],    # c d e o
                        [3, 4],           # c d
                        [4, 5, 15],       # d e o
                        [5, 15],          # e o
                        [9, 15, 16],      # i o p
                        [9, 16],          # i p
                        [8, 9, 15],       # h i o
                        [9, 15],          # i o
                        [15, 16],         # o p
                        [16, 17],         # p q
                        [17],             # q
                        [],               #
                    ]

                    @test all(1:17) do i
                        tree[i] == [residual(tree, i); separator(tree, i)]
                    end

                    rg = relatives(tree)
                    @test isa(rg, BipartiteGraph{V,E})

                    @test all(1:16) do i
                        j = parentindex(tree, i)
                        tree[j][neighbors(rg, i)] == separator(tree, i)
                    end

                    @test neighbors(rg, 17) == []
                end

                @testset "maximal" begin
                    # Figure 4.7 (left)
                    label, tree = cliquetree(graph; alg=1:17, snd=Maximal())
                    @test isa(label, Vector{V})
                    @test isa(tree, CliqueTree{V,E})
                    @test length(tree) == 8
                    @test rootindex(tree) === V(8)
                    @test treewidth(tree) === V(4)
                    @test nv(tree) === V(17)
                    @test ne(tree) === E(42)
                    @test eliminationgraph(tree) ==
                        reverse(sympermute(__completion, invperm(label), Forward))

                    @test map(i -> parentindex(tree, i), 1:8) ==
                        [8, 3, 6, 6, 6, 7, 8, nothing]

                    @test map(i -> collect(childindices(tree, i)), 1:8) ==
                        [[], [], [2], [], [], [3, 4, 5], [6], [1, 7]]

                    @test map(i -> view(label, residual(tree, i)), 1:8) == [
                        [10, 11],             # j k
                        [2],                  # b
                        [1, 3, 4],            # a c d
                        [6],                  # f
                        [7, 8],               # g h
                        [5, 9],               # e i
                        [15],                 # o
                        [12, 13, 14, 16, 17], # l m n p q
                    ]

                    @test map(i -> view(label, separator(tree, i)), 1:8) == [
                        [13, 14, 17], # m n q
                        [3, 4],       # c d
                        [5, 15],      # e o
                        [9, 16],      # i p
                        [9, 15],      # i o
                        [15, 16],     # o p
                        [16, 17],     # p q
                        [],           #
                    ]

                    @test all(1:8) do i
                        tree[i] == [residual(tree, i); separator(tree, i)]
                    end

                    rg = relatives(tree)
                    @test isa(rg, BipartiteGraph{V,E})

                    @test all(1:7) do i
                        j = parentindex(tree, i)
                        tree[j][neighbors(rg, i)] == separator(tree, i)
                    end

                    @test neighbors(rg, 8) == []
                end

                @testset "fundamental" begin
                    # Figure 4.9
                    label, tree = cliquetree(graph; alg=1:17, snd=Fundamental())
                    @test isa(label, Vector{V})
                    @test isa(tree, CliqueTree{V,E})
                    @test length(tree) == 12
                    @test rootindex(tree) === V(12)
                    @test treewidth(tree) === V(4)
                    @test nv(tree) === V(17)
                    @test ne(tree) === E(42)
                    @test eliminationgraph(tree) ==
                        reverse(sympermute(__completion, invperm(label), Forward))

                    @test map(i -> parentindex(tree, i), 1:12) ==
                        [3, 3, 12, 6, 6, 7, 10, 10, 10, 11, 12, nothing]

                    @test map(i -> collect(childindices(tree, i)), 1:12) == [
                        [],
                        [],
                        [1, 2],
                        [],
                        [],
                        [4, 5],
                        [6],
                        [],
                        [],
                        [7, 8, 9],
                        [10],
                        [3, 11],
                    ]

                    @test map(i -> view(label, residual(tree, i)), 1:12) == [
                        [10, 11], # j k
                        [12],     # l
                        [13, 14], # m n
                        [1],      # a
                        [2],      # b
                        [3, 4],   # c d
                        [5],      # e
                        [6],      # f
                        [7, 8],   # g h
                        [9],      # i
                        [15],     # o
                        [16, 17], # p q
                    ]

                    @test map(i -> view(label, separator(tree, i)), 1:12) == [
                        [13, 14, 17],     # m n q
                        [13, 14, 16, 17], # m n p q
                        [16, 17],         # p q
                        [3, 4, 5, 15],    # c d e o
                        [3, 4],           # c d
                        [5, 15],          # e o
                        [9, 15, 16],      # i o p
                        [9, 16],          # i p
                        [9, 15],          # i o
                        [15, 16],         # o p
                        [16, 17],         # p q
                        [],               #
                    ]

                    @test all(1:12) do i
                        tree[i] == [residual(tree, i); separator(tree, i)]
                    end

                    rg = relatives(tree)
                    @test isa(rg, BipartiteGraph{V,E})

                    @test all(1:11) do i
                        j = parentindex(tree, i)
                        tree[j][neighbors(rg, i)] == separator(tree, i)
                    end

                    @test neighbors(rg, 12) == []
                end
            end
        end
    end
end
