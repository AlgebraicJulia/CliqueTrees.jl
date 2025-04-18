using AbstractTrees
using Base: oneto
using Base.Order
using CliqueTrees
using CliqueTrees: DoublyLinkedList, sympermute
using Graphs
using Graphs: SimpleEdge
using JET
using LinearAlgebra
using SparseArrays
using Test

import Catlab
import libpicosat_jll
import CryptoMiniSat_jll
import FlowCutterPACE17_jll

@static if Sys.iswindows()
    import libpicosat_jll as PicoSAT_jll
    import libpicosat_jll as Lingeling_jll
else
    import PicoSAT_jll
    import Lingeling_jll
end

const TYPES = (
    (BipartiteGraph{Int8, Int16}, Int8, Int16),
    (Matrix{Float64}, Int, Int),
    (SparseMatrixCSC{Float64, Int16}, Int16, Int16),
    (Graph{Int8}, Int8, Int),
    (DiGraph{Int8}, Int8, Int),
    (Catlab.Graph, Int, Int),
    (Catlab.SymmetricGraph, Int, Int),
)

@testset "errors" begin
    matrix = [
        0 1 1 0 0 0 0 0
        1 0 1 0 1 1 1 0
        1 1 0 1 1 0 0 0
        0 0 1 0 1 0 0 0
        0 1 1 1 0 0 1 1
        0 1 0 0 0 0 1 0
        0 1 0 0 1 1 0 1
        0 0 0 0 1 0 1 0
    ]

    @test_throws ArgumentError permutation(matrix; alg = AMD())
    @test_throws ArgumentError permutation(matrix; alg = SymAMD())
    @test_throws ArgumentError permutation(matrix; alg = METIS())
    @test_throws ArgumentError permutation(matrix; alg = Spectral())
    @test_throws ArgumentError permutation(matrix; alg = BT())
end

using AMD: AMD as AMDJL
using Laplacians: Laplacians
using Metis: Metis
using TreeWidthSolver: TreeWidthSolver

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
        graph = BipartiteGraph{Int8, Int16}(
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

        label, tree = eliminationtree(graph; alg = 1:8)
        @test isequal(Tree(tree), tree)
        @test isequal(Tree{Int32}(tree), tree)

        label, tree = supernodetree(graph; alg = 1:8)
        @test isequal(Tree(tree), tree.tree)
        @test isequal(Tree{Int32}(tree), tree.tree)

        label, tree = cliquetree(graph; alg = 1:8)
        @test isequal(Tree(tree), tree.tree.tree)
        @test isequal(Tree{Int32}(tree), tree.tree.tree)
    end
end

@testset "bipartite graphs" begin
    graph = BipartiteGraph{Int8, Int16}(
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

    label, tree = cliquetree(graph)
    filledgraph = FilledGraph(tree)

    @testset "conversion" begin
        @test isa(
            convert(BipartiteGraph{Int32, Int64, Vector{Int64}, Vector{Int32}}, graph),
            BipartiteGraph{Int32, Int64, Vector{Int64}, Vector{Int32}},
        )
        @test convert(BipartiteGraph{Int8, Int16, Vector{Int16}, Vector{Int8}}, graph) ===
            graph
    end

    @testset "construction" begin
        @test allequal(
            (
                graph,
                BipartiteGraph(graph),
                BipartiteGraph{Int32}(graph),
                BipartiteGraph{Int32, Int64}(graph),
            )
        )

        @test allequal(
            (
                graph,
                BipartiteGraph(Matrix(graph)),
                BipartiteGraph{Int8}(Matrix{Float64}(graph)),
                BipartiteGraph{Int8, Int16}(Matrix{Float64}(graph)),
            )
        )

        @test allequal(
            (
                graph,
                BipartiteGraph(SparseMatrixCSC(graph)),
                BipartiteGraph{Int8}(SparseMatrixCSC{Float64}(graph)),
                BipartiteGraph{Int8, Int16}(SparseMatrixCSC{Float64, Int32}(graph)),
            )
        )

        @test allequal(
            (
                graph,
                BipartiteGraph(sparse(graph)),
                BipartiteGraph{Int8}(sparse(Float64, graph)),
                BipartiteGraph{Int8, Int16}(sparse(Float64, Int32, graph)),
            )
        )

        @test allequal(
            (
                graph,
                BipartiteGraph(Graph(graph)),
                BipartiteGraph{Int8}(Graph{Int64}(graph)),
                BipartiteGraph{Int8, Int16}(Graph{Int32}(graph)),
            )
        )

        @test allequal(
            (
                graph,
                BipartiteGraph(DiGraph(graph)),
                BipartiteGraph{Int8}(DiGraph{Int64}(graph)),
                BipartiteGraph{Int8, Int16}(DiGraph{Int32}(graph)),
            )
        )

        @test allequal(
            (
                graph,
                BipartiteGraph(Catlab.Graph(graph)),
                BipartiteGraph{Int8}(Catlab.Graph(graph)),
                BipartiteGraph{Int8, Int16}(Catlab.Graph(graph)),
            )
        )

        @test allequal(
            (
                graph,
                BipartiteGraph(Catlab.SymmetricGraph(graph)),
                BipartiteGraph{Int8}(Catlab.SymmetricGraph(graph)),
                BipartiteGraph{Int8, Int16}(Catlab.SymmetricGraph(graph)),
            )
        )

        @test DiGraph(filledgraph) == DiGraph(BipartiteGraph(filledgraph))
        @test Graph(filledgraph) == Graph(BipartiteGraph(filledgraph))
    end

    @testset "interface" begin
        nullgraph = zero(BipartiteGraph{Int8, Int16, Vector{Int16}, Vector{Int8}})
        @test nv(nullgraph) === zero(Int8)
        @test ne(nullgraph) === zero(Int16)

        @test is_directed(graph)
        @test nv(graph) === Int8(8)
        @test ne(graph) === Int16(26)
        @test eltype(graph) === Int8
        @test edgetype(graph) === SimpleEdge{Int8}

        @test has_edge(graph, 1, 2)
        @test !has_edge(graph, 1, 4)
        @test SimpleEdge(1, 2) ∈ edges(graph)
        @test SimpleEdge(1, 4) ∉ edges(graph)

        @test vertices(graph) === oneto(Int8(8))
        @test src.(edges(graph)) ==
            [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8]
        @test dst.(edges(graph)) ==
            [2, 3, 1, 3, 5, 6, 7, 1, 2, 4, 5, 3, 5, 2, 3, 4, 7, 8, 2, 7, 2, 5, 6, 8, 5, 7]

        @test neighbors(graph, 1) == [2, 3]
        @test collect(inneighbors(graph, 1)) == [2, 3]
        @test all_neighbors(graph, 1) == [2, 3]

        @test outdegree(graph, 1) === Int8(2)
        @test indegree(graph, 1) === Int8(2)
        @test degree(graph, 1) === Int8(4)
    end
end

@testset "filled graphs" begin
    graph = BipartiteGraph{Int8, Int16}(
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

    label, tree = cliquetree(graph; alg = [1, 4, 6, 8, 3, 7, 2, 5])
    filledgraph = FilledGraph(tree)

    @testset "construction" begin
        @test allequal(
            (
                BipartiteGraph(filledgraph),
                BipartiteGraph{Int8}(filledgraph),
                BipartiteGraph{Int8, Int16}(filledgraph),
                BipartiteGraph(SparseMatrixCSC(filledgraph)),
                BipartiteGraph(SparseMatrixCSC{Float64}(filledgraph)),
                BipartiteGraph(SparseMatrixCSC{Float64, Int32}(filledgraph)),
                BipartiteGraph(sparse(filledgraph)),
                BipartiteGraph(sparse(Float64, filledgraph)),
                BipartiteGraph(sparse(Float64, Int16, filledgraph)),
                BipartiteGraph(Matrix(filledgraph)),
                BipartiteGraph(Matrix{Float64}(filledgraph)),
            )
        )
    end

    @testset "interface" begin
        @test is_directed(filledgraph)
        @test nv(filledgraph) === Int8(8)
        @test ne(filledgraph) === 13
        @test eltype(filledgraph) === Int8
        @test edgetype(filledgraph) === SimpleEdge{Int8}

        @test has_edge(filledgraph, 1, 7)
        @test !has_edge(filledgraph, 7, 1)
        @test SimpleEdge(1, 7) ∈ edges(filledgraph)
        @test SimpleEdge(7, 1) ∉ edges(filledgraph)
        @test Base.hasfastin(edges(filledgraph))

        @test vertices(filledgraph) === oneto(Int8(8))
        @test src.(edges(filledgraph)) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7]
        @test dst.(edges(filledgraph)) == [6, 7, 6, 8, 5, 7, 5, 8, 7, 8, 7, 8, 8]

        @test neighbors(filledgraph, 5) == [7, 8]
        @test collect(inneighbors(filledgraph, 5)) == [3, 4]
        @test collect(all_neighbors(filledgraph, 5)) == [3, 4, 7, 8]

        @test outdegree(filledgraph, 5) === Int8(2)
        @test indegree(filledgraph, 5) === Int8(2)
        @test degree(filledgraph, 5) === Int8(4)
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
    for alg in (
            BFS(),
            MCS(),
            LexBFS(),
            RCMMD(),
            RCMGL(),
            LexM(),
            MCSM(),
            AMD(),
            SymAMD(),
            AMF(),
            MF(),
            MMD(),
            METIS(),
            Spectral(),
            # FlowCutter(),
            BT(),
            MinimalChordal(),
            CompositeRotations([1, 2, 3]),
            SafeRules(),
            SafeSeparators(),
            ConnectedComponents(),
        )
        @test isa(repr("text/plain", alg), String)
    end

    for L in (SinglyLinkedList, DoublyLinkedList)
        list = prepend!(L{Int}(10), [1, 2, 3, 4, 5, 6])
        @test isa(repr("text/plain", list), String)
    end

    graph = BipartiteGraph(
        [
            0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 1 0 0 0 0 0
            0 0 0 0 0 1 0 1 0 0 0 0
            0 0 0 0 0 0 1 0 1 0 0 0
            0 0 0 0 0 0 0 1 0 1 0 0
            0 0 0 0 0 0 0 0 1 0 1 0
            0 0 0 0 0 0 0 0 0 1 0 1
            0 0 0 0 0 0 0 0 0 0 1 0
        ],
    )

    @test isa(repr("text/plain", graph), String)
    @test isa(repr("text/plain", edges(graph)), String)
    label, tree = eliminationtree(graph)
    @test isa(repr("text/plain", tree), String)
    label, tree = supernodetree(graph)
    @test isa(repr("text/plain", tree), String)
    label, tree = cliquetree(graph)
    @test isa(repr("text/plain", tree), String)

    filledgraph = FilledGraph(tree)
    @test isa(repr("text/plain", filledgraph), String)
    @test isa(repr("text/plain", edges(filledgraph)), String)
end

@testset "null graph" begin
    graph = spzeros(0, 0)
    @test ischordal(graph)
    @test iszero(treewidth(graph))

    for alg in (
            BFS(),
            MCS(),
            LexBFS(),
            RCMMD(),
            RCMGL(),
            LexM(),
            MCSM(),
            AMD(),
            SymAMD(),
            AMF(),
            MF(),
            MMD(),
            # METIS(),
            # Spectral(),
            # FlowCutter(),
            BT(),
            MinimalChordal(),
            CompositeRotations([]),
            SafeRules(),
            SafeSeparators(),
            ConnectedComponents(),
        )
        @test permutation(graph; alg) == ([], [])
    end

    for S in (Nodal, Maximal, Fundamental)
        label, tree = cliquetree(graph; snd = S())
        filledgraph = FilledGraph(tree)
        @test iszero(length(tree))
        @test isnothing(rootindex(tree))
        @test iszero(treewidth(tree))
        @test iszero(nv(filledgraph))
        @test iszero(ne(filledgraph))
    end
end

@testset "singleton graph" begin
    graph = spzeros(1, 1)
    @test ischordal(graph)
    @test iszero(treewidth(graph))

    for alg in (
            BFS(),
            MCS(),
            LexBFS(),
            RCMMD(),
            RCMGL(),
            LexM(),
            MCSM(),
            MinimalChordal(),
            AMD(),
            SymAMD(),
            AMF(),
            MF(),
            MMD(),
            METIS(),
            # Spectral,
            # FlowCutter(),
            BT(),
            CompositeRotations([1]),
            SafeRules(),
            SafeSeparators(),
            ConnectedComponents(),
        )
        @test permutation(graph; alg) == ([1], [1])
    end

    for S in (Nodal, Maximal, Fundamental)
        label, tree = cliquetree(graph; snd = S())
        filledgraph = FilledGraph(tree)
        @test isone(length(tree))
        @test isone(rootindex(tree))
        @test iszero(treewidth(tree))
        @test isone(nv(filledgraph))
        @test iszero(ne(filledgraph))
        @test isnothing(parentindex(tree, 1))
        @test isempty(childindices(tree, 1))
        @test isempty(separator(tree, 1))
        @test isempty(neighbors(relatives(tree), 1))
        @test isone(only(residual(tree, 1)))
        @test isone(only(tree[1]))
    end
end

# Chordal Graphs and Semidefinite Optimization
# Vandenberghe and Andersen
@testset "vandenberghe and andersen" begin
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

    for (G, V, E) in TYPES
        @testset "$(nameof(G))" begin
            graph = G(__graph)
            completion = G(__completion)

            @testset "inference" begin
                @inferred ischordal(graph)
                @inferred CliqueTrees.bfs(graph)
                @inferred CliqueTrees.mcs(graph)
                @inferred CliqueTrees.mcs(graph, [1, 3])
                @inferred CliqueTrees.lexbfs(graph)
                @inferred CliqueTrees.rcmmd(graph)
                @inferred CliqueTrees.rcmgl(graph)
                @inferred CliqueTrees.lexm(graph)
                @inferred CliqueTrees.mcsm(graph)
                @inferred CliqueTrees.mcsm(graph, [1, 3])
                @inferred CliqueTrees.amf(graph)
                @inferred CliqueTrees.mf(graph)
                @inferred CliqueTrees.mmd(graph)
                @inferred CliqueTrees.minimalchordal(graph, 1:17)
                @inferred CliqueTrees.pr3(graph, lowerbound(graph))
                @inferred CliqueTrees.pr3(ones(17), graph, lowerbound(ones(17), graph))
                @inferred CliqueTrees.connectedcomponents(graph)
                @inferred treewidth(graph; alg = 1:17)
                @inferred eliminationtree(graph; alg = 1:17)
                @inferred supernodetree(graph; alg = 1:17, snd = Nodal())
                @inferred supernodetree(graph; alg = 1:17, snd = Maximal())
                @inferred supernodetree(graph; alg = 1:17, snd = Fundamental())
                @inferred cliquetree(graph; alg = 1:17, snd = Nodal())
                @inferred cliquetree(graph; alg = 1:17, snd = Maximal())
                @inferred cliquetree(graph; alg = 1:17, snd = Fundamental())

                label, tree = cliquetree(graph; alg = 1:17)
                @inferred treewidth(tree)
            end

            @testset "JET error analysis" begin
                @test_call target_modules = (CliqueTrees,) ischordal(graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.bfs(graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.mcs(graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.mcs(graph, [1, 3])
                @test_call target_modules = (CliqueTrees,) CliqueTrees.lexbfs(graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.rcmmd(graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.rcmgl(graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.lexm(graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.mcsm(graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.mcsm(graph, [1, 3])
                @test_call target_modules = (CliqueTrees,) CliqueTrees.amf(graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.mf(graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.mmd(graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.minimalchordal(graph, 1:17)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.pr3(graph, lowerbound(graph))
                @test_call target_modules = (CliqueTrees,) CliqueTrees.pr3(ones(17), graph, lowerbound(ones(17), graph))
                @test_call target_modules = (CliqueTrees,) CliqueTrees.connectedcomponents(graph)
                @test_call target_modules = (CliqueTrees,) treewidth(graph; alg = 1:17)
                @test_call target_modules = (CliqueTrees,) eliminationtree(graph; alg = 1:17)
                @test_call target_modules = (CliqueTrees,) supernodetree(
                    graph; alg = 1:17, snd = Nodal()
                )
                @test_call target_modules = (CliqueTrees,) supernodetree(
                    graph; alg = 1:17, snd = Maximal()
                )
                @test_call target_modules = (CliqueTrees,) supernodetree(
                    graph; alg = 1:17, snd = Fundamental()
                )
                @test_call target_modules = (CliqueTrees,) cliquetree(
                    graph; alg = 1:17, snd = Nodal()
                )
                @test_call target_modules = (CliqueTrees,) cliquetree(
                    graph; alg = 1:17, snd = Maximal()
                )
                @test_call target_modules = (CliqueTrees,) cliquetree(
                    graph; alg = 1:17, snd = Fundamental()
                )

                label, tree = cliquetree(graph; alg = 1:17)
                @test_call target_modules = (CliqueTrees,) treewidth(tree)
            end

            @testset "JET optimization analysis" begin
                @test_opt target_modules = (CliqueTrees,) ischordal(graph)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.bfs(graph)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.mcs(graph)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.mcs(graph, [1, 3])
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.lexbfs(graph)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.rcmmd(graph)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.rcmgl(graph)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.lexm(graph)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.mcsm(graph)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.mcsm(graph, [1, 3])
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.amf(graph)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.mf(graph)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.mmd(graph)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.minimalchordal(graph, 1:17)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.pr3(graph, lowerbound(graph))
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.pr3(ones(17), graph, lowerbound(ones(17), graph))
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.connectedcomponents(graph)
                @test_opt target_modules = (CliqueTrees,) treewidth(graph; alg = 1:17)
                @test_opt target_modules = (CliqueTrees,) eliminationtree(graph; alg = 1:17)
                @test_opt target_modules = (CliqueTrees,) supernodetree(
                    graph; alg = 1:17, snd = Nodal()
                )
                @test_opt target_modules = (CliqueTrees,) supernodetree(
                    graph; alg = 1:17, snd = Maximal()
                )
                @test_opt target_modules = (CliqueTrees,) supernodetree(
                    graph; alg = 1:17, snd = Fundamental()
                )
                @test_opt target_modules = (CliqueTrees,) cliquetree(
                    graph; alg = 1:17, snd = Nodal()
                )
                @test_opt target_modules = (CliqueTrees,) cliquetree(
                    graph; alg = 1:17, snd = Maximal()
                )
                @test_opt target_modules = (CliqueTrees,) cliquetree(
                    graph; alg = 1:17, snd = Fundamental()
                )

                label, tree = cliquetree(graph; alg = 1:17)
                @test_opt target_modules = (CliqueTrees,) treewidth(tree)
            end

            @testset "chordality" begin
                @test !ischordal(graph)
                @test !isperfect(graph, permutation(graph, MCS()))
                @test !isperfect(graph, permutation(graph, LexBFS()))
                @test !isperfect(graph, permutation(graph, LexM()))
                @test !isperfect(graph, permutation(graph, MCSM()))
                @test !isperfect(graph, permutation(graph, MF()))

                @test ischordal(completion)
                @test isperfect(completion, permutation(completion, MCS()))
                @test isperfect(completion, permutation(completion, LexBFS()))
                @test isperfect(completion, permutation(completion, LexM()))
                @test isperfect(completion, permutation(completion, MCSM()))
                @test isperfect(completion, permutation(completion, MF()))

                coloring = CliqueTrees.color(completion)
                @test coloring.num_colors == 5
                @test unique(sort(coloring.colors)) == 1:5

                @test all(edges(__completion)) do edge
                    v = src(edge)
                    w = dst(edge)
                    return coloring.colors[v] != coloring.colors[w]
                end
            end

            @testset "permutations" begin
                for alg in (
                        BFS(),
                        MCS(),
                        LexBFS(),
                        RCMMD(),
                        RCMGL(),
                        LexM(),
                        MCSM(),
                        AMD(),
                        SymAMD(),
                        AMF(),
                        MF(),
                        MMD(),
                        METIS(),
                        Spectral(),
                        # FlowCutter(),
                        BT(),
                        MinimalChordal(),
                        CompositeRotations([1, 3]),
                        SafeRules(),
                        SafeSeparators(),
                        ConnectedComponents(),
                    )
                    order, index = permutation(graph; alg)
                    order, index = permutation(ones(17), graph; alg)
                    @test isa(order, Vector{V})
                    @test isa(index, Vector{V})
                    @test length(order) == 17
                    @test order[index] == 1:17
                end
            end

            @testset "clique trees" begin
                @testset "nodal" begin
                    # Figure 4.3
                    label, tree = cliquetree(graph; alg = 1:17, snd = Nodal())
                    filledgraph = FilledGraph(tree)
                    @test isa(label, Vector{V})
                    @test isa(tree, CliqueTree{V, E})
                    @test isa(filledgraph, FilledGraph{V, E})
                    @test length(tree) == 17
                    @test rootindex(tree) === V(17)
                    @test treewidth(tree) === V(4)
                    @test nv(filledgraph) === V(17)
                    @test ne(filledgraph) === 42
                    @test Symmetric(sparse(filledgraph), :L) ==
                        sparse(__completion)[label, label]

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
                    @test isa(rg, BipartiteGraph{V, E})

                    @test all(1:16) do i
                        j = parentindex(tree, i)
                        tree[j][neighbors(rg, i)] == separator(tree, i)
                    end

                    @test neighbors(rg, 17) == []
                end

                @testset "maximal" begin
                    # Figure 4.7 (left)
                    label, tree = cliquetree(graph; alg = 1:17, snd = Maximal())
                    filledgraph = FilledGraph(tree)
                    @test isa(label, Vector{V})
                    @test isa(tree, CliqueTree{V, E})
                    @test isa(filledgraph, FilledGraph{V, E})
                    @test length(tree) == 8
                    @test rootindex(tree) === V(8)
                    @test treewidth(tree) === V(4)
                    @test nv(filledgraph) === V(17)
                    @test ne(filledgraph) === 42
                    @test Symmetric(sparse(filledgraph), :L) ==
                        sparse(__completion)[label, label]

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
                    @test isa(rg, BipartiteGraph{V, E})

                    @test all(1:7) do i
                        j = parentindex(tree, i)
                        tree[j][neighbors(rg, i)] == separator(tree, i)
                    end

                    @test neighbors(rg, 8) == []
                end

                @testset "fundamental" begin
                    # Figure 4.9
                    label, tree = cliquetree(graph; alg = 1:17, snd = Fundamental())
                    filledgraph = FilledGraph(tree)
                    @test isa(label, Vector{V})
                    @test isa(tree, CliqueTree{V, E})
                    @test isa(filledgraph, FilledGraph{V, E})
                    @test length(tree) == 12
                    @test rootindex(tree) === V(12)
                    @test treewidth(tree) === V(4)
                    @test nv(filledgraph) === V(17)
                    @test ne(filledgraph) === 42
                    @test Symmetric(sparse(filledgraph), :L) ==
                        sparse(__completion)[label, label]

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
                    @test isa(rg, BipartiteGraph{V, E})

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

@testset "treewidth" begin
    __graph1 = BipartiteGraph(
        [
            0 1 0 1 0 1 1 0 0 0 0 0 0 0 0 0
            1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0
            0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0
            1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0
            0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0
            1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0
            1 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 1
            0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0
            0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 1
            0 0 0 0 0 0 0 0 1 1 0 0 0 1 0 0
            0 0 0 0 0 0 0 1 0 0 0 0 1 0 1 1
            0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0
            0 0 0 0 0 0 0 1 0 0 1 0 0 0 1 1
            0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0
            0 0 0 0 0 0 0 1 0 1 0 1 0 1 0 0
        ]
    )

    __graph2 = BipartiteGraph(
        [
            0 1 1 0 1 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 0 1 1 1 1 1 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 1 0 0 1 1 0 1 0 0 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 1 0 0 0 0 1 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 1 1 0 0 1 1 1 0 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 1 1 0 1 0 1 1 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 1 0 1 1 1 0 1 0 0 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 0 1 1 1 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 0 0 0 0 0 0 0 0 1 1 0 1 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 1 0 0 0 0 0 0 1 0 1 1 1 1 1 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0
            1 1 1 0 1 1 0 0 1 1 0 0 1 1 0 1 0 0 1 0 1 1 0 0 0 0 0 0 0 0 0 0
            0 1 0 1 0 0 1 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0
            0 0 1 0 1 0 0 1 1 1 1 0 0 1 1 1 0 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0
            0 0 1 0 0 1 0 1 1 1 1 0 1 0 1 1 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0
            0 0 0 1 0 0 1 0 0 1 0 1 1 1 0 1 0 0 0 1 0 0 1 1 0 0 0 0 0 0 0 0
            0 0 0 0 1 1 1 1 0 0 1 0 1 1 1 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 0 1 1 0 0 1 0 1 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 1 1 1 1 1 0 0 1 1 1 0 0 0 0
            0 0 0 0 0 0 0 0 1 1 1 0 1 1 0 0 1 1 0 0 1 1 0 1 0 0 1 0 1 1 0 0
            0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 1 0
            0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 1 1 1 0 0 1 1 1 0 0 1 0 1 0 0 1
            0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 1 1 1 0 1 0 1 1 0 0 1 0 0 1 0 1
            0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 1 1 1 0 1 0 0 0 1 0 0 1 1
            0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 1 0 1 1 1 0 0 0 0 0 1 1 0 1
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0
        ]
    )

    weights1 = log2.([2, 2, 4, 2, 4, 2, 2, 2, 2, 2, 4, 2, 4, 2, 4, 2])
    weights2 = log2.([4, 3, 4, 3, 4, 4, 3, 4, 4, 3, 4, 3, 4, 4, 3, 4, 4, 3, 4, 3, 4, 4, 3, 4, 4, 3, 4, 3, 4, 4, 3, 4])

    for (G, V, E) in TYPES
        @testset "$(nameof(G))" begin
            graph1 = G(__graph1)
            graph2 = G(__graph2)

            width1 = V(4)
            lb1 = lowerbound(graph1)
            ub1 = treewidth(graph1)

            @test isa(lb1, V)
            @test isa(ub1, V)
            @test lb1 <= width1 <= ub1
            #@test treewidth(graph1; alg=ConnectedComponents(BT())) === width1
            @test treewidth(graph1; alg = SAT{libpicosat_jll}()) === width1
            @test treewidth(graph1; alg = SAT{Lingeling_jll}()) === width1
            @test treewidth(graph1; alg = SAT{PicoSAT_jll}()) === width1
            @test treewidth(graph1; alg = SAT{CryptoMiniSat_jll}()) === width1
            #@test treewidth(graph1; alg=SafeRules(ConnectedComponents(BT()))) === width1
            @test treewidth(graph1; alg = SafeRules(SAT{libpicosat_jll}())) === width1
            @test treewidth(graph1; alg = SafeRules(SAT{Lingeling_jll}())) === width1
            @test treewidth(graph1; alg = SafeRules(SAT{PicoSAT_jll}())) === width1
            @test treewidth(graph1; alg = SafeRules(SAT{CryptoMiniSat_jll}())) === width1
            @test treewidth(graph1; alg = SafeSeparators(BT())) === width1
            @test treewidth(graph1; alg = SafeSeparators(SAT{libpicosat_jll}())) === width1
            @test treewidth(graph1; alg = SafeSeparators(SAT{Lingeling_jll}())) === width1
            @test treewidth(graph1; alg = SafeSeparators(SAT{PicoSAT_jll}())) === width1
            @test treewidth(graph1; alg = SafeSeparators(SAT{CryptoMiniSat_jll}())) === width1

            width1 = 5.0
            lb1 = lowerbound(weights1, graph1)
            ub1 = treewidth(weights1, graph1)

            @test isa(lb1, Float64)
            @test isa(ub1, Float64)
            @test lb1 <= width1 <= ub1
            #@test treewidth(weights1, graph1; alg=ConnectedComponents(BT())) === width1
            #@test treewidth(weights1, graph1; alg=SafeRules(ConnectedComponents(BT()))) === width1
            @test treewidth(weights1, graph1; alg = SafeSeparators(BT())) === width1

            width2 = V(9)
            lb2 = lowerbound(graph2)
            ub2 = treewidth(graph2)

            @test isa(lb2, V)
            @test isa(ub2, V)
            @test lb2 <= width2 <= ub2
            @test treewidth(graph2; alg = BT()) === width2
            @test treewidth(graph2; alg = SAT{libpicosat_jll}()) === width2
            @test treewidth(graph2; alg = SAT{Lingeling_jll}()) === width2
            @test treewidth(graph2; alg = SAT{PicoSAT_jll}()) === width2
            @test treewidth(graph2; alg = SAT{CryptoMiniSat_jll}()) === width2
            @test treewidth(graph2; alg = SafeRules(BT())) === width2
            @test treewidth(graph2; alg = SafeRules(SAT{libpicosat_jll}())) === width2
            @test treewidth(graph2; alg = SafeRules(SAT{Lingeling_jll}())) === width2
            @test treewidth(graph2; alg = SafeRules(SAT{PicoSAT_jll}())) === width2
            @test treewidth(graph2; alg = SafeRules(SAT{CryptoMiniSat_jll}())) === width2
            @test treewidth(graph2; alg = SafeSeparators(BT())) === width2
            @test treewidth(graph2; alg = SafeSeparators(SAT{libpicosat_jll}())) === width2
            @test treewidth(graph2; alg = SafeSeparators(SAT{Lingeling_jll}())) === width2
            @test treewidth(graph2; alg = SafeSeparators(SAT{PicoSAT_jll}())) === width2
            @test treewidth(graph2; alg = SafeSeparators(SAT{CryptoMiniSat_jll}())) === width2

            width2 = 19.169925001442312
            lb2 = lowerbound(weights2, graph2)
            ub2 = treewidth(weights2, graph2)

            @test isa(lb2, Float64)
            @test isa(ub2, Float64)
            @test lb2 <= width2 <= ub2
            @test treewidth(weights2, graph2; alg = BT()) === width2
            @test treewidth(weights2, graph2; alg = SafeRules(BT())) === width2
            @test treewidth(weights2, graph2; alg = SafeSeparators(BT())) === width2
        end
    end
end
