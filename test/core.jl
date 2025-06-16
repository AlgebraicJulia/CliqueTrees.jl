using AbstractTrees
using Base: @kwdef, oneto
using Base.Order
using CliqueTrees
using CliqueTrees: DoublyLinkedList, EliminationAlgorithm, sympermute, cliquetree!
using CliqueTrees.Utilities
using Graphs
using Graphs: SimpleEdge
using JET
using LinearAlgebra
using MatrixMarket
using SparseArrays
using SuiteSparseMatrixCollection
using Test

const SSMC = ssmc_db()

function readmatrix(name::String)
    path = joinpath(fetch_ssmc(SSMC[SSMC.name .== name, :]; format = "MM")[1], "$(name).mtx")
    return mmread(path)
end

@testset "errors" begin
    weights = [1]
    @test_throws Exception lowerbound(weights, 1)
    @test_throws Exception lowerbound(weights, MMW())
    @test_throws Exception permutation(weights, [1])
    @test_throws Exception permutation(weights, ([1], [1]))
    @test_throws Exception permutation(weights, BFS())
    @test_throws Exception permutation(weights, MCS())
    @test_throws Exception permutation(weights, LexBFS())
    @test_throws Exception permutation(weights, RCMMD())
    @test_throws Exception permutation(weights, RCMGL())
    @test_throws Exception permutation(weights, LexM())
    @test_throws Exception permutation(weights, MCSM())
    @test_throws Exception permutation(weights, AMF())
    @test_throws Exception permutation(weights, MF())
    @test_throws Exception permutation(weights, MMD())
    @test_throws Exception permutation(weights, AMD())
    @test_throws Exception permutation(weights, SymAMD())
    @test_throws Exception permutation(weights, METIS())
    @test_throws Exception permutation(weights, ND())
    @test_throws Exception permutation(weights, Spectral())
    @test_throws Exception permutation(weights, BT())
    @test_throws Exception permutation(weights, SAT{CryptoMiniSat_jll}())
    @test_throws Exception permutation(weights, MinimalChordal())
    @test_throws Exception permutation(weights, CompositeRotations([1]))
    @test_throws Exception permutation(weights, SafeRules())
    @test_throws Exception permutation(weights, SafeSeparators())
    @test_throws Exception permutation(weights, ConnectedComponents())
    @test_throws Exception permutation(weights, BestWidth())
    @test_throws Exception permutation(weights, BestFill())
    @test_throws Exception eliminationtree(weights, [1])
    @test_throws Exception treewidth(weights, [1])
    @test_throws Exception treefill(weights, LexM())
end

import Catlab
import libpicosat_jll
import CryptoMiniSat_jll

@static if Sys.iswindows()
    import libpicosat_jll as PicoSAT_jll
    import libpicosat_jll as Lingeling_jll
    const METIS_OR_KAHYPAR = METISND
else
    import PicoSAT_jll
    import Lingeling_jll
    const METIS_OR_KAHYPAR = KaHyParND
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

@kwdef struct SafeFlowCutter <: EliminationAlgorithm
    time::Int = 5
    seed::Int = 0
end

function CliqueTrees.permutation(graph, alg::SafeFlowCutter)
    time = alg.time
    seed = alg.seed

    try
        return permutation(graph, FlowCutter(; time, seed))
    catch
        @warn "FlowCutter failed"
        return permutation(graph)
    end
end

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
    @test_throws ArgumentError permutation(matrix; alg = ND(MMD(), METISND()))
    @test_throws ArgumentError permutation(matrix; alg = ND(MMD(), KaHyParND()))
    @test_throws ArgumentError permutation(matrix; alg = Spectral())
    @test_throws ArgumentError permutation(matrix; alg = FlowCutter())
    @test_throws ArgumentError permutation(matrix; alg = BT())
end

import AMD as AMDLib
import FlowCutterPACE17_jll
import KaHyPar
import Laplacians
import Metis
import TreeWidthSolver

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

        label, tree = supernodetree(graph; alg = 1:8)
        @test isequal(Tree(tree), tree.tree)

        label, tree = cliquetree(graph; alg = 1:8)
        @test isequal(Tree(tree), tree.tree.tree)
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
                BipartiteGraph{Int32, Int64}(graph),
            )
        )

        @test allequal(
            (
                graph,
                BipartiteGraph(Matrix(graph)),
                BipartiteGraph{Int8, Int16}(Matrix{Float64}(graph)),
            )
        )

        @test allequal(
            (
                graph,
                BipartiteGraph(SparseMatrixCSC(graph)),
                BipartiteGraph{Int8, Int16}(SparseMatrixCSC{Float64, Int32}(graph)),
            )
        )

        @test allequal(
            (
                graph,
                BipartiteGraph(sparse(graph)),
                BipartiteGraph{Int8, Int16}(sparse(Float64, Int32, graph)),
            )
        )

        @test allequal(
            (
                graph,
                BipartiteGraph(Graph(graph)),
                BipartiteGraph{Int8, Int16}(Graph{Int32}(graph)),
            )
        )

        @test allequal(
            (
                graph,
                BipartiteGraph(DiGraph(graph)),
                BipartiteGraph{Int8, Int16}(DiGraph{Int32}(graph)),
            )
        )

        @test allequal(
            (
                graph,
                BipartiteGraph(Catlab.Graph(graph)),
                BipartiteGraph{Int8, Int16}(Catlab.Graph(graph)),
            )
        )

        @test allequal(
            (
                graph,
                BipartiteGraph(Catlab.SymmetricGraph(graph)),
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

        @test outdegree(graph, 1) === 2
        @test indegree(graph, 1) === 2
        @test degree(graph, 1) === 4
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
        @test ne(filledgraph) === Int16(13)
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
        @test outdegree(filledgraph, 5) === 2
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

@testset "clique tree" begin
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

    label1, tree1 = cliquetree(matrix)
    label2 = copy(label1); tree2 = copy(tree1)
    permute!(label2, cliquetree!(tree2, 1))
    isomorphism = Int[]

    for bag1 in tree1
        bag1 = label1[bag1]

        for (i, bag2) in enumerate(tree2)
            bag2 = label2[bag2]

            if issetequal(bag1, bag2)
                push!(isomorphism, i)
                break
            end
        end
    end

    graph1 = Graph(length(tree1))
    graph2 = Graph(length(tree2))

    for j in Tree(tree1), i in childindices(tree1, j)
        add_edge!(graph1, isomorphism[i], isomorphism[j])
    end

    for j in Tree(tree2), i in childindices(tree2, j)
        add_edge!(graph2, i, j)
    end

    @test tree1 == tree1
    @test tree2 == tree2
    @test tree1 != tree2
    @test graph1 == graph2
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
            ND{1}(MMD(), METISND(); width = 5),
            ND{2}(MMD(), KaHyParND(); width = 5),
            ND{1}(MCS(), METISND(); width = 5),
            ND{2}(MCS(), KaHyParND(); width = 5),
            Spectral(),
            FlowCutter(; time = 1),
            BT(),
            SAT{CryptoMiniSat_jll}(),
            MinimalChordal(),
            CompositeRotations([1, 2, 3]),
            SafeRules(),
            SimplicialRule(),
            SafeSeparators(),
            ConnectedComponents(),
            BestWidth(MCS(), MF()),
            BestFill(MCS(), MF()),
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
    weights = Float64[]; graph = spzeros(0, 0)
    @test ischordal(graph)

    for alg in (
            MMW{1}(),
            MMW{2}(),
            MMW{3}(),
        )
        @test iszero(lowerbound(graph))
        @test iszero(lowerbound(weights, graph))
    end

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
            ND{1}(MMD(), METISND(); width = 5),
            ND{2}(MMD(), METIS_OR_KAHYPAR(); width = 5),
            ND{1}(MCS(), METISND(); width = 5),
            ND{2}(MCS(), METIS_OR_KAHYPAR(); width = 5),
            # Spectral(),
            SafeFlowCutter(; time = 1),
            BT(),
            MinimalChordal(),
            CompositeRotations([]),
            SafeRules(),
            SimplicialRule(),
            SafeSeparators(),
            ConnectedComponents(),
            BestWidth(MCS(), MF()),
        )

        @test permutation(graph; alg) == ([], [])
        @test permutation(weights, graph; alg) == ([], [])
        @test iszero(treewidth(graph; alg))
        @test iszero(treewidth(weights, graph; alg))
        @test iszero(treefill(graph; alg))
        @test iszero(treefill(weights, graph; alg))
    end

    for S in (Nodal, Maximal, Fundamental)
        label, tree = cliquetree(graph; snd = S())
        filledgraph = FilledGraph(tree)
        @test iszero(length(tree))
        @test isnothing(rootindex(tree))
        @test iszero(treewidth(tree))
        @test iszero(treefill(tree))
        @test iszero(nv(filledgraph))
        @test iszero(ne(filledgraph))
    end
end

@testset "singleton graph" begin
    weights = Float64[2]; graph = spzeros(1, 1)
    @test ischordal(graph)

    for alg in (
            MMW{1}(),
            MMW{2}(),
            MMW{3}(),
        )
        @test iszero(lowerbound(graph))
        @test istwo(lowerbound(weights, graph))
    end

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
            ND{1}(MMD(), METISND(); width = 5),
            ND{2}(MMD(), METIS_OR_KAHYPAR(); width = 5),
            ND{1}(MCS(), METISND(); width = 5),
            ND{2}(MCS(), METIS_OR_KAHYPAR(); width = 5),
            # Spectral,
            SafeFlowCutter(; time = 1),
            BT(),
            CompositeRotations([1]),
            SafeRules(),
            SimplicialRule(),
            SafeSeparators(),
            ConnectedComponents(),
            BestWidth(MCS(), MF()),
            BestFill(MCS(), MF()),
        )
        @test permutation(graph; alg) == ([1], [1])
        @test permutation(weights, graph; alg) == ([1], [1])
        @test iszero(treewidth(graph; alg))
        @test istwo(treewidth(weights, graph; alg))
        @test iszero(treefill(graph; alg))
        @test isfour(treefill(weights, graph; alg))
    end

    for S in (Nodal, Maximal, Fundamental)
        label, tree = cliquetree(graph; snd = S())
        filledgraph = FilledGraph(tree)
        @test isone(length(tree))
        @test isone(rootindex(tree))
        @test iszero(treewidth(tree))
        @test iszero(treefill(tree))
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
    weights = Float64[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]

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
                @inferred CliqueTrees.pr3(weights, graph, lowerbound(weights, graph))
                @inferred CliqueTrees.pr4(graph, lowerbound(graph))
                @inferred CliqueTrees.pr4(weights, graph, lowerbound(weights, graph))
                @inferred CliqueTrees.connectedcomponents(graph)
                @inferred CliqueTrees.twins(graph, Val(true))
                @inferred CliqueTrees.simplicialrule(graph)
                @inferred CliqueTrees.qcc(V, E, graph, 1.0, Forward)
                @inferred treewidth(graph; alg = 1:17)
                @inferred treefill(graph; alg = 1:17)
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
                @test_call target_modules = (CliqueTrees,) CliqueTrees.pr3(weights, graph, lowerbound(weights, graph))
                @test_call target_modules = (CliqueTrees,) CliqueTrees.pr4(graph, lowerbound(graph))
                @test_call target_modules = (CliqueTrees,) CliqueTrees.pr4(weights, graph, lowerbound(weights, graph))
                @test_call target_modules = (CliqueTrees,) CliqueTrees.connectedcomponents(graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.twins(graph, Val(true))
                @test_call target_modules = (CliqueTrees,) CliqueTrees.simplicialrule(graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.qcc(V, E, graph, 1.0, Forward)
                @test_call target_modules = (CliqueTrees,) treewidth(graph; alg = 1:17)
                @test_call target_modules = (CliqueTrees,) treefill(graph; alg = 1:17)
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
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.pr4(graph, lowerbound(graph))
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.pr4(ones(17), graph, lowerbound(ones(17), graph))
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.connectedcomponents(graph)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.twins(graph, Val(true))
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.simplicialrule(graph)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.qcc(V, E, graph, 1.0, Forward)
                @test_opt target_modules = (CliqueTrees,) treewidth(graph; alg = 1:17)
                @test_opt target_modules = (CliqueTrees,) treefill(graph; alg = 1:17)
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

            uwidth = treewidth(graph; alg = BT())
            wwidth = treewidth(weights, graph; alg = BT())

            @testset "permutations" begin
                for alg in (
                        MMW{1}(),
                        MMW{2}(),
                        MMW{3}(),
                    )
                    @test lowerbound(graph; alg) <= uwidth
                    @test lowerbound(weights, graph; alg) <= wwidth
                end

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
                        ND{1}(MMD(), METISND(); width = 5),
                        SimplicialRule(ND{2}(MMD(), METIS_OR_KAHYPAR(); width = 5)),
                        ND{1}(MCS(), METISND(); width = 5),
                        SimplicialRule(ND{2}(MCS(), METIS_OR_KAHYPAR(); width = 5)),
                        Spectral(),
                        SafeFlowCutter(; time = 1),
                        BT(),
                        MinimalChordal(),
                        CompositeRotations([1, 3]),
                        SafeRules(),
                        SimplicialRule(),
                        SafeSeparators(),
                        ConnectedComponents(),
                        BestWidth(MCS(), MF()),
                        BestFill(MCS(), MF()),
                    )
                    order, index = permutation(graph; alg)
                    @test isa(order, Vector{V})
                    @test isa(index, Vector{V})
                    @test length(order) == 17
                    @test order[index] == 1:17

                    order, index = permutation(weights, graph; alg)
                    @test isa(order, Vector{V})
                    @test isa(index, Vector{V})
                    @test length(order) == 17
                    @test order[index] == 1:17

                    @test treewidth(graph; alg) >= uwidth
                    @test treewidth(weights, graph; alg) >= wwidth

                    @test isa(treefill(graph; alg), E)
                    @test isa(treefill(weights, graph; alg), Float64)
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
                    @test treefill(tree) === E(42)
                    @test nv(filledgraph) === V(17)
                    @test ne(filledgraph) === E(42)
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
                    @test treefill(tree) === E(42)
                    @test nv(filledgraph) === V(17)
                    @test ne(filledgraph) === E(42)
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
                    @test treefill(tree) === E(42)
                    @test nv(filledgraph) === V(17)
                    @test ne(filledgraph) === E(42)
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

@testset "fill" begin
    algs = (
        MMD(),
        MMD(; delta = 5),
        AMF(),
        SimplicialRule(ND{2}(MMD(), METISND())),
        SimplicialRule(ND{2}(MMD(), METIS_OR_KAHYPAR())),
        MF(),
    )

    matrices = (
        ("bcspwr09", 5200, [1144, 1135, 1402]),
        ("bcspwr10", 25300, [4986, 4354, 2870]),
        ("bcsstk08", 31100, [1028, 1029, 1014]),
    )

    for (name, fill, clique) in matrices
        matrix = readmatrix(name)

        for alg in algs
            order1, index1 = permutation(matrix; alg = MinimalChordal(alg))
            order2, index2 = permutation(matrix; alg = CompositeRotations(clique, alg))
            order3, index3 = permutation(matrix; alg)

            label1, tree1 = cliquetree(matrix; alg = order1)
            label2, tree2 = cliquetree(matrix; alg = order2)
            label3, tree3 = cliquetree(matrix; alg = order3)

            fill1 = treefill(matrix; alg = order1)
            fill2 = treefill(matrix; alg = order2)
            fill3 = treefill(matrix; alg = order3)

            @test order2[(end - 2):end] == clique
            @test fill1 == treefill(tree1)
            @test fill2 == treefill(tree2)
            @test fill3 == treefill(tree3)
            @test fill1 <= fill2 <= fill3 <= fill
        end
    end
end

@testset "exact treewidth" begin
    lalgs = (
        MMW{1}(),
        MMW{2}(),
        MMW{3}(),
    )

    ualgs = (
        SAT{libpicosat_jll}(),
        SAT{Lingeling_jll}(),
        SAT{PicoSAT_jll}(),
        SAT{CryptoMiniSat_jll}(),
        SimplicialRule(SAT{CryptoMiniSat_jll}()),
        SafeRules(SAT{CryptoMiniSat_jll}()),
        SafeSeparators(SAT{CryptoMiniSat_jll}()),
        SafeRules(SafeSeparators(BT())),
        SafeSeparators(BT()),
    )

    walgs = (
        SafeRules(SafeSeparators(BT())),
        SafeSeparators(BT()),
    )

    weights1 = log2.([2, 2, 4, 2, 4, 2, 2, 2, 2, 2, 4, 2, 4, 2, 4, 2])
    weights2 = log2.([4, 3, 4, 3, 4, 4, 3, 4, 4, 3, 4, 3, 4, 4, 3, 4, 4, 3, 4, 3, 4, 4, 3, 4, 4, 3, 4, 3, 4, 4, 3, 4])

    graph1 = BipartiteGraph(
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

    graph2 = BipartiteGraph(
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

    graphs = (
        (weights1, graph1, 4, 0, 0, 5.0, 0, 0),
        (weights2, graph2, 9, 21, 94, 19.169925001442312, 21, 94),
    )

    for (G, V, E) in TYPES
        @testset "$(nameof(G))" begin
            for (weights, graph, uwidth, un, um, wwidth, wn, wm) in graphs
                graph = G(graph); uwidth = V(uwidth); un = V(un); wn = V(wn)

                ukernel, urest... = CliqueTrees.pr4(graph, lowerbound(graph))
                wkernel, wrest... = CliqueTrees.pr4(weights, graph, lowerbound(weights, graph))
                @test nv(ukernel) === un
                @test nv(wkernel) === wn
                @test ne(ukernel) === um
                @test ne(wkernel) === wm

                for alg in lalgs
                    u = lowerbound(graph; alg)
                    w = lowerbound(weights, graph; alg)
                    @test isa(u, V)
                    @test isa(w, Float64)
                    @test u <= uwidth
                    @test w <= wwidth
                end

                for alg in ualgs
                    order, tree = cliquetree(graph; alg)
                    @test treewidth(graph; alg = order) === uwidth
                    @test treewidth(tree) === uwidth
                    @test treefill(graph; alg = order) === treefill(tree)
                end

                for alg in walgs
                    order, tree = cliquetree(weights, graph; alg)
                    @test treewidth(weights, graph; alg = order) === wwidth
                    @test treewidth(view(weights, order), tree) === wwidth
                    @test treefill(weights, graph; alg = order) ≈ treefill(view(weights, order), tree)
                end
            end
        end
    end
end
