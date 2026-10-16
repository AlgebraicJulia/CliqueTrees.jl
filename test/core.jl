using AbstractTrees
using Base: @kwdef, oneto
using Base.Order
using CliqueTrees
using CliqueTrees: SinglyLinkedList, DoublyLinkedList, EliminationAlgorithm, Parent, cliquetree!, simplegraph, connectedcomponents
using CliqueTrees.Multifrontal
using CliqueTrees.Utilities
using Graphs
using Graphs: SimpleEdge
using JET
using Krylov
using LinearAlgebra
using MatrixMarket
using Random
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
    @test_throws Exception permutation(weights, [1])
    @test_throws Exception eliminationtree(weights, [1])
    @test_throws Exception treewidth(weights, [1])
    @test_throws Exception treefill(weights, LexM())
end

import Catlab

@static if Sys.iswindows()
    const METIS_OR_KAHYPAR = METISND
else
    const METIS_OR_KAHYPAR = KaHyParND
end

const TYPES = (
    (BipartiteGraph{Int16, Int32}, Int16, Int32),
    (Graph{Int16}, Int16, Int),
    (DiGraph{Int16}, Int16, Int),
)

@kwdef struct SafeFlowCutter <: EliminationAlgorithm
    time::Int = 5
    seed::Int = 0
end

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph, alg::SafeFlowCutter)
    time = alg.time
    seed = alg.seed

    try
        return permutation(weights, graph, FlowCutter(; time, seed))
    catch error
        @warn "FlowCutter failed: $error"
        return permutation(weights, graph)
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
        parent = Parent(Int16(5), Int16[2, 5, 4, 5, 0])
        tree = Tree(parent)
        @test rootindex(tree) === Int16(5)
        @test childindices(tree, 1) == Int16[]
        @test childindices(tree, 5) == Int16[2, 4]

        setrootindex!(parent, 1)
        tree = Tree(parent)
        @test rootindex(tree) === Int16(1)
        @test childindices(tree, 1) == Int16[2]
        @test childindices(tree, 5) == Int16[4]

        node = IndexNode(tree)
        @test ParentLinks(node) === StoredParents()
        @test NodeType(node) === HasNodeType()
        @test nodetype(node) === typeof(node)
    end

    @testset "construction" begin
        graph = BipartiteGraph{Int16, Int32}(
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
    graph = BipartiteGraph{Int16, Int32}(
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
        @test convert(BipartiteGraph{Int16, Int32, Vector{Int32}, Vector{Int16}}, graph) ===
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
                BipartiteGraph{Int16, Int32}(Matrix{Float64}(graph)),
            )
        )

        @test allequal(
            (
                graph,
                BipartiteGraph(SparseMatrixCSC(graph)),
                BipartiteGraph{Int16, Int32}(SparseMatrixCSC{Float64, Int64}(graph)),
            )
        )

        @test allequal(
            (
                graph,
                BipartiteGraph(sparse(graph)),
                BipartiteGraph{Int16, Int32}(sparse(Float64, Int64, graph)),
            )
        )

        @test allequal(
            (
                graph,
                BipartiteGraph(Graph(graph)),
                BipartiteGraph{Int16, Int32}(Graph{Int64}(graph)),
            )
        )

        @test allequal(
            (
                graph,
                BipartiteGraph(DiGraph(graph)),
                BipartiteGraph{Int16, Int32}(DiGraph{Int64}(graph)),
            )
        )

        @test allequal(
            (
                graph,
                BipartiteGraph(Catlab.Graph(graph)),
                BipartiteGraph{Int16, Int32}(Catlab.Graph(graph)),
            )
        )

        @test allequal(
            (
                graph,
                BipartiteGraph(Catlab.SymmetricGraph(graph)),
                BipartiteGraph{Int16, Int32}(Catlab.SymmetricGraph(graph)),
            )
        )

        @test DiGraph(filledgraph) == DiGraph(BipartiteGraph(filledgraph))
        @test Graph(filledgraph) == Graph(BipartiteGraph(filledgraph))
    end

    @testset "interface" begin
        nullgraph = zero(BipartiteGraph{Int16, Int32, Vector{Int32}, Vector{Int16}})
        @test nv(nullgraph) === zero(Int16)
        @test ne(nullgraph) === zero(Int32)

        @test is_directed(graph)
        @test nv(graph) === Int16(8)
        @test ne(graph) === Int32(26)
        @test eltype(graph) === Int16
        @test edgetype(graph) === SimpleEdge{Int16}

        @test has_edge(graph, 1, 2)
        @test !has_edge(graph, 1, 4)
        @test SimpleEdge(1, 2) ∈ edges(graph)
        @test SimpleEdge(1, 4) ∉ edges(graph)

        @test vertices(graph) === oneto(Int16(8))
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
    graph = BipartiteGraph{Int16, Int32}(
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
                BipartiteGraph{Int16, Int32}(filledgraph),
                BipartiteGraph(SparseMatrixCSC(filledgraph)),
                BipartiteGraph(SparseMatrixCSC{Float64}(filledgraph)),
                BipartiteGraph(SparseMatrixCSC{Float64, Int64}(filledgraph)),
                BipartiteGraph(sparse(filledgraph)),
                BipartiteGraph(sparse(Float64, filledgraph)),
                BipartiteGraph(sparse(Float64, Int32, filledgraph)),
                BipartiteGraph(Matrix(filledgraph)),
                BipartiteGraph(Matrix{Float64}(filledgraph)),
            )
        )
    end

    @testset "interface" begin
        @test is_directed(filledgraph)
        @test nv(filledgraph) === Int16(8)
        @test ne(filledgraph) === Int32(13)
        @test eltype(filledgraph) === Int16
        @test edgetype(filledgraph) === SimpleEdge{Int16}

        @test has_edge(filledgraph, 1, 7)
        @test !has_edge(filledgraph, 7, 1)
        @test SimpleEdge(1, 7) ∈ edges(filledgraph)
        @test SimpleEdge(7, 1) ∉ edges(filledgraph)
        @test Base.hasfastin(edges(filledgraph))

        @test vertices(filledgraph) === oneto(Int16(8))
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

@testset "twins" begin
    function relation(nitem, cols)
        ptr = cumsum([1; length.(cols)]); tgt = reduce(vcat, cols; init = Int[])
        return BipartiteGraph{Int, Int}(nitem, length(cols), length(tgt), ptr, tgt)
    end

    # rows[i] is the set of columns that contain item i
    function check(svar, partition, rows)
        n = length(rows)
        ok = CliqueTrees.nov(partition) == n && nv(partition) == length(unique(rows)) && ne(partition) == n
        ok &= all((svar[i] == svar[k]) == (rows[i] == rows[k]) for i in 1:n, k in 1:n)
        ok &= all(svar[i] == s for s in vertices(partition) for i in neighbors(partition, s))
        ok &= all(issorted(neighbors(partition, s)) for s in vertices(partition))
        return ok
    end

    rng = Xoshiro(1)

    # relations (:N), with repeated columns so that twins are likely
    for (nitem, ncol, q) in ((0, 3, 0.5), (1, 1, 0.5), (5, 30, 0.2), (30, 5, 0.3), (40, 40, 0.1), (200, 12, 0.4), (12, 200, 0.05))
        cols = [sort!(randsubseq(rng, 1:nitem, q)) for _ in 1:ncol]
        cols = [cols; cols[1:ncol ÷ 3]]
        rows = [[j for j in eachindex(cols) if i in cols[j]] for i in 1:nitem]
        @test check(CliqueTrees.twins(relation(nitem, cols), Val(:N))..., rows)
    end

    # graphs (:F and :T): blow up each vertex of a random graph into a few
    # copies that form a clique (true twins) or an independent set (false
    # twins), and add some self loops
    for (k, p) in ((1, 0.0), (8, 0.3), (20, 0.15), (30, 0.5))
        H = erdos_renyi(k, p; seed = k)
        copies = [rand(rng, 1:3) for _ in 1:k]; clique = rand(rng, Bool, k)
        owner = reduce(vcat, [fill(v, copies[v]) for v in 1:k])
        n = length(owner)

        lists = map(1:n) do u
            [w for w in 1:n if (w != u && (has_edge(H, owner[u], owner[w]) || (owner[u] == owner[w] && clique[owner[u]]))) || (w == u && rand(rng) < 0.3)]
        end

        graph = relation(n, lists)
        @test check(CliqueTrees.twins(graph, Val(:F))..., [setdiff(lists[u], u) for u in 1:n])
        @test check(CliqueTrees.twins(graph, Val(:T))..., [sort!(union(lists[u], u)) for u in 1:n])
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
            AMF(; exact = true),
            MF(),
            MF(1),
            MMD(),
            MD(),
            METIS(),
            ND{1}(MMD(), METISND(); width = 5),
            ND{2}(MMD(), KaHyParND(); width = 5),
            ND{1}(MCS(), METISND(); width = 5),
            ND{2}(MCS(), KaHyParND(); width = 5),
            Spectral(),
            FlowCutter(; time = 1),
            BT(),
            PIDBT(),
            HBT(; time = 0.1),
            MinimalChordal(),
            CompositeRotations([1, 2, 3]),
            Compression(),
            SafeRules(),
            SimplicialRule(),
            SafeSeparators(),
            SafeSeparators(; maxmissing = 100),
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
            LBN{1}(),
            LBN{2}(),
            LBN{3}(),
            LBP{1}(),
            LBP{2}(),
            LBP{3}(),
        )
        @test isone(-lowerbound(graph; alg))
        @test iszero(lowerbound(weights, graph; alg))
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
            AMF(; exact = true),
            MF(),
            MF(1),
            MMD(),
            # METIS(),
            NDS{1}(MMD(), METISND(); width = 5),
            NDS{2}(MMD(), METIS_OR_KAHYPAR(); width = 5),
            NDS{1}(MCS(), METISND(); width = 5),
            NDS{2}(MCS(), METIS_OR_KAHYPAR(); width = 5),
            # Spectral(),
            SafeFlowCutter(; time = 1),
            BT(),
            PIDBT(),
            HBT(; time = 0.1),
            MinimalChordal(),
            CompositeRotations([]),
            Compression(; tau=1.0),
            Compression(; tau=0.9),
            SafeRules(),
            SafeRules(),
            SimplicialRule(),
            SimplicialRule(),
            SafeSeparators(),
            SafeSeparators(; maxmissing = 100),
            ConnectedComponents(),
            BestWidth(MCS(), MF()),
        )

        @test permutation(graph; alg) == ([], [])
        @test permutation(weights, graph; alg) == ([], [])
        @test isone(-treewidth(graph; alg))
        @test iszero(treewidth(weights, graph; alg))
        @test iszero(treefill(graph; alg))
        @test iszero(treefill(weights, graph; alg))
    end

    for S in (Nodal, Maximal, Fundamental)
        label, tree = cliquetree(graph; snd = S())
        filledgraph = FilledGraph(tree)
        @test iszero(length(tree))
        @test isnothing(rootindex(tree))
        @test isone(-treewidth(tree))
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
            LBN{1}(),
            LBN{2}(),
            LBN{3}(),
            LBP{1}(),
            LBP{2}(),
            LBP{3}(),
        )
        @test iszero(lowerbound(graph; alg))
        @test istwo(lowerbound(weights, graph; alg))
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
            AMF(; exact = true),
            MF(),
            MF(1),
            MMD(),
            METIS(),
            NDS{1}(MMD(), METISND(); width = 5),
            NDS{2}(MMD(), METIS_OR_KAHYPAR(); width = 5),
            NDS{1}(MCS(), METISND(); width = 5),
            NDS{2}(MCS(), METIS_OR_KAHYPAR(); width = 5),
            # Spectral,
            SafeFlowCutter(; time = 1),
            BT(),
            PIDBT(),
            HBT(; time = 0.1),
            CompositeRotations([1]),
            Compression(; tau=1.0),
            Compression(; tau=0.9),
            SafeRules(),
            SafeRules(),
            SimplicialRule(),
            SimplicialRule(),
            SafeSeparators(),
            SafeSeparators(; maxmissing = 100),
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
        @test isone(only(residual(tree, 1)))
        @test isone(only(tree[1]))
    end
end

@testset "fill-in" begin
    # the fill-in of a vertex is the weight of the missing edges in its neighborhood
    function bruteforce(graph, weights)
        return map(vertices(graph)) do v
            fill = zero(eltype(weights)); N = neighbors(graph, v)

            for i in eachindex(N), j in (i + 1):lastindex(N)
                if !has_edge(graph, N[i], N[j])
                    fill += weights[N[i]] * weights[N[j]]
                end
            end

            fill
        end
    end

    rng = MersenneTwister(3)

    for n in (1, 2, 10, 30, 60), p in (0.05, 0.2, 0.5, 0.9)
        graph = erdos_renyi(n, p; rng)
        simple = simplegraph(graph)
        ptr = pointers(simple); tgt = targets(simple); m = length(tgt)
        V = eltype(tgt); E = eltype(ptr)
        number = V[ptr[v + 1] - ptr[v] for v in 1:n]
        begptr = view(ptr, 1:n); endptr = view(ptr, 2:(n + 1))
        marker = Vector{V}(undef, n); order = Vector{V}(undef, n)
        tail = Vector{E}(undef, n); adjtgt = Vector{V}(undef, m)

        fill = Vector{Int}(undef, n)
        fillin!(fill, marker, order, tail, adjtgt, number, tgt, begptr, endptr, V(n))
        @test fill == bruteforce(graph, ones(Int, n))

        weights = rand(rng, 1:5, n); wdegree = Vector{Int}(undef, n)
        fillin!(fill, weights, wdegree, marker, order, tail, adjtgt, number, tgt, begptr, endptr, V(n))
        @test fill == bruteforce(graph, weights)

        weights = rand(rng, n) .+ 0.5; ffill = Vector{Float64}(undef, n); fdegree = Vector{Float64}(undef, n)
        fillin!(ffill, weights, fdegree, marker, order, tail, adjtgt, number, tgt, begptr, endptr, V(n))
        @test ffill ≈ bruteforce(graph, weights)
    end
end

@testset "chordality" begin
    @test  ischordal(smallgraph(:bull))
    @test  ischordal(smallgraph(:diamond))
    @test  ischordal(smallgraph(:housex))
    @test  ischordal(barbell_graph(10, 10))
    @test  ischordal(binary_tree(10))
    @test !ischordal(smallgraph(:chvatal))
    @test !ischordal(smallgraph(:cubical))
    @test !ischordal(smallgraph(:desargues))
    @test !ischordal(smallgraph(:dodecahedral) )
    @test !ischordal(smallgraph(:frucht) )
end

# Chordal Graphs and Semidefinite Optimization
# Vandenberghe and Andersen
@testset "vandenberghe and andersen" begin
    weights = Float64[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]

    _graph = BipartiteGraph(
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
    _completion = BipartiteGraph(
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
            graph = G(_graph)
            completion = G(_completion)

            @testset "inference" begin
                @inferred ischordal(graph)
                @inferred CliqueTrees.bfs(graph)
                @inferred CliqueTrees.mcs(graph)
                @inferred CliqueTrees.mcs(graph, V[1, 3])
                @inferred CliqueTrees.lexbfs(graph)
                @inferred CliqueTrees.rcmmd(graph, QuickSort)
                @inferred CliqueTrees.rcmgl(graph, QuickSort)
                @inferred CliqueTrees.lexm(graph)
                @inferred CliqueTrees.mcsm(graph)
                @inferred CliqueTrees.mcsm(graph, V[1, 3])
                @inferred CliqueTrees.amf(weights, graph)
                @inferred CliqueTrees.amf(weights, graph; exact = true)
                @inferred CliqueTrees.mlf(weights, graph)
                @inferred CliqueTrees.mlf(CliqueTrees.MinAvgFill, weights, graph)
                @inferred CliqueTrees.mmd(weights, graph)
                @inferred CliqueTrees.mcs_etree(weights, graph, 1:17)
                @inferred CliqueTrees.pr3(weights, graph, lowerbound(weights, graph))
                @inferred CliqueTrees.pr4(weights, graph, lowerbound(weights, graph))
                @inferred CliqueTrees.connectedcomponents(graph)
                @inferred CliqueTrees.twins(graph, Val(:T), 1.0)
                @inferred CliqueTrees.twins(graph, Val(:T), 0.9)
                @inferred CliqueTrees.sr(weights, graph, zero(V))
                @inferred CliqueTrees.asr(weights, graph, zero(V))
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
                @test_call target_modules = (CliqueTrees,) CliqueTrees.mcs(graph, V[1, 3])
                @test_call target_modules = (CliqueTrees,) CliqueTrees.lexbfs(graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.rcmmd(graph, QuickSort)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.rcmgl(graph, QuickSort)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.lexm(graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.mcsm(graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.mcsm(graph, V[1, 3])
                @test_call target_modules = (CliqueTrees,) CliqueTrees.amf(weights, graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.amf(weights, graph; exact = true)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.mlf(weights, graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.mlf(CliqueTrees.MinAvgFill, weights, graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.mmd(weights, graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.mcs_etree(weights, graph, 1:17)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.pr3(weights, graph, lowerbound(weights, graph))
                @test_call target_modules = (CliqueTrees,) CliqueTrees.pr4(weights, graph, lowerbound(weights, graph))
                @test_call target_modules = (CliqueTrees,) CliqueTrees.connectedcomponents(graph)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.twins(graph, Val(:T), 1.0)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.twins(graph, Val(:T), 0.9)
                @test_call target_modules = (CliqueTrees,) CliqueTrees.sr(weights, graph, zero(V))
                @test_call target_modules = (CliqueTrees,) CliqueTrees.asr(weights, graph, zero(V))
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
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.mcs(graph, V[1, 3])
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.lexbfs(graph)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.rcmmd(graph, QuickSort)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.rcmgl(graph, QuickSort)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.lexm(graph)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.mcsm(graph)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.mcsm(graph, V[1, 3])
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.amf(weights, graph)
                #@test_opt target_modules = (CliqueTrees,) CliqueTrees.mlf(weights, graph)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.mmd(weights, graph)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.mcs_etree(weights, graph, 1:17)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.pr3(weights, graph, lowerbound(weights, graph))
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.pr4(weights, graph, lowerbound(weights, graph))
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.connectedcomponents(graph)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.twins(graph, Val(:T), 1.0)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.twins(graph, Val(:T), 0.9)
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.sr(weights, graph, zero(V))
                @test_opt target_modules = (CliqueTrees,) CliqueTrees.asr(weights, graph, zero(V))
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

                @test all(edges(completion)) do edge
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
                        LBN{1}(),
                        LBN{2}(),
                        LBN{3}(),
                        LBP{1}(),
                        LBP{2}(),
                        LBP{3}(),
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
                        AMF(; exact = true),
                        MF(),
                        MF(1),
                        MMD(),
                        MD(),
                        MD(; mass = true),
                        METIS(),
                        NDS{1}(MMD(), METISND(); width = 5),
                        SimplicialRule(NDS{2}(MMD(), METIS_OR_KAHYPAR(); width = 5)),
                        NDS{1}(MCS(), METISND(); width = 5),
                        SimplicialRule(NDS{2}(MCS(), METIS_OR_KAHYPAR(); width = 5)),
                        Spectral(),
                        SafeFlowCutter(; time = 1),
                        BT(),
                        PIDBT(),
                        MinimalChordal(),
                        CompositeRotations([1, 3]),
                        Compression(; tau=1.0),
                        Compression(; tau=0.9),
                        SafeRules(),
                        SafeRules(),
                        SimplicialRule(),
                        SimplicialRule(),
                        SafeSeparators(),
                        SafeSeparators(; maxmissing = 100),
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
                        sparse(completion)[label, label]

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
                        sparse(completion)[label, label]

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
                        sparse(completion)[label, label]

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
                end
            end
        end
    end
end

@testset "minimal triangulations" begin
    # Fast Computation of Minimal Fill inside a Given Elimination Ordering
    # Heggernes and Peyton
    # Figure 2.1
    matrix = [
        0 0 1 1 0 0
        0 0 1 0 1 0
        1 1 0 0 0 1
        1 0 0 0 0 1
        0 1 0 0 0 1
        0 0 1 1 1 0
    ]

    # A Practical Algoithm for Making Fille Graphs Minimal
    # Blair, Heggernes, and Telle
    # Figure 2
    matrix = [
        0 1 1 0 1 1 0
        1 0 1 0 1 0 0
        1 1 0 1 0 0 0
        0 0 1 0 0 0 1
        1 1 0 0 0 1 0
        1 0 0 0 1 0 1
        0 0 0 1 0 1 0
    ]

    order = axes(matrix, 2)
    @test treefill(matrix; alg=MinimalChordal(order)) == 13

    # Minimal Triangulations of Graphs: a Survey
    # Heggernes
    # Figure 1
    matrix = [
        0 1 0 0 1
        1 0 1 1 0
        0 1 0 0 1
        0 1 0 0 1
        1 0 1 1 0
    ]
 
    order = axes(matrix, 2)
    @test treefill(matrix; alg=MinimalChordal(order)) == 7
   
    # The Minimum Degree Heuristic and the Minimal Triangulation Process
    # Berry, Heggernes, and Simonet
    # Figure 1
    matrix = [
        0 0 0 1 1 0 0
        0 0 0 1 1 0 0
        0 0 0 1 1 0 0
        1 1 1 0 0 1 0
        1 1 1 0 0 1 1
        0 0 0 1 1 0 0
        0 0 0 0 1 0 0
    ]

    order = axes(matrix, 2)
    @test treefill(matrix; alg=MinimalChordal(order)) == 10

    # Minimal Triangulation of a Graph and Optimal Pivoting Order in a Sparse Matrix
    # Ohtsuki, Cheung, and Fujisawa
    # Figure 3
    matrix = [
        0 0 0 0 0 1 1 0 0
        0 0 0 0 0 0 1 1 0
        0 0 0 0 0 0 0 1 1
        0 0 0 0 0 1 1 0 0
        0 0 0 0 0 0 0 1 1
        1 0 0 1 0 0 0 0 0
        1 1 0 1 0 0 0 0 0
        0 1 1 0 1 0 0 0 0
        0 0 1 0 1 0 0 0 0
    ]

    order = axes(matrix, 2)
    @test treefill(matrix; alg=MinimalChordal(order)) == 12

    # the filled graph of `order`
    function filledgraph(graph, order)
        index = invperm(order); F = copy(graph)

        for v in order
            higher = [u for u in neighbors(F, v) if index[u] > index[v]]

            for a in higher, b in higher
                a < b && add_edge!(F, a, b)
            end
        end

        return F
    end

    # `order` is a minimal elimination ordering of `graph`,
    # and its fill is a subset of the fill of `initial`
    function isminimalsandwich(graph, order, initial)
        F = filledgraph(graph, order)
        G = filledgraph(graph, initial)

        for e in Graphs.edges(F)
            has_edge(G, e) || return false

            if !has_edge(graph, e)
                H = copy(F); rem_edge!(H, e)
                ischordal(H) && return false
            end
        end

        return true
    end

    # regression: after a root change, `mcs_etree_postorder!` can place a
    # former ancestor of the new root among the unchanged subtrees, so its
    # stale skeleton adjacency set was used to compute its cardinality
    edgelist = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14),
        (2, 3), (2, 4), (2, 5), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 13), (2, 14), (2, 15), (3, 4), (3, 7),
        (3, 8), (3, 10), (3, 11), (3, 13), (3, 14), (4, 7), (4, 8), (4, 11), (4, 13), (4, 14), (4, 15), (5, 6), (5, 7),
        (5, 8), (5, 9), (5, 10), (5, 11), (5, 14), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14), (7, 15),
        (8, 9), (8, 10), (8, 11), (8, 14), (9, 10), (9, 11), (9, 12), (9, 14), (9, 15), (10, 11), (10, 13), (10, 14),
        (11, 14), (11, 15), (13, 14), (14, 15)]

    graph = SimpleGraph(15)

    for (u, v) in edgelist
        add_edge!(graph, u, v)
    end

    initial = [6, 12, 15, 13, 14, 2, 7, 10, 3, 1, 5, 11, 4, 9, 8]
    order, _ = permutation(graph, MinimalChordal(initial))
    @test isminimalsandwich(graph, order, initial)

    rng = MersenneTwister(1)

    for _ in 1:2000
        n = rand(rng, 2:20); p = rand(rng)
        graph = SimpleGraph(n)

        for u in 1:n, v in (u + 1):n
            rand(rng) < p && add_edge!(graph, u, v)
        end

        initial = randperm(rng, n)
        order, _ = permutation(graph, MinimalChordal(initial))
        @test isminimalsandwich(graph, order, initial)
    end
end

@testset "fill" begin
    algs = (
        MMD(),
        MMD(; delta = 5),
        AMF(),
        Compression(AMF()),
        SimplicialRule(AMF()),
        SafeRules(AMF()),
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
            @test fill1 <= fill3
            @test fill2 <= fill3
            @test fill3 <= fill
        end
    end

    name = "mycielskian14"; fill = 14000000
    matrix = readmatrix(name)
    @test treefill(matrix; alg=AMF()) <= fill
end

@testset "exact minimum degree" begin
    # test whether `order` eliminates, at every step, a vertex of minimum
    # weighted closed degree in the elimination graph
    function isminimumdegree(graph, weights, order)
        n = nv(graph); eliminated = falses(n)
        nbrs = [Set(neighbors(graph, v)) for v in vertices(graph)]
        key(v) = weights[v] + sum((weights[w] for w in nbrs[v]); init = 0)

        for v in order
            k = key(v)

            for w in vertices(graph)
                !eliminated[w] && key(w) < k && return false
            end

            for w in nbrs[v]
                delete!(nbrs[w], v)

                for x in nbrs[v]
                    w != x && push!(nbrs[w], x)
                end
            end

            eliminated[v] = true
        end

        return true
    end

    rng = MersenneTwister(0)

    for n in (10, 30, 60), trial in 1:5
        # sparse graphs, power-law graphs, and sparse graphs with a hub
        hub = erdos_renyi(n, 2 / n; rng)

        for v in 2:n
            add_edge!(hub, 1, v)
        end

        for graph in (erdos_renyi(n, 3 / n; rng), barabasi_albert(n, 2; rng), hub)
            for mass in (false, true), external in (false, true)
                # `debug` recomputes every degree after every elimination
                for weights in (ones(Int, n), rand(rng, 1:3, n))
                    index = CliqueTrees.MDLib.md(weights, graph; mass, external, debug = true)
                    @test sort(index) == 1:n

                    if !mass && !external
                        @test isminimumdegree(graph, weights, invperm(index))
                    end
                end
            end
        end
    end
end

@testset "exact treewidth" begin
    lb_algs = (
        MMW{1}(),
        MMW{2}(),
        MMW{3}(),
        LBN{1}(),
        LBN{2}(),
        LBN{3}(),
        LBP{1}(),
        LBP{2}(),
        LBP{3}(),
    )

    tw_algs = (
        BT(),
        PIDBT(),
    )

    uwidth1 = 4; wwidth1 = 8.0
    uwidth2 = 9; wwidth2 = 19.2
    uwidth3 = 6; wwidth3 = 8.6
    uwidth4 = 8; wwidth4 = 9.0
    uwidth5 = 4; wwidth5 = 6.8
    uwidth6 = 7; wwidth6 = 22.6

    weights1 = log2.([2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 4, 4, 4, 2, 4, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 4, 4, 4, 2, 4, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 4, 4, 4, 2, 4, 2, 4, 2])
    weights2 = log2.([4, 3, 4, 3, 4, 4, 3, 4, 4, 3, 4, 3, 4, 4, 3, 4, 4, 3, 4, 3, 4, 4, 3, 4, 4, 3, 4, 3, 4, 4, 3, 4])
    weights3 = log2.([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 4, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2, 4, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 4, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 4, 4, 3, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2])
    weights4 = log2.([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    weights5 = log2.([2, 2, 2, 3, 3, 3, 3, 2, 3, 2, 2, 3, 4, 4, 4, 4, 3, 2, 2, 3, 2, 2, 3, 2, 3, 2, 3, 3, 2, 3, 3, 4, 4, 3, 4, 3, 3])
    weights6 = log2.([9, 5, 3, 6, 11, 5, 8, 9, 2, 6, 6, 10, 6, 10, 10, 8, 5, 7, 6, 67, 9, 3, 7, 10, 10, 10, 9, 7, 9, 8, 9, 10, 7, 9, 8, 7, 6, 9, 7, 5, 8, 9, 4, 8, 4, 4, 20, 6])

    graph1 = BipartiteGraph(
        [
            0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 1 0 1 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 1 0 1 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0
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

    graph3 = BipartiteGraph(
        [
            1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 1 0 1 0
            1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 0 1 1 0 0 0 0 0 0 1 0 0 1 0 0 0 1 1 1 0 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 1 0 1 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 1 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 1 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 1 1 0 0 1 0 0 1 0 0 0 0 1 1 0 0 0 1 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 1 0 0 1 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 1 0 0 1 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 1 0 1 0 0 1 0 1 1 1 0 0 1 1 0 0 1 1 1 0 1 1 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 0 1 0
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 1 0 0 1 0 0 0 1 1 1 1 0 0 0 1 1 1 0 1 0 1 0 0 0 0 1 0 0 1 1 1 1 0 0 1 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
            0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 0 0 1 0 0 0 0 0 0 1 0 0 1 1 0 1 1 1 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 1 0 0 1 0 0 0 1 0 0 1 1 0 0 1 1 1 0 1 0 1 0 0 0 0 0 1 0 0 0 0 0 1 1 1 1 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 1 0 1 1 1 0 0 1 0 1
            0 0 0 1 0 0 1 0 0 0 1 1 1 1 0 0 0 1 1 1 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 1 0 0 0 0 0 0 1 0 0 1 0 0 0 1 1 1 0 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0 1 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 1 0 0 0 1 0 0 1 0 0 0 0 1 1 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 1 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 1 0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
            0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
            0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
            0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
        ]
    )

    graph4 = BipartiteGraph(
        [
            1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 0 1 1 1 0 1 1 1 0 0 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 1 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 0 1 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
            0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
            0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 1 0 1 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0
            0 0 1 0 0 0 0 0 1 0 0 1 1 1 1 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 1 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 1 1 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 0 1 0 0 0 1 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 1 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 1 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 1 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 0 0 0 1 0 0 0 1 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
            0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 1 1 0 0 1 0 0 1 1 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0
            0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 1 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1 1 1 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 1 1 1 1 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 1 1 1 1 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 1 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 1 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
            0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
            0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 1 1 0 0 0 0 0 0 0 0 1 0 0 1 1 1 1 1 0 0 1 1 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 1 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0
            0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0
            0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0
            0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
            0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
        ]
    )

    graph5 = BipartiteGraph(
        [
            1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 0 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0
            0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 1 1 0 0 0 1 1 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0
            0 0 0 0 0 0 0 0 1 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
            0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 1 1 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
            0 0 0 0 0 0 0 0 1 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0
            0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 1 0 1 1 0 0 0 0 0 1 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
            0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 1 0
            0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 1 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
            0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
            0 0 0 0 0 0 0 0 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
            0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 1
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1
        ]
    )

    graph6 = BipartiteGraph(
        [
            1 1 1 1 1 1 1 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 0 1 1 1 0 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 1 0 1 1 0 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 0 0 0 0 1 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 1 1 0 1 1 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 0 0 0 0 1 0 0 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 0 0 0 0 1 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 1 1 1 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 1 0 0 0 1 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 1 1 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 1 0 0 0 1 0 1 1 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 1 0 0 0 1 0 0 0 0 0 0 0 1 1 1 1 1 0 0 1 1 0 0 0 0 0 0 1
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 1 0 0 1 1 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 1 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 0 1 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 1 0 1 1 1 0 0 0 1 1 1 1 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 1 1 0 1 0 0 1 0 0 0 0 1
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 1 0 0 1 1 1 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 1 0 0 0 0 1 1 1 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 1 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 1 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 1 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 1 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 1 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1
        ]
    )

    graphs = (
        (weights1, graph1, uwidth1, wwidth1),
        (weights2, graph2, uwidth2, wwidth2),
        (weights3, graph3, uwidth3, wwidth3),
        (weights4, graph4, uwidth4, wwidth4),
        (weights5, graph5, uwidth5, wwidth5),
        (weights6, graph6, uwidth6, wwidth6)
    )

    for (G, V, E) in TYPES
        @testset "$(nameof(G))" begin
            for (weights, graph, uwidth, wwidth) in graphs
                graph = G(graph); uwidth = V(uwidth)

                for alg in lb_algs
                    ulb = lowerbound(graph; alg)
                    wlb = lowerbound(weights, graph; alg)
                    @test isa(ulb, V)
                    @test isa(wlb, Float64)
                    @test ulb <= uwidth
                    @test wlb <= wwidth
                end

                for outer_alg in (SafeRules, SafeSeparators)
                    for inner_alg in tw_algs
                        alg = outer_alg(inner_alg)
                        @test treewidth(graph; alg) === uwidth
                    end

                    alg = outer_alg(BT())
                    @test round(treewidth(weights, graph; alg); digits=1) === wwidth
                end
            end
        end
    end
end

@testset "heuristic treewidth (HBT)" begin
    # brute-force components of G - X, and their neighborhoods
    function bruteforce_components(graph, X)
        n = nv(graph); inX = falses(n); inX[X] .= true; seen = falses(n); out = []

        for v0 in 1:n
            (inX[v0] || seen[v0]) && continue
            C = Int[]; queue = [v0]; seen[v0] = true

            while !isempty(queue)
                u = pop!(queue); push!(C, u)

                for x in neighbors(graph, u)
                    if !inX[x] && !seen[x]
                        seen[x] = true; push!(queue, x)
                    end
                end
            end

            S = sort!(unique!([x for u in C for x in neighbors(graph, u) if inX[x]]))
            push!(out, (sort!(C), S))
        end

        return out
    end

    function ispmc(graph, X)
        cmps = bruteforce_components(graph, X)
        any(S == X for (_, S) in cmps) && return false

        for a in X, b in X
            (a < b && !has_edge(graph, a, b)) || continue
            any(a in S && b in S for (_, S) in cmps) || return false
        end

        return true
    end

    # check every PMC of every solution: it is a PMC, and its components,
    # their separators, and its superblocks are right
    function validate(hbt, graph)
        ctx = hbt.srh.ctx; seq(b) = collect(CliqueTrees.bt_tab_seq(ctx.seps, ctx.bsep[b]))

        for st in hbt.srh.sols, p in 1:CliqueTrees.hbt_len(st.sol)
            sol = st.sol; X = collect(CliqueTrees.hbt_pmc(sol, p))
            ispmc(graph, X) || return false
            cmps = bruteforce_components(graph, X); cs = sol.cfst[p]:sol.cfst[p + 1] - 1
            length(cmps) == length(cs) || return false

            for c in cs
                b = sol.csub[c]
                C, S = cmps[findfirst(c -> ctx.bmin[b] in c[1], cmps)]
                minimum(C) == ctx.bmin[b] || return false
                length(C) == ctx.bsize[b] || return false
                seq(b) == S || return false
                b = sol.csup[c]; rest = setdiff(X, S)
                D, T = only(filter(c -> rest[1] in c[1], bruteforce_components(graph, S)))
                issubset(rest, D) && T == S || return false
                minimum(D) == ctx.bmin[b] && length(D) == ctx.bsize[b] || return false
            end
        end

        return true
    end

    for (n, p) in ((30, 0.10), (40, 0.08), (50, 0.15), (64, 0.05))
        Random.seed!(n)
        graph = erdos_renyi(n, p)
        weights = rand(1:3, n)
        exact = treewidth(graph; alg = ConnectedComponents(PIDBT()))
        wexact = treewidth(weights, graph; alg = ConnectedComponents(PIDBT()))

        # base = 8 forces the heuristic machinery onto small graphs
        for alg in (
                HBT(; time = 0.2, base = 8),
                HBT(; time = 0.2, base = 8, merge = false),
                HBT(; time = 0.2, base = 8, diversify = false),
                HBT(; time = 0.2, base = 8, refined = false, near = 0),
                HBT(; time = 0.2, base = 20, dsize = 25, seed = 7, xtime = 0.01),
                HBT(; time = 0.2, base = 8, margin = 0, pexact = 0.0),
                HBT(; time = 0.2, base = 8, margin = 10, pexact = 1.0, near = 0),
                HBT(; time = 0.2, base = 8, threads = 3),
                HBT(; time = 0.2, base = 8, init = AMF(), ninit = 0),
            )
            order, index = permutation(graph; alg = ConnectedComponents(alg))
            @test sort(order) == 1:n
            @test index == invperm(order)
            @test treewidth(graph; alg = order) >= exact

            order, index = permutation(weights, graph; alg = ConnectedComponents(alg))
            @test sort(order) == 1:n
            @test treewidth(weights, graph; alg = order) >= wexact
        end

        # internal invariants of the solutions
        for merge in (true, false), margin in (0, 40)
            if is_connected(graph)
                hbt = CliqueTrees.hbtnew(weights, graph; seed = 2, alg = AMF(), base = 8, ntry = 10, ninit = 3, merge, margin)
                @test validate(hbt, graph)

                for _ in 1:20
                    CliqueTrees.hbtstep!(hbt)
                end

                @test validate(hbt, graph)
                order, invp, k = CliqueTrees.hbtord(hbt)
                @test sort(order) == 1:n && invp == invperm(order)
                @test treewidth(weights, graph; alg = order) <= k
                @test CliqueTrees.pmcdp(weights, graph, CliqueTrees.hbtpmc(hbt))[3] <= k

                # a start from an ordering: no wider than it
                init = randperm(n)
                hbt = CliqueTrees.hbtnew(weights, graph; seed = 2, alg = AMF(), base = 8, ninit = 0, merge, margin, init)
                @test validate(hbt, graph)
                @test CliqueTrees.hbtord(hbt)[3] <= treewidth(weights, graph; alg = init)

                for _ in 1:20
                    CliqueTrees.hbtstep!(hbt)
                end

                @test validate(hbt, graph)

                # the workers
                hbt = CliqueTrees.hbtnew(weights, graph; seed = 3, alg = AMF(), base = 8, ntry = 10, ninit = 3, merge, margin)
                order, invp, k = CliqueTrees.hbtrun!(hbt; mt = 0.2, nt = 3)
                @test validate(hbt, graph)
                @test sort(order) == 1:n
                @test treewidth(weights, graph; alg = order) <= k
            end
        end
    end

    # disconnected graphs and isolated vertices
    Random.seed!(1)
    graph = blockdiag(blockdiag(erdos_renyi(40, 0.1), erdos_renyi(70, 0.05)), Graph(3))
    order, index = permutation(graph; alg = ConnectedComponents(HBT(; time = 0.2, base = 10)))
    @test sort(order) == 1:nv(graph)
    @test treewidth(graph; alg = order) >= treewidth(graph; alg = ConnectedComponents(PIDBT()))

    # complete graph: the lower bound is reached immediately
    graph = complete_graph(80)
    @test treewidth(graph; alg = HBT(; time = 10.0)) == 79

    # septw with a cap on the bag weight, as used by HBT
    for (n, p) in ((20, 0.2), (30, 0.1), (40, 0.05))
        Random.seed!(n)
        graph = erdos_renyi(n, p)

        for v in 1:n - 1
            add_edge!(graph, v, v + 1)
        end

        weights = rand(1:3, n)
        w = treewidth(weights, graph; alg = PIDBT())
        order, _, k, status = CliqueTrees.septw(weights, graph, 0, w)
        @test status == CliqueTrees.BT_FOUND && k == w && treewidth(weights, graph; alg = order) == w
        @test CliqueTrees.septw(weights, graph, 0, w - 1)[4] == CliqueTrees.BT_ABOVE
        @test CliqueTrees.septw(weights, graph, w, w - 1)[4] == CliqueTrees.BT_ABOVE

        # a target at or above the treewidth is met
        for target in w:w + 3
            @test treewidth(weights, graph; alg = PIDBT(; target)) <= target
            @test treewidth(weights, graph; alg = PIDBT(w; target)) <= target
        end
    end

    # PIDBT with a target that is not a bag weight: the bags of this graph
    # can weigh 38 or 40 but not 39, and its treewidth is 37 (found by HBT
    # in a local graph of PACE 2017 instance he016)
    weights = [1, 20, 3, 6, 3, 4, 4]
    graph = Graph(7)

    for (v, list) in enumerate(([3, 4, 5, 6, 7], [3, 4, 5, 6, 7], [1, 2, 4, 6, 7], [1, 2, 3, 5, 7], [1, 2, 4, 6, 7], [1, 2, 3, 5, 7], [1, 2, 3, 4, 5, 6]))
        for w in list
            add_edge!(graph, v, w)
        end
    end

    @test treewidth(weights, graph; alg = PIDBT()) == 37

    for lb in (0, 37), target in (38, 39)
        @test treewidth(weights, graph; alg = PIDBT(lb; target)) <= 38
        @test treewidth(weights, graph; alg = CliqueTrees.septw(weights, graph, max(lb, target), typemax(Int))[1]) <= 38
    end
end

@testset "safe separators: weighted almost cliques" begin
    # S = {1, 2, 3} is an almost clique: the edge {2, 3} is missing. Its full
    # components are {4} and {5}; the component {6} is not full. Vertex 6 is
    # lighter than both candidate centers, 2 and 3, so S is filled only if the
    # full components are at least as heavy as the center.
    graph = Graph(6)

    for (u, v) in ((1, 2), (1, 3), (4, 1), (4, 2), (4, 3), (5, 1), (5, 2), (5, 3), (6, 1))
        add_edge!(graph, u, v)
    end

    order = [6, 4, 5, 1, 2, 3]

    for (w, filled) in ((5, true), (3, true), (2, false))
        weights = [1, 3, 3, w, w, 1]
        cmpgraph, _, _ = CliqueTrees.acsd(weights, graph, MinimalChordal(order))
        @test (3 in neighbors(cmpgraph, 2)) == filled
        @test treewidth(weights, graph; alg = SafeSeparators(PIDBT())) == treewidth(weights, graph; alg = PIDBT())
    end
end

@testset "safe separators: minor test" begin
    # S = {1, 2, 3} is independent, so not an almost clique, and neither is
    # any other minimal separator. Each component of G - S is a triangle
    # {x, y, z} with x adjacent to 1 and 2, y to 2 and 3, and z to 1 and 3:
    # contracting x into 1, y into 2, and z into 3 makes S a clique, so S is
    # safe.
    graph = Graph(9)

    for (x, y, z) in ((4, 5, 6), (7, 8, 9))
        for (u, v) in ((x, 1), (x, 2), (y, 2), (y, 3), (z, 1), (z, 3), (x, y), (y, z), (x, z))
            add_edge!(graph, u, v)
        end
    end

    weights = ones(Int, 9)
    min = MinimalChordal([4, 5, 6, 7, 8, 9, 1, 2, 3])
    isclique(g) = all(u == v || v in neighbors(g, u) for u in 1:3, v in 1:3)
    # (S has three missing edges)
    @test !isclique(first(CliqueTrees.acsd(weights, graph, (min,), 0)))
    @test !isclique(first(CliqueTrees.acsd(weights, graph, (min,), 2)))
    @test isclique(first(CliqueTrees.acsd(weights, graph, (min,), 3)))
    @test isclique(first(CliqueTrees.acsd(weights, graph, (min,), 100)))
end

@testset "safe separators: exactness" begin
    rng = Xoshiro(2)
    mins = (MinimalChordal(AMF()), MinimalChordal(MF()), MCSM(), LexM())

    for _ in 1:100
        n = rand(rng, 10:22)
        graph = erdos_renyi(n, rand(rng, (0.15, 0.25, 0.35, 0.5)); rng)
        is_connected(graph) || continue
        weights = rand(rng, Bool) ? ones(Int, n) : rand(rng, 1:4, n)
        width = treewidth(weights, graph; alg = PIDBT())
        @test treewidth(weights, graph; alg = SafeSeparators(PIDBT())) == width
        @test treewidth(weights, graph; alg = SafeSeparators(PIDBT(); maxmissing = 100)) == width
        @test treewidth(weights, graph; alg = SafeSeparators(PIDBT(), mins; maxmissing = 100)) == width
    end
end

@testset "safe separators: solver" begin
    # the leaves are the atoms of the safe completion (the graph with every
    # leaf filled in), the separator of a leaf is a clique of the completion,
    # in the leaf and in a later one, and the orderings of the leaves glue
    # into an ordering no wider than the widest leaf, optimal if they are
    rng = Xoshiro(5)
    mins = (CliqueTrees.DEFAULT_MINIMAL_ALGORITHM,)
    nsplit = 0

    for _ in 1:100
        graph = erdos_renyi(rand(rng, 10:30), rand(rng, (0.05, 0.1, 0.2, 0.3)); rng)
        rand(rng, Bool) && (graph = blockdiag(graph, erdos_renyi(rand(rng, 3:8), 0.4; rng)))
        n = nv(graph); weights = rand(rng, Bool) ? ones(Int, n) : rand(rng, 1:4, n)
        V = rand(rng, (Int32, Int)); maxmissing = rand(rng, (0, 100))
        weights30, graphs30, result = CliqueTrees.safesolve(weights, BipartiteGraph{V, V}(graph), mins, maxmissing)
        nleaf = length(result.lgraph); nsplit += nleaf > 1

        classes(k) = [result.prjtgt[result.prjptr[c]:result.prjptr[c + 1] - 1] for c in result.leafptr[k]:result.leafptr[k + 1] - 1]
        leaf(k) = reduce(vcat, classes(k))
        sep(k) = result.septgt[result.sepptr[k]:result.sepptr[k + 1] - 1]
        completion = Graph(n)

        for k in 1:nleaf
            cs = classes(k); g = result.lgraph[k]

            for c in cs, u in c, v in c
                u < v && add_edge!(completion, u, v)
            end

            if !iszero(g)
                for x in vertices(graphs30[g]), y in neighbors(graphs30[g], x), u in cs[x], v in cs[y]
                    add_edge!(completion, u, v)
                end
            end
        end

        @test all(e -> has_edge(completion, e), edges(graph))
        order, tree = CliqueTrees.atomtree(completion)
        @test Set(Set(order[i] for i in bag) for bag in tree) == Set(Set(leaf(k)) for k in 1:nleaf)

        for k in 1:nleaf
            K = sep(k)
            @test K ⊆ leaf(k)
            @test all(u == v || has_edge(completion, u, v) for u in K, v in K)
            @test isempty(K) || any(K ⊆ leaf(j) for j in k + 1:nleaf)
        end

        # one step at a time
        solver = CliqueTrees.SafeSolver(weights, BipartiteGraph{V, V}(graph), mins, maxmissing)
        while !isnothing(CliqueTrees.safe_step!(solver)) end
        @test CliqueTrees.SafeResult(solver).lgraph == result.lgraph
        @test CliqueTrees.SafeResult(solver).prjtgt == result.prjtgt

        orders = [randperm(rng, nv(g)) for g in graphs30]
        order = CliqueTrees.safe_pull(result, graphs30, orders)
        width = maximum(treewidth(w, g; alg = o) for (w, g, o) in zip(weights30, graphs30, orders); init = 0)
        @test eltype(order) == V
        @test sort(order) == 1:n
        @test treewidth(weights, graph; alg = order) <= max(width, result.width)

        orders = [first(permutation(w, g; alg = PIDBT())) for (w, g) in zip(weights30, graphs30)]
        order = CliqueTrees.safe_pull(result, graphs30, orders)
        @test treewidth(weights, graph; alg = order) == treewidth(weights, graph; alg = ConnectedComponents(PIDBT()))
    end

    @test nsplit >= 50
end

@testset "safe separators: components from the clique tree" begin
    # the components of G - S, and their neighborhoods, for every separator S
    # of a minimal triangulation, against a search of G
    rng = Xoshiro(3)

    for _ in 1:50
        n = rand(rng, 8:30)
        graph = BipartiteGraph(erdos_renyi(n, rand(rng, (0.05, 0.1, 0.2, 0.35)); rng))
        order, tree = cliquetree(graph, MinimalChordal())
        index = invperm(order)
        at = CliqueTrees.ACSDTree(tree, n)
        comps = Tuple{Int, Int}[]; ancs = Int[]

        for j in eachindex(tree)
            CliqueTrees.acsd_components!(comps, ancs, at, tree, j)
            S = Set(order[i] for i in CliqueTrees.separator(tree, j))
            nbrs = Dict(top => sort([order[i] for i in CliqueTrees.separator(tree, q)]) for (top, q) in comps)
            groups = Dict{Int, Vector{Int}}()

            for v in 1:n
                v in S && continue
                push!(get!(groups, CliqueTrees.acsd_top!(at, at.home[index[v]]), Int[]), v)
            end

            found = Set((sort(C), get(nbrs, top, Int[])) for (top, C) in groups)
            expected = Set{Tuple{Vector{Int}, Vector{Int}}}()
            seen = falses(n)

            for v in 1:n
                (v in S || seen[v]) && continue
                C = [v]; N = Set{Int}(); seen[v] = true; h = 1

                while h <= length(C)
                    u = C[h]; h += 1

                    for x in neighbors(graph, u)
                        if x in S
                            push!(N, x)
                        elseif !seen[x]
                            seen[x] = true; push!(C, x)
                        end
                    end
                end

                push!(expected, (sort(C), sort(collect(N))))
            end

            @test found == expected
        end
    end
end

@testset "safe reduction rules: lower bound" begin
    # the bound returned with the kernel must not exceed the largest bag
    # weight of an optimal decomposition (the almost simplicial rule used to
    # scan entries of its work arrays left over from `pr3` and from earlier
    # rounds when raising it)
    rng = Xoshiro(1)

    for _ in 1:200
        n = rand(rng, 10:25)
        graph = erdos_renyi(n, rand(rng, (0.15, 0.3, 0.5, 0.7)); rng)
        is_connected(graph) || continue
        weights = rand(rng, Bool) ? ones(Int, n) : rand(rng, 1:3, n)
        lb = lowerbound(weights, graph)
        kerlb = CliqueTrees.compressreduce(5, weights, graph, lb)[5]
        @test kerlb <= treewidth(weights, graph; alg = PIDBT())
    end
end

@testset "quasi-clique covers" begin
    # with β < 1, `qcc!` used to add a member of C to C again once no vertex
    # outside C was adjacent to C, repeating vertices and writing past the
    # end of the cover
    graphs = (
        complete_graph(3),
        path_graph(4),
        reduce(blockdiag, [path_graph(4) for _ in 1:20]),
        complete_bipartite_graph(30, 30),
    )

    for graph in graphs, beta in (1.0, 0.8, 0.5, 0.1)
        cover = CliqueTrees.qcc(Int, Int, graph, beta, Forward)
        covered = Set{Tuple{Int, Int}}()

        for c in vertices(cover)
            clique = neighbors(cover, c)
            @test allunique(clique)

            for i in clique, j in clique
                i < j && has_edge(graph, i, j) && push!(covered, (i, j))
            end
        end

        @test length(covered) == ne(graph)
    end
end

@static if !Sys.iswindows()
    @testset "KaHyPar dissection with nothing to cut" begin
        # KaHyPar crashed on hypergraphs in which no net has two or more pins:
        # isolated vertices, disjoint edges, and quasi-clique covers with β < 1
        # that collapse to a single quasi-clique
        graphs = (
            SimpleGraph(200),
            reduce(blockdiag, [complete_graph(2) for _ in 1:200]),
            complete_bipartite_graph(130, 130),
        )

        for graph in graphs, beta in (1.0, 0.5)
            order, index = permutation(graph; alg = ND{1}(AMF(), KaHyParND(; beta)))
            @test isperm(order)
            @test index == invperm(order)
        end
    end
end

@testset "lowrank" begin
    matrices = ("nos4", "mesh3e1", "494_bus")

    for name in matrices
        M = readmatrix(name); n = size(M, 2)
        b = rand(n)

        nz = [1, 5, 10, 100]
        M[nz, nz] .+= 1
        M[nz, nz] .-= 1
        v = zeros(n); v[nz] .= randn(length(nz))
        M_updated = copy(M); M_updated[nz, nz] .+= v[nz] * v[nz]'

        F1 = cholesky!(ChordalCholesky{:L}(copy(M)), NoPivot())
        F2 = cholesky!(ChordalCholesky{:U}(copy(M)), NoPivot())
        F3 = ldlt!(ChordalLDLt{:L}(copy(M)))
        F4 = ldlt!(ChordalLDLt{:U}(copy(M)))

        for Fi in (F1, F2, F3, F4)
            lowrankupdate!(Fi, v)
            @test isapprox(M_updated * (Fi \ b), b; rtol=1e-6, atol=1e-14)
        end

        G1 = cholesky!(ChordalCholesky{:L}(copy(M_updated)), NoPivot())
        G2 = cholesky!(ChordalCholesky{:U}(copy(M_updated)), NoPivot())
        G3 = ldlt!(ChordalLDLt{:L}(copy(M_updated)))
        G4 = ldlt!(ChordalLDLt{:U}(copy(M_updated)))

        for Gi in (G1, G2, G3, G4)
            lowrankdowndate!(Gi, v)
            @test isapprox(M * (Gi \ b), b; rtol=1e-6, atol=1e-14)
        end
    end

    M = SparseMatrixCSC{BigFloat}(readmatrix("685_bus")); n = size(M, 2)
    b = rand(BigFloat, n)
    nz = [1, 5, 10, 100]
    M[nz, nz] .+= 1
    M[nz, nz] .-= 1
    v = zeros(BigFloat, n); v[nz] .= randn(length(nz))
    M_updated = copy(M); M_updated[nz, nz] .+= v[nz] * v[nz]'

    F1 = cholesky!(ChordalCholesky{:L}(copy(M)), NoPivot())
    F2 = cholesky!(ChordalCholesky{:U}(copy(M)), NoPivot())
    F3 = ldlt!(ChordalLDLt{:L}(copy(M)))
    F4 = ldlt!(ChordalLDLt{:U}(copy(M)))

    for Fi in (F1, F2, F3, F4)
        lowrankupdate!(Fi, v)
        @test isapprox(M_updated * (Fi \ b), b; rtol=1e-6, atol=1e-14)
    end

    G1 = cholesky!(ChordalCholesky{:L}(copy(M_updated)), NoPivot())
    G2 = cholesky!(ChordalCholesky{:U}(copy(M_updated)), NoPivot())
    G3 = ldlt!(ChordalLDLt{:L}(copy(M_updated)))
    G4 = ldlt!(ChordalLDLt{:U}(copy(M_updated)))

    for Gi in (G1, G2, G3, G4)
        lowrankdowndate!(Gi, v)
        @test isapprox(M * (Gi \ b), b; rtol=1e-6, atol=1e-14)
    end
end

@testset "permutation" begin
    using CliqueTrees.Multifrontal: Permutation

    n = 50

    perm = randperm(n)
    P = Permutation(perm, invperm(perm))

    perm = randperm(n)
    Q = Permutation(perm, invperm(perm))

    D = rand(n, n)
    S = sprand(n, n, 0.3)

    for M in (Q, D, S)
        @assert P * M ≈ Matrix(P) * M
        @assert M * P ≈ M * Matrix(P)
        @assert P \ M ≈ Matrix(P) \ M
        @assert M \ P ≈ M \ Matrix(P)
        @assert P / M ≈ Matrix(P) / M
        @assert M / P ≈ M / Matrix(P)
    end
end
