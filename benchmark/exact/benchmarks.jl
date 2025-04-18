using BenchmarkTools
using CliqueTrees
using CryptoMiniSat_jll
using TreeWidthSolver
using Graphs

const EXCLUDE = Dict(
    "bt" => [
        "DyckGraph",
        "GrayGraph",
        "MeredithGraph",
        "NonisotropicUnitaryPolarGraph_3_3",
        "OddGraph_4",
        "RandomBarabasiAlbert_100_2",
        "RingedTree_6",
        "SylvesterGraph",
        "SzekeresSnarkGraph",
        "contiki_dhcpc_handle_dhcp",
        "dimacs_miles1000",
        "dimacs_queen7_7",

    ],
    "cms" => [
        "AhrensSzekeresGeneralizedQuadrangleGraph_3",
        "DyckGraph",
        "GrayGraph",
        "KneserGraph_10_2",
        "McGeeGraph",
        "MeredithGraph",
        "NonisotropicUnitaryPolarGraph_3_3",
        "OddGraph_4",
        "RandomBarabasiAlbert_100_2",
        "RingedTree_6",
        "SchlaefliGraph",
        "SquaredSkewHadamardMatrixGraph_2",
        "SylvesterGraph",
        "SzekeresSnarkGraph",
        "TaylorTwographDescendantSRG_3",
        "TaylorTwographSRG_3",
        "dimacs_myciel5",
        "dimacs_queen6_6",
        "dimacs_queen7_7",
    ],
)

const SUITE = BenchmarkGroup()

function readgraph(io::IO)
    lines = readlines(io)

    # remove comments
    filter!(lines) do line
        !startswith(line, "c")
    end

    # parse statistics
    words = split(popfirst!(lines))
    nv = parse(Int, words[3])
    ne = parse(Int, words[4])

    # parse graph
    graph = Graph(nv)

    while !isempty(lines)
        words = split(popfirst!(lines))
        v = parse(Int, words[1])
        w = parse(Int, words[2])
        add_edge!(graph, v, w)
    end

    return graph
end

path = joinpath(@__DIR__, "graphs")
bt = SafeRules(BT(), MMW(), MF())
cms = SafeRules(SAT{CryptoMiniSat_jll}(MF()), MMW(), MF())

for file in readdir(path)
    if endswith(file, ".gr")
        name = file[begin:(end - 3)]
        graph = open(readgraph, joinpath(path, file))

        if name ∉ EXCLUDE["bt"]
            SUITE[name]["bt"] = @benchmarkable treewidth($graph; alg = $bt)
        end

        if name ∉ EXCLUDE["cms"]
            SUITE[name]["cms"] = @benchmarkable treewidth($graph; alg = $cms)
        end
    end
end
