using BenchmarkTools
using CliqueTrees
using CryptoMiniSat_jll
using TreeWidthSolver
using Graphs

const SUITE = BenchmarkGroup()

function readgraph(io::IO)
    lines = readlines(io)

    # remove comments
    filter!(lines) do line
        !startswith(line, "c")
    end
    
    # parse statistics
    words = split(popfirst!(lines))
    nv = parse(Int, words[2])
    ne = parse(Int, words[3])

    # parse graph
    graph = Graph(nv)
    
    while !isempty(lines)
        words = split(popfirst!(lines))
        v = parse(Int, words[2]) + 1
        w = parse(Int, words[3]) + 1
        add_edge!(graph, v, w)
    end

    return graph
end

path = joinpath(@__DIR__, "graphs")
bt = SafeRules(BT())
cms = SafeRules(SAT{CryptoMiniSat_jll}(MMW(), MF()))

for name in readdir(path)
    if endswith(name, ".graph")
        graph = open(readgraph, joinpath(path, name))
        SUITE[name[begin:end - 6]]["bt"] = @benchmarkable treewidth($graph; alg=$bt)
        SUITE[name[begin:end - 6]]["cms"] = @benchmarkable treewidth($graph; alg=$cms)
    end
end
