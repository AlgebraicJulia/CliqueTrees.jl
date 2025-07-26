module FlowCutterPACE17_jllExt

using ArgCheck
using Base: oneto
using CliqueTrees
using CliqueTrees: Parent, postorder, nov
using CliqueTrees.Utilities
using FlowCutterPACE17_jll
using Graphs
using SparseArrays

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph, alg::FlowCutter)
    index = flowcutter(graph, alg.time, alg.seed)
    return invperm(index), index
end

function flowcutter(graph::AbstractGraph{V}, time::Int, seed::Int) where {V}
    @argcheck !isnegative(time)
    @argcheck !isnegative(seed)
    dir = dirname(FlowCutterPACE17_jll.flow_cutter_pace17_path)

    hypergraph, tree = mktempdir(dir) do tmp
        input = tmp * "/input.gr"
        output = tmp * "/output.td"

        open(input; write = true) do io
            writegr(io, graph)
        end

        execute = flow_cutter_pace17()
        command = `$execute -s $seed`
        process = run(pipeline(input, command, output); wait = false)

        while !process_running(process)
            sleep(1)
        end

        sleep(time)
        kill(process)
        return open(readtd, output)
    end

    n = nov(hypergraph)
    index = zeros(V, n)
    order = reverse!(invperm(postorder(tree)))

    for i in order, v in neighbors(hypergraph, i)
        if iszero(index[v])
            index[v] = n
            n -= 1
        end
    end

    return index
end

function writegr(io::IO, graph::AbstractGraph)
    m = 0

    for v in vertices(graph), w in neighbors(graph, v)
        if (!is_directed(graph) && v != w) || v < w
            m += 1
        end
    end

    n = nv(graph)
    println(io, "p tw $n $m")

    for v in vertices(graph), w in neighbors(graph, v)
        if (!is_directed(graph) && v != w) || v < w
            println(io, "$v $w")
        end
    end

    return
end

function readtd(io::IO)
    lines = readlines(io)

    # remove comments
    filter!(lines) do line
        !startswith(line, "c")
    end

    isempty(lines) && error("FlowCutter failed")

    # parse statistics
    words = split(lines[1])
    nb = parse(Int, words[3])
    tw = parse(Int, words[4])
    nv = parse(Int, words[5])

    # parse hypergraph
    index = Vector{Int}(undef, nb)
    hypergraph = BipartiteGraph{Int, Int}(nv, nb, 0)
    pointers(hypergraph)[begin] = p = 1

    for i in oneto(nb)
        words = split(lines[i + 1])

        j = parse(Int, words[2])
        index[j] = i

        for word in words[3:end]
            p += 1
            push!(targets(hypergraph), parse(Int, word))
        end

        pointers(hypergraph)[i + 1] = p
    end

    # parse tree
    I = sizehint!(Int[], nb)
    J = sizehint!(Int[], nb)

    for line in lines[(nb + 2):end]
        words = split(line)
        i = parse(Int, words[1])
        j = parse(Int, words[2])
        push!(I, index[i], index[j])
        push!(J, index[j], index[i])
    end

    root = 1
    parent = dfs_parents(BipartiteGraph(sparse(I, J, J, nb, nb)), root)
    parent[root] = 0

    tree = Parent(nb, parent)
    return hypergraph, tree
end

end
