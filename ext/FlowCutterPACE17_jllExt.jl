module FlowCutterPACE17_jllExt

using CliqueTrees
using CliqueTrees: postorder, nov, readeo, writegr
using FlowCutterPACE17_jll
using Graphs

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph, alg::FlowCutter)
    index = flowcutter(graph, alg.time, alg.seed)
    return invperm(index), index
end

function flowcutter(graph::AbstractGraph{V}, time::Int, seed::Int) where {V}
    @assert time >= 0
    @assert seed >= 0

    index = mktempdir() do tmp
        input = tmp * "/input.gr"
        output = tmp * "/output.td"

        open(input; write = true) do io
            writegr(io, graph)
        end

        execute = flow_cutter_pace17()
        command = `$execute -s $seed`
        process = run(pipeline(command; stdin = input, stdout = output); wait = false)

        while !process_running(process)
            sleep(1)
        end

        sleep(time)
        kill(process)
        return open(io -> readeo(io, V), output)
    end

    return index
end

end
