module HTD_jllExt

using CliqueTrees
using CliqueTrees: postorder, nov, readeo, writegr
using Graphs
using HTD_jll

const ALGORITHM = (
    "random",
    "min-fill",
    "min-degree",
    "min-separator",
    "max-cardinality",
    "max-cardinality-enhanced",
    "challenge",
)

const STRATEGY = (
    "none",
    "simple",
    "advanced",
    "full",
)

const CRITERION = (
    "none",
    "width",
)

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph, alg::HTD)
    index = htd(graph, alg.seed, alg.strategy, alg.preprocessing,
        alg.triangulation_minimization, alg.opt, alg.iterations, alg.patience)

    return invperm(index), index
end

function htd(
        graph::AbstractGraph{V},
        seed::Int,
        strategy::Int,
        preprocessing::Int,
        triangulation_minimization::Int,
        opt::Int,
        iterations::Int,
        patience::Int,
    ) where {V}
    input = IOBuffer()
    output = IOBuffer()

    writegr(input, graph)    
    seekstart(input)

    execute = htd_main()
    command = `$execute`

    command = addseed(command, seed)
    command = addstrategy(command, strategy)
    command = addpreprocessing(command, preprocessing)
    command = addtriangulationminimization(command, triangulation_minimization)
    command = addopt(command, opt)
    command = additerations(command, iterations)
    command = addpatience(command, patience)

    process = run(pipeline(command, stdin = input, stdout = output))
    seekstart(output)
    return readeo(output, V)
end

function addseed(command::Cmd, option::Int)
    if option >= 0
        command = `$command -s $option`
    end

    return command
end

function addstrategy(command::Cmd, option::Int)
    if option >= 0
        enum = ALGORITHM[option + 1]
        command = `$command --strategy $enum`
    end

    return command
end

function addpreprocessing(command::Cmd, option::Int)
    if option >= 0
        enum = STRATEGY[option + 1]
        command = `$command --preprocessing $enum`
    end

    return command
end

function addtriangulationminimization(command::Cmd, option::Int)
    if option >= 0
        command = `$command --triangulation-minimization`
    end

    return command
end

function addopt(command::Cmd, option::Int)
    if option >= 0
        enum = CRITERION[option + 1]
        command = `$command --opt $enum`
    end

    return command
end

function additerations(command::Cmd, option::Int)
    if option >= 0
        command = `$command --iterations $option`
    end

    return command
end

function addpatience(command::Cmd, option::Int)
    if option >= 0
        command = `$command --patience $option`
    end

    return command
end

end
