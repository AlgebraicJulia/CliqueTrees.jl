using CairoMakie
using CSV
using DataFrames
using PkgBenchmark
using PkgBenchmark: benchmarkgroup

function makeplots()
    results = benchmarkgroup(readresults(joinpath("exact", "results.json")))

    times = Dict(
        "bt" => Float64[],
        "cms" => Float64[],
    )

    for file in readdir(joinpath(@__DIR__, "exact", "graphs"))
        if endswith(file, ".gr")
            name = file[begin:(end - 3)]
            group = results[name]

            for (alg, vector) in times
                if haskey(group, alg)
                    trial = group[alg]
                    t = time(minimum(trial)) / 1.0e9
                else
                    t = 100.0
                end

                push!(vector, t)
            end
        end
    end

    perm = sortperm(times["bt"])
    permute!(times["bt"], perm)
    permute!(times["cms"], perm)

    figure = Figure(size = (450, 250))
    axis = Axis(figure[1, 1], ylabel = "time (s)", yscale = log10, xticksvisible = false, xticklabelsvisible = false)
    bt_plot = scatter!(axis, times["bt"]; marker = :circle, color = :red)
    cms_plot = scatter!(axis, times["cms"]; marker = :cross, color = :blue)

    Legend(
        figure[1, 2],
        [bt_plot, cms_plot],
        ["BT", "SAT"]
    )

    return save("exact/figure.png", figure)
end

function makereadme()
    makeplots()

    return open("README.md"; write = true) do io
        println(io, "# Benchmarks")
        println(io)
        println(io, "To regenerate this file, navigate to the `benchmark` directory and run the following command.")
        println(io, "```")
        println(io, "julia --project make.jl")
        println(io, "```")
        println(io)
        println(io, "## Exact Treewidth")
        println(io)
        println(io, "To run the exact treewidth benchmarks, navigate to the `benchmark` directory and run the following command.")
        println(io)
        println(io, "```")
        println(io, "julia --project exact/make.jl")
        println(io, "```")
        println(io)
        println(io, "The algorithms `BT` and `SAT` are benchmarked on the PACE16 competition medium instances.")
        println(io)
        println(io, "![](exact/figure.png)")
    end
end

makereadme()
