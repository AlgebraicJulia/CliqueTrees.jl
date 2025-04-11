using CairoMakie
using CSV
using DataFrames
using PkgBenchmark
using PkgBenchmark: benchmarkgroup

function makeplots()
    names = String[]
    bt = Float64[]
    cms = Float64[]

    for (name, group) in benchmarkgroup(readresults("exact/results.json"))
        push!(names, name)
        push!(bt, time(minimum(group["bt"])) / 1e9)
        push!(cms, time(minimum(group["cms"])) / 1e9)
    end

    perm = sortperm(bt)
    permute!(names, perm)
    permute!(bt, perm)
    permute!(cms, perm)

    figure = Figure(size=(450, 250))
    axis = Axis(figure[1, 1], ylabel = "time (s)", yscale = log10, xticksvisible=false, xticklabelsvisible=false)
    bt_plot = scatter!(axis, bt, color = :red)
    cms_plot = scatter!(axis, cms, color = :blue)

    Legend(figure[1, 2],
        [bt_plot, cms_plot],
        ["BT", "SAT"])

    save("exact/figure.png", figure)
end

function makereadme()
    makeplots()
    graphs = CSV.read("exact/graphs/graphs.csv", DataFrame)
    results = benchmarkgroup(readresults("exact/results.json"))

    open("README.md"; write=true) do io
        println(io, "# CliqueTrees.jl Benchmarks")
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
        println(io, "The algorithms `bt` and `sat` are benchmarked on the DIMACS graph coloring instances.")
        println(io, "```julia-repl")
        println(io, "julia> using CliqueTrees, TreeWidthSolver, CryptoMiniSat_jll")
        println(io)
        println(io, "julia> bt = SafeRules(MMW(), BT()) # BT")
        println(io, "SafeRules{MMW, BT}:")
        println(io, "    MMW")
        println(io, "    BT")
        println(io)
        println(io, "julia> sat = SafeRules(MMW(), SAT{CryptoMiniSat_jll}(MMW(), MF())) # SAT")
        println(io, "SafeRules{MMW, SAT{CryptoMiniSat_jll, MMW, MF}}:")
        println(io, "    SAT{CryptoMiniSat_jll, MMW, MF}:")
        println(io, "        MMW")
        println(io, "        MF")
        println(io, "```")
        println(io)
        println(io, "![](exact/figure.png)")
        println(io)
        println(io, "| name | vertices | edges | treewidth | BT | SAT |")
        println(io, "| :--- | :------- | :---- | :-------- | :- | :-- |")
        
        for row in axes(graphs, 1)
            name = String(graphs[row, :name])
            nv = graphs[row, :nv]
            ne = graphs[row, :ne]
            tw = graphs[row, :tw]        
            time_bt = time(minimum(results[name]["bt"])) / 1e9
            time_cms = time(minimum(results[name]["cms"])) / 1e9
            print(io, "| $name | $nv | $ne | $tw ")
            
            if time_bt < time_cms
                println(io, "| **$time_bt** | $time_cms |")
            else
                println(io, "| $time_bt | **$time_cms** |")
            end
        end
    end
end

makereadme()
