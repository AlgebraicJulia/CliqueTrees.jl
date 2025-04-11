using PkgBenchmark
using CliqueTrees

script = joinpath("benchmark", "exact", "benchmarks.jl")
resultfile = joinpath(@__DIR__, "results.json")
benchmarkpkg(CliqueTrees; script, resultfile)
