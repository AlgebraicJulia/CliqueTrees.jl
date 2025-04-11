# CliqueTrees.jl Benchmarks

This file was automatically generated on 2025-04-11. To regenerate it, navigate to the `benchmark` directory and run the following command.
```
julia --project make.jl
```

## Exact Treewidth

The algorithms `bt` and `sat` are benchmarked on the DIMACS graph coloring instances.
```julia-repl
julia> using CliqueTrees, TreeWidthSolver, CryptoMiniSat_jll

julia> bt = SafeRules(MMW(), BT()) # BT
SafeRules{MMW, BT}:
    MMW
    BT

julia> sat = SafeRules(MMW(), SAT{CryptoMiniSat_jll}(MMW(), MF())) # SAT
SafeRules{MMW, SAT{CryptoMiniSat_jll, MMW, MF}}:
    SAT{CryptoMiniSat_jll, MMW, MF}:
        MMW
        MF
```

To run these benchmarks, navigate to the `benchmark` directory, enter the Julia REPL, and run the following commands.

```julia-repl
julia> using PkgBenchmark, CliqueTrees

julia> script = joinpath("benchmark", "exact", "benchmarks.jl");

julia> resultfile = joinpath(@__DIR__, "exact", "results.json");

julia> benchmarkpkg(CliqueTrees; script, resultfile)
PkgBenchmark: Running benchmarks...
```

![](exact/figure.png)

| name | vertices | edges | treewidth | BT | SAT |
| :--- | :------- | :---- | :-------- | :- | :-- |
| anna | 138 | 493 | 12 | **0.001205916** | 0.023199875 |
| david | 87 | 406 | 13 | **0.007648416** | 0.017865042 |
| fpsol2.i.1 | 496 | 11654 | 66 | 0.214960708 | **0.009796209** |
| fpsol2.i.2 | 451 | 8691 | 31 | 1.979331125 | **0.006072125** |
| fpsol2.i.3 | 425 | 8688 | 31 | 1.983972166 | **0.006065583** |
| huck | 74 | 301 | 10 | 3.3166e-5 | **3.1042e-5** |
| inithx.i.1 | 864 | 18707 | 56 | **0.634499417** | 11.572270334 |
| inithx.i.2 | 645 | 13979 | 31 | 3.866544208 | **0.009192959** |
| inithx.i.3 | 621 | 13969 | 31 | 4.051543084 | **0.009221667** |
| jean | 80 | 254 | 9 | 2.8875e-5 | **2.6833e-5** |
| miles250 | 128 | 387 | 9 | 0.004899 | **7.6042e-5** |
| miles750 | 128 | 2113 | 36 | **36.463336167** | 68.370938459 |
| miles1000 | 128 | 3216 | 49 | 93.015928375 | **10.554019416** |
| miles1500 | 128 | 5198 | 77 | 0.122255917 | **0.002365625** |
| mulsol.i.1 | 197 | 3925 | 50 | 0.004172333 | **0.00153075** |
| mulsol.i.2 | 188 | 3885 | 32 | 0.035491 | **0.00185675** |
| mulsol.i.3 | 184 | 3916 | 32 | 0.036973291 | **0.001798708** |
| mulsol.i.4 | 185 | 3946 | 32 | 0.03735625 | **0.001825041** |
| mulsol.i.5 | 186 | 3973 | 31 | **0.039066167** | 2.790137458 |
| myciel2 | 5 | 5 | 2 | 3.552125e-6 | **1.8e-6** |
| myciel3 | 11 | 20 | 5 | **0.0001545** | 0.003108708 |
| myciel4 | 23 | 71 | 10 | **0.008309042** | 4.117594875 |
| zeroin.i.1 | 211 | 4100 | 50 | **0.002198959** | 0.057723083 |
| zeroin.i.2 | 211 | 3541 | 32 | **0.003567166** | 0.109104458 |
| zeroin.i.3 | 206 | 3540 | 32 | **0.003548709** | 0.10769625 |
