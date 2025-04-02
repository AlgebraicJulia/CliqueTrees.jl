# IPASIR.jl

An wrapper for IPASIR-compliant SAT solvers.

```julia-repl
julia> using CliqueTrees.IPASIR, PicoSAT_jll, CryptoMiniSAT_jll

julia> open(Solver{PicoSAT_jll}, 5) do solver
           # add clauses
           clause!(solver, 1, -5, 4)
           clause!(solver, -1, 5, 3, 4)
           clause!(solver, -3, -4)

           # add assumption
           solver[5] = -1

           # solve
           solve!(solver)

           # get assignments
           return collect(solver)
       end
5-element Vector{Int32}:
  1
 -1
 -1
  1
 -1

julia> open(Solver{CryptoMiniSat_jll}, 5) do solver
           # add clauses
           clause!(solver, 1, -5, 4)
           clause!(solver, -1, 5, 3, 4)
           clause!(solver, -3, -4)

           # add assumption
           solver[5] = -1

           # solve
           solve!(solver)

           # get assignments
           return collect(solver)
       end
5-element Vector{Int32}:
 -1
 -1
 -1
 -1
 -1
```
