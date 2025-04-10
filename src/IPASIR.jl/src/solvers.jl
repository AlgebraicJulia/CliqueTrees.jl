"""
    Solver{H} <: AbstractVector{Int32}

    Solver{H}(num::Integer)

An IPASIR-compliant SAT solver. This type implements the [abstract vector interface](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array).
"""
mutable struct Solver{H} <: AbstractVector{Int32}
    handle::Val{H}
    solver::Ptr{Cvoid}
    num::Int32

    function Solver{H}(num::Integer) where {H}
        handle = Val(H)
        solver = init(handle)
        return new{H}(handle, solver, num)
    end
end

function clause!(solver::Solver, clause::Integer...)
    return clause!(solver, clause)
end

"""
    clause!(solver::Solver, clause)

    clause!(solver::Solver, clause::Integer...)

Add a clause to a solver.
"""
function clause!(solver::Solver, clause)
    for lit in clause
        @argcheck one(Int32) <= abs(lit) <= length(solver)
        add(solver.handle, solver.solver, Int32(lit))
    end

    add(solver.handle, solver.solver, zero(Int32))
    return solver
end

"""
    solve!(solver::Solver)

Solve a SAT problem. Returns `:sat`, `:unsat`, or `:unknown`.
"""
function solve!(solver::Solver)
    state = solve(solver.handle, solver.solver)

    return state == Cint(10) ? :sat :
        state == Cint(20) ? :unsat :
        :unknown
end

function Base.open(f::Function, ::Type{Solver{H}}, num::Integer = zero(Int32)) where {H}
    solver = Solver{H}(num)

    try
        return f(solver)
    finally
        close(solver)
    end
end

function Base.close(solver::Solver)
    release(solver.handle, solver.solver)
    return
end

function Base.show(io::IO, solver::Solver{H}) where {H}
    println(io, "Solver{$H}:")
    print(io, "    solver: $(signature(solver.handle))")
    return
end

#############################
# abstract vector interface #
#############################

function Base.size(solver::Solver)
    return (solver.num,)
end

function Base.setindex!(solver::Solver, val::Integer, lit::Integer)
    @argcheck one(Int32) <= lit <= length(solver)
    @argcheck isone(val) || isone(-val)
    assume(solver.handle, solver.solver, Int32(val * lit))
    return solver
end

function Base.getindex(solver::Solver, lit::Integer)
    @argcheck one(Int32) <= lit <= length(solver)
    return sign(val(solver.handle, solver.solver, Int32(lit)))
end

function Base.resize!(solver::Solver, num::Integer)
    @argcheck !isnegative(num)
    solver.num = num
    return solver
end

#######################
# low-level interface #
#######################

function signature(handle)
    error()
end

function init(handle)
    error()
end

function release(handle, solver::Ptr{Cvoid})
    error()
end

function add(handle, solver::Ptr{Cvoid}, lit_or_zero::Int32)
    error()
end

function assume(handle, solver::Ptr{Cvoid}, lit::Int32)
    error()
end

function solve(handle, solver::Ptr{Cvoid})
    error()
end

function val(handle, solver::Ptr{Cvoid}, lit::Int32)
    error()
end
