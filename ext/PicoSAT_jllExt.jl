module PicoSAT_jllExt

using CliqueTrees
using PicoSAT_jll

const library = PicoSAT_jll.libpicosat

function CliqueTrees.signature(::Val{PicoSAT_jll})
    version = unsafe_string(ccall((:picosat_version, library), Ptr{Cchar}, ()))
    return "picosat-$(version)"
end

function CliqueTrees.init(::Val{PicoSAT_jll})
    solver = ccall((:picosat_init, library), Ptr{Cvoid}, ())
    return solver
end

function CliqueTrees.release(::Val{PicoSAT_jll}, solver::Ptr{Cvoid})
    ccall((:picosat_reset, library), Cvoid, (Ptr{Cvoid},), solver)
    return
end

function CliqueTrees.add(::Val{PicoSAT_jll}, solver::Ptr{Cvoid}, lit_or_zero::Int32)
    ccall((:picosat_add, library), Cvoid, (Ptr{Cvoid}, Int32), solver, lit_or_zero)
    return
end

function CliqueTrees.assume(::Val{PicoSAT_jll}, solver::Ptr{Cvoid}, lit::Int32)
    ccall((:picosat_assume, library), Cvoid, (Ptr{Cvoid}, Int32), solver, lit)
    return
end

function CliqueTrees.solve(::Val{PicoSAT_jll}, solver::Ptr{Cvoid})
    state = ccall((:picosat_sat, library), Cint, (Ptr{Cvoid}, Cint), solver, -one(Cint))
    return state
end

function CliqueTrees.val(::Val{PicoSAT_jll}, solver::Ptr{Cvoid}, lit::Int32)
    value = ccall((:picosat_deref, library), Int32, (Ptr{Cvoid}, Int32), solver, lit)

    return value > zero(value) ? lit :
        value < zero(value) ? -lit :
        zero(Int32)
end

end
