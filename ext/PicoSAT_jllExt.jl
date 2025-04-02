module PicoSAT_jllExt

import CliqueTrees.IPASIR
import PicoSAT_jll

@static if !Sys.iswindows()
    const library = PicoSAT_jll.libpicosat

    function IPASIR.signature(::Val{PicoSAT_jll})
        version = unsafe_string(ccall((:picosat_version, library), Ptr{Cchar}, ()))
        return "picosat-$(version)"
    end

    function IPASIR.init(::Val{PicoSAT_jll})
        solver = ccall((:picosat_init, library), Ptr{Cvoid}, ())
        return solver
    end

    function IPASIR.release(::Val{PicoSAT_jll}, solver::Ptr{Cvoid})
        ccall((:picosat_reset, library), Cvoid, (Ptr{Cvoid},), solver)
        return
    end

    function IPASIR.add(::Val{PicoSAT_jll}, solver::Ptr{Cvoid}, lit_or_zero::Int32)
        ccall((:picosat_add, library), Cvoid, (Ptr{Cvoid}, Int32), solver, lit_or_zero)
        return
    end

    function IPASIR.assume(::Val{PicoSAT_jll}, solver::Ptr{Cvoid}, lit::Int32)
        ccall((:picosat_assume, library), Cvoid, (Ptr{Cvoid}, Int32), solver, lit)
        return
    end

    function IPASIR.solve(::Val{PicoSAT_jll}, solver::Ptr{Cvoid})
        state = ccall((:picosat_sat, library), Cint, (Ptr{Cvoid}, Cint), solver, -one(Cint))
        return state
    end

    function IPASIR.val(::Val{PicoSAT_jll}, solver::Ptr{Cvoid}, lit::Int32)
        value = ccall((:picosat_deref, library), Int32, (Ptr{Cvoid}, Int32), solver, lit)

        return value > zero(value) ? lit :
            value < zero(value) ? -lit :
            zero(Int32)
    end
end

end
