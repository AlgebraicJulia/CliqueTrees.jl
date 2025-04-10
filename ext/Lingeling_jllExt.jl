module Lingeling_jllExt

import CliqueTrees.IPASIR
import Lingeling_jll

@static if !Sys.iswindows()
    const library = Lingeling_jll.liblgl

    function IPASIR.signature(::Val{Lingeling_jll})
        version = unsafe_string(ccall((:lglversion, library), Ptr{Cchar}, ()))
        return "lingeling-$(version)"
    end

    function IPASIR.init(::Val{Lingeling_jll})
        solver = ccall((:lglinit, library), Ptr{Cvoid}, ())
        return solver
    end

    function IPASIR.release(::Val{Lingeling_jll}, solver::Ptr{Cvoid})
        ccall((:lglrelease, library), Cvoid, (Ptr{Cvoid},), solver)
        return
    end

    function IPASIR.add(::Val{Lingeling_jll}, solver::Ptr{Cvoid}, lit_or_zero::Int32)
        if !iszero(lit_or_zero)
            ccall((:lglfreeze, library), Cvoid, (Ptr{Cvoid}, Int32), solver, lit_or_zero)
        end

        ccall((:lgladd, library), Cvoid, (Ptr{Cvoid}, Int32), solver, lit_or_zero)
        return
    end

    function IPASIR.assume(::Val{Lingeling_jll}, solver::Ptr{Cvoid}, lit::Int32)
        ccall((:lglfreeze, library), Cvoid, (Ptr{Cvoid}, Int32), solver, lit)
        ccall((:lglassume, library), Cvoid, (Ptr{Cvoid}, Int32), solver, lit)
        return
    end

    function IPASIR.solve(::Val{Lingeling_jll}, solver::Ptr{Cvoid})
        state = ccall((:lglsat, library), Cint, (Ptr{Cvoid}, Cint), solver, -one(Cint))
        return state
    end

    function IPASIR.val(::Val{Lingeling_jll}, solver::Ptr{Cvoid}, lit::Int32)
        value = ccall((:lglderef, library), Int32, (Ptr{Cvoid}, Int32), solver, lit)

        return value > zero(value) ? lit :
            value < zero(value) ? -lit :
            zero(Int32)
    end
end

end
