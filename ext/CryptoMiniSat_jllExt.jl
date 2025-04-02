module CryptoMiniSat_jllExt

import CliqueTrees.IPASIR
import CryptoMiniSat_jll

const library = CryptoMiniSat_jll.libipasircryptominisat5

function IPASIR.signature(::Val{CryptoMiniSat_jll})
    string = unsafe_string(ccall((:ipasir_signature, library), Ptr{Cchar}, ()))
    return string
end

function IPASIR.init(::Val{CryptoMiniSat_jll})
    solver = ccall((:ipasir_init, library), Ptr{Cvoid}, ())
    return solver
end

function IPASIR.release(::Val{CryptoMiniSat_jll}, solver::Ptr{Cvoid})
    ccall((:ipasir_release, library), Cvoid, (Ptr{Cvoid},), solver)
    return
end

function IPASIR.add(::Val{CryptoMiniSat_jll}, solver::Ptr{Cvoid}, lit_or_zero::Int32)
    ccall((:ipasir_add, library), Cvoid, (Ptr{Cvoid}, Int32), solver, lit_or_zero)
    return
end

function IPASIR.assume(::Val{CryptoMiniSat_jll}, solver::Ptr{Cvoid}, lit::Int32)
    ccall((:ipasir_assume, library), Cvoid, (Ptr{Cvoid}, Int32), solver, lit)
    return
end

function IPASIR.solve(::Val{CryptoMiniSat_jll}, solver::Ptr{Cvoid})
    state = ccall((:ipasir_solve, library), Cint, (Ptr{Cvoid},), solver)
    return state
end

function IPASIR.val(::Val{CryptoMiniSat_jll}, solver::Ptr{Cvoid}, lit::Int32)
    value = ccall((:ipasir_val, library), Int32, (Ptr{Cvoid}, Int32), solver, lit)
    return value
end

end
