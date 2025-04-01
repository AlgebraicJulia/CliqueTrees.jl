module CryptoMiniSat_jllExt

using CliqueTrees
using CryptoMiniSat_jll

const library = CryptoMiniSat_jll.libipasircryptominisat5

function CliqueTrees.signature(::Val{CryptoMiniSat_jll})
    string = unsafe_string(ccall((:ipasir_signature, library), Ptr{Cchar}, ()))
    return string
end

function CliqueTrees.init(::Val{CryptoMiniSat_jll})
    solver = ccall((:ipasir_init, library), Ptr{Cvoid}, ())
    return solver
end

function CliqueTrees.release(::Val{CryptoMiniSat_jll}, solver::Ptr{Cvoid})
    ccall((:ipasir_release, library), Cvoid, (Ptr{Cvoid},), solver)
    return
end

function CliqueTrees.add(::Val{CryptoMiniSat_jll}, solver::Ptr{Cvoid}, lit_or_zero::Int32)
    ccall((:ipasir_add, library), Cvoid, (Ptr{Cvoid}, Int32), solver, lit_or_zero)
    return
end

function CliqueTrees.assume(::Val{CryptoMiniSat_jll}, solver::Ptr{Cvoid}, lit::Int32)
    ccall((:ipasir_assume, library), Cvoid, (Ptr{Cvoid}, Int32), solver, lit)
    return
end

function CliqueTrees.solve(::Val{CryptoMiniSat_jll}, solver::Ptr{Cvoid})
    state = ccall((:ipasir_solve, library), Cint, (Ptr{Cvoid},), solver)
    return state
end

function CliqueTrees.val(::Val{CryptoMiniSat_jll}, solver::Ptr{Cvoid}, lit::Int32)
    value = ccall((:ipasir_val, library), Int32, (Ptr{Cvoid}, Int32), solver, lit)
    return value
end

end
