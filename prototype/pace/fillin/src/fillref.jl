# CPU reference: gpupid's bitsets/pidref/devfuns (read-only) + the fill-in PID (fillcore.jl).
const GPUPID_SRC = joinpath(@__DIR__, "..", "..", "..", "gpupid", "src")
include(joinpath(GPUPID_SRC, "pidref.jl"))     # bitsets, PGraph, comps, cliquish, vorder, relabel
include(joinpath(GPUPID_SRC, "devfuns.jl"))    # dv_outlet (shared with the GPU)
include("fillcore.jl")
