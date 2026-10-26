module MLFLib

using Base: oneto
using FillArrays
using ..Utilities

export mlf!, MLFScore, MinFill, MinAvgFill, MinMeanFill, MinIncDeg, MinSqrtFill

function maxint(::Type{I}) where {I}
    return typemax(I) - convert(I, 100000)
end

# bit-mask type used by mfupd_def; the number of new clique members
# marked at once (the group size) is its width
const GMASK = UInt64
const GROUPSIZE = 8 * sizeof(GMASK)

include("mlfscore.jl")
include("mlfwh.jl")
include("genmf_wh.jl")
include("compress.jl")
include("mfinit_def.jl")
include("mfinit_vanilla.jl")
include("mfinit_heap.jl")
include("build_heap.jl")
include("del_heap.jl")
include("mod_heap.jl")
include("move_up.jl")
include("move_down.jl")
include("elmclq.jl")
include("elmtra.jl")
include("garbg2.jl")
include("mfupd_def.jl")
include("mfupd_mat.jl")
include("mfnumn.jl")

end
