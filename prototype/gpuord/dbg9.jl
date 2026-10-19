using CUDA
include(joinpath(@__DIR__, "src", "GPUOrd.jl")); using .GPUOrd
import .GPUOrd: jobkey, warpmin
function k!(bstart, width, lcost)
    lane = threadIdx().x - 1; b = blockIdx().x
    j0 = bstart[b]; j1 = bstart[b + 1] - Int32(1)
    key = typemax(UInt64); j = j0 + lane
    while j <= j1
        key = min(key, jobkey(width[j], lcost[j])); j += Int32(32)
    end
    key = warpmin(key)
    jb = typemax(Int32)
    j = j0 + lane
    while j <= j1
        jb = ifelse(jobkey(@inbounds(width[j]), @inbounds(lcost[j])) == key, min(jb, j), jb)
        j += Int32(32)
    end
    jb0 = jb
    sync_warp()
    for o in (16, 8, 4, 2, 1)
        jb = min(jb, shfl_xor_sync(0xffffffff, jb, o))
    end
    if lane < 5
        @cuprintln("b=$b lane=$lane key=$(key) jb0=$jb0 jb=$jb")
    end
    return
end
w = CuArray(Int32[7,7,7,7,7,7,7,7]); c = CuArray(Float32[4.9,4.8,4.7,4.6,4.5,4.4,4.3,4.2])
@cuda threads=32 blocks=2 k!(CuArray(Int32[1,5,9]), w, c); synchronize()
open("dbg9.ptx", "w") do io
CUDA.code_ptx(io, k!, Tuple{CuDeviceVector{Int32,1}, CuDeviceVector{Int32,1}, CuDeviceVector{Float32,1}})
end
