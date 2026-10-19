using CUDA
include(joinpath(@__DIR__, "src", "GPUOrd.jl")); using .GPUOrd
import .GPUOrd: fillbag!, isset
function k!(::Val{W}, out, b1, b2, res) where {W}
    lane = threadIdx().x - 1
    sm = CuDynamicSharedArray(UInt64, 64W * W)
    for i in lane:32:64W*W-1; sm[i+1] = 0; end
    sync_warp()
    bag = (b1, b2)
    for i in 1:2W; res[i, lane+1] = isset(bag, i, lane); end
    s = fillbag!(sm, bag, ntuple(_->Int32(1), Val(2W)), lane)
    sync_warp()
    for i in lane:32:64W*W-1; out[i+1] = sm[i+1]; end
    return
end
b1 = (UInt64(1) << 56) | 1; b2 = UInt64(1) << 31
out = CUDA.zeros(UInt64, 256); res = CUDA.zeros(Bool, 4, 32)
@cuda threads=32 shmem=2048 k!(Val(2), out, b1, b2, res)
o = Array(out); r = Array(res)
@show findall(r)
for u in 0:127; (o[u+1] != 0 || o[u+129] != 0) && println(u, " ", bitstring(o[u+1]), " ", bitstring(o[u+129])); end
