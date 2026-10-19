using CUDA
function k!(out, T)
    lane = threadIdx().x - 1
    jb = lane == 3 ? T(4) : typemax(T)
    for o in (16, 8, 4, 2, 1)
        jb = min(jb, shfl_xor_sync(0xffffffff, jb, o))
    end
    out[lane+1] = jb
    return
end
for T in (Int32, UInt32, Int64)
  out = CUDA.zeros(T, 32); @cuda threads=32 k!(out, T); println(T, " ", Array(out)')
end
function k2!(out)
    lane = threadIdx().x - 1
    jb = lane == 3 ? Int32(4) : typemax(Int32)
    jb = min(jb, shfl_xor_sync(0xffffffff, jb, 16))
    jb = min(jb, shfl_xor_sync(0xffffffff, jb, 8))
    jb = min(jb, shfl_xor_sync(0xffffffff, jb, 4))
    jb = min(jb, shfl_xor_sync(0xffffffff, jb, 2))
    jb = min(jb, shfl_xor_sync(0xffffffff, jb, 1))
    out[lane+1] = jb
    return
end
out = CUDA.zeros(Int32, 32); @cuda threads=32 k2!(out); println("unrolled ", Array(out)')
