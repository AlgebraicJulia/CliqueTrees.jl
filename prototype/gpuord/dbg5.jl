using CUDA, CliqueTrees, Graphs, Random
include(joinpath(@__DIR__, "src", "GPUOrd.jl")); using .GPUOrd
import .GPUOrd: getbag, fillbag!, clearbit, warpsum, loadpiece!
function k!(::Val{W}, adj, wts, ord, n, out, bags) where {W}
    N = 64W; lane = threadIdx().x - 1
    sm = CuDynamicSharedArray(UInt64, N * W)
    loadpiece!(sm, adj, 1, Val(W), lane)
    wl = ntuple(i -> wts[lane + 32(i - 1) + 1, 1], Val(2W))
    alive = ntuple(_ -> typemax(UInt64), Val(W))
    sync_warp()
    for t in 1:n
        v = Int32(ord[t])
        bag = getbag(sm, v, alive, lane)
        for j in 1:W; bags[j, lane+1, t] = bag[j]; end
        s = warpsum(fillbag!(sm, bag, wl, lane))
        alive = clearbit(alive, v)
        sync_warp()
        for i in lane:32:N*W-1; out[i+1, t] = sm[i+1]; end
        sync_warp()
    end
    return
end
rng = Xoshiro(1); n=100; W=2; N=128
g = erdos_renyi(n, 0.08; rng); ord = randperm(rng, n); w = ones(Int32, n)
b = pack([g], [w], W)
out = CUDA.zeros(UInt64, N*W, n); bags = CUDA.zeros(UInt64, W, 32, n)
@cuda threads=32 shmem=N*W*8 k!(Val(W), b.adj, b.wts, CuArray(Int16.(ord .- 1)), n, out, bags)
o = Array(out); bg = Array(bags)
sm = vec(Array(b.adj)); alive = ntuple(_ -> typemax(UInt64), Val(W))
for t in 1:n
  v = Int32(ord[t]-1); bag = getbag(sm, v, alive, lane)
  for l in 1:32; (bg[1,l,t], bg[2,l,t]) == bag || println("t=$t lane $(l-1) bag differs"); end
  for lane in Int32(0):Int32(31); fillbag!(sm, bag, ntuple(i->Int32(1),Val(2W)), lane); end
  global alive = clearbit(alive, v)
  if sm != o[:, t]
     d = findall(sm .!= o[:, t]); println("t=$t v=$v bag=$(bitstring.(bag)) rows differ at word idx ", [( (i-1)%N, (i-1)÷N) for i in d]); 
     for i in d; println("  cpu ", bitstring(sm[i]), "\n  gpu ", bitstring(o[i,t])); end
     break
  end
end
