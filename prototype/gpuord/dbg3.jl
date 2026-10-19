using CUDA, CliqueTrees, Graphs, Random
include(joinpath(@__DIR__, "src", "GPUOrd.jl")); using .GPUOrd
import .GPUOrd: getbag, fillbag!, clearbit, warpsum, loadpiece!
function k!(::Val{W}, adj, wts, ord, n, out) where {W}
    N = 64W; lane = threadIdx().x - 1
    sm = CuDynamicSharedArray(UInt64, N * W)
    loadpiece!(sm, adj, 1, Val(W), lane)
    wl = ntuple(i -> wts[lane + 32(i - 1) + 1, 1], Val(2W))
    alive = ntuple(_ -> typemax(UInt64), Val(W))
    sync_warp()
    for t in 1:n
        v = Int32(ord[t])
        bag = getbag(sm, v, alive)
        if lane == 0
            for j in 1:W; out[j, t] = bag[j]; end
        end
        sync_warp()
        s = warpsum(fillbag!(sm, bag, wl, lane))
        alive = clearbit(alive, v)
        sync_warp()
    end
    return
end
rng = Xoshiro(1); n=100; W=2; N=128
g = erdos_renyi(n, 0.08; rng); ord = randperm(rng, n); w = ones(Int32, n)
b = pack([g], [w], W)
out = CUDA.zeros(UInt64, W, n)
@cuda threads=32 shmem=N*W*8 k!(Val(W), b.adj, b.wts, CuArray(Int16.(ord .- 1)), n, out)
o = Array(out)
# CPU
sm = vec(Array(b.adj)); alive = ntuple(_ -> typemax(UInt64), Val(W))
for t in 1:n
  v = Int32(ord[t]-1); bag = getbag(sm, v, alive)
  if (bag[1], bag[2]) != (o[1,t], o[2,t]); println("first diff at t=$t v=$v cpu=$(bitstring.(bag)) gpu=$(bitstring.(o[:,t]))"); break; end
  for lane in Int32(0):Int32(31); fillbag!(sm, bag, ntuple(i->Int32(1),Val(2W)), lane); end
  global alive = clearbit(alive, v)
end
