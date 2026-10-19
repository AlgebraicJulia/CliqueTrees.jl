using CliqueTrees, Graphs, Random
include(joinpath(@__DIR__, "src", "GPUOrd.jl")); using .GPUOrd
rng = Xoshiro(1)
n=100; W=2; N=128
g = erdos_renyi(n, 0.08; rng); w = ones(Int32, n)
rng = Xoshiro(1); g = erdos_renyi(n, 0.08; rng); ord = randperm(rng, n)
adj = zeros(UInt64, N*W, 1); for v in 1:n, x in [v; neighbors(g, v)]; GPUOrd.setbit!(adj, 1, N, v-1, x-1); end
sm = vec(adj)
alive = ntuple(_ -> typemax(UInt64), Val(W)); wmax = 0
for t in 1:n
  v = Int32(ord[t]-1)
  bag = GPUOrd.getbag(sm, v, alive)
  s = 0
  for lane in Int32(0):Int32(31)
    wl = ntuple(i -> w[min(n, lane + 32(i-1) + 1)], Val(2W))
    s += GPUOrd.fillbag!(sm, bag, wl, lane)
  end
  global wmax = max(wmax, s); global alive = GPUOrd.clearbit(alive, v)
end
@show wmax treewidth(w, g; alg = ord)
