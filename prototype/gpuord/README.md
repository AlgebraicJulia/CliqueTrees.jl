# gpuord: batched bit-parallel elimination on the GPU

Prototype code (CUDA.jl 6, Julia 1.12). The env is `env/` (CliqueTrees dev'ed from `CliqueTrees.jl-mt`, plus CUDA, Metis and Graphs).
Run with `julia --project=env`. The laptop GPU is shared, so keep runs short.

## Kernels: `src/GPUOrd.jl` (module `GPUOrd`)

Pieces of at most N = 64W vertices are padded into buckets. W = 1, 2, 4 or 8 (N up to 512).
Rows are stored word-major (word j of row u is `adj[u + j*N + 1, p]`), and the diagonal is set.
One warp handles one job, a (piece, ordering) pair. Lane l owns rows l, l+32, and so on.
Widths are exact Int32 sums of the weights; cost is log2 Σ_t 2^(bag_t · invq) in Float32 (|error| ≤ 2e-4 bits at N = 512).

```julia
include("src/GPUOrd.jl"); using .GPUOrd
b = pack(graphs, int32_weights, W; nints = nothing, sweights = int32_weights)
# graphs: Graphs.jl graphs with nv <= 64W. int32_weights[i]::Vector{Int32} are the exact weights.
# nints[i]: only vertices 1:nint are eliminated; the rest are "boundary" and count in bags only.
# sweights: small nonnegative ints, used only in greedy scores (stored as bit planes).

pidx  = CuArray(Int32[...])               # job g works on piece pidx[g]
width = CUDA.zeros(Int32, J); lcost = CUDA.zeros(Float32, J); tmax = CUDA.zeros(Int32, J)

score!(width, lcost, tmax, b, pidx, orders::CuMatrix{Int16}; invq)   # orders[t, g]: 0-based vertex eliminated at step t
greedy!(orders, width, lcost, tmax, b, pidx, params::CuVector{GreedyParams}; invq)
#   GreedyParams(cdeg, cfill, cavg, noise, seed): score = cdeg*w(N[u]) + cfill*fill + cavg*fill/w(N[u]),
#   times (1 + noise*U), with ties broken at random.
mutate!(cand, width, lcost, tmax, b, pidx, best, besttmax, params::CuVector{MutateParams}; invq)
#   moves applied to the incumbent best[:, piece], then the result is scored
selectbest!(bstart, bpiece, cand, width, lcost, tmax, best, bestw, bestc, besttmax, b)
#   per block of jobs: keep the least (width, cost) as the incumbent
```

Launch configuration (warps per block) comes from device attributes only.

Verified (`test_correct.jl`): widths equal CliqueTrees `treewidth(w, g; alg = order)` exactly on 2,560 random orderings (W = 1..8).
Boundary pieces match the CPU bitset reference. Greedy orderings are valid permutations, and their widths are rechecked on the CPU.

**Compiler pitfalls found (read before writing similar kernels):**
1. `x & ~(1 << (v & 63))` becomes a rotate whose NVPTX lowering gives shl/shr the unmasked amount, so the word is zeroed for v ≥ 64.
   Clear a known-set bit by subtraction instead.
2. A `Union{Int32, Int64}` index (from `threadIdx().x - 1`, which is Int64) gets union-split into lane-divergent paths with different shuffles, which return garbage.
   Keep warp-reduction inputs concretely typed.
3. Read shared incumbents before any lane writes them: put a `sync_warp()` between the read and the write.

## Hybrid ordering: `src/Hybrid.jl`, `src/GPUND.jl`

`Hybrid.GPUND(; time, pmax, width, level, rounds, big)` is a CliqueTrees `EliminationAlgorithm`.
It is ND{1}, and every ND node graph with at most `pmax` vertices is ordered by one batched GPU search (record and replay, with cached METIS separators).

## Exact local triangulations (HBT's `septw`) on the GPU: `exact/`

HBT's exact local triangulation is `hbt_exa`, which runs a lower bound, then `compressreduce`, then SafeRules/SafeSeparators.
Each atom then goes to `septw`, which tries the widths k one round at a time; each round is `sepenum` (the minimal separators lighter than k) followed by `sepdp` (Tamaki's separator DP).
Both are ported to the GPU, and the results are bit-identical: the same separator lists in the same order, the same block values and caps, the same permutation and k/status.

| file | what |
|---|---|
| `Bits.jl` | fixed-size bitsets for device code: `UInt64` (n ≤ 64) or `Bits{W}` (n ≤ 64W), same operations |
| `EnumW.jl` | `sepenum`. One thread per start vertex. The tree is split at depth 1–3 into tasks for persistent threads; listings carry DFS-path keys, so the sorted output is the CPU list. |
| `DPW.jl` | `sepdp` block values: one thread per block, one launch per block-size class, persistent threads, explicit-stack cap/uni search, then the roots. Preprocessing and the final ordering stay on the CPU. |
| `SeptwGPU.jl` | `septw_gpu(w, g, lo, hi, dl)`: CliqueTrees' `bt_tw`/`septw_run` with GPU rounds. `hook_hbt_escalate!()` sends HBTExact atoms to the GPU when the CPU has not finished within δ, where δ = 2× the measured GPU round overhead (`calibrate!()`). |
| `SepEnumGPU.jl`, `SepDPGPU.jl` | the first one-word versions; the reference for the generic ones |
| `capture.jl` | in-process instrumentation of HBT (records `hbt_exa` calls and `septw` atoms and rounds; the tree is not modified) |

```julia
include("exact/SeptwGPU.jl")       # loads EnumW and DPW
perm, invp, k, status = septw_gpu(weights, graph, lo, hi, deadline)   # same as CliqueTrees.septw(..., deadline, 1)
r = gpu_round(weights, graph, k, deadline)   # one round: (perm, invp, mink, :found | :above), :timeout, or nothing (GPU stack overflow)
```

Deadlines use cooperative cancellation. The host sets an abort flag in device memory from a second stream, and the kernels poll it with an atomic read every 1,024 search steps.
Device buffers persist between calls and grow geometrically; only the counters are zeroed.

Verified (`test_w.jl`, `test_septw.jl`, `gpu_exact.jl`): all captured rounds match CliqueTrees, on the UInt64 path, on `Bits{2}`/`Bits{4}` forced on n ≤ 64 data, and on real atoms of 84–164 vertices.

## Exact separator enumeration: `exact/SepEnumGPU.jl`

This is a GPU port of CliqueTrees `sepenum` (Tamaki's minimal-separator listing, w(S) < k) for n ≤ 64.
Each thread runs the search from one start vertex. The search tree can be split at depth 1–3 into tasks, which persistent threads take from a queue.
Every listing carries a DFS-path key, so sorting the keys gives `sepenum`'s list exactly (same separators, same order).

```julia
include("exact/SepEnumGPU.jl"); using .SepEnumGPU
b = prepare(graphs, float64_weights, maxks); lists, ovf, t, info = gpu_sepenum(b; split = 3)
```

## Status (2026-10-18)

- Exact GPU `septw` rounds are bit-identical to CliqueTrees on all 4,939 captured rounds (B200 and RTX PRO 6000).
- Kernel time is 5.6–7.2× one laptop CPU thread for n ≤ 64 (W=1) and 3.6–4.4× for the 256-bit bucket (3 rounds).
- The bottleneck has moved to host glue (CliqueTrees' sepdp preprocessing on the CPU, the key sort, transfers) and to the long single-thread tails of the block searches in sepdp.
- Integrated into HBT through `hook_hbt_escalate!()`, it does not yet pay off. Per-call latency (0.6–2.5 s on heavy calls) halves the number of exact calls HBT completes within its cutoffs (3 seeds × 7 instances, 20 s, B200; `results/hpg_logs/itay-gpu-46600537.out`).
- Next steps: move sepdp preprocessing (blocks, relabel, trie) to the GPU, use warp-cooperative candidate scans in the cap search, and batch local graphs across HBT calls.
