# GPU semiring solver (experimental)

CUDA.jl kernels for `CliqueTrees.Multifrontal.Semiring`, the solver of the APSP-LU paper. They cover the solve
phase, a hybrid CPU+GPU numeric factorization, and the full closure A* (APSP). Every kernel is generic over the
semiring: it calls only `splus`, `sprod`, `smuladd`, `sstar`, `szero` and `sone`. Everything is checked against
the CPU code: bit for bit for min-plus with integer weights and for max-min, and to about 1e-15 for plus-times.

All numbers below are from one laptop: RTX 5060 Laptop GPU (sm_120, 8 GB, 50 W), AMD Ryzen AI 7 350 (8 cores,
16 threads), Float32, CUDA.jl 6.1, Julia 1.12. **CPU numbers use all 16 threads.** HiPerGator (A100/B200 vs EPYC)
runs are in progress.

## Bottom line

| workload | GPU vs best CPU (ours) | vs other people |
|---|---|---|
| blocks of single-source queries (k = 256) | **11–17×** with the same algorithm (18–31× with the GPU fast path) | — |
| single queries (k = 1) | **CPU wins, 2–38×** (the CPU uses the `tmp/sssp_tmp.jl` fast path and subtree parallelism) | PHAST is faster still |
| numeric factorization, high-fill graphs | **1.9–5.3×** vs our subtree-parallel CPU, 3–6× vs upstream | cuDSS (real arithmetic) is 1.5–2.2× faster than us; we beat CHOLMOD and UMFPACK |
| numeric factorization, road networks | ~1× (fronts too small) | — |
| full APSP closure | **17–40×** vs our CPU closure | **ROME (PPoPP'26): we're 1.05–1.76× faster end to end** on the 4 test grids |

## Layout and how to run

```
src/SemiringGPU.jl   semiring GEMM (sgemx_gpu!), module
src/sgetrs.jl        solve phase: GPUSLU, rmul_gpu!, sssp_gpu!, SSSPPlan, closure_gpu!, dense GPU TRSM
src/sgetrf.jl        dense GPU LU (sgetrf_gpu!), hybrid factorization (FactorPlan, factorize!, mlu_gpu, factorize_cpu!)
test/                test_solve.jl, test_factor.jl, test_closure.jl (all against the CPU solver)
bench/               benchmark scripts and results_*.txt
external/            ROME notes, patch, run script and logs (ROME itself is not vendored)
```

```
cd src/Multifrontal.jl/src/Semiring.jl/tmp/gpu
julia --project=. -e 'using Pkg; Pkg.develop(path="../../../../../.."); Pkg.instantiate()'
julia --project=. -t auto test/test_solve.jl          # likewise test_factor.jl, test_closure.jl
julia --project=. -t auto bench/bench_solve.jl         # blocked queries; also bench_single, bench_factor, bench_closure, bench_gemm
```

- **Graph data is not committed.** `bench_solve.jl` reads `data/roadNet-PA.txt.gz` (SNAP) and
  `data/USA-road-t.{NY,FLA}.gr.gz` (9th DIMACS challenge). `bench/export_mtx.jl` writes the synthetic grids used
  for the ROME comparison to `data/mtx/`.
- **Undirected graphs only for now:** directed graphs with coupled strongly connected components are not
  supported on the GPU yet.

## API

```julia
F = mlu(MinPlus(), A)                    # CPU factorization (upstream)
G = GPUSLU(F)                            # upload the factor and the level schedule

rmul_gpu!(B, G)                          # B ← B A*, B is k × n on the GPU (row layout)
sssp_gpu!(X, G, sources)                 # rows of A* for the given sources (fast path: U sweep along root paths)
P = SSSPPlan(G, k); P(sources)           # the same, recorded as a CUDA graph
D = closure_gpu(G)                       # all of A*, n × n, in elimination coordinates: D[i, j] = A*[p[i], p[j]], p = G.rperm

P = FactorPlan(F; large = 256)           # hybrid factorization plan (F must hold A: copyto!(F, A))
factorize!(P); G = GPUSLU(P)             # CPU bottom subtrees ∥ GPU top fronts; later calls replay a CUDA graph
sgetrf_gpu!(MinPlus(), A_gpu)            # dense semiring LU on the GPU
```

## How it works

- **Solve.** Level schedule over the elimination tree: the U sweep goes leaves to root, the L sweep root to
  leaves. Each level is one batched launch with one block per (front, chunk of 64 right-hand sides) and one thread
  per right-hand side.
  - The row layout (each vertex's k values contiguous) makes the accesses coalesced.
  - Sibling scatters into ancestors use a generic atomic ⊕: a compare-and-swap loop around `splus`.
  - Large fronts use dense kernels: a blocked TRSM with diagonal-block inversion, plus the semiring GEMM.
  - For unit-vector sources, the U sweep walks only each source's root path. This is the paper's fast path, and
    the upward search of GPHAST and contraction hierarchies.
- **Factorization.** The top of the tree (fronts with nn+na ≥ `large` and all their ancestors) goes to the GPU.
  - The bottom forest is split into independent subtrees, factored on CPU threads with upstream's own
    `sgetrf_loop!`.
  - Boundary updates are uploaded. The top is factored front by front (extend-add, dense LU, both TRSMs, Schur
    GEMM), with the fronts of each level spread over 8 CUDA streams, and captured as a CUDA graph for
    refactorization.
- **Closure.** All n sources in one block, with the n×n result as its own work matrix: n root-path walks, then
  the top and L sweeps with n rows. Its work is n·nnz(L), against n²|S| for supernodal Floyd–Warshall
  (SuperFW, ROME).

## Results

### Blocked queries, k = 256, MinPlus (ms per query; `bench/results_solve_v2_rtx5060_laptop.txt`)

| graph | n | CPU best (16 thr) | GPU, same algorithm (`rmul_gpu!`) | GPU fast path (`sssp_gpu!`) |
|---|---:|---:|---:|---:|
| USA-road-t.NY | 264k | 1.139 | 0.086 (13×) | 0.052 (22×) |
| roadNet-PA | 1.09M | 4.698 | 0.366 (13×) | 0.249 (19×) |
| USA-road-t.FLA | 1.07M | 4.371 | 0.264 (17×) | 0.142 (31×) |
| grid2d-500 | 250k | 1.282 | 0.363 (3.5×) | 0.326 (3.9×) |

The CPU column is the best of `rmul!`/`lmul!` with a dense B (there is no blocked CPU fast path). Copying results
to the host costs about as much as the solve (0.07–0.3 ms/query pinned), so the GPU pays off when results stay on
the device.

### Single queries, k = 1 (ms; `bench/results_single.txt`, median of 100 sources)

| graph | CPU fast path, 1 thr | **CPU fast path, 16 thr** | GPU (`SSSPPlan`) |
|---|---:|---:|---:|
| USA-road-t.NY | 3.24 | **0.83** | 3.18 |
| USA-road-t.FLA | 13.2 | **3.27** | 6.87 |
| roadNet-PA | 14.8 | **8.24** | 30.6 |
| grid2d-500 | 5.50 | **2.23** | 84.9 |

The CPU uses `tmp/sssp_tmp.jl` + `tmp/subtree_tmp.jl`, which should be wired into upstream. **Single queries
belong on the CPU.**

### Numeric factorization (ms; `bench/results_factor_v2.txt`, symbolic phase excluded)

MinPlus Float32, same AMF ordering for all ("hybrid" = refactorization by CUDA graph replay):

| graph | n | upstream CPU | ours CPU ∥ (subtree-parallel) | hybrid CPU+GPU | vs ours CPU ∥ |
|---|---:|---:|---:|---:|---:|
| USA-road-t.NY | 264k | 25.2 | 12.3 | 13.6 | 0.9× |
| grid2d-500 | 250k | 152.9 | 89.2 | 47.5 | 1.9× |
| grid2d-1000 | 1M | 720.9 | 507.9 | 183.6 | 2.8× |
| grid3d-30 | 27k | 155.8 | 129.8 | 33.2 | 3.9× |
| grid3d-40 | 64k | 561.1 | 512.8 | 96.9 | 5.3× |

Real arithmetic, M = I − W (ours factors W in `PlusProd`; each library uses its own ordering):

| graph | ours CPU∥ F64 | ours GPU F64 | ours GPU F32 | UMFPACK | CHOLMOD | cuDSS LU F64 | cuDSS LU F32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| USA-road-t.NY | 11.3 | 12.3 | 10.1 | 203.7 | 33.7 | 7.8 | 6.6 |
| grid2d-1000 | 653.6 | 431.9 | 183.3 | 2083.1 | 565.5 | 221.6 | 94.3 |
| grid3d-40 | 689.5 | 369.0 | 93.4 | 1231.3 | 314.2 | 208.6 | 59.6 |

Dense LU, n = 4096 (G multiply-adds/s):
- **MinPlus:** our GPU 649 vs our CPU 113.
- **Real F32:** our GPU 697 vs cuSOLVER 1331 vs LAPACK 293.

### Full APSP closure vs ROME (s; `bench/results_closure_laptop.txt`, `external/ROME_NOTES.md`)

ROME is built locally from https://github.com/LyleLuo/ROME, with `external/rome_gatas.patch`. **Stock ROME
ignores the file's weights and uses `rand()%10+1`**; the patch makes it read them. Same graphs and weights for
both; our distances match ROME's exactly (grid3d-25, 46,875 entries checked).

| graph | n | ours: symbolic + numeric + closure | ROME: ordering + setup + GPU compute | GPU compute, ours / ROME | ROME work ÷ ours |
|---|---:|---:|---:|---:|---:|
| grid3d-25 | 15,625 | **0.205** | 0.216 | 0.161 / **0.122** | 5.5× |
| grid2d-150 | 22,500 | **0.103** | 0.181 | 0.084 / 0.082 | 10.2× |
| grid3d-30 | 27,000 | **0.724** | 0.831 | **0.644** / 0.677 | 7.7× |
| grid2d-180 | 32,400 | **0.209** | 0.273 | **0.182** / 0.209 | 10.9× |

Our CPU closure (`Matrix(F)`, 16 threads): 3.42 s (grid3d-25) and 4.07 s (grid2d-150).

How to read this:
- **We win because we do 5–11× less work, not because our kernels are better.** Per operation, ROME is about
  7–10× more efficient (about 1.4 T vs 0.2 T min-plus ops/s).
- **The work ratio ≈ n·|S|/nnz(L).** It grows with n on small-separator graphs (2D ~√n/log n, 3D ~n^(1/3)), so
  larger meshes should favor us more.
- **On large-separator graphs** (social, random, dense 3D; work ratio ~0.5–3× in the paper's SuperFW table), ROME
  should win. A dense blocked closure switched on by the fill ratio would be needed there; our dense GEMM already
  runs at about 1.75 T/s.
- **Only 4 synthetic grids, on one consumer GPU.** ROME was tuned for an RTX 4090.
- **Our thresholds were tuned on these graphs, and Julia's JIT compile time is excluded.** ROME is compiled ahead
  of time.
- **Output and un-permutation.** Both sides exclude copying the result to the host and the final un-permutation;
  our D is in elimination coordinates.
- **ROME's limit is GPU memory.** It needs n² on the GPU (about 45k vertices on 8 GB). Blocked queries with the
  same factorization run on 1M+ vertices.

## Semiring GEMM (`sgemx_gpu!`; `bench/results_rtx5060_laptop.txt`)

- **Tiles:** 128×128 (64×64, 256×16 and 128×32 for small or skinny outputs), BK = 8, 256 threads, register tiles
  unrolled with `@generated`.
- **Speed:** MinPlus F32 reaches about 1.75–2.2 T/s, about 50% of cuBLAS SGEMM. That's near the ceiling, because
  min-plus is 2 instructions (FADD + FMNMX) against 1 FFMA.
- **Use F32/I32 on consumer GPUs;** FP64 runs at 1/64 rate.
- **CuTropicalGEMM.jl doesn't work here:** there's no CUDA 13 binary.

## What is still weak in the kernels

1. **Latency, not arithmetic.**
   - About 40 small launches per dense front, each with a 3–5 µs floor even in a CUDA graph.
   - Small fronts get one 64-thread block each.
   - One launch per tree level.
2. **Unfused memory passes in the dense path:** zero, assemble, copy into L/U, copy the upper triangle, and the
   TRSM scratch round trip. The 64×64 diagonal LU runs on a single SM.
3. **No batching across fronts and no merging of contiguous subtrees** (ROME's batch and merge engines), and no
   supernode amalgamation (mean front size 1.3–4 here).
4. **The batched solve re-reads the factor once per 64 right-hand sides,** and uses Int64 indices. Holding columns
   in registers (`DOWN_VARIANT[] = :blocked`) was slower: masked instructions still issue.
5. **GEMM:** single-buffered shared memory, scalar loads, bounds checks on every tile, and no warp tiling.

## Upstream issues found

- **The tropical `szero`/`sone` methods have an unbounded `T`.** They are ambiguous with the generic
  `Vec{W,T}` lift, which crashes `lmul!` and Int32 `mlu` on AVX-512 CPUs. This commit fixes it with
  `T <: Real` in `src/semiring/tropical.jl`.
- **Row-layout `rmul!` doesn't get faster with threads.**
- **The single-query fast path and subtree parallelism live in `tmp/`,** not in `sgetrs!`/`rmul!`.
- **The numeric factorization processes fronts in sequence.** Subtree parallelism (`factorize_cpu!` here) gives
  1.2–1.8×.
- **`CuVector(::FixedSizeArray)` is about 40× slower than a raw copy;** use `unsafe_copyto!`.

## Next

- HiPerGator: A100/B200 against a full EPYC node, larger graphs (up to n²-memory limits), real graphs, ROME's own
  test set and the SuperFW set.
- A dense blocked closure for high-fill graphs, chosen by the fill ratio.
- Batched or fused front kernels, and supernode amalgamation, to close the 7–10× efficiency gap to ROME.
- The residuated solve (`ldiv!`, `trans = :C`) and BTF coupling for directed graphs.
