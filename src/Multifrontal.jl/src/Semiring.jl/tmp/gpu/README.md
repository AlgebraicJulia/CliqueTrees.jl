# GPU semiring solver (experimental)

CUDA.jl kernels for `CliqueTrees.Multifrontal.Semiring`, the solver of the APSP-LU paper. They cover the solve
phase, a hybrid CPU+GPU numeric factorization, and the full closure A* (APSP). The algorithm is the paper's: the
same symbolic phase (upstream `ssymbolic`, AMF), the same Backhouse–Carré multifrontal factorization (bottom fronts
literally run upstream's `sgetrf_loop!`), the same U-then-L triangular sweeps and fast path, and the same closure
bound n·nnz(L+U). The GPU work only changes scheduling (tree levels, streams, CUDA graphs) and, in dense fronts,
the grouping of the same semiring products (diagonal-block inversion, precomputed per-front operators). Every
kernel is generic over the semiring: it calls only `splus`, `sprod`, `smuladd`, `sstar`, `szero` and `sone`.

Machines:
- **Laptop:** RTX 5060 Laptop GPU (sm_120, 8 GB, 50 W) and AMD Ryzen AI 7 350 (8 cores, 16 threads).
- **HiPerGator:** NVIDIA B200 (sm_100, 179 GB) and RTX PRO 6000 Blackwell (sm_120, 95 GB); CPU baselines there
  use 16 cores of the GPU node.

Unless stated otherwise: Float32, CUDA.jl 6.1, Julia 1.12. See `HPG_NOTES.md` for how jobs run on HiPerGator.

## Bottom line

| workload | GPU vs best CPU (ours) | vs other people |
|---|---|---|
| blocks of single-source queries (k = 256) | **11–17×** on the laptop with the same algorithm (18–31× with the GPU fast path); **up to 69×** vs 16 CPU threads on the 24M-vertex USA network (RTX PRO 6000) | — |
| single queries (k = 1) | **CPU wins, 2–38×** (the CPU uses the `tmp/sssp_tmp.jl` fast path and subtree parallelism) | PHAST is faster still |
| numeric factorization, high-fill graphs | **1.9–5.3×** vs our subtree-parallel CPU, 3–6× vs upstream | cuDSS (real arithmetic) is 1.5–2.2× faster than us; we beat CHOLMOD and UMFPACK |
| numeric factorization, road networks | ~1× (fronts too small) | — |
| full APSP closure | **17–40×** vs our CPU closure | **ROME (PPoPP'26):** it wins below ~30–60k vertices on the big GPUs (by 1.1–2.8×); we win above, by up to **4.6×** (B200, 195k vertices); only we fit grid2d-450 (202k vertices, 164 GB) |

The wins over ROME come from doing 5–11× less work, not from better kernels: per operation we are still 5–22×
less efficient (see below). Everything so far is on 2D and 3D grids, which have small separators. On
large-separator graphs (social, random) the work advantage disappears, and ROME should win.

## Layout and how to run

```
src/SemiringGPU.jl   semiring GEMM (sgemx_gpu!, pipelined v2), device overrides, module
src/sgetrs.jl        solve phase: GPUSLU, rmul_gpu!, sssp_gpu!, SSSPPlan, closure_gpu!, precompute_ops!,
                     persistent L sweep, warp-cooperative path walk, dense GPU TRSM
src/sgetrf.jl        dense GPU LU (sgetrf_gpu!), hybrid factorization (FactorPlan, factorize!, mlu_gpu, factorize_cpu!)
test/                test_solve.jl, test_factor.jl, test_closure.jl (against the CPU solver);
                     stress.jl, stress_large.jl (against independent oracles); sass_guard.jl; sanitize_small.jl
bench/               benchmark scripts and results_*.txt
external/            ROME notes, patch, run script and logs (ROME itself is not vendored)
HPG_NOTES.md         HiPerGator setup, partitions, sbatch templates, pitfalls
```

```
cd src/Multifrontal.jl/src/Semiring.jl/tmp/gpu
julia --project=. -e 'using Pkg; Pkg.develop(path="../../../../../.."); Pkg.instantiate()'
julia --project=. -t auto test/test_solve.jl            # likewise test_factor.jl, test_closure.jl
julia --project=. -t auto test/stress.jl 300            # random cases vs the Floyd–Warshall oracle (rerun one: stress.jl 1 <seed>)
julia --project=. -t auto test/stress_large.jl 30       # 2k–30k vertices vs a Dijkstra oracle, plus determinism
julia --project=. test/sass_guard.jl                     # no CALL / spills, FMNMX in the hot kernels
julia --project=. -t auto bench/bench_closure.jl grid3d-30 256 8192 simple ops   # closure; also bench_solve, bench_single, bench_factor, bench_gemm
```

- **Graph data is not committed.**
  - `bench_solve.jl` reads `data/roadNet-PA.txt.gz` (SNAP) and `data/USA-road-t.{NY,FLA,USA}.gr.gz` (9th DIMACS
    challenge).
  - `bench/export_mtx.jl` and `bench/export_more.jl` write the synthetic grids used for the ROME comparison to
    `data/mtx/`.
  - `bench/export_real.jl` fetches SNAP and DIMACS10 graphs.
- **Undirected graphs only for now.** Directed graphs whose strongly connected components reach one another are
  not supported on the GPU, and are rejected with an error.
- **Precision.** Float32 represents integers exactly only up to 2²⁴ ≈ 16.7M. When distances can exceed that
  (continental road networks: USA distances reach 6.45 × 10⁷), use Int32 (same speed on the GPU) or Float64; see
  "Hardening".

## API

### All pairs in one call: `apsp_gpu`

```julia
using SemiringGPU, CUDA
D = apsp_gpu(A)                          # A: n × n SparseMatrixCSC{Float32}, A[i, j] = weight of the arc i → j;
                                         # D[i, j] = A*[i, j] = distance i → j, a CuMatrix in the labels of A
H = apsp_gpu(A; output = :host)          # the same, as a Matrix
E, cols = apsp_gpu(A; columns = :elimination)   # E[i, k] = A*[i, cols[k]]: skips the final relabel pass
X = apsp_gpu(A, sources)                 # k × n: X[t, :] = A*[sources[t], :] (any order, repeats allowed)

using SemiringGPU.Semiring: MinPlus, MaxPlus, MaxMin, PlusProd
B = apsp_gpu(A; semiring = MaxMin())     # widest paths; any semiring of CliqueTrees.Multifrontal.Semiring

devs = collect(CUDA.devices())
H = apsp_gpu(A; devices = devs, output = :host)   # rows split over the GPUs, assembled on the host
blocks = apsp_gpu(A; devices = devs)              # [(sources_g, D_g)]: D_g[t, j] = A*[sources_g[t], j] on devs[g]

with_config(() -> apsp_gpu(A); merge = 1)         # solver settings (see below)
```

- **What a call does:** the whole pipeline with the tuned settings of `bench/portable.jl`: symbolic phase on
  the CPU, hybrid numeric factorization, and the closure. It then relabels the result from elimination order
  in place on the GPU, through a buffer of at most 5% of free memory and 1 GiB, and frees its workspace and
  the factor before returning.
- **Memory:** n² elements on one GPU (about n²/g each with g devices), plus the factor. `:host` also needs n²
  on the host. If the result can't fit, the call fails before doing any work and suggests more devices or
  blocks of `sources`.
- **Input:** Float32 or Float64 weights for min-plus and max-plus (Int32 and Int64 are rejected because their
  infinity overflows), Float64 for plus-times. `ArgumentError`s cover a non-square `A`, an unsupported element
  type or semiring, and a directed graph whose strongly connected components reach one another.

### Lower level: keep the factorization

Use this for repeated solves on one graph, such as new sources or new weights on the same pattern. Results of
the closure are in elimination coordinates.

```julia
F = ChordalSLU(MinPlus(), A); copyto!(F, A)        # symbolic phase; F holds the entries of A
P = FactorPlan(F; large = 256, graph = false, nstreams = 8)
factorize!(P)                                      # CPU bottom subtrees ∥ GPU top fronts
G = GPUSLU(P; large = 8192)                        # the factor and the level schedule on the GPU
precompute_ops!(G)                                 # fold the large fronts' triangular solves into one operator each
closure_gpu!(D, G; M)                              # all of A*, n × n, in elimination coordinates (M: n × G.maxna):
                                                   #   D[i, j] = A*[p[i], p[j]], p = F.rperm (= Array(G.rperm))
copyto!(F, A₂); factorize!(P)                      # new weights, same pattern (with graph = true: CUDA graph replay)

sssp_gpu!(X, G, sources)                           # rows of A* for sources::CuVector, in the labels of A (fast path:
                                                   #   U sweep along root paths)
S = SSSPPlan(G, k); S(sources)                     # the same, recorded as a CUDA graph
rmul_gpu!(B, G)                                    # B ← B A*, B is k × n on the GPU (row layout)
D = closure_gpu(G)                                 # closure_gpu! with its own D and M
MG = MultiGPUSLU(P; devices)                       # a copy of the factor on every device
closure_multigpu!(MG)                              # [(rows_g, D_g)]: D_g = D[rows_g, :] (elimination coordinates)

F = mlu(MinPlus(), A); G = GPUSLU(F)               # or: CPU factorization (upstream), then upload
sgetrf_gpu!(MinPlus(), A_gpu)                      # dense semiring LU on the GPU
```

### Settings

Every tunable choice lives in `GPUConfig` (`?GPUConfig` lists them and their defaults). `with_config(f; kw...)`
changes settings only for the code inside `f`, and `SEMIRINGGPU_<SETTING>` environment variables set the
process defaults:

```julia
with_config(merge = 1, skip_fill = false) do      # e.g. no solve amalgamation, fill the result first
    apsp_gpu(A)
end
```

```
SEMIRINGGPU_TUNE=off julia --project=. script.jl  # no GEMM autotuning (heuristic kernel choice)
SEMIRINGGPU_ORDERING=amf julia --project=. script.jl   # fill-reducing ordering: auto (default), amf, hub, bfsnd, amd, metis
```

Host threads: start Julia with about half the cores, e.g. `julia -t 8` on a 16-core allocation. The host
phases (ordering, symbolic analysis, the CPU part of the factorization) use the Julia threads, and with one
per core the CUDA driver's and CUDA.jl's synchronization threads find no free core: on HPG's 16-core
allocations, `-t 16` stalled a third of the calls on small graphs by 10–15 ms, while `-t 8` was as fast or
faster on all 30 benchmark graphs.

## How it works

- **Solve.** Level schedule over the elimination tree: the U sweep goes leaves to root, the L sweep root to
  leaves.
  - Each level is one batched launch with one block per (front, chunk of right-hand sides) and one thread per
    right-hand side. The row layout (each vertex's k values contiguous) keeps accesses coalesced.
  - Sibling scatters into ancestors use a generic atomic ⊕: a compare-and-swap loop around `splus`.
  - Large fronts use dense kernels: a blocked TRSM with diagonal-block inversion, plus the semiring GEMM. With
    `precompute_ops!`, the operators K_L = [L₁₁*; L₂₁L₁₁*] and K_U = [U₁₁* | U₁₁*U₁₂] are formed once, so each
    large front costs a gather and one GEMM.
  - For unit-vector sources, the U sweep walks only each source's root path. This is the paper's fast path, and
    the upward search of GPHAST and contraction hierarchies. One warp per source splits each front's separator
    update over its 32 lanes.
  - Below the top of the tree, with many rows (the closure), the L sweep runs as a **layered walk**
    (`src/layered.jl`). The fronts are cut into layers of subtree regions of about √(2 nf) fronts, and each
    (row, region) pair gets a thread that walks its region parents first, one launch per layer (2–3 layers).
    Rows are independent and so are disjoint subtrees, so this needs no synchronization. Each thread walks
    O(√nf) fronts instead of nf, and there are k × (number of regions) threads instead of k.
  - **Supernode amalgamation for the solve** (`src/amalgamate.jl`): chains of small fronts, each the last child
    of the next, are merged into fronts of up to 8 residual vertices, padded with the semiring zero (exact). It
    halves the number of levels and cuts fronts by ~30%. The merge is computed once per symbolic factorization
    as index maps, and each numeric factor is rearranged by a GPU gather, so it costs 0–3 ms.
  - Sibling scatters use native reductions (`RED.MIN`/`MAX`/`ADD`) when ⊕ is min, max or + on a machine type.
    Floats use the sign split: signed-integer min of the bits for x ≥ 0, unsigned max for x < 0, which is exact.
    Other semirings keep the compare-and-swap loop.
  - **Fused large fronts:** the GEMMs read the separator columns and accumulate into them through index views
    (no gather or scatter buffers), and an overwrite epilogue drops the fills. The residual block is updated in
    place when it is one tile wide; that is race-free because each block reads all of its rows before writing.
  - An earlier persistent launch (tickets in depth order, per-front completion counters) is still available but
    slower.
- **Factorization.** The top of the tree (fronts with nn+na ≥ `large` and all their ancestors) goes to the GPU.
  - The bottom forest is split into independent subtrees, factored on CPU threads with upstream's `sgetrf_loop!`.
  - Boundary updates are uploaded. The top is factored front by front (extend-add, dense LU, both TRSMs, Schur
    GEMM), with the fronts of each level spread over 8 CUDA streams, and captured as a CUDA graph for
    refactorization.
- **Closure.** All n sources in one block, with the n×n result as its own work matrix: n root-path walks, then
  the top and L sweeps with n rows.

## Results

### Full APSP closure vs ROME

ROME is built from https://github.com/LyleLuo/ROME with `external/rome_gatas.patch`.
- **Weights:** stock ROME ignores the file's weights and uses `rand()%10+1`; the patch makes it read them.
- **Inputs and correctness:** both solvers get the same graphs and weights, and the distances match exactly
  (checked on grid3d-25, 46,875 entries).
- **How we time:** "ours" is the GPU closure (`closure_gpu!`, with `precompute_ops!`, solve threshold 8192),
  after a warm-up run. Julia's JIT time is excluded; ROME is compiled ahead of time. Totals add our symbolic and
  numeric phases. ROME's compute time is the one it prints, which excludes its ordering, allocation and download.

GPU compute time (s), 2026-10-02 (`jobs/final.sbatch`; this round's kernels, all defaults):

| graph | n | RTX PRO 6000: ours | ROME | ratio | B200: ours | ROME | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| grid3d-25 | 15,625 | **0.021** | 0.022 | 1.07× | **0.022** | 0.033 | 1.5× |
| grid2d-150 | 22,500 | **0.014** | 0.016 | 1.13× | **0.012** | 0.022 | 1.8× |
| grid3d-30 | 27,000 | **0.071** | 0.108 | 1.5× | **0.059** | 0.163 | 2.8× |
| grid2d-180 | 32,400 | **0.023** | 0.032 | 1.4× | **0.019** | 0.044 | 2.3× |
| grid2d-250 | 62,500 | **0.076** | 0.144 | 1.9× | **0.060** | 0.220 | 3.7× |
| grid3d-40 | 64,000 | **0.407** | 0.942 | 2.3× | **0.358** | 1.42 | 4.0× |

Before this round (2026-10-01), ours was 0.062 / 0.034 / 0.123 / 0.043 / 0.130 / 0.586 s (RTX PRO 6000) and
0.045 / 0.025 / 0.106 / 0.035 / 0.103 / 0.491 s (B200), so ROME won the four smallest on RTX PRO 6000 and the
two smallest on B200. Earlier, larger runs (old kernels): grid2d-380 0.611 vs 1.14, grid3d-52 3.18 vs 7.02
(RTX PRO 6000); grid3d-58 5.62 vs 26.0, grid2d-450 1.02 vs did not complete (B200).

**Caveat, numeric factorization.** The table compares our closure with ROME's "computing", and ROME's computing
includes all of its numeric work. Ours also needs the numeric LU first (hybrid CPU + GPU; 0.032 s on grid3d-25 and
0.058 s on grid3d-30 on the RTX PRO 6000 node, 0.052 / 0.084 s on the B200 node). Numeric + closure vs ROME:
RTX PRO 6000 0.053 vs 0.022, 0.024 vs 0.016, 0.129 vs 0.108, 0.037 vs 0.032, **0.106 vs 0.144**, **0.579 vs 0.942**;
B200 0.074 vs 0.033, 0.026 vs 0.022, **0.143 vs 0.163**, **0.040 vs 0.044**, **0.103 vs 0.220**, **0.587 vs 1.42**.
So with the factorization included, ROME still wins below about 30k vertices on the RTX PRO 6000 and below about
25k on the B200. The factorization is now the bottleneck there; its CPU/GPU split threshold (`large = 256`) is
already the best of 16…∞ (laptop grid3d-25: 22 ms at 256, 68 ms all-CPU).

Our totals (symbolic + numeric + closure) on B200 range from 0.053 s (grid2d-150) to 7.1 s (grid3d-58).

On the laptop (`bench/results_closure_laptop.txt`), end to end we're 1.05–1.76× faster than ROME on the 4 grids
it can hold.

- **We do less work:** ROME's measured work divided by our n·nnz(L) is 5.5× (grid3d-25), 7.7× (grid3d-30),
  10.2× (grid2d-150) and 10.9× (grid2d-180), and it grows with n.
- **Our efficiency is lower:** per operation, we run at 0.3–1.2 T min-plus ops/s against ROME's 5–10 T/s, so 5–22×
  less efficient, and worst on small graphs.
- **Why the small graphs are slow for us:** the time there is a latency floor of tens of milliseconds (per-level
  launches, root-path walks), not arithmetic.
- **The memory wall:** full APSP needs n² entries (about 90 GB at 150k vertices, 160 GB at 200k), which limits
  every APSP code, ROME included. Beyond that, stream the closure in row blocks (`sssp_gpu!`) or answer queries.

### Blocked queries, k = 256, MinPlus (ms per query)

Laptop (`bench/results_solve_v2_rtx5060_laptop.txt`):

| graph | n | CPU best (16 thr) | GPU, same algorithm (`rmul_gpu!`) | GPU fast path (`sssp_gpu!`) |
|---|---:|---:|---:|---:|
| USA-road-t.NY | 264k | 1.139 | 0.086 (13×) | 0.052 (22×) |
| roadNet-PA | 1.09M | 4.698 | 0.366 (13×) | 0.249 (19×) |
| USA-road-t.FLA | 1.07M | 4.371 | 0.264 (17×) | 0.142 (31×) |
| grid2d-500 | 250k | 1.282 | 0.363 (3.5×) | 0.326 (3.9×) |

RTX PRO 6000, against 16 CPU threads on the same node:

| graph | n | k | CPU | GPU (`SSSPPlan`) | speedup |
|---|---:|---:|---:|---:|---:|
| USA-road-t.FLA | 1.07M | 256 | 2.65 | 0.042 | 63× |
| USA-road-t.USA | 23.9M | 16 | 130.4 | 7.18 | 18× |
| USA-road-t.USA | 23.9M | 64 | 70.2 | 1.97 | 36× |
| USA-road-t.USA | 23.9M | 256 | 62.7 | 0.91 | 69× |

The full USA network factors in 27 s on 16 CPU threads, and uploading the factor takes 0.6 s.

### Single queries, k = 1 (ms; median of 100 sources)

| graph | CPU fast path, 1 thr | **CPU fast path, 16 thr** | GPU (`SSSPPlan`) |
|---|---:|---:|---:|
| USA-road-t.NY | 3.24 | **0.83** | 3.18 |
| USA-road-t.FLA | 13.2 | **3.27** | 6.87 |
| roadNet-PA | 14.8 | **8.24** | 30.6 |
| grid2d-500 | 5.50 | **2.23** | 84.9 |
| USA-road-t.USA (RTX PRO 6000 node) | 326 | **42.8** | 109 |

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

## Kernel changes (2026-10-02)

Measured with per-kernel GPU times (`bench/prof_closure.jl`, CUDA.jl's CUPTI profiler; the synced phase timer of
`bench_closure.jl` inflates small phases, e.g. it shows ~15 ms for a 1.5 ms root-path walk).

1. **Layered L sweep** (`src/layered.jl`). The row-major walk (one thread per row through every front) was at
   the laptop's DRAM peak but a 2× regression on the big GPUs, where 22k rows fill ~7% of a B200's thread slots
   and each row is a serial chain of ~nf dependent gathers. The layered walk restores parallelism: B200 grid2d-150
   15 → 12 ms, grid2d-250 87 → 64 ms (vs the old level schedule).
2. **Amalgamation** on the GPU (`src/amalgamate.jl`): 5–12% on the closure, with no download of the factor.
3. **Native atomic ⊕:** the top-of-tree U kernel is 25% faster.
4. **Fused large fronts:** the closure is 14–15% faster on the laptop (grid3d-25 100.9 → 86.7 ms, grid3d-30
   390.6 → 332.0 ms), and 8–10% on HPG.
5. **Dead ends (measured, removed):**
   - a per-thread shared-memory row cache for subtree clusters: 2.5–15× slower, because the shared memory per
     thread caps occupancy and takes L1 away from the hardware cache;
   - launching the row-major sweep in row chunks for more L2 per row: slower, since it is latency-bound per row;
   - GEMM v3 with the C++ port's ideas (fully unrolled k-panels, BK = 16, paired k-steps for FMNMX3): 255
     registers and spills, slower than v2 even without spills (1874 vs 2157 G/s; sm_120 has no FMNMX3 anyway).
   - A register cap of 48 for the row-major kernel (one wave of rows on sm_120) is kept: small, within noise.

## Kernel changes (2026-10-01)

1. **Min-plus is 2 instructions per multiply-add again, not 5.**
   - The cause: upstream's scalar `vmin` is `ifelse(x < y, x, y)`, picked by an `@static if X86` host check that
     the GPU compilation inherits. ptxas turns it into FSETP + FSEL.
   - The fix: a GPU-only `CUDA.@device_override` maps `vmin`/`vmax` on floats to `llvm.minnum`/`maxnum`
     (`FMNMX`), and the CPU is unchanged.
   - Semantics: the same as upstream's whenever the accumulator isn't NaN (+∞ + −∞ is still absorbed).
2. **Pipelined GEMM v2** (cuASR/CUTLASS and TropicalGEMM ideas):
   - double-buffered shared memory with register prefetch, so there is one barrier per k-panel;
   - 4-contiguous fragments, and a padded n-contiguous B tile;
   - skinny tiles 256×16 and 128×32.

   Result: min-plus F32 reaches 2.26 T/s on the laptop (the 2-instruction ceiling, about 51% of cuBLAS SGEMM), and
   is 1.3–2.2× faster than v1 on the solver's skinny and short-k shapes. On HiPerGator at n = 4096 it reaches
   6.35 T/s (B200) and 9.65 T/s (RTX PRO 6000).
3. **Removed a stale duplicate block in `sgetrs.jl`.** Because it came later in the file, its older definitions
   overrode the new ones, so the inversion-based TRSM had never run. It is active now.
4. **Precomputed per-front operators** (`precompute_ops!`, the "partitioned inverse"): 5–15% on the dense phases.
5. **Warp-cooperative path walk:** about 30% faster on that phase.
6. **Persistent L sweep:** one launch below the top of the tree, using dependency counters and ordered tickets
   (the sync-free SpTRSV idea). See below for its status.
7. **No emulated 64-bit division in hot code.** Helper kernels use 2D launches, and the persistent kernel uses
   32-bit unchecked division. That removed every CALL and spill flagged by the SASS guard.
8. **Loud failures:**
   - directed graphs with coupled components;
   - element types that aren't 4 or 8 bytes (the atomic ⊕ would silently corrupt them);
   - closures or workspaces that don't fit in GPU memory, reported with the sizes.

## Hardening

| check | result |
|---|---|
| `test/stress.jl`: random graphs vs an independent Floyd–Warshall–Kleene oracle (shares no code with the solver). Covers trees, paths, stars, grids, dense and disconnected graphs; zero, negative and negative-cycle weights; duplicates and self-loops; MinPlus F32/F64/I32, MaxPlus, MaxMin and PlusProd; and every GPU path and switch (GEMM v1/v2, persistent on/off, warp on/off, ops on/off, CPU or hybrid factorization with 1 or 8 streams and graph replay, thresholds 1…∞, k = 1…100; since 2026-10-02 also the row-major / layered L sweep at any k, region sizes 1…32, amalgamation widths 1…8 and fused / buffered large fronts) | **3,000 cases, 0 failures** (2026-10-02, laptop; plus 1,500 cases, 0 failures on the RTX PRO 6000), after fixing the two bugs it found: region size 1 left no region, and the amalgamation cache was keyed by array contents (`WeakKeyDict` uses `isequal`), so two factorizations with equal separator arrays shared a merge (illegal address). The cache is now identity-keyed and checks the structure on every hit. Earlier: 890 cases, 0 failures. |
| `test/stress_large.jl`: 2k–30k vertices vs a binary-heap Dijkstra / widest-path oracle, plus the same solve repeated and required bit-identical | **30 cases, 0 failures, deterministic** (re-run 2026-10-02 with the new switches randomized: 30 laptop + 60 RTX PRO 6000 cases, 0 failures) |
| `compute-sanitizer` on HPG (RTX PRO 6000): memcheck, racecheck, synccheck and initcheck over `test/sanitize_small.jl` | **0 errors and 0 hazards in all four** |
| `test/sass_guard.jl`: no device function calls, no local-memory spills, `FMNMX` present in the hot kernels | **all clean** |
| USA-road-t.USA, k = 256, Float32: GPU ≠ CPU (`bench/diag_usa.jl`) | **Float32 rounding, not a bug.** Every mismatch is a distance above 2²⁴, off by at most 16 (relative error 4.7 × 10⁻⁷). The CPU's own Float32 run is also inexact vs Float64. **In Float64, GPU = CPU on all 6.1 × 10⁹ entries** (so there is no index overflow at k·n > 2³¹). |

## Status of the kernels: what is still weak

1. **Per-operation efficiency is 5–22× below ROME's**, mostly from latency and memory traffic, not arithmetic.
   - Small fronts get one launch per tree level and a short block each.
   - The batched sweep is memory-bound, at about 2.5–4.5× above its own bandwidth floor.
   - The dense fronts' GEMMs are skinny.
2. **The persistent L sweep is not yet a win.** It was slower on the laptop (69 vs 54 ms on grid3d-25, 98 vs
   59 ms on grid2d-150) and hasn't been measured on the big GPUs. Its tickets wait on whole parent fronts.
   Per-(front, chunk) flags would fix that.
3. **Next steps:**
   - subtree blocking: keep a subtree's columns in shared memory or L2 for a block of rows;
   - relaxed supernode amalgamation (fatter fronts);
   - ROME-style batched work lists for the dense fronts of a level;
   - a dense blocked closure for high-fill graphs, chosen by the fill ratio;
   - Nsight Compute rooflines on HiPerGator;
   - Int32 / DPX on B200.
4. **Not yet benchmarked:** large-separator graphs, ROME's and SuperFW's own test sets, n × GPHAST, and an
   all-cores CPU node (`bench/bench_cpu.jl` is ready).

## Upstream issues found

- **The tropical `szero`/`sone` methods have an unbounded `T`.** They are ambiguous with the generic
  `Vec{W,T}` lift, which crashes `lmul!` and Int32 `mlu` on AVX-512 CPUs. The first commit on this branch fixes it
  with `T <: Real` in `src/semiring/tropical.jl`.
- **`szero(s, a::T, op)` recurses forever (stack overflow)** when no specific method exists, e.g. for `T = Any`.
  It should throw a `MethodError`.
- **Upstream's `vmin` / `vmax`** use an `@static if X86` host check that also decides the GPU code; see kernel
  change 1.
- **Row-layout `rmul!` doesn't get faster with threads.**
- **The single-query fast path and subtree parallelism live in `tmp/`,** not in `sgetrs!`/`rmul!`.
- **The numeric factorization processes fronts in sequence.** Subtree parallelism (`factorize_cpu!` here) gives
  1.2–1.8×.
- **`CuVector(::FixedSizeArray)` is about 40× slower than a raw copy;** use `unsafe_copyto!`.
