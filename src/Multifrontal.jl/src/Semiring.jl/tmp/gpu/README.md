# SemiringGPU

GPU experiments for `CliqueTrees.Multifrontal.Semiring`, the solver behind the APSP-LU paper.

`src/SemiringGPU.jl` defines `sgemx_gpu!(s, C, A, B)`, which computes C ← C ⊕ A ⊗ B in semiring `s`. It has the
same semantics as the CPU `Semiring.sgemx!` and is the GEMM line of Algorithm 2 (`M = F22 + L21 U12`). The kernel
is generic: it calls only `smuladd`, `splus`, `szero`, and `sone`. Any semiring whose operations compile for the
GPU works without changes.

```
julia --project=. -t auto bench/bench_gemm.jl
```

## Kernel

- 128×128 tiles (64×64 for small outputs), a k-panel depth of 8, and 256 threads, each holding an 8×8 (or 4×4)
  register tile.
- Out-of-range entries are padded with 0 in A and 1 in B. 0 ⊗ 1 = 0 holds exactly, with no Int overflow, and ⊕
  absorbs it.
- The register-tile update and the epilogue are unrolled with `@generated`, the same way as the CPU
  `sgemx_col_kernel!`.

## Pitfalls found (each cost about 10×)

1. **A closure that reassigns `acc` boxes it.** The kernel then fails to compile (`jl_f_tuple`, dynamic dispatch).
2. **`splus` was not inlined**, so every multiply-add became a device function call. The fix is `@inline` on
   `splus`/`sprod`/`smuladd`/`szero`/`sone` in CliqueTrees `semiring/{tropical,real,bottleneck,boolean,semiring}.jl`
   (old branch; upstream main now has these `@inline`s itself).
3. **A 64-term `ntuple` closure was not inlined.** `@generated` unrolling fixed it.
4. **A runtime-indexed tuple in the epilogue spilled to local memory** (`STL`). Unrolling fixed it.

Check SASS with `CUDA.@device_code_sass`. A healthy inner loop is TM·TN `FMNMX.NAN` + `FADD` (min-plus F32), with
no `CALL` and no `STL`.

## Results: RTX 5060 Laptop (sm_120, 50 W), CUDA.jl 6.1

See `bench/results_rtx5060_laptop.txt`. G multiply-adds/s:

| n    | MinPlus F32 | MaxMin F32 | MinPlus I32 | MinPlus F64 | PlusProd F32 | cuBLAS SGEMM |
|------|------------:|-----------:|------------:|------------:|-------------:|-------------:|
| 1024 | 2235        | 2176       | 2191        | 25          | 2625         | 4726         |
| 2048 | 2211        | 2184       | 2326        | 30          | 2698         | 4282         |

- MinPlus F32 reaches about 48% of cuBLAS SGEMM. Min-plus needs 2 instructions (FADD + FMNMX) where FMA needs 1,
  so this is near the naive ceiling. If FMNMX issues on the ALU pipe alongside FADD, the real ceiling is higher.
- Use Float32/Int32 on consumer GPUs: Float64 runs at 1/64 rate, plus NaN emulation in `min`.
- n = 4096 and the last frontal shapes drop because the laptop throttles thermally (cuBLAS drops too).
- The CPU column is noisy on the laptop. Compare against HiPerGator EPYC numbers, not these.
- CuTropicalGEMM.jl does not work here: `TropicalGemmC_jll` has no binary for CUDA 13, and it pins CUDA.jl to 5.x.

## Solve phase on the GPU: `rmul_gpu!(B, GPUSLU(F))`

`src/sgetrs.jl` is the GPU version of `rmul!(B, F)`, which computes X = B A* = B U* L*: k single-source queries in
the row layout, with B of size k × n. Factorization stays on the CPU (`mlu`), and `GPUSLU(F)` uploads the factor
once.

- **Level schedule over the elimination tree.** The U sweep goes leaves to root by height; the L sweep goes root
  to leaves by depth. Each level is one batched launch for its small fronts, with one block per (front, chunk of
  ≤64 right-hand sides) and one thread per right-hand side, so access is coalesced in the row layout.
- **U sweep scatter** into separator columns uses a generic atomic ⊕: a CAS loop around `splus`, for any 4- or
  8-byte element type. The L sweep only gathers from finished ancestors, so it needs no atomics.
- **Large fronts** (nn·(nn+na) ≥ `large`, default 2048) use dense kernels: a blocked TRSM `strsx_gpu!` (64-column
  diagonal-block kernel, with `sgemx_gpu!` for the trailing update), `sgemx_gpu!` for the separator update, and
  gather/scatter kernels. This took grid2d-500 from 1.25 to 0.33 ms/query.
- **Not supported yet:** `trans` other than `:N` (the residuated `ldiv!`/`rdiv!`); BTF coupling between strongly
  connected components (undirected graphs have none); and the column layout (`lmul!`).
- `test/test_solve.jl` checks the result against the CPU `rmul!` for MinPlus F32/F64, MaxMin, and PlusProd, at
  k = 1, 7, 64, 100 and every threshold (all-batched, mixed, all-dense).

`bench/bench_solve.jl` results (`bench/results_solve_rtx5060_laptop.txt`), MinPlus Float32, ms per query, at
k = 256. The CPU column is the best of `rmul!` and `lmul!` at 1 or 16 threads; B stays resident on the GPU:

| graph          | n     | fill | CPU 16 thr | GPU   | speedup | D→H pinned per query |
|----------------|-------|-----:|-----------:|------:|--------:|---------------------:|
| USA-road-t.NY  | 264k  | 3.5  | 0.991      | 0.089 | 11.1×   | 0.073                |
| roadNet-PA     | 1.09M | 4.2  | 4.387      | 0.368 | 11.9×   | 0.307                |
| USA-road-t.FLA | 1.07M | 2.6  | 4.128      | 0.283 | 14.6×   | 0.302                |
| grid2d-500     | 250k  | 17.7 | 1.280      | 0.366 | 3.5×    | 0.070                |

At k = 1 the GPU is 0.9–1.9× on the road networks and loses on the grid (0.3×), as expected for latency-bound
single queries.

Caveats:
- **This laptop CPU looks slow, so the speedups above are probably inflated.** Single-core `rmul!` on FLA is
  6.6 ms/query here, against 2.34 for the paper's EPYC run (F64). Row-layout `rmul!` on current upstream doesn't
  scale with threads (1.0×, sometimes slower). The paper's 16-core EPYC region is about 2.34/2.4 ≈ 1 ms/query on
  FLA, so the GPU is more like 3–4× ahead of that.
- **Copying results back to the host costs about as much as the solve** at k = 256, even with pinned memory. The
  GPU pays off when the results are consumed on the device (reductions, centrality, the next solve).

Upstream issues found:
- **The tropical `szero`/`sone` methods had an unbounded `T`.** That made them ambiguous with the generic
  `Vec{W,T}` lift, which broke `lmul!` and Int32 `mlu` on this AVX-512 laptop. Fixed on branch `gpu-semiring` with
  `T <: Real`.
- **`rmul!` thread scaling** is the issue described in the first caveat.

## v2: path-walk U sweep and CUDA graphs (`sssp_gpu!`, `SSSPPlan`)

Phase profile of `rmul_gpu!` (`bench/profile_solve.jl`, CUDA events): on FLA at k=256, 50% of the time is the U
sweep, 29% the L sweep and 20% the permutations. On the grid, 80% is the dense path for large fronts, and it costs
the same at k=1 and k=256, so it is latency, not work.

- **`sssp_gpu!(X, G, sources)`:** the sources are unit vectors, so B·U* is nonzero only on each source's
  leaf-to-root path. This is the paper's fast path, and the upward search of GPHAST and contraction hierarchies.
  - One thread per source walks its own path and owns its row, so there are no atomics.
  - The walk stops at the "top" of the tree (large fronts and their ancestors), which is swept level by level for
    all rows.
  - It also skips the input permutation: W is filled with 0 and a 1 is placed at each source.
- **`SSSPPlan(G, k)`:** captures `sssp_gpu!` as a CUDA graph and replays it with new sources. This gave almost
  nothing (≤8%), because host launch overhead was not the bottleneck.

Results: `bench/results_solve_v2_rtx5060_laptop.txt`, ms/query, at k=256 (k=1 in parentheses):

| graph | CPU best | rmul_gpu! | sssp_gpu! / plan | vs CPU |
|---|---:|---:|---:|---:|
| USA-road-t.NY | 1.139 | 0.086 | 0.052 | 22× (3.0×) |
| roadNet-PA | 4.698 | 0.366 | 0.249 | 18× (1.6×) |
| USA-road-t.FLA | 4.371 | 0.264 | 0.142 | 31× (4.7×) |
| grid2d-500 | 1.282 | 0.363 | 0.326 | 3.9× (0.3×) |

**The CPU baseline doesn't use the fast path, which inflates these ratios.** It solves with a dense B; upstream's
CPU fast path is only in `tmp/sssp_tmp.jl`. The like-for-like algorithm comparison is the rmul_gpu! column (11–17×
at k=256). The paper reports that the fast path saves up to half of a solve on the CPU too.

## v3: GPU factorization (`sgetrf_gpu!`, `FactorPlan`, `mlu_gpu`) and comparisons with others

`src/sgetrf.jl`.

- **Dense semiring LU on the GPU** (`sgetrf_gpu!(s, A)`): right-looking, with 64-wide panels.
  - Each 64×64 diagonal block is factored in shared memory.
  - The triangular solves use **diagonal-block inversion**: T = A[J,J]* is formed by solving with the identity,
    and the block solve is then a GEMM. This only paid off once launch latency was addressed (see below).
  - The Schur update is `sgemx_gpu!`.
- **Hybrid sparse factorization:**
  - The "top" (fronts with nn+na ≥ `large` and all their ancestors) goes to the GPU. The bottom forest is split
    into independent subtrees, factored on CPU threads with upstream's own `sgetrf_loop!`, each thread with its
    own update stack.
  - Boundary updates are uploaded, and the top is factored front by front: extend-add, LU, both TRSMs, GEMM.
  - With `nstreams = 8`, the fronts of each top level run concurrently on 8 streams with event barriers; each top
    front gets its own update slot.
- **`FactorPlan` + CUDA graph:** the GPU part is captured after the first factorization. Refactorization (new
  weights, same pattern) is: CPU bottom → upload into persistent buffers → graph replay.
- **Exactness:** the factor matches the CPU bit for bit (MinPlus F32/F64 with integer weights, MaxMin) and to
  ~1e-15 (PlusProd). Also checked: solves, graph replays after weight changes, and the subtree-parallel CPU
  reference `factorize_cpu!` (`test/test_factor.jl`).

How the GPU part got fast (grid3d-40, GPU part of the factorization):

| step | ms |
|---|---:|
| single stream, direct launches | 92–110 |
| plus CUDA graph replay | 91 (host launch cost gone, but the GPU executes ~40 tiny kernels per front serially) |
| plus 8 streams, level-concurrent graph | 72–75 |
| plus subtree-parallel CPU bottom (was 58 ms) | 6.5 (CPU part) |
| fast uploads (`unsafe_copyto!`; `CuVector(::FixedSizeArray)` was ~40× slower) | 15 (was 50–130) |

Pitfalls in the multi-stream graph:
- **CUDA.jl's implicit cross-stream synchronization** isn't allowed inside capture. It's disabled on the plan's
  buffers with `CUDA.enable_synchronization!(x, false)`, which is safe because the plan orders all cross-stream
  access with events.
- **TRSM scratch has to be per stream.**

Results: `bench/results_factor_v2.txt`, numeric factorization only, same AMF ordering for ours.

MinPlus Float32:

| graph | upstream CPU | ours CPU ∥ | hybrid refac | vs upstream | vs ours CPU ∥ |
|---|---:|---:|---:|---:|---:|
| USA-road-t.NY | 25.2 | 12.3 | 13.6 | 1.9× | 0.9× |
| grid2d-500 | 152.9 | 89.2 | 47.5 | 3.2× | 1.9× |
| grid2d-1000 | 720.9 | 507.9 | 183.6 | 3.9× | 2.8× |
| grid3d-30 | 155.8 | 129.8 | 33.2 | 4.7× | 3.9× |
| grid3d-40 | 561.1 | 512.8 | 96.9 | 5.8× | 5.3× |

Real arithmetic, M = I − W (ms). Each library uses its own ordering:

| graph | upstream F64 | ours CPU∥ F64 | ours GPU F64 | ours GPU F32 | UMFPACK | CHOLMOD | cuDSS LU F64 | cuDSS LU F32 | cuDSS LLᵀ F64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| USA-road-t.NY | 24.9 | 11.3 | 12.3 | 10.1 | 203.7 | 33.7 | 7.8 | 6.6 | 6.7 |
| grid2d-500 | 173.2 | 108.3 | 95.1 | 47.2 | 295.1 | 83.6 | 36.2 | 21.9 | 25.4 |
| grid2d-1000 | 865.9 | 653.6 | 431.9 | 183.3 | 2083.1 | 565.5 | 221.6 | 94.3 | 131.7 |
| grid3d-30 | 193.9 | 154.4 | 86.5 | 32.6 | 289.3 | 53.9 | 44.4 | 16.6 | 25.9 |
| grid3d-40 | 797.8 | 689.5 | 369.0 | 93.4 | 1231.3 | 314.2 | 208.6 | 59.6 | 113.8 |

- **Ours vs cuDSS:** cuDSS (NVIDIA's real-arithmetic sparse direct solver) is still 1.5–2.2× faster in F32. Our
  kernels are generic over semirings.
- **Ours vs the CPU libraries:** our GPU F32 beats CHOLMOD everywhere and UMFPACK by 6–20×.
- **Dense LU, n=4096:** ours 697 G/s (real F32) against cuSOLVER's 1331 and LAPACK's 293. MinPlus: 649 G/s on the
  GPU against 113 G/s on the CPU.

### APSP (full closure) against ROME (PPoPP'26)

Built locally in `external/` (see `ROME_NOTES.md`; it needed a reader patch to use the file's weights instead of
`rand()%10+1`). Ours = symbolic + hybrid factorization + all-sources solves into an n×n matrix on the GPU, with no
D→H (`bench/results_apsp.txt`). Our rows match ROME's exactly (grid3d-25, 3 rows, 46,875 distances).

| graph | n | ours, GPU total | ROME GPU compute (+ordering, setup) | ours CPU closure |
|---|---:|---:|---:|---:|
| grid3d-25 | 15625 | 1.02 s | 0.12 s (0.22 s) | 3.42 s |
| grid2d-150 | 22500 | 0.28 s | 0.08 s (0.18 s) | 4.07 s |
| grid3d-30 | 27000 | 2.40 s | 0.68 s (0.83 s) | — |
| grid2d-180 | 32400 | 0.65 s | 0.21 s (0.27 s) | — |

- **ROME wins the full closure, 1.5–4.6× end to end.** Our closure is n sources pushed through the solve sweeps,
  which is bandwidth-bound and does one thread per right-hand side. ROME's is compute-bound tiled min-plus GEMM.
- **What would close the gap:** compute the closure from the factors with GEMM-based supernodal triangular
  inversion (X ← U*, then X ← X L*, paper §III-d) instead of solves.
- **Where we're ahead:** ROME needs n² on the GPU (n ≲ 45k on 8 GB), while blocked queries scale to 1M+
  vertices. Our pipeline is also generic over the semiring, and our GPU closure is 3–15× faster than our own CPU
  closure.

## Next

- Double-buffered shared memory and vectorized loads, to close the gap to the 2-instruction ceiling.
- GEMM-based closure on the GPU (supernodal triangular inversion), to compete with ROME on APSP.
- Fewer, fused kernels per front, or ROME-style batching of small fronts into one launch over a tile work list.
  The GPU part is still latency-bound (~40 kernels per front).
- Close the gap to cuDSS: better GEMM (double buffering), and fused assembly.
- L sweep (now 60% on roads): fuse the output permutation into its final writes; use visited bits instead of
  filling W (GPHAST); DFS-within-level renumbering for locality of separator gathers.
- Overlap result downloads with the next block's solve, using two streams (ROME's async pipeline).
- The residuated solve (`trans = :C`, `ldiv!`) and BTF coupling for directed graphs.
- The closure (B = I) on the GPU, compared against ROME on the same device.
- The same benchmark on HiPerGator (A100/B200 against EPYC), which is the comparison that belongs in a paper.
