# CUDA C++ backend: is C++ faster than the Julia (CUDA.jl) kernels?

These are hand-written CUDA C++ versions of the hot kernels of the semiring solver. Julia calls them with
`@ccall` on CUDA.jl's stream, and they run the **same algorithms and schedules** as `sgemx_gpu!` and
`closure_gpu!`/`sssp_gpu!`. Results are bit-identical to the Julia kernels for MinPlus and MaxMin, and
for PlusProd closures as well (PlusProd GEMM agrees to rounding). The test is `test/test_cuda_backend.jl`.

Hardware: RTX 5060 Laptop (sm_120, 26 SMs, 50 W cap, 384 GB/s, 32 MB L2). Software: nvcc 12.9;
Julia 1.12 / CUDA.jl 6.1 (ptxas 13.3). Date: 2026-10-01.

## Bottom line

- **C++ is not faster because it is C++.** Wherever a C++ kernel beats a Julia kernel here, the cause is
  a design choice that is also expressible in CUDA.jl:
  - kernels with the same design run at the same speed;
  - the ptxas version makes no difference;
  - a Julia version of the best C++ sweep kernel runs as fast as the C++ one.
- **GEMM: the C++ kernel is 1.23–1.57× faster than Julia's best tile** on compute-bound shapes (MinPlus
  ~3.6 T vs ~2.7 T mul-adds/s). It is 1.0–1.05× on the skinny, memory-bound shapes and 0.95–0.99× when
  the leading dimension is odd (no 128-bit loads possible).
  - Against Julia's *default* tile choice it is 1.3–2.9× (`choose_tiling` picks 128×128 for n = 64).
  - PlusProd C++ reaches cuBLAS SGEMM: 0.97–1.06× on squares, faster on skinny shapes.
  - The causes are the mainloop structure and the register budget (see Analysis), not the compiler.
- **Closure (same schedule, bit-identical output): C++ is 1.06–1.32× faster end to end.**
  - A line-by-line port of the Julia sweep kernels is at parity: 0.98–1.11× on the level-by-level
    schedule. On the persistent schedule it is 0.89–1.02×, a compiler-heuristic effect explained below.
  - The gain comes from the GEMM in the dense fronts (12–45% less time there) and from the
    register-blocked L kernel ("reg").
  - The same register-blocked kernel written in Julia is just as fast as the C++ one (44–47 vs 47 ms on
    grid3d-25). So this part of the gain is available in Julia today.
- **The L sweep is DRAM-bound** at about 285–290 GB/s of 384 GB/s. Neither language can move that
  much; only less traffic can.
- **Side findings for the main code:**
  1. `choose_tiling` should prefer 64×64 / 128×64 when n ≤ 64: 2.2× on (27000, 64, 64).
  2. The register-blocked L kernel with exact-nn dispatch (no masks) and explicitly batched separator
     loads cuts L-sweep time by 8–32%, in either language.
  3. On this laptop the developer's new persistent L sweep is **slower** than the level-by-level
     schedule in almost every case, for both backends (by up to 25%).
  4. On B200 (sm_100), pairing k-steps lets ptxas emit the 3-input FMNMX3, so min-plus needs 1.5 instead
     of 2 instructions per multiply-add (up to 1.33×). This is in the C++ GEMM, verified in the SASS but
     not yet timed; the Julia GEMM could do the same.

## Results (RTX 5060 Laptop, 2026-10-01; raw logs: cuda/results_*_rtx5060_laptop.txt)

### GEMM, G multiply-adds/s (`bench/bench_cuda_backend.jl gemm`)

Measurement method:
- The laptop's SM clock drifts between 1.35 and 2.06 GHz under the 50 W cap.
- So all candidates for one shape (Julia: 4 tilings + auto; C++: 10 tilings + auto; cuBLAS) are timed in
  ~10 ms bursts, interleaved round-robin over 7 rounds after 50 ms idle, and each keeps its best burst.
- The data are L2-resident for the smaller shapes, as in the solver's repeated column blocks.

MinPlus Float32 (min-plus peak at 2.05 GHz: 26 SMs × 64 × 2.05 G ≈ 3.4 T/s):

| m × n × k | Julia auto (tile) | Julia best (tile) | C++ auto (tile) | C++ / Julia auto | C++ / Julia best |
|---|---:|---:|---:|---:|---:|
| 1024³ | 2257 (128×128) | 2604 (64×64) | **3486** (128×64×16) | 1.54× | 1.34× |
| 2048³ | 2724 (128×128) | 2722 (128×128) | **3638** (128×128×16) | 1.34× | 1.34× |
| 4096³ | 2768 (128×128) | 2704 (128×128) | **3597** (128×128×16) | 1.30× | 1.28× |
| 27000×64×64 | 928 (128×128) | 2075 (64×64) | **2603** (128×64×16) | 2.80× | 1.26× |
| 27000×20×500 | 1149 (128×32) | 1150 (128×32) | **1210** (128×32) | 1.05× | 1.05× |
| 27000×500×20 | 359 (128×128) | 622 (256×16) | **624** (128×64×16) | 1.74× | 1.04× |
| 8192×8192×64 | 1303 (128×128) | 2037 (64×64) | **2508** (128×64) | 1.92× | 1.23× |
| 2048×512×512 | 2164 (128×128) | 2569 (64×64) | **3431** (128×64×16) | 1.59× | 1.34× |
| 15625×64×64 (odd ld) | 916 (128×128) | **1870** (64×64) | 1727 (best 1846) | 1.88× | 0.99× |
| 15625×300×64 (odd ld) | 1313 (128×128) | **2061** (64×64) | 1917 (best 1975) | 1.46× | 0.96× |

PlusProd Float32, against cuBLAS SGEMM (`mul!(C, A, B, true, true)`, which does an FFMA where the
semiring kernels use FADD+FMNMX):

| m × n × k | Julia auto | Julia best | C++ auto | cuBLAS | C++ / Julia best | C++ / cuBLAS |
|---|---:|---:|---:|---:|---:|---:|
| 1024³ | 3092 | 3668 | **4947** | 4723 | 1.35× | 1.05× |
| 2048³ | 3719 | 3760 | **5894** | 5539 | 1.57× | 1.06× |
| 4096³ | 3764 | 3773 | 5618 | **5786** | 1.49× | 0.97× |
| 27000×64×64 | 1152 | 2496 | **3349** | 3150 | 1.34× | 1.06× |
| 27000×20×500 | 1177 | 1176 | **1229** | 1174 | 1.05× | 1.05× |
| 27000×500×20 | 378 | **659** | 628 | 619 | 0.95× | 1.01× |
| 8192×8192×64 | 1592 | 2327 | **2529** | 2025 | 1.09× | 1.25× |
| 2048×512×512 | 2927 | 3591 | 4835 | **4964** | 1.35× | 0.97× |
| 15625×64×64 (odd ld) | 1128 | 2135 | 2026 (best 2194) | **2819** | 0.95× (1.03×) | 0.72× |
| 15625×300×64 (odd ld) | 1588 | 2438 | 2324 | **3237** | 0.95× | 0.72× |

Notes:
- cuBLAS isn't limited by odd leading dimensions; the C++ kernel falls back to scalar loads there.
- Noise between bursts is about ±3% on the large shapes, and up to ±10% on 8192×8192×64, which is
  DRAM-bound and power-heavy.

### Closure (`bench/bench_cuda_backend.jl closure`)

Setup and measurement:
- Same factor for every backend: `FactorPlan(F; large = 256)`, `GPUSLU(P; large = 8192)`, MinPlus Float32.
- Wall time, `closure_*!` + `synchronize`, min of 3 runs.
- Every C++ result was checked bit-identical to Julia's with a per-column GPU hash; the test file checks
  full matrices on smaller graphs.
- The GPU was idle during this run. Two earlier runs, one with heavy contention from the developer's
  tests, gave the same picture within ±5%.

| graph (n) | schedule | ops | Julia (s) | C++ port (s) | C++ reg (s) | port / Julia | reg / Julia |
|---|---|---|---:|---:|---:|---:|---:|
| grid3d-25 (15,625) | levels | – | 0.1590 | 0.1470 | **0.1463** | 1.08× | 1.09× |
| | levels | ops | 0.1461 | 0.1489 | **0.1322** | 0.98× | 1.11× |
| | persistent | – | 0.1652 | 0.1640 | **0.1430** | 1.01× | 1.16× |
| | persistent | ops | 0.1570 | 0.1618 | **0.1407** | 0.97× | 1.12× |
| grid2d-150 (22,500) | levels | – | 0.1006 | 0.1014 | **0.0894** | 0.99× | 1.12× |
| | levels | ops | 0.1188 | 0.1068 | **0.0899** | 1.11× | 1.32× |
| | persistent | – | 0.1214 | 0.1270 | **0.1053** | 0.96× | 1.15× |
| | persistent | ops | 0.1172 | 0.1315 | **0.1028** | 0.89× | 1.14× |
| grid3d-30 (27,000) | levels | – | 0.5402 | 0.5133 | **0.4742** | 1.05× | 1.14× |
| | levels | ops | 0.5113 | 0.5188 | **0.4838** | 0.99× | 1.06× |
| | persistent | – | 0.5913 | 0.5802 | **0.5029** | 1.02× | 1.18× |
| | persistent | ops | 0.5743 | 0.5846 | **0.5011** | 0.98× | 1.15× |
| grid2d-180 (32,400) | levels | – | 0.1927 | 0.1905 | **0.1701** | 1.01× | 1.13× |
| | levels | ops | 0.2171 | 0.2190 | **0.1884** | 0.99× | 1.15× |
| | persistent | – | 0.2491 | 0.2749 | **0.2134** | 0.91× | 1.17× |
| | persistent | ops | 0.2462 | 0.2777 | **0.2128** | 0.89× | 1.16× |

Phase breakdown, levels schedule without ops (ms). One synchronized run per phase, so the phases add up to
more than the wall time:

| graph | | L_batched | U_top_dense | L_dense | U_path | U_top_batched | fill |
|---|---|---:|---:|---:|---:|---:|---:|
| grid3d-25 | Julia | 57.9 | 48.1 | 37.3 | 21.7 | 11.9 | 3.6 |
| | C++ port | 57.1 | 36.1 | 32.9 | 10.7 | 12.4 | 3.6 |
| | C++ reg | **46.0** | 36.4 | 32.8 | 13.6 | 11.4 | 3.5 |
| grid2d-150 | Julia | 60.9 | 5.9 | 4.7 | 19.6 | 1.5 | 6.1 |
| | C++ port | 66.1 | 4.3 | 3.3 | 18.9 | 1.5 | 6.2 |
| | C++ reg | **55.8** | 4.0 | 3.3 | 19.9 | 1.5 | 6.3 |
| grid3d-30 | Julia | 183.0 | 167.2 | 165.2 | 25.7 | 29.4 | 9.0 |
| | C++ port | 174.2 | 126.0 | 133.9 | 25.7 | 35.8 | 9.5 |
| | C++ reg | **156.9** | 124.1 | 139.4 | 25.6 | 28.4 | 9.2 |
| grid2d-180 | Julia | 173.7 | 18.6 | 16.4 | 17.2 | 1.4 | 12.4 |
| | C++ port | 174.2 | 15.0 | 11.3 | 26.8 | 1.6 | 14.4 |
| | C++ reg | **118.4** | 13.5 | 9.0 | 21.6 | 1.6 | 13.9 |

- **Dense fronts (GEMM-bound):** C++ takes 12–45% less time.
- **Batched sweeps:** port ≈ Julia (single synced runs are noisy: the wall times above are the reliable
  number), and reg takes 8–32% less time.
- **U path, one synced run each:** noisy. In isolation the path kernels are at parity:

  | | thread: Julia | thread: C++ | warp: Julia | warp: C++ |
  |---|---:|---:|---:|---:|
  | grid2d-180 | 10.1 | 11.5 / 10.9 | 7.5 | 7.1 / 6.8 |
  | grid3d-25 | 8.5 | 8.8 | 1.4 | 1.4 |

### Control experiments (`cuda/experiments/`)

| question | experiment | result |
|---|---|---|
| Is it the compiler version? (Julia: ptxas 13.3, nvcc: 12.9) | same .cu as PTX-only build JIT-compiled by the driver (CUDA 13.x) vs nvcc 12.9 SASS, interleaved (`libcmp.jl`) | identical within 1% on every shape and tiling |
| Is it Julia's register count? | Julia's own `sgemx_kernel2!` 128×128 launched with `maxregs = 128` (2 blocks/SM; `maxregs.jl`) | no help: 2697 → 2634 (2048³), 2636 → 2456 (4096³) |
| Is the "reg" L-sweep gain a C++ effect? | the same kernel written in Julia (`@generated`, exact-nn dispatch, 4 loads in flight; `jl_reg.jl`), batched L levels | **no**: Julia reg 44.0–46.9 ms vs C++ reg 46.8–47.4 (grid3d-25); 54–61 vs 60–63 (grid2d-150). Julia simple 58.4 vs C++ port 61.4 |
| Does `__ldg` matter for the sweeps? | `-DSR_NO_LDG` build | no (47.0 vs 46.8 ms) |
| Is a faithful port as fast as the Julia kernel? | persistent L sweep only, Julia vs the C++ port with 32-bit ticket math (as Julia now) vs 64-bit (as before 23:35), alternated (`persist.jl`, `-DSR_PERSIST_I64`) | grid3d-25: Julia 54.3, C++ 32-bit **69.4**, C++ 64-bit 57.5, C++ reg 50.7 ms; grid2d-150: 84.8 / **101.8** / 90.2 / 74.4. Same SASS mix, but in the 32-bit build nvcc chose 40 instead of 53 registers and no longer hoists the 4 unrolled `Stgt` loads (one register, R30, reused), so the separator gathers are serialized. The reg variant issues its 4 loads explicitly and is immune |
| Occupancy of the small GEMM tiles | `__launch_bounds__` 3 vs 2 blocks/SM for 4×4 tiles (`libcmp.jl`) | +6–20% for BK ≤ 16 (e.g. 128×32: 1019 → 1211 on 27000×20×500), −10% for BK = 32 → used for BK ≤ 16 only |

## Analysis

### GEMM: why the C++ kernel is faster

Registers and per-k-tile instruction mix, MinPlus Float32, 128×128×8 tiles (`cuobjdump -res-usage` /
`CUDA.registers`, SASS from `cuobjdump -sass` / `CUDA.code_sass`):

| | Julia `sgemx_kernel2!` 128×128 | C++ 128×128×8 | C++ 128×64×16 |
|---|---|---|---|
| registers / blocks per SM | 162 / 1 (8 warps) | 128 / 2 (16 warps) | 118 / 2 |
| k-step loop | runtime loop over BK: 4 LDS.128 + 64 FADD + 64 FMNMX + 6 loop/address instructions per step | fully unrolled: 32 LDS.128 + 512 FADD + 512 FMNMX per tile, LDS of step k+1 scheduled under step k | same, BK = 16 |
| global → shared per tile | 8 predicated scalar LDG + 8 STS, ~60 integer/predicate instructions (Int64 index math) | 2 LDG.128 (interior) + STS.128 / 4 STS | same |
| epilogue | load-⊕-store per element, serialized (no alias proof) | 128-bit RMW | 128-bit RMW (8×4 tiles: one pass) |

Neither the register budget alone (`maxregs` experiment) nor the compiler version explains the gap. What
does, plausibly, is the combination of four things:
- the unrolled, software-pipelined k-loop: the LDS latency of each step is hidden under the previous
  step's 128 math instructions instead of exposed at every iteration;
- 2 blocks per SM, so one block's barrier is covered by the other block's math;
- vector global loads and stores;
- BK = 16 (half the barriers per multiply-add) and better tiles for skinny n.

Each of these can be written in CUDA.jl: unrolling via `@generated` (as already done for the register
tile), `maxregs`, vector loads through `reinterpret`ed `NTuple`/`VecElement` pointers. **Not verified:**
I did not rewrite the Julia GEMM.

Julia's 64×64 tile (72 registers, 3 blocks per SM) is the better Julia default for skinny problems. It
slightly beats the C++ kernels when the leading dimension is odd (15625), where C++ loses its 128-bit
loads.

### Sweeps: memory-bound, language-neutral

L-sweep front statistics (`GPUSLU(P; large = 8192)`):

| graph | batched L fronts | levels | fronts with nn = 1 | Σ na | DRAM traffic (reg) |
|---|---:|---:|---:|---:|---:|
| grid3d-25 | 10,881 | 95 | 96% | 200,674 | 14 GB |
| grid2d-150 | 17,136 | 81 | 90% | 127,182 | 15 GB |
| grid3d-30 | 18,646 | 126 | 96% | 369,058 | 45 GB |
| grid2d-180 | 24,620 | 67 | 90% | 182,520 | 32 GB |

- 90–96% of the batched fronts have nn = 1. Each row then does about 7 separator loads and 1 store per
  front, all coalesced 256-byte column segments.
- The reg variant moves about 32 GB in 112 ms on grid2d-180 and 45 GB in 154 ms on grid3d-30. That is
  about 285–290 GB/s, about 75% of the 384 GB/s DRAM peak, counting no L2 reuse.
- The port reads `C[t, Stgt[r]]` nn times. Those re-reads hit L1/L2, which is why it is only about 15%
  slower on the graphs with larger fronts.

Registers:

| kernel | Julia | C++ port | C++ reg |
|---|---:|---:|---:|
| downward | 40 | 40 | 47 |
| upward | 40 | 40 | 40 |
| thread path | 64 | 56 | 56 |
| warp path | 64 | 56 | 56 |
| persistent | 40 | 40 (53 with 64-bit tickets) | 56 |

The SASS mixes are similar: Int64 address math, LDG, FADD/FMNMX. The C++ port unrolls slightly more.

**Going faster needs less traffic, not another language.**
- Rows are independent in the L sweep. A block could carry one row chunk through many fronts in
  depth-first order, so a parent's freshly written columns are re-read from L1/L2 by its children.
  Today they are re-read from DRAM one level later.
- Fusing levels this way is an algorithmic change (not done here, since the task was "same algorithm").
  The developer's persistent kernel removes the launches but keeps the level order, which explains why
  it doesn't help.

### Persistent vs level-by-level schedule

- On this laptop the persistent L sweep is slower for both backends in 7 of 8 cases each: Julia levels
  0.159 vs persistent 0.165 s on grid3d-25, and 0.19 vs 0.25 s on grid2d-180.
- Its per-item overhead is not repaid by the saved launches: two `__syncthreads`, an atomic ticket, a
  spin on the parent's counter, and `__threadfence` per 128-row chunk of a 1-column front.
- The 67–126 level launches cost only about 0.5 ms.

## Validation (`test/test_cuda_backend.jl`; all pass)

- **GEMM:** 16 shapes, from 1×1×1 to 2048×2048×64, ragged and skinny (n = 5, 7, 16, 17, 30, 33, 64,
  65, k = 1), × 11 tiling codes. Run for MinPlus F32/F64, MaxMin F32, PlusProd F32/F64.
  - Data are integer-valued, with 5% +∞ and 2% −∞ to exercise NaN absorption.
  - Exact equality with the CPU `Semiring.sgemx!` and with `sgemx_gpu!` (`isapprox` for PlusProd).
  - Strided views with ld = 999 and unaligned offsets.
- **Closure:** `closure_cuda` vs `closure_gpu` and vs the CPU `Matrix(mlu(s, A))` on 5 grids: grid
  60×60, grid3 14³, grid3 12³ MaxMin, grid 30×30 PlusProd F64, grid3 10³ F64.
  - Crossed with `large` ∈ {∞, 64} (dense path on/off), with and without `precompute_ops!`, both
    schedules (levels / persistent + warp path) and both variants.
  - Results: bit-identical to Julia for MinPlus and MaxMin, ≤ 1e-12 relative for PlusProd (in practice
    also identical), and identical to the CPU closure (1e-10 for PlusProd).

## Caveats

- One consumer laptop GPU with a 50 W cap, so clocks drift by ±20%. GEMM timings use interleaved bursts
  and the closure uses min of 3, with ±3–5% run-to-run noise left.
- The GPU is shared: the scripts wait while another process is on it, and `[GPU busy]` lines in the logs
  mark it. A process can still start mid-measurement.
- Only 4 synthetic grids. B200 / RTX PRO 6000 are untested: `ARCH="100 120" BUILD=build-hpg
  cuda/build.sh` builds, and the sm_100 SASS shows FMNMX3.
- Julia JIT time is excluded (warm-up runs); C++ is compiled ahead of time.
- The Julia baseline is a moving target. It was compared at the `src/sgetrs.jl` md5 given in the
  Driver section, with its `PERSISTENT[]` / `PATH_WARP[]` switches covering both schedules.

## Files

```
cuda/semiring.cuh        the semirings (MinPlus, MaxMin, PlusProd × float, double), exact Julia semantics
cuda/semiring_gemm.cu    sr_gemm: C ← C ⊕ A ⊗ B, column-major, leading dimensions, 10 tile configurations
cuda/semiring_solve.cu   batched sweeps (U, L, U-path, persistent L, warp U-path) + dense-path helpers
cuda/build.sh            builds cuda/build/libsemiring_cuda.so (ARCH, BUILD, PTXONLY, NVCC, EXTRA variables)
src/cuda_backend.jl      module SemiringCUDA: sgemx_cuda!, closure_cuda!, sssp_cuda! (the Julia driver)
test/test_cuda_backend.jl
bench/bench_cuda_backend.jl   `gemm` | `closure [graph ...]`
cuda/experiments/*.jl    control experiments (libcmp, maxregs, jl_reg, path, persist, jlregs)
cuda/results_{gemm,closure}_rtx5060_laptop.txt   raw logs of the tables below
```

## Build and run

```
cuda/build.sh                                  # sm_120 → cuda/build/libsemiring_cuda.so
ARCH="100 120" BUILD=build-hpg cuda/build.sh   # HiPerGator: B200 (sm_100) + RTX PRO 6000 (sm_120)
PTXONLY=1 BUILD=build-ptx cuda/build.sh        # PTX only (the driver's ptxas compiles it at load)
SEMIRING_CUDA_LIB=$PWD/cuda/build-hpg/libsemiring_cuda.so julia ...   # use another build
julia --project=. -t 16 test/test_cuda_backend.jl
julia --project=. -t 16 bench/bench_cuda_backend.jl gemm      # VERBOSE=1 prints every tiling
julia --project=. -t 16 bench/bench_cuda_backend.jl closure   # 4 grids, both schedules
```

```julia
include("src/SemiringGPU.jl"); include("src/cuda_backend.jl")
using .SemiringGPU, .SemiringCUDA
sgemx_cuda!(s, C, A, B)                  # same contract as sgemx_gpu!; strided views are fine
D = closure_cuda(G)                      # == closure_gpu(G); keywords variant = 0 | 1, tiling
sssp_cuda!(X, G, sources)                # == sssp_gpu!(X, G, sources)
```

The flags are `-O3 -std=c++17 --fmad=false -cudart static`, without `--use_fast_math`. Fast math would turn
the 1/(1 − a) in PlusProd's `sstar` into an approximation and flush denormals. `--fmad=false` keeps every
fma explicit, as in Julia, where `a*b + c` is fused only when written as `muladd`. Every entry point is
`extern "C"`, takes raw device pointers, sizes, leading dimensions and a `cudaStream_t`, and returns a
`cudaError_t` (0 = OK). Julia passes `CUDA.stream().handle`, so kernel order and `CUDA.@elapsed` work as
for Julia kernels. cudart is linked statically and attaches to CUDA.jl's primary context.

## Semantics (cuda/semiring.cuh)

The device code reproduces what the Julia kernels compute. That is the X86 host branch of
`Semiring.jl`, plus the `CUDA.@device_override` of `vmin`/`vmax` → `llvm.minnum`/`maxnum`:

| | ⊕ | smuladd(a, b, c) | sprod (scale) | sstar(a) | zero / one | integral |
|---|---|---|---|---|---|---|
| MinPlus | `fminf` | `fminf(a + b, c)` | `isnan(a+b) ? +∞ : a+b` | `a ≥ 0 ? 0 : −∞` | +∞ / 0 | no |
| MaxMin | `fmaxf` | `fmaxf(fminf(a, b), c)` | `fminf(a, b)` | one | −∞ / +∞ | yes |
| PlusProd | `+` | `fma(a, b, c)` | `a * b` | `a < 1 ? 1/(1−a) : +∞` | 0 / 1 | no |

- `fminf`/`fmaxf` are PTX `min.f32`/`max.f32`, which is exactly what `llvm.minnum`/`maxnum` lower to. So
  +∞ + −∞ = NaN is absorbed and the result is the other operand.
- In the GEMM, A is padded with zero and B with one, and the epilogue is `C = acc ⊕ C`.
- The U-sweep scaling (`SCALE = !integral`) and the CAS loop of `atomic_splus!` (compare-and-swap on the
  32/64-bit pattern, stopping when `new` and `old` are bitwise equal, like `===`) are the same.
- SASS check for MinPlus float: the GEMM mainloop is only FADD + FMNMX, with no FSETP/FSEL. The 128×64×16
  kernel has 512 FADD and 576 FMNMX (512 in the mainloop, the rest in the epilogue).

## Design

### GEMM (`semiring_gemm.cu`)

One templated kernel, `sgemm_kernel<S, T, Cfg<BM, BN, BK, WARPS_M, WARPS_N, LM>>`:

1. **Two-stage pipeline (CUTLASS/cuASR).**
   - The prologue loads k-tile 0 into registers, stores it to shared buffer 0 and syncs.
   - Each iteration first issues the global loads of tile q+1 into registers, then computes tile q from
     shared memory (BK steps, fully unrolled, so ptxas software-pipelines the LDS of step k+1 under the
     math of step k), then stores the registers into the other buffer.
   - One `__syncthreads` per k-tile.
   - Source: cuASR `srmma_pipelined.h:176-188` (prologue), `:235-239` (store + one barrier), `:263-284`
     (fragment double buffer, global loads issued at `warp_mma_k == 0`).
2. **Warp tiling with 128-bit fragments (cuASR).**
   - A warp owns a WM × WN tile. Its lanes form an LM × LN grid, with 8 lanes along M when WM > WN.
   - Each lane owns 4-wide groups of rows and columns, so every fragment read is one LDS.128 with at most
     one wavefront per warp.
   - Source: cuASR `default_srmma_core_simt.h:40-42` (`simt_get_warp_threads_m`), `:179-185` (lane
     fragment = 128 bits / element size), `default_srgemm_configuration.h:90-91` (128×128×8 block, 64×32
     warp).
3. **Shared layouts.**
   - `As[k][m]`: A is m-contiguous, so it is stored without a transpose.
   - `Bs[k][n + 4]`: B is k-contiguous, so the store transposes, padded by 4 words. The pad makes both the
     scalar and the vector store patterns bank-conflict-free.
   - Source: cuASR's transpose skew, `default_srmma_core_simt.h:45-50` and `:383-384`.
4. **Interior fast path (TropicalGemm).**
   - A k-tile whose rows (A) or columns (B) are fully in range, with a full k extent and 16-byte-aligned
     columns (`ld % 4 == 0`), is fetched with 128-bit `__ldg` loads and no bounds checks.
   - Any other tile is fetched with coalesced, checked scalar loads, padded with zero or one.
   - The branch is uniform per block, and A and B are tested separately, so A still gets vector loads when
     n < BN.
   - Source: TropicalGemm `tropicalgemm_kernels.cu:167` and `:625` (bounds checks only on the last block
     row/column or last k-tile), `:56` (padding with the tropical zero).
5. **Epilogue.**
   - Interior blocks with aligned C use 128-bit read-modify-write (ROME `kernel.cu:729` `FETCH_FLOAT4`,
     `:793` and `:985`).
   - For tiles of ≤ 32 accumulators, the epilogue issues **all loads of C before any store**. Written as
     load-⊕-store per element (as in `sgemx_kernel2!`), the compiler can't prove that a store doesn't alias
     the next load, because ldc is a runtime value. Every load then waits for the previous store, and for
     small k the GEMM is bound by this.
   - The 8×8 tiles keep the one-pass epilogue, because the extra registers would make the mainloop spill.
6. **Two k-steps per expression.** The update is written `acc = (acc ⊕ a₀b₀) ⊕ a₁b₁`. On **sm_100 (B200)**,
   ptxas then fuses the two `min`s into one 3-input **FMNMX3**:
   - 512 FADD + 256 FMNMX3 per 512 multiply-adds, i.e. 1.5 instead of 2 instructions;
   - up to 1.33× on HiPerGator, verified in the sm_100 SASS but not timed (no B200 here);
   - bit-identical, since it is the same pair of minnum operations;
   - sm_120 has no FMNMX3, so the code there stays 2 × FMNMX.
7. **Tile configurations** (`tiling` codes), all 256 threads unless noted:

   | code | tile | per thread | notes |
   |---|---|---|---|
   | 1 | 128×128×8 | 8×8 | |
   | 2 | 128×64×8 | 8×4 | |
   | 3 | 64×64×8 | 4×4 | |
   | 4 | 128×32×8 | 4×4 | |
   | 5 | 256×16×8 | 4×4 | |
   | 6 | 128×128×16 | 8×8 | |
   | 7 | 128×64×16 | 8×4 | |
   | 8 | 128×16×8 | 4×4 | 128 threads |
   | 9 | 128×32×32 | 4×4 | BK = 32 as TropicalGemm's FP32 (`:71`) |
   | 10 | 128×16×32 | 4×4 | BK = 32, 128 threads |

   - `__launch_bounds__` targets 2 blocks/SM (≤ 128 registers) for the 8×8 and 8×4 tiles and for BK = 32.
   - It targets 3 blocks/SM (≤ 80 registers) for the 4×4 tiles with BK ≤ 16, which gave +6–20% on the
     skinny, memory-bound shapes.
   - `choose_tiling(m, n, k)` was tuned on this GPU (see the code).
8. **FP64.** It uses the same kernels with `V4<double>` = 2 × 16-byte accesses. Tilings 1, 6, 7, 9 and 10
   fall back to 2, 2, 2, 4 and 8, because 8×8 double tiles spill and BK ≥ 16 exceeds 48 KB of static smem.

**Considered and not used:**
- **cp.async:** with a 2-stage register-staged pipeline and 2–3 blocks per SM, the GEMM is
  issue-bound on compute shapes and L2-bound on skinny ones, and B needs a transpose through registers
  anyway.
- **3-input `min.f32` on sm_120:** ptxas lowers it to 2 × FMNMX.

### Sweeps (`semiring_solve.cu`)

Line-for-line ports of `src/sgetrs.jl`:
- `upward_kernel!` + `upward_front!` (U sweep, atomic CAS ⊕);
- `upward_path_kernel!`, `downward_kernel_simple!`;
- the developer's newer `persistent_down_kernel!` (ticketed persistent L sweep, per-front counters) and
  `upward_path_warp_kernel!` (one warp per source).

They use the same grids, block sizes, thread-to-row mapping, loop order and memory accesses, with Int64
1-based index arrays read as they are. Factor and index arrays go through `__ldg`. C is never read
through `__ldg`, so the persistent kernel's cross-block reads stay coherent. Each kernel has two variants:
- **variant 0, "port":** the Julia kernel transcribed.
- **variant 1, "reg":** for fronts with nn ≤ 8 (a `switch` on nn to a template instance; nn is uniform
  per block, so there is no divergence and no masking):
  - the row's residual values stay in registers, so each separator value `C[t, Stgt[r]]` is loaded once
    instead of nn times;
  - the separator values are loaded explicitly 4 at a time, for 4 loads in flight. `#pragma unroll`
    alone is not enough: see the persistent-port experiment;
  - the order of every ⊕-accumulation is unchanged, so results are bit-identical to variant 0;
  - in the warp path kernel, every lane solves the small U₁₁ redundantly in registers and lane 0 stores.

Helpers in C++ too, so the whole closure runs on C++ kernels: fill, strided copy, identity,
gather, scatter-⊕, and the one-thread-per-row diagonal solve `strsx_diag_kernel!`.

### Driver (`src/cuda_backend.jl`)

`closure_cuda!` → `sssp_cuda!(D, G, rperm; W = D, permute = false)` follows `sssp_gpu!` step by step:
- fill, U path walk, then the top of the tree level by level: batched upward, then `upward_large_cuda!`
  per large front;
- then the L sweep, either level by level (`downward_large_cuda!` + batched) or the persistent schedule
  (`SemiringGPU.persistent_plan(G)`'s top levels, then one persistent launch).

The dense large-front path is reproduced exactly:
- `strsx_cuda!` follows `strsx_gpu!`: 64-column diagonal blocks, inverted with identity + diagonal
  kernel + GEMM when there are more than 64 rows, plus the trailing GEMMs. It uses SemiringGPU's own
  `trsm_workspace`.
- With `precompute_ops!`, the driver uses the `G.ops[]` operators (copy, fill, one GEMM, gather/scatter)
  and `ops_workspace`.

The schedule follows `SemiringGPU.PERSISTENT[]`/`PATH_WARP[]` by default. Keywords `persistent`,
`path_warp`, `variant` (0/1) and `tiling` override it. Nothing in `src/`, `test/` or `bench/` was edited;
the backend only reads SemiringGPU's internals.

**The Julia side changed while this was written.** The developer added the persistent L sweep and the
warp path walk, and made them the default. The benchmarks therefore compare **both** schedules, each
against the identical C++ schedule.

The md5 of the `src/sgetrs.jl` behind the final numbers is `08478344c20a495b1fe68a6c69f17d10`
(23:35). That version's persistent kernel uses 32-bit ticket arithmetic, and the C++ port was updated to
match. The previous version was `2ba642c2756e33a51699227a356eb215`. The level-by-level kernels are
the same in both.
