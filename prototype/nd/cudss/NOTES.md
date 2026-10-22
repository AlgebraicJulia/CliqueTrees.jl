# cuDSS end to end: do our GPU orderings make a real sparse Cholesky faster? (Oct 20)

Question: GPU ND hybrid (`gndd:256:hyb=1;rmax=4`) and GPU-MF1 as `user_perm` of NVIDIA cuDSS on a B200, against
cuDSS's own default reordering (METIS-based ND, host multithreaded), end to end: order + analysis + factor + solve.

## Setup
- Julia CUDSS.jl 0.8.1 + cuDSS 0.8.0 (CUDSS_jll, CUDA 13.0 artifact, already in the HPG depot). HPG env
  `/blue/fairbanksj/itaykadosh/gpu/cudss/env` = `rounds/nd/henv` (rounds/env + AMD.jl; CUDA_Runtime/Compiler_jll 13.0
  pinned, both under [extras]) + CUDSS, Statistics. Laptop env `tn-work/cudss/lenv` (= rounds/genv + CUDSS).
- Code: `cudss_bench.jl` (driver), `ndtree.jl` (partition tree, below), `perms.jl` (CPU reference orders),
  `analyze.py` (tables), `hpg/cudss.sbatch`, `hpg/perms.sbatch`, `dbg*.jl` (laptop probes). HPG: `cudss/code`, sources
  of GND/GRounds copied unchanged to `cudss/src` (md5 = rounds/nd/code3 + grd3 = local tn-work/rounds).
- Matrix: the graph pattern of each of the 28 nd/results3 graphs with SPD values, graph Laplacian + I (off-diagonal -1,
  diagonal deg + 1), lower triangle CSR Int32 on the device, cuDSS "SPD"/'L' (LL^T, no pivoting, so time depends only
  on pattern + order). b = A * 1, relative residual on the host. Real SuiteSparse values are not used (the .bin files
  are patterns); for SPD Cholesky without pivoting the values do not change the work.
- Threads: Julia -t32 (the GND hybrid's CPU FM) and cuDSS's threading layer `libcudss_mtlayer_gomp.so` (shipped in the
  artifact) with `host_nthreads` = 32, OMP_NUM_THREADS=32. `cudss1` = same with host_nthreads 1 (does MT help the
  reordering?). The layer is set on the handle after the GND warm-up (setting it before the GND warm-up crashed once on
  the laptop with "double free"; after: fine).
- Timing (wall, CUDA.synchronize after each phase): cuDSS phases called separately: "reordering", "symbolic_factorization",
  "factorization", "solve". Global warm-up of all paths; 3 timed cuDSS runs per config (fresh solver each), median.
  Our orderings: 1 untimed + 3 timed runs, median, host CSR in -> host permutation out (graph upload included; the
  matrix is also on the device already, a device-CSR entry point would save the upload). cuDSS default's ordering cost
  sits in its "reordering" phase (orders 0).
- Fill: cuDSS `lu_nnz` and CUDSS_DATA_FLOPS (int64, available after analysis; CUDSS.jl does not wrap it, read with
  cudssDataGet), plus our exact nnz(L)/flops (fillq). lu_nnz is ~5-9% above exact nnz(L) (supernode amalgamation).
- Memory guard: a config is skipped if 8 * nnz(L) (ours before, lu_nnz after analysis) > 0.8 x GPU memory.

## Findings so far (laptop RTX 5060, tiny matrices; correctness only)
1. user_perm convention: elimination order, 1-based (same base as the matrix; `perm_reorder_row` returns the same
   convention, feeding it back reproduces lu_nnz exactly). The inverse is 7-10x more fill.
2. **user_perm alone is a slow path.** cuDSS's own order fed back as user_perm: lat2_200 factor 48 ms vs 5 ms, analysis
   0.14 vs 0.03 s; lat3_20 fact 14 vs 8 ms. cuDSS uses its ND "partition tree" for the factorization's parallel schedule
   and has no tree for a user perm. Fix: also pass `CUDSS_DATA_USER_ND_PARTITION_TREE` (not wrapped by CUDSS.jl; raw
   cudssDataSet). With the tree, own perm = default speed exactly (lat2_200 fact 5.0 ms).
3. Partition-tree format (found by experiment, `dbg2.jl`, `ndtree.jl` check_layout): 2^L - 1 Int32 block sizes,
   L = nd_nlevels = 10; array = level L-1 (512 leaf domains) left to right, then each level up, root separator last; the
   **permutation lists the blocks in this same array order** (not a postorder; a postordered perm with the right sizes
   made cuDSS fail with an illegal address). Unsplit nodes: whole region in the node's slot, children 0 (cuDSS does
   this for small graphs); cuDSS never has an empty separator with nonempty children.
4. `nd_tree(ptr, adj, order)` derives such a tree for ANY elimination order: etree, postorder, then top-down: a node owns
   independent etree subtrees; separator = chain of single-child vertices at the top of the largest one (incl. its first
   branching vertex); its children = the remaining independent subtrees in two size-balanced groups; emitted in cuDSS's
   block layout. Same fill as the input order (checked), valid layout (no edge between unrelated tree nodes), cuDSS
   accepts it, and on cuDSS's own order it reproduces cuDSS's tree exactly up to left/right swaps (first version was
   identical). Cost O(n + m) on the host, counted in our ordering time (`;tree=` in the ttag column).
5. Laptop (4 host threads) small cases: cuDSS's reordering is most of its analysis (lat2_200: 0.115 of 0.148 s total).
   gnd+t on lat2_200: order 0.077 + analysis 0.029 + fact 0.005 = 0.112 s vs cudss 0.148 s; lat3_20 0.083 vs 0.085;
   fill within 1-3%. cuDSS algo3 (its AMD) gives the same lu_nnz as our AMD order (consistency check).

## Configs (cudss_bench.jl CFGS)
cudss (default reordering, 32 host threads), cudss1 (1 thread), own (cuDSS's perm as user_perm, no tree: slow-path
control, only n <= 2M), own+t (with derived tree: control for the tree path), gnd / gnd+t (GPU ND hybrid), gmf1+t
(GPU-MF1 whole graph, mf1 relax 0.1), amd+t / metis+t / mf1+t (CPU orders from perms.jl: SuiteSparse AMD and METIS_NodeND
via CliqueTrees in cudss/env; serial MF1 = Richard's origin/main, env rounds/rr/renv; times = milan serial, median of 3,
1 above 1-2M; BipartiteGraph built outside the timer as in pbase).

## HPG jobs (submitted Oct 20; all job name itay-cudss, logs cudss/logs/)
- 46907939 perms (hpg-milan, 8 cpus, 200G, 8 h; 46907637 failed in 3 s: env.sh defines G, renamed GRAPHS): writes `cudss/perms/NAME.{AMD,METIS,MF1}.perm`.
- 46907646 smoke (B200, 45 min): lat2_224, thermomech_TC, parabolic_fem, CFGS cudss,cudss1,own,own+t,gnd,gnd+t,gmf1+t
  -> `cudss/results/smoke_b200.tsv`.
- 46907647 main (B200, afterok smoke, 10 h): all 28 graphs (small to large, Bump/Queen last), same CFGS ->
  `cudss/results/cudss_b200.tsv`.
- 46907648 ref (B200, afterany main + afterok perms 46907939, 10 h): CFGS cudss,amd+t,metis+t,mf1+t -> `cudss/results/ref_b200.tsv`.
If smoke fails, main is never run (DependencyNeverSatisfied): read logs/itay-cudss-46907646.out, fix, resubmit
(`cd cudss; CFGS=... OUT=... sbatch hpg/cudss.sbatch`; commas: pass CFGS/NAMES as env vars, not via --export).

## Harvest
```
scp hpg:/blue/fairbanksj/itaykadosh/gpu/cudss/results/*.tsv tn-work/cudss/results/
python3 tn-work/cudss/analyze.py tn-work/cudss/results/cudss_b200.tsv tn-work/cudss/results/ref_b200.tsv > tn-work/cudss/results/tables.md
```
analyze.py: per matrix medians; ratios vs the cudss rows of the same file (total = T(1), T(10), factor+solve = k->inf,
factor, lu_nnz, flops, order/total, cuDSS reorder/total); bucket geomeans (100k-500k, 500k-1M, 1-3M, >=3M) with
#wins; regimes table (k factorizations with one ordering + analysis: "all k", "k<=K", "k>=K", "never").
What to look at: (a) own+t vs cudss factor ~1.0 (tree path is not handicapped at scale; if not, the comparison is
unfair to us and must be reported as such); own vs own+t = cost of the missing tree; (b) gnd+t total vs cudss per
bucket; cudss's reorder share of its total (if the factorization dominates, ordering speed cannot matter much and the
fill/flops ratio decides); (c) gnd+t factor ratio vs flops ratio; (d) cudss vs cudss1 = how much MT helps cuDSS reorder.

## Results
(pending: fill in from results/tables.md)
