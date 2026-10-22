# Experiment B with a FAIR executor (cuTensorNet), Oct 20

STATUS: full job 46879308 (hpg-b200, 1 B200, itay-exact, 8 h limit) was STILL RUNNING when this was written (273 of ~330 rows;
only the heavy syc54_m12/m14, rqc7x7_m14/m16 networks remain, ~50 s per execution). The tables below are from a snapshot
(`res/gpu_cutn_fair_partial.tsv`, full tables `res/B_fair_tables_partial.md`, 24 networks have both ours60 and ctg_hyper60 executed).

## Harvest (when the job is done)
    E=/blue/fairbanksj/itaykadosh/gpu/exactness
    ssh hpg "squeue -u itaykadosh -n itay-exact; tail -3 $E/logs/BFair-46879308.out; grep -c . $E/resB/gpu_cutn_fair.tsv"
    ssh hpg "cd $E && venv/bin/python src/analyzeBfair.py resB/gpu_cutn_fair.tsv resB/B_fair_tables.md"
    rsync -a hpg:$E/resB/gpu_cutn_fair.tsv hpg:$E/resB/B_fair_tables.md tn-work/exactness/res/
(if the job died/timed out: `cd $E; sbatch --time=08:00:00 --export=ALL hpg/B_fair.sbatch` resumes, finished rows are skipped.)
Code: `src/contract_cutn.py` (rewritten), `src/analyzeBfair.py`, `hpg/B_fair.sbatch`, `src/probe_fpe.py`. Rows: net tag log10C W nslices search_s t_med ... relerr ref status.

## What was wrong / fixed in contract_cutn.py
1. BUG: slicing was passed as `(mode, full_extent)` pairs. In cuTensorNet the pair is (mode, extent INSIDE a slice), so full extent = no
   slicing at all. That produced the MemoryLimitExceeded errors (Promedas_33, pedigree50, ...) of the first run. Now modes only (extent 1).
2. Crash-proof driver: child process runs the items and writes each row; the driver watches `<out>.cur`, on a native signal writes
   `status=crash:<SIGNAME>:<last python frame>` for the in-flight item and relaunches (resumable); per-item timeout 900 s (own path
   search 300 s) -> `status=timeout`; RLIMIT_CORE=0; NetworkOptions(memory_limit=120 GiB); trees above 10^13.5 -> `skip_cost`.
3. Reference = cuTensorNet complex128 of the cheapest tree (C <= 10^12.5), rescaling of the data when |ref| outside (1e-6,1e6) as in contractB.py.
   All executed rows agree with it: relerr median 2.4e-6, max 2.1e-4, none above 1e-3.
4. Fair slice loop: cuTensorNet runs the slice loop on the GPU (no Python launches). Same autotune/warm-up/median-of-3 for every tree.
   Second executor (cotengra batched/vectorized slices): NOT done; cotengra 0.8.2 `tree.contract` loops slices in Python (that is the
   torch artifact itself) and has no batched-slice path, so it was skipped.

## SIGFPE (not reproduced)
No SIGFPE in this run (0 crashes so far; 2 timeouts, 1 INTERNAL_ERROR, 4 MemoryLimitExceeded). Probe (`probe_fpe.py`, 46877775):
contract_path on mc_2020_arjun_102 (1205 tensors) is fine for samples 1/8/default, threads 4. On mc_2022_085 (4974 tensors) cuTensorNet's
own path search takes > 300 s for any samples>=1 (default OptimizerOptions: 163 s) -> the old crash was at exactly this item
(`cutn_own` of mc_2022_085, right after ctg_hyper60; 4 CPUs). The baseline worker's SIGFPE is documented (STATUS_oct19) as int32 sample-count
overflow with huge `samples`; the exactness driver used samples=8, so a different trigger (very long multithreaded path search on a
4974-tensor network) is likely but unproven. Not a property of our/cotengra trees: all 233 tree executions ran. Remaining non-ok rows:
cutn_own mc_2022_085 and qc_maxcut_n29_r4_p10 (timeout), cutn_own mc_2020_arjun_102 (INTERNAL_ERROR), ctg_greedy on rnd_oe_04,
gm_pedigree50, str_ctg_regular_04, rnd_oe_03 (MemoryLimitExceeded with 120 GiB although width <= 2^30: cuTensorNet's minimum workspace
for these sliced greedy trees is larger than predicted; unexplained).

## Results (snapshot; speedup = ctg time / ours time; win >1.05, loss <0.95)
Ours60 vs ctg_hyper60 (24 networks both executed):
| class | n | W/T/L | median speedup | geomean | median TTS speedup | geomean TTS |
|---|---|---|---|---|---|---|
| (a) both unsliced | 6 | 3/1/2 | 1.07 | 1.00 | 2.44 | 4.79 |
| (b) ctg sliced, ours fewer/no slices | 16 | 16/0/0 | 26.8 | 18.2 | 3.80 | 6.79 |
| (c) other (equal slices) | 2 | 1/1/0 | 1.40 | 1.34 | 12.4 | 4.86 |
| all | 24 | 20/2/2 | 5.05 | 7.10 | 2.61 | 6.05 |
Ours10 vs ctg_hyper10 (17): all 12/3/2, median 5.0, geomean 5.9, TTS median 2.46; (a) 5 nets 3/2/0 median 1.29; (b) 10 nets 9/0/1 median 21.4.
TTS = (search + one contraction); the networks where ours60 search is 60 s tie with ctg search 60 s, so TTS gains come mostly from execution.

Interpretation (honest):
- (a) parity, as found before: median 1.07, losses qaoazz_N120 (0.67) and supremacy2d_1_12 (0.57) despite lower predicted C.
- The "5x is an executor artifact" claim is only PARTLY right. Under cuTensorNet the slice loop is on the GPU and the speedup persists:
  log10 speedup vs d log10 C (ctg - ours) Pearson 0.69 over 24 nets; big wins track a real flop gap (pedigree50 dC 2.05 -> 115x, rnd_oe_04
  2.49 -> 392x, wmc_2021_187 2.31 -> 89x, mc_2022_085 1.94 -> 36x). The torch run exaggerated the small-dC cases: lm_* (dC 0.13-0.27)
  is still 34-92x because ctg slices them 1900-1996x into tiny tensors (width 14-18), where even cuTensorNet's slice loop is launch bound.
  So the defensible statement is: cotengra's capped-memory search returns sliced trees with 10-300x more flops OR thousands of tiny slices;
  ours stays unsliced or lightly sliced. Wins in (b) are structural (slicing), not better contraction orders per se.
- Where cotengra is not sliced and C is equal, no gap (a).

## Slice counts at the memory cap (2^30 elements per intermediate; full list in res/B_fair_tables_partial.md)
ours60 vs ctg_hyper60 #slices (30 nets): ours fewer in 21, equal 8, more 1 (str_ctg_lattice_01: 210 vs 48); unsliced: ours 23, cotengra 6;
median log10(ctg/ours slices) = 0.65. Ours10 vs ctg_hyper10: fewer 23, equal 6, more 1; unsliced 22 vs 5; median 1.98.
Examples ours60/ctg60: rnd_oe_04 3/1152, wmc_2021_187 2/256, pedigree50 1/180, supremacy2d_4_4_100 1/64, mc_2022_085 1/64, lm_* 1/~1900-1996,
gm_1kp6 1/27, Promedas_33 16/2e24 (ctg60 C 10^35.7, not executable).
