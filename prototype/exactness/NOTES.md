# Exactness experiments (Oct 20): do better TDs give better contraction orders, in cost and in real time?

HPG dir: `/blue/fairbanksj/itaykadosh/gpu/exactness` (code = `src/`, jobs = `hpg/`, results `resA/`, `resB/`).
Local copy of the code: `tn-work/exactness/{src,hpg,queueA.txt,queueB.txt}`. Nothing committed.
CliqueTrees: own copy of the pipeline worktree (itay/pipeline @ 90964ec) at `$E/CliqueTrees.jl`, env `$E/env`
(copy of pipeline/env, depot chain in `$E/common.sh`), precompiled on the login node. Python: `$E/venv`
(cotengra 0.8.2, cotengrust 0.2.1, kahypar 1.3.7, torch 2.x cu128, cupy-cuda12x).

## Jobs (submitted 2026-10-20 00:51)
| job | what | where |
|---|---|---|
| 46787880 | A: Julia TD generation on queueA.txt (181 nets, n <= 800) then costA.py | hpg-dev, 32 c, 6 h |
| 46787881 | B: Julia orders (queueB.txt, 38 nets) then costB.py trees | hpg-dev, 32 c, 4 h |
| 46787882 | B GPU: contractB.py (torch) + contract_cutn.py if the cuquantum venv imports | hpg-b200, 1 GPU, afterany:46787881 (queue estimate said Oct 23) |

All three are named `itay-exact`. Re-submitting any of them resumes (finished .td / cost / tree / GPU rows are skipped).
`sbatch hpg/A_cpu.sbatch`, `sbatch hpg/B_cpu.sbatch`, `sbatch hpg/B_gpu.sbatch` from `$E`.
(The first attempt on hpg-milan, 46787539-41, was cancelled: the queue estimate was 9-29 h.)

## Analyze
    E=/blue/fairbanksj/itaykadosh/gpu/exactness
    ssh hpg "cd $E && venv/bin/python src/analyzeA.py resA resA/A_tables.md"     # partial results are fine
    ssh hpg "cd $E && venv/bin/python src/analyzeB.py resB resB/B_tables.md"
    rsync -a hpg:$E/resA/A_tables.md hpg:$E/resA/A_points.tsv hpg:$E/resB/B_tables.md tn-work/exactness/res/
Raw: `resA/td/<net>.td` (orderings, widths, lcost, branchtree/branchrefine SSA paths), `resA/cost/<net>.tsv`
(cotengra costs per ordering x stage), `resB/trees/{trees.tsv,<net>.pkl,<net>.json}`, `resB/gpu_torch.tsv`,
`resB/gpu_cutn.tsv`. Logs: `logs/{A,B,Bgpu}-<job>.out`, `resA/logs`, `resB/logs`.
Plot data for "width gap vs cost gap": `resA/A_points.tsv` (one row per network x TD x stage).

## Experiment A design (src/tdgen.jl mode A + src/costA.py)
- Networks: every corpus network with line-graph n <= 800 (181: einsum gm/lm/mc/qc/rnd/str/wmc, quimb rqc 7x7,
  Sycamore-54 m10-20, QAOA zz, random regular; ladder small ones). A network enters the exact set when
  HybridTW (CPU, 2 threads, 300 s; safe rules + safe separators + septw/RTW exact on atoms + HBT) certifies
  its weighted width (bag weight = sum log2 dims; = log2 of the largest bag's contraction size).
- TDs per network (one ordering each):
  - exact (the solver's own optimal TD); when it took < 30 s, 7 more exact solves on randomly relabeled copies
    (exact_r1..7: other optimal TDs, the solver breaks ties by labels);
  - greedy MF, MF1, AMF, MMD, MCS, bw (best of the 4); random tie-breaking MF_r1..4, MF1_r1..4 (relabeled graph);
  - METIS ND nd_s1..3 (ND with bw leaves), nd8 (best of 8 seeds);
  - HBT (SafeRules+SafeSeparators) at 0.1 / 1 / 10 s x seeds 1..3.
- Per TD: width, lcost = log2 sum_bags 2^w(bag) (maximal cliques), and contraction trees:
  bucket (cotengra from_path edge_path, greedy binarization; the TD's plain induced tree), bt (branchtree),
  br (branchtree + branchrefine = our best conversion), bucket+ctg and br+ctg (cotengra subtree_reconfigure,
  subtree 8, to a fixed point, <= 8 passes). Baselines: cotengra greedy and hyper 10 s (1 thread), raw and +ctg.
  All costs re-computed by cotengra (log10 C = contraction_cost, multiply-adds); the test run matched
  CliqueTrees branchstats to 1e-6.
- Analysis (analyzeA.py): Spearman(width, C) per network (all TDs; TDs within 2 bits of optimal), whether the
  cheapest TD-tree has optimal width and by how many bits it misses, exact (best over relabels) vs best heuristic
  TD (W/T/L), spread of C among width-optimal TDs, and cost-aware selection among optimal TDs (by lcost; by our
  exact C of the br tree) vs the best optimal one; per family; vs cotengra hyper 10 s.

## Experiment B design (src/tdgen.jl mode B + costB.py + contractB.py + contract_cutn.py)
- 38 candidate networks (queueB.txt): einsum qc/lm/str/rnd/gm/mc/wmc with published best flops 10^9.4-10^13.3,
  plus quimb Sycamore-54 m12/m14, rqc 7x7 m14/m16, QAOA zz, random regular, QAOA amp. The GPU job executes
  every tree with predicted C <= 10^13.5 (others are logged as skip_cost).
- Trees: ours10/ours60 = hybridcost(HybridTW(T, 1 thread, CPU), refine = branchrefine) [pipeline ordering +
  MF/MF1/AMF/MMD portfolio, cheapest by exact C] + cotengra subtree_reconfigure; TD ladder td_MF1, td_hbt1,
  td_exact (td_hyb300 when not certified) via branchtree+branchrefine+reconf; cotengra hyper 10/60 s
  (parallel=False, minimize='flops'; cotengra's own sliced search when greedy is wider than 2^30), cotengra greedy.
  Every tree wider than 2^30 is sliced (slice_and_reconfigure, target 2^30). search_s = all CPU seconds that
  made the tree (Julia + conversion + reconf + slicing; cotengra wall).
- GPU: torch complex64 on one B200, random data (tensor scaled by prod d^(-1/(2k)) over summed indices so the
  value stays O(1)), 1 warm-up + 3 timed runs (median), peak memory, agreement vs complex128 result of the
  cheapest tree (if log10 C <= 12.5). cuTensorNet: same paths through cuquantum Network + its own path search
  (8 samples), only if `/blue/.../baselines/tn/venv_cutn` imports cuquantum when the job runs.
- Analysis (analyzeB.py): pooled and within-network Spearman(C, time), time model a*#contractions + b*C,
  fraction of tree pairs where lower C is faster, ours vs cotengra W/T/L in real time and time to solution
  (search + 1 contraction), exact vs MF1/HBT TDs in real time.

## Test run (46786772, hpg-dev, 2 small nets; GPU part tested with cupy on the laptop and torch-CPU on HPG)
- qaoazz_N30_p4_s0: width-optimal is NOT cost-optimal already here: width-17 TDs give br log2 C 20.69-21.25,
  the width-18 MF1 TD gives 20.42; hybridcost picks the MF1 tree (portfolio). cotengra hyper 3 s: 6.80 log10
  vs our TD trees 6.15-6.40.
- branchstats cost == cotengra contraction_cost (6.440583 vs 6.440584 log10).
- relerr vs complex128: ~1e-6 for all trees.

## Interim (Oct 20 ~01:15, 23/181 A networks done, 7 certified; Julia branchstats br-tree cost, cotengra-consistent)
`src/interimA.py` (run on HPG: `venv/bin/python src/interimA.py`, reads resA/td only, no costA needed):
| net | #TD | #opt-width TDs | rho(width,C) | width gap of cheapest TD | d log10C exact(best) - best heuristic | spread among opt TDs |
|---|---|---|---|---|---|---|
| gm_pedigree38 | 28 | 4 | 0.71 | 0 | -0.099 | 0.211 |
| qaoazz_N2000_p6_s0 | 28 | 9 | 0.93 | 0 | +0.066 | 0.212 |
| qc_rand_google53_26_15 | 28 | 4 | 0.91 | **3 bits** | +0.142 | 0.082 |
| qc_rand_supremacy2d_1_12_400 | 35 | 17 | 0.93 | 0 | -0.046 | 0.055 |
| rnd_mixed_05 | 28 | 6 | 0.96 | 0.1 | +0.111 | 0.111 |
| rnd_mixed_07 | 28 | 5 | 0.98 | 0 | +0.002 | 0.017 |
| rnd_mixed_08 | 35 | 12 | 0.42 | 0 | +0.169 | 0.190 |
Median rho 0.93 (width is a strong predictor of C across the quality range); the cheapest TD-tree has optimal
width on 5/7; but the exact solver's TD vs the best heuristic TD (mostly HBT 1-10 s, which also reaches optimal
width) is 2 wins / 1 tie / 4 losses, median +0.07 log10 C. Among width-optimal TDs C spreads by a median 0.11
log10 (up to 0.21) -> exactness is necessary-ish but not sufficient; cost-aware tie-breaking matters as much as
proving optimality. Too few networks yet; wait for the full table.
B (CPU side, 38 nets, Julia orders done for ~25): exact certified on the small/tree-like ones only; on rqc/syc/
peps/lattice the 300 s lower bound stays well below the width (gaps of 2-20 bits in the B .td files).

## Results
(fill in from resA/A_tables.md and resB/B_tables.md when the jobs finish)
