# gpucost: GPU mass scoring by total cost C + batched exact subtree reconfiguration (Oct 19)

## Report (15 lines)
1. **Proxy (step 1).** lcost = log2 Σ 2^bag is *not* faithful where it matters. On 20 networks × 96 orderings each, the Spearman correlation with cotengra's C of the induced tree has median 0.90 over all orderings, but only 0.76 in the top quartile (range −0.51 to 0.98; lm 0.39–0.65).
   So I wrote an exact-C GPU kernel. Its median top-quartile ρ is 0.99. It gives exactly cotengra's C on 96/96 orderings of every network with neither hyperedges nor open indices, and on 94–95/96 with open indices, where ftn re-roots the PEO. With hyperedges, only the bucket binarization differs.
2. **Gap closed (32 einsum networks, every tree re-costed by cotengra).** At 10 s, (b) has median gap 0.000 to the best tree known in every family except qc 0.002 and wmc 0.026.
   For comparison, (a) our pipeline + cotengra refine has lm 0.053, qc 0.174, rnd 0.043, str 0.110 (log10 C), the same gap as draft Table 7. hyper-sa 10 s has 0.000/0.020/0.043/0.132; TreeSA has 0.003/0.017/0.011/0.035.
3. **Head to head.** (b) against the better of hyper-sa and TreeSA at the same budget, wins/ties/losses:
   - 10 s: 15/16/1. lm 0/5/0, qc 3/4/1, rnd 2/3/0, str 5/2/0, gm 2/1/0, mc 2/0/0, wmc 1/1/0. Median ΔW is 0.
   - 1 s: 20/8/4, median ΔC −0.020.
   - Against TreeSA with 200 iterations (median 8 s, up to 35 s): 17/11/4. The four losses are google53 26_15/27_20, Promedas_33 and wmc_2021_187, each by 0.01–0.17.
4. **Speed.** Mass scoring (perturbation + tree build + exact C) runs at 0.1–3.4 M candidates/s, median 0.5 M/s, on the RTX 5060 laptop GPU: median 1.4 M candidates per network in a 4.4 s GPU stage.
   The GPU subtree DP solves 1.0e5 (K=12) to 3.2e5 (K=8) subtrees/s (median).
   Reconfiguration alone from the same hbt seed: at K=8 (cotengra's size) it beats cotengra's subtree_reconfigure on 28/32 networks (median −0.17 log10 C) and is a median 104× faster.
5. **Verification.** 549 trees re-costed with cotengra 0.8.2 (from_path on the ssa path). All 325 trees that carry a producer-reported C/W match it: 0 mismatches, max |Δlog2 C| = 5e-7.
   The 224 TreeSA trees have no stored value in cotengra's convention and are reported with cotengra's number. On all 32 networks, the GPU kernel equals the Float64 host reference to ≤2.7e-5 bits on 2,560 trees, with identical structure. The GPU DP equals an independent set-based CPU DP on 192 sampled chunks (K=10) and 72 (K=8), and no chunk optimum exceeds the chunk's current cost.
6. **Caveats.**
   - (b) uses a GPU plus one CPU thread; the baselines use one CPU thread. TreeSA is ntrials=1.
   - hyper-sa "1 s" really ran a median 5.9 s, because cotengra finishes a trial before it checks the clock.
   - Each method ran once; at 1 s, (b) varies by about 0.01–0.1 between runs on some qc/gm networks.
   - auto-hq was not run (it is not time-bounded, starts a 16-process pool, and the laptop was near OOM); it was done on 4 lm networks only.
   - "Best known" means the best over all trees made here.
7. **Next steps.**
   - Fix the remaining qc-circuit and Promedas losses with a W-capped objective (slicing) and ILS with SA-style acceptance.
   - Move chunk building to the GPU; the host is now about half of each reconfiguration round.
   - Run `hpg/gpucost.sbatch` on a B200 at 1/10/60 s (written, not submitted); any further evaluation goes to HPG, since the laptop is out of memory.

## What was built
| file | what |
|---|---|
| `src/Host.jl` | network loader (.tnc), the tree induced by an ordering (bucket elimination, open indices last via a port of `ftn.open_last`), exact cotengra C/W by Steiner walks, legs, ssa I/O, renumbering |
| `src/GPUTree.jl` | `perturb!`: insertion moves on orderings, one thread per job. `build!`: union-find bucket elimination to a binary tree with preorder, one thread per job. `gcost!`: Steiner walks with atomic exponent sums and warp reductions to log2 C, W, max contraction, one warp per job |
| `src/GPUReconf.jl` | batched exact subtree DP, one block per subtree of ≤12 leaves. Zeta transforms of the leg masks give flops(S1,S2) = 2^(Zall(F) − Zall(F∖S) − Zint(S1) − Zint(S2)). Subsets are processed by popcount level; the argmin table is returned for rebuilding |
| `src/Reconf.jl` | host driver. Every round grows a chunk at every internal node (expanding the costliest or a random frontier node), solves all chunks of all trees in one launch, and applies a disjoint set of improving chunks, largest gain first. Also local re-solving around dirty nodes, `kick!` (random rotations), and population ILS |
| `src/Search.jl` | (μ+λ) mass search over perturbed orderings scored by exact C; `gpu_pipeline` = mass search → reconfigure the top 8 (K=10) → ILS → K=12 polish |
| `scripts/` | `mkorders.jl` (CliqueTrees seeds), `corr_orders.jl` + `corr_ctg.py` + `corr_analyze.py` (step 1), `run_gpu.jl` (method b), `baselines.py` (a, c), `treesa_batch.jl` (d), `ablate_reconf.jl` + `ablate_table.py`, `verify.py`, `analyze.py`, `stages.py`, `times.py` |

Method (b) per network:
- Seeds: our hbt (0.5 s) or hbt-nd (5 s) ordering, plus the 4 greedy and 8 nd1 orderings. Their CPU time is charged to the budget.
- The GPU stage gets B minus that. The stage split is 40% mass search, 20% reconfiguration, 30% ILS, 10% polish.
- Median wall time is 0.98 s at B=1 and 9.5 s at B=10. For comparison, (a) took 3.3 s and 11.5 s, hyper-sa 5.9 s and 12.3 s.

Stage contributions (median drop in log10 C, from `stages.py`), measured from the raw caterpillar-binarized seed tree:

| B | mass search | reconfiguration | ILS (median / mean) |
|---|---|---|---|
| 1 s | 0.135 | 0.240 | 0 / 0.027 |
| 10 s | 0.188 | 0.049 | 0 / 0.023 |

## Results
- `results/table.md`: per-network gaps for all methods, the family medians, W/T/L, and ΔW.
- `results/corr/summary.tsv`: step 1.
- `results/verified.tsv`: the cotengra re-costing of every tree.
- `results/ablate/`: reconfiguration only.
- Logs are in `logs/`.

Median gap to the best tree known (log10 C):

| family (n) | (a) 1 s | **(b) 1 s** | hyper-sa 1 s* | TreeSA@1 | (a) 10 s | **(b) 10 s** | hyper-sa 10 s | hyper 10 s | TreeSA@10 | TreeSA 200 it |
|---|---|---|---|---|---|---|---|---|---|---|
| gm (3) | 0.733 | **0.000** | 0.989 | 0.466 | 0.295 | **0.000** | 0.353 | 1.115 | 0.167 | 0.159 |
| lm (5) | 0.053 | **0.000** | 0.003 | 0.005 | 0.053 | **0.000** | 0.000 | 0.052 | 0.003 | 0.003 |
| mc (2) | 0.661 | **0.122** | 0.533 | 0.115 | 0.192 | **0.000** | 0.661 | 7.974 | 0.074 | 0.075 |
| qc (8) | 0.376 | **0.014** | 0.170 | 0.043 | 0.174 | **0.002** | 0.020 | 0.955 | 0.017 | 0.009 |
| rnd (5) | 0.010 | **0.000** | 0.167 | 0.008 | 0.043 | **0.000** | 0.043 | 0.284 | 0.011 | 0.031 |
| str (7) | 0.455 | **0.001** | 0.219 | 0.283 | 0.110 | **0.000** | 0.132 | 0.829 | 0.035 | 0.022 |
| wmc (2) | 0.687 | **0.026** | 0.264 | 0.236 | 0.342 | **0.026** | 0.131 | 7.451 | 0.037 | 0.000 |
| all (32) | 0.259 | **0.000** | 0.190 | 0.068 | 0.134 | **0.000** | 0.043 | 0.671 | 0.021 | 0.009 |

\* the realized time of hyper-sa "1 s" has a median of 5.9 s. TreeSA@B is the most iterations (2–200) that fit in B seconds, single-threaded, ntrials=1.

Step 1 (Spearman with cotengra's C of `ftn.tree_from_order`, 96 orderings per network):

| | all orderings | perturbed variants only | top quartile |
|---|---|---|---|
| lcost | 0.90 | 0.88 | 0.76 |
| exact GPU C | 0.99 | 0.99 | 0.99 |

The GPU C equals cotengra's C exactly on 96/96 orderings of every network with neither hyperedges nor open indices, and on 94–95/96 with open indices. Elsewhere it differs only in how hyperedge buckets are binarized.
