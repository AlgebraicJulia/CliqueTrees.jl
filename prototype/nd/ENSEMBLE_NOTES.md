# GPU-MF1 fill gap vs serial MF1: diagnosis, fixes, ensembles

Changelog for closing the nnz(L) gap between GPU-MF1 (GRounds.jl, fill config:
`RoundsAlg(score=:mf1, relax=0.1)`, unit weights) and serial MF1
(CliqueTrees `MF(strategy=1)`). Scripts and raw rows are in `ens/`.
Metrics: nnz(L) = sum of column counts (diagonal included), flops = sum of
squared column counts, tw = max column count - 1 (as in fill/RESULTS.md).
Graphs: the 14 fill-config graphs of perf/lsweep.tsv.

## Oct 19

### Setup
- `ens/base.jl` -> `ens/base.tsv`: serial MF1, AMF, AMD, METIS-ND on the 14 graphs
  (laptop, `-t1`), plus serial MF1 on 4 random relabelings of each graph.
- Laptop caveat: the machine was oversubscribed by other jobs (load 8-13, swap
  full, earlyoom killing the largest Julia processes every few seconds), so
  laptop timings are noisy and long runs go through `ens/runwait.sh` (waits for
  free memory, retries a run that was killed).
- On request, all local runs were stopped at 20:10. Everything after that runs on HPG.
  Big-graph laptop rows are therefore incomplete:
  - ens/diag2.tsv has no parabolic_fem (partial), ecology2 or lat2_724 rows;
  - ens/fcap.tsv has only one row.

### Diagnosis (laptop RTX 5060; ens/diag.tsv, diag2.tsv, xdiag*.tsv, base.tsv)
The gap has three separate causes.

1. **Serial MF1's own tie-breaking luck.** Serial MF1 on 4 random relabelings of each graph:
   - median nnz(L) 1.000 × the natural-label run, max 1.053;
   - lat3_32: +4 to +13%; lat2_724: +4.4 to +5.3%; apache1: -2 to -4%;
   - the best of the 4 relabelings has median 0.991.

   So a few % of the reported gap is tie luck in the reference, not parallelism. lat2_724
   (GPU 1.046) and parabolic_fem (GPU 1.01-1.03) fall inside serial MF1's own relabeling spread.
   - Serial MF1 with the GPU's hash tie-break, emulated on the CPU (one vertex per round), gives
     1.0007 on jnlbrng1. So hash ties alone do not cost fill.
2. **Relaxation admits untouched vertices of the next score level (lattices).**
   - lat3_32 nnz/MF1 by relax: 0 -> 1.014; 0.02 -> 1.023; 0.05 -> 1.095; 0.1 -> 1.35; 0.2 -> 1.33.
     lat3_20 shows the same pattern.
   - Changing the seed does not help: relax 0.1 over seeds 1-8 gives lat3_32 between 1.21 and 1.42.
     This loss is systematic, not noise.
   - The candidates that hurt are vertices never in a clique yet ("untouched"). They are
     interior vertices at the next discrete score level, which the relax window lets in
     everywhere at once.
   - **Fix (`urelax=0`):** untouched vertices are candidates only at the minimum key; touched
     vertices keep relax 0.1. CPU reference: lat3_32 1.40 -> 1.05 and lat3_20 1.16 -> 0.94,
     with 1.0-1.5× the rounds of relax 0.1.
3. **Parallel processing of ties on 2D grids (jnlbrng1, ecology2).**
   - The gap stays at every relax value, relax 0 included: jnlbrng1 +3 to +6%;
     ecology2 +7 to +9% (relax 0) and +7 to +11% (relax 0.1).
   - Serial MF1 is stable on these graphs under relabeling (ecology2 0.985-0.99).
   - The cause is that a round commits all far-apart minimum ties at once. Serial MF1 instead
     follows the chain of neighbours whose scores drop after each elimination.
   - Partial fix, "fresh cap" (CPU, `freshmax`): at most F untouched candidates per round, the
     lowest priorities. jnlbrng1: F=4 -> 1.010, F=16 -> 1.016, against 1.05.
     - Rounds rise 2-8×.
     - Choosing a random subset each round instead does not help (1.044).
     - On the other graphs the effect is within noise (both signs, ±3%).
   - GPU version: `fcap=F` keeps untouched candidates with hash <= H, where H is set from the
     previous round's untouched count, so that about F pass.
   - Not adopted. It is in the B200 sweep as `u0.1f16`.
   - Other things that did not help:
     - CPU tail on the last 1/5/20%: no gain, since the damage is done early;
     - absolute delta: no better than a small relax;
     - touched-first ties or touched-first candidates: worse (jnlbrng1 1.07);
     - index ties: implemented (`ties=:index`), not measured, because of the laptop.

### Changes to GRounds.jl (defaults unchanged; v2 fill config gives the same nnz/flops/tw as perf/GRounds_v2.jl)
- `urelax`, `udelta` kwargs: a separate candidate threshold for untouched vertices.
  - Per-vertex touched flag `tch`, set in `k_apply!` for committed rounds only, so the redo
    after a capacity overflow stays deterministic.
- Device fill counters: nnz(L), flops = sum cc^2, and tw are accumulated per eliminated
  supervariable in `k_apply!`.
  - A member-count array `nm` follows absorptions.
  - The CPU tail is counted through its column counts.
  - Returned in `timing[:nnz]`, `timing[:flops]`, `timing[:tw]`.
  - Checked equal to CliqueTrees column counts on every 4th diag2 run and in ens/check.jl.
- `grounds_ens(weights, graph, members; metric=:nnz, conc)`:
  - `members` are `(RoundsAlg, kwargs)` pairs;
  - one Julia task (so one CUDA stream) per member, with a shared CSR;
  - the member with the smallest device-side metric wins.
  - Each member's host loop yields while it syncs, so the CUDA-graph replays of the members
    overlap on the device.
- `fcap` kwarg (fresh cap, above). After a round that selects nothing, the next round has no
  cap, so the loop always progresses.
- `ties=:index` is now honoured on the GPU (the low bits of the priority are the vertex index).
- Old version 2: perf/GRounds_v2.jl (module GRoundsV2).

### Ensembles, offline (best member per graph from ens/diag2.tsv; 12 of the 14 graphs; nnz(L)/MF1)
| ensemble | median | max |
|---|---|---|
| v2 single (relax 0.1) | 1.022 | 1.352 |
| u0.1 single | 1.016 | 1.055 |
| u0.1 × seeds 1-4 | 1.005 | 1.051 |
| u0.1 × seeds 1-8 | 1.005 | 1.036 |
| u0.1:1-2, u0.05:1, b0.02:1 (k=4) | 1.005 | 1.047 |
| u0.1:1-3, u0.05:1-3, b0.02:1-2 (k=8) | **1.000** | 1.047 (jnlbrng1) |
| u0.1:1-4, u0.05:1-4 (k=8) | 1.003 | 1.045 |
| relax 0 × seeds 1-4 | 0.998 | 1.032 (but 3-10× the rounds) |

- k=8 mixed ensemble against MF1, per graph:
  - flops median about 0.99;
  - tw is within ±10%.
- Cost:
  - the u0.1 members need 1.0-1.5× the rounds of v2;
  - b0.02 members need 1.4-3.8×.

  The wall time of a concurrent ensemble is bounded by its slowest member, so the B200 sweep
  also times a cheaper family without b0.02 (`fast-` rows).
- Laptop smoke test (lat2_40): 4 members took 11 ms run concurrently and 26 ms run one after
  the other.

### B200 sweep
- **itay-gpue, job 46747676** (ens/hpg/gpue.sbatch, driver ens/hpg/esweep.jl)
  - Output: /blue/fairbanksj/itaykadosh/gpu/rounds/ens/results/ens_b200.tsv.
  - Graphs: the 14 fill graphs, the 8 other small SuiteSparse graphs, and the 19 large fill graphs.
  - cfgs: v2, u0.1, u0.05, u0.1f16, ens2/4/8 (family u0.1:1,u0.05:1,b0.02:1,u0.1:2,u0.05:2,b0.02:2,u0.1:3,u0.05:3) and seq4.
  - Then `fast-ens2/4/8` (u0.1/u0.05 seeds only) on the small sets.
  - Each row has the member nnz(L) list, so best-of-k for any prefix can be read off the ens8 row.
- **itay-ecpu, job 46747636** (hpg-milan)
  - Serial MF1/AMF/AMD/METIS-ND plus 4 MF1 relabelings on the 14 graphs.
  - Output: results/base_milan.tsv.
  - Serial baselines for the large set: job 46702562 (still pending).
- Still open: the ensemble wall time and speed vs serial MF1 above 100k vertices, and fill on
  ecology2, lat2_724 and parabolic_fem with the fixes. All come from these jobs.
