# GPU-MF1 round latency and round count (LATENCY_NOTES)

Changelog for making GPU minimum-average-fill (`RoundsAlg(score=:mf1)`, replay driver of
GRounds.jl) faster per round and in fewer rounds on 30k-3M vertex graphs, without worse
fill. Work copy and scripts: `lat/` (lat/GRounds.jl is the working version; `lat/base/` is
GRounds.jl + GSmall.jl as of Oct 20 00:00, module GRoundsBase, for before/after). HPG:
`/blue/fairbanksj/itaykadosh/gpu/rounds/lat/` (code bundles code/, code2/, ...; results/;
logs/). Configs: **v2** = relax 0.1 (fast config), **u0.1** = relax 0.1 with urelax 0
(quality config). Serial baseline = Richard's new serial MF1 (main_table.tsv).

## Oct 20

### Step 1: per-round phase profile (B200, job 46782009, results/prof1/prof_b200.txt)
Instrumentation: `grounds(...; rprof = R)` (replay only) records, for the first R rounds
and each of the 12 round kernels, the first block start and last block end (globaltimer),
the number of work items (a warp's clique row / candidate / list chunk; a block's selected
vertex in detect/apply), the sum and the maximum of the item times (clock64), and item
sizes. `pfsum.jl` summarises; per-round matrices are in results/prof1/rounds/*.f32
(Float32, header = rows, cols; cols = round µs, then (wall µs, max item µs, items) per
phase, then nal, nc, ns). The probes cost 1.2-1.9x run time (per-block end atomics), so the
walls below are inflated; ratios between phases are what matter. With rprof = 0 the probes
compile away: same time as lat/base (tbase column of the job, ±2%) and identical fill.
The clock64-to-ns rate (from block durations) came out at 1.0-1.6 GHz although the SM ran at
1965 MHz (nvidia-smi during the job), so item times are approximate (±40%).

Per-round wall (µs, profiled) by phase:

| graph | cfg | rounds | ns/round mean / p50 | round | gaps | scan | cand | luby2 | sizes | elim12 | ob+whg | elim34 | detect | apply |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bcsstk17 (11k) | v2 | 574 | 4.5 / 2 | 127 | 21 | 8 | 12 | 6 | 17 | 29 | - | 12 | 9 | 12 |
| crystm03 (25k) | v2 | 1087 | 7.9 / 2 | 178 | 19 | 9 | 11 | 7 | 19 | 73 | - | 16 | 13 | 12 |
| jnlbrng1 (40k) | v2 | 1723 | 17.6 / 5 | 111 | 20 | 9 | 14 | 7 | 18 | 16 | - | 10 | 7 | 11 |
| ct20stif (52k) | u0.1 | 2674 | 3.3 / 1 | 137 | 21 | 8 | 12 | 6 | 17 | 38 | - | 13 | 10 | 12 |
| apache1 (81k) | u0.1 | 9450 | 5.9 / 2 | 125 | 21 | 8 | 12 | 7 | 17 | 32 | - | 12 | 7 | 10 |
| parabolic_fem (526k) | u0.1 | 1522 | 210 / 9 | 146 | 17 | 10 | 20 | 9 | 20 | 31 | - | 16 | 9 | 13 |
| ecology2 (1M) | u0.1 | 10066 | 75 / 9 | 115 | 21 | 8 | 13 | 7 | 18 | 18 | - | 13 | 7 | 11 |
| audikw_1 (944k) | u0.1 | 5051 | 24 / 3 | 285 | 20 | 10 | 14 | 8 | 20 | 120 | 113* | 27 | 16 | 13 |
| Geo_1438 (1.4M) | u0.1 | 29138 | 7.6 / 2 | 249 | 26 | 9 | 14 | 7 | 16 | 90 | 36 | 25 | 17 | 11 |
| StocF-1465 (1.5M) | u0.1 | 63881 | 10.8 / 2 | 272 | 22 | 12 | 20 | 8 | 18 | 93 | 44 | 30 | 17 | 11 |
| Queen_4147 (4.1M) | u0.1 | 140859 | 3.2 / 2 | 259 | 25 | 8 | 11 | 6 | 14 | 141 | 12* | 20 | 14 | 11 |
| Bump_2911 (2.9M) | u0.1 | 126379 | 4.2 / 1 | 433 | 10 | 16 | 30 | 14 | 32 | 93 | 122 | 72 | 30 | 13 |

(* dense-mode kernels run in only part of the rounds: per-round averages over those rounds.)

**What sets the phase time.** Per phase and round: the max item time vs the mean item time
and vs the phase wall (sums over rounds):
- elim12 (warp per clique row; Wing-Huang pairs walked one pair at a time): max item 3-7x
  the mean item (crystm03 122 vs 28 µs, Geo 129 vs 21, Queen 249 vs 37) and about equal to
  the phase wall. **The heaviest row sets the phase: load imbalance.** It is the largest
  phase on every FEM graph (30-55% of the round).
- whg (dense mode, warp per row) and elim34 (warp per row, list merge): the same pattern,
  max item 2-5x the mean (Bump whg 108 vs 16 µs; elim34 83 vs 54).
- cand: max item (a 32-entry chunk whose long lists the warp walks one after another) 5-10x
  the mean, about the wall.
- luby2, sizes, detect, apply, scan: items are uniform (max ≈ 1.2-1.5x mean) and the phase
  is a fixed 6-18 µs: a chain of dependent global loads/atomics (~0.7-1 µs each on the
  B200) plus the last-block decision. The total work is tiny: balshare (Σ item time /
  resident warps / wall) is 0.001-0.4, i.e. the GPU is >60% idle in every phase.
- gaps between the ~12 graph nodes: ~20 µs per round (~2 µs per node, two of them the
  rebuild kernels that return at once in almost every round).
- Rounds are threshold-bound, not independence-bound: median 1-2 selected per round on
  every FEM graph, nc ≈ 2 ns (sel/cand 0.45-0.7).

**Why u0.1 collapses on 3D FEM: rounds, not round cost.**

| graph | v2 rounds | u0.1 rounds | ratio | v2 ns mean | u0.1 ns mean | v2 round µs | u0.1 round µs | v2 t s | u0.1 t s | serial MF1 s |
|---|---|---|---|---|---|---|---|---|---|---|
| Geo_1438 | 10658 | 29138 | 2.7 | 18.8 | 7.6 | 274 | 249 | 2.76 | 6.10 | 6.66 |
| StocF-1465 | 19099 | 63881 | 3.3 | 33.1 | 10.8 | 308 | 272 | 5.13 | 14.29 | 17.2 |
| Queen_4147 | 18229 | 140859 | 7.7 | 22.4 | 3.2 | 276 | 259 | 5.43 | 31.34 | 11.0 |
| Bump_2911 | 22680 | 126379 | 5.6 | 19.8 | 4.2 | 424 | 433 | 9.05 | 47.68 | 237 |

- urelax 0 admits untouched vertices only at the global minimum key; FEM keys are nearly
  continuous, so a round has ~6-20 candidates (Queen u0.1: nc 5.8/round) against 40-70 in v2.
- Round cost is the same in both configs (±10%). Times are without the profile; the dense
  mode (grouped Wing-Huang, rr port) already cut Bump u0.1 from 451 s (ens job) to 48 s and
  Queen from 43 to 31 s.

**Grid size is not the cost** (job 46782204, results/grid1): blocks per SM capped at 1, 2, 4
(`bpsm` kwarg) vs the occupancy default 8: small graphs ±5-12% (bcsstk17 v2 0.060 → 0.053 s at
1/SM), large graphs slower (parabolic_fem 0.19 → 0.30 s, Geo v2 2.76 → 3.41 s at 1/SM).
Default kept; `bpsm` stays as a kwarg.

### Steps 2-4: changes (all in lat/GRounds.jl; main GRounds.jl NOT touched yet)
All exact (integer atomics, same selections): `lat/check.jl`-style runs on the laptop, lat vs
lat/base, configs mf1 v2 / u0.1, md relax 0, mf relax 0.5 no-compress, mf1 scale config with GPU
tail; variants default, esub=1, dgroup=2, dgroup=2 + ocap=64 + poolfactor 1, and a stress build
that splits every elim34 row (E34MIN=2): **150/150 identical** nnz/flops/tw on lat2_30, lat3_8,
bcsstk17, crystm03, nasasrb, apache1. Persistent :block/:grid/batch modes were NOT checked
locally (their compile takes >15 min on the laptop); hcheck.jl in the pending job does it.
1. **Row split in k_elim12!** (`esub`, default 32): when a round has fewer rows than warps,
   min(esub, warps ÷ rows) warps share each clique row; sub-warp q does the missing pairs
   t ≡ q (mod nsub). Rows > 32 words keep one warp. Bug found on the way: the new loop counter
   was named `s2`, which the pair walk already uses (state of c2) → nondeterministic fill;
   renamed `qrow_`. Lesson: kernel bodies reuse short names; check new locals against them.
2. **Row split in k_whg!** (same nsub): sub-warp q takes the q-th slice of N(a) (outside part)
   and the words kw ≡ q (mod nsub) (inside part).
3. **Row split in k_elim34!** (`e34`, default on, rows with ≥ E34MIN = 128 entries): each
   sub-warp walks the whole list once (count, hash, O count and last O entry before its slice),
   the first to arrive allocates the new list (round-tagged CAS slot `rpos`), each writes its
   slice of the merge, the last to finish (`rdone`) publishes astart/alen/hsh/cnt. New Dev
   arrays rpos/rdone (grown in growk!).
4. k_cand!: one candidate per warp when |L| ≤ warps/4 (long lists no longer walked one after
   another inside a chunk). k_sizes!: lists ≤ 32 loaded once with state/weight/count/length in
   flight together. detect_one! (d ≤ 64): (hash, count) staged in shared memory before the
   quadratic scan. Laptop: no measurable effect of these three (esub1 ≈ base).
5. The splits and the cand change apply only under GridCx (replay, :grid); :block and the batch
   keep their code paths. API: new trailing optional kernel args (pf, esub, rpos, rdone,
   ucount, uadapt, umax) with defaults, so GSmall.jl's calls are unchanged; **NCTL 40 → 41**
   (new slot C_UREL). New grounds_csr kwargs: rprof, bpsm, esub, e34, uadapt, umax.
6. **Adaptive untouched window** (`uadapt = F`, `umax = cap`; off by default): after a round
   with < F untouched candidates the untouched window widens ×2 (from 0.25% up to cap), after
   one with > 4F it halves. Lattices keep their big tie sets at the minimum, so the window
   stays ~0 there; FEM keys are continuous, so it opens.

Laptop (RTX 5060), min of 4, base / new (all changes), unit weights:

| graph | v2 base s | v2 new s | speedup | u0.1 base s | u0.1 new s | speedup |
|---|---|---|---|---|---|---|
| bcsstk17 | 0.0281 | 0.0238 | 1.18 | 0.0359 | 0.0333 | 1.08 |
| crystm03 | 0.0736 | 0.0650 | 1.13 | 0.0909 | 0.0760 | 1.20 |
| ct20stif | 0.1076 | 0.0986 | 1.09 | 0.1438 | 0.1252 | 1.15 |
| nasasrb | 0.0970 | 0.0877 | 1.11 | 0.1753 | 0.1459 | 1.20 |
| apache1 | 0.2658 | 0.2342 | 1.13 | 0.3575 | 0.3262 | 1.10 |
| jnlbrng1 | 0.0554 | 0.0551 | 1.01 | 0.0568 | 0.0536 | 1.06 |

(B200 elim12 is a larger share of the round, so the B200 gain should be larger; not measured.)

Round count vs fill, laptop (results/uexp_laptop.tsv, uexp2_laptop.tsv; nnz(L)/new serial MF1):

| graph | u0.1 rounds / nnz | urelax 0.02 | adaptive F=64 cap 0.05 | v2 |
|---|---|---|---|---|
| bcsstk17 | 817 / 0.998 | 743 / 1.006 | 652 / 1.009 | 574 / 1.013 |
| cbuckle | 1721 / 1.028 | 1383 / 1.004 | 1322 / 0.998 | 1231 / 0.991 |
| olafu | 1246 / 1.032 | 1224 / 1.011 | 1227 / 0.995 | 1144 / 0.969 |
| crystm03 | 1473 / 1.009 | 1410 / 1.050 | 1344 / 1.053 | 1087 / 1.021 |
| ct20stif | 2674 / 1.005 | 2179 / 1.002 | 1932 / 0.997 | 1779 / 1.000 |
| nasasrb | 3331 / 1.029 | 2457 / 1.009 | 1860 / 1.016 | 1506 / 1.028 |
| apache1 | 9450 / 1.097 | 9450 / 1.097 | 5118 / 1.107 | 5750 / 1.101 |
| s3dkt3m2 | 1035 / 1.018 | 1007 / 1.007 | 992 / 1.004 | 937 / 1.006 |
| lat3_20 | 260 / 0.943 | 260 / 0.943 | 233 / 0.936 | 203 / 1.166 |
| lat3_32 | 376 / 1.053 | 368 / 1.050 | 354 / 1.064 | 303 / 1.385 |
| jnlbrng1, lat2_100, thermal1 | = | = | = (±1 round) | = |

Adaptive F=64/cap 0.05: 10-45% fewer rounds than u0.1 on the FEM graphs, fill median about the
same (better on cbuckle/olafu/nasasrb, worse on crystm03 +4%), lattices kept (unlike static
urelax 0.05: lat3_32 1.13). Needs the 3D FEM (Queen/Bump/StocF/Geo) on the B200 before adoption.

### Pending (NOT submitted: HPG ssh control master died at ~03:00 Oct 20, new login needs 2FA)
Everything is staged in `lat/code3/` (bundle) and `lat/hpg/ab.sbatch`. To run, after logging in
to HPG again (opens the ssh master):

    cd ~/Projects/GATAS/tn-work/rounds/lat && hpg/submit.sh code3 hpg/ab.sbatch TAGD=ab1

The job (itay-gpus, 1 B200, ≤4 h, ~1.5 h expected) writes to
`/blue/fairbanksj/itaykadosh/gpu/rounds/lat/results/ab1/`:
1. `ab_b200.tsv` — cols: graph, n, cfg (v2/u0.1), variant (base = code before this work,
   esub1 = all splits off, no34 = without the elim34 split, new = everything), rounds,
   device nnz, nnz==base (must be true everywhere), best time s, gpu s, base/variant speedup.
   37 graphs (16 small fill graphs, lattices, the 19 large). Analysis: median speedup of `new`
   per size bucket (<30k, 30-100k, 100k-1M, 1-3M, >3M) and cfg; new vs esub1 = gain of the
   splits, new vs no34 = gain of the elim34 split. Any `false` in col 7 = a fill change: stop.
2. `prof_b200.txt` — phase profile of `new` (same format as prof1): compare elim12/whg/elim34
   walls and maxit_us with results/prof1 (expect elim12 max item ≈ mean item).
3. `rsel_b200.tsv` — graph, n, m, cfg, rounds, nnz, flops, tw, time, gpu; cfgs v2, r0.1u0 (= u0.1),
   r0.1u0.02, r0.1u0.05, r0.1a16c0.05, r0.1a64c0.05, r0.1a16c0.02, r0.2u0. Divide nnz by
   serial-MF1-new from harvest_oct19b/mf/main_table.tsv; criterion: median ≤ u0.1's median
   +1%, no graph > u0.1 + 5% (3D lattices, Queen, apache1 are the ones to watch), then pick
   the cfg with the fewest rounds on Queen/Bump/StocF/Geo.
4. The log ends with hcheck.jl: "TOTAL same N diff 0" expected (persistent :block8/:grid3 and
   batch against lat/base). If it shows diffs in block/grid/batch only, the cause is in code
   shared with GSmall (k_sizes single-pass load, NCTL), not the GridCx-only splits.

**Port to main GRounds.jl** (not done; main is still md5 b7d4809b, unchanged since this work
started): after the job passes, copy lat/GRounds.jl over ../GRounds.jl (module name is the same),
keep GSmall.jl as the GSmall agent has it (it changed meanwhile; my changes are only trailing
optional kernel arguments and NCTL), and rerun small/check.jl.
