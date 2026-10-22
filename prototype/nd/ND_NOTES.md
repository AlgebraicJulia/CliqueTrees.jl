# GPU nested dissection + batched GPU MF1 leaves (GND)

Goal: GPU-parallel nested dissection whose leaves are ordered by batched GPU MF1
(`grounds_batch`), one GPU pipeline, for 100k-8M-vertex graphs: much faster than Richard's
serial MF1 (aim >= 10x at 1M+) and faster than METIS-ND + serial MF1, with fill as good as
min(serial MF1, METIS-ND) per graph.

Code (new files only; GRounds.jl / GSmall.jl untouched, called through `grounds_batch_csr`):
- `nd/GND.jl`: module GND, `gnd_perm(ptr, adj; opts=GNDOpts(...), leafsolve)`.
- `nd/cpund.jl`: CPU recursive METIS separators + the same leaf solver (baseline
  "CPU METIS-ND + GPU batched leaves").
- `nd/ndbench.jl`: B200 driver (configs gnd / gndcpu / metis), `nd/hpg/ndgpu.sbatch`.
- `nd/hpg/ndcpu.jl`, `nd/hpg/ndcpu.sbatch`: CPU baselines on milan (METIS_NodeND,
  CliqueTrees `ND{3}(MF1, METISND; width=L)`).
- `nd/ndutil.jl` (loading, CSR, fill recount = sbench.jl `fillq`), `nd/cpufm.jl` (CPU
  reference node FM used to debug the GPU refinement), diagnostics `nd/*diag*.jl`, `nd/ltest*.jl`.
- HPG: `/blue/fairbanksj/itaykadosh/gpu/rounds/nd/{code,results,logs,cenv}`; `nd/` itself
  holds copies of Rounds.jl / GRounds.jl / GSmall.jl / gens.jl (GRDIR).

## Design (Oct 20)

All data stays on the device; the host only reads a few counts per step (sizes for
allocation, per-part weights for the split decisions) and downloads the leaf union CSR once.

1. **Twin compression** (closed neighbourhoods): order-independent hash, open-addressing table
   (representative = smallest vertex), verified by a merge of the sorted lists. Quotient graph
   with vertex weights = class sizes, edge weights = original edge counts. 3-dof FEM graphs
   shrink 3x (9x in edges).
2. **One ND level at a time, all parts together.** The parts of a level are contiguous vertex
   ranges of one disjoint-union graph, so each step is a few launches for all parts:
   - *coarsening*: locally dominant heavy-edge matching (edge priority = (weight, symmetric
     hash); every locally maximal edge is mutual, 4 sweeps), then 2-hop pairing of unmatched
     vertices that share their heaviest neighbour when > 10% stay unmatched (METIS rule;
     without it FEM coarse graphs became stars and coarsening stalled at ~4k vertices).
     Contraction from member lists (per-row sort/merge). Coarse vertex weight capped at
     1.5 W(part)/C0; a part stops at C0 = 100 vertices; contiguity of parts is preserved
     (representative = smaller endpoint, prefix-sum numbering).
   - *initial separator* (coarsest graph): one warp per part, one trial per lane (32
     trials): BFS-grown half, frontier = separator (the lane-serial node FM there is off by
     default, `npass=0`: it was O(m^2) per lane and the warp FM below follows anyway).
   - *uncoarsening*: projection, then **1-sided node FM (METIS FM_2WayNodeRefine1Sided)
     with one warp per part**: alternating target side per pass (2 x 10 passes), every move
     goes into the target side (so no locks), stop after min(3|S|, 300) non-improving moves
     (or limit and S > 1.1 best), rollback to the best prefix. Gains are kept incrementally
     per separator-list position (moving v changes no gain; pulling u raises its separator
     neighbours' gains by w(u)); the warp scans gains (coalesced) instead of neighbourhoods.
   - *replicas* (METIS MlevelNodeBisectionL2): the level is coarsened 4 times once, then
     the coarse graph is copied R times into one union graph (different hashes -> different
     matchings, BFS seeds, tie-breaks), the R trials run to the end, the best separator per
     part is projected down the 4 shared levels. R = clamp(4 nSM / parts, 1, 8): free at the
     top levels, where the GPU is otherwise idle and quality matters most.
   - *speed defaults* (Oct 20, B200 profiles): FM limit 150 (METIS 300); FM at the two finest
     levels, skip levels 3-4, FM again above; at levels with parts >= 20k vertices, 3 parallel
     positive-gain passes thin the projected separator before the FM (FM thinned 60-100k
     vertices one move at a time on lat3_140); chunked argmax (max of every 32 list entries)
     for lists > 256.
   - *split*: per part A | B | S ranges of the final order (top-down positions); a child of
     at most Lb original vertices becomes a leaf, else a part of the next level
     (renumbered contiguously with prefix sums, edges filtered on the device).
3. **Leaves**: cut out of the original (uncompressed) graph on the device as one union CSR
   (local ids), downloaded once, ordered by one `grounds_batch_csr` call (MF1, relax 0.1,
   unit weights), orders written into each leaf's range on the device; perm downloaded.

## Changelog

### Separator quality (laptop, top separator vs METIS_ComputeVertexSeparator)
| refinement | lat3_10 (METIS 100) | lat2_100 (101) | lat3_20 (400) | thermal1 (119) | bcsstk17 (144) |
|---|---|---|---|---|---|
| greedy positive-gain moves (Jacobi, one side per part) | 304 | 159 | - | 143 | 144 |
| jet-style (negative moves + afterburner) | 145-177 | 120-149 | 627-739 | 127-145 | 144-342 |
| warp 2-sided FM (own rules) | 157-182 | 108-160 | 653-730 | 119-139 | 144-375 |
| warp 2-sided FM, METIS rules | 126-150 | 100-130 | 540-645 | 118-127 | 126-216 |
| **warp 1-sided FM (METIS ND default)** | **100** (8/8 seeds) | 100-110 | 400-406 (7/8) | 114-119 | 126-222 |
| 1-sided FM, 8 replicas | 100 | 100 | 400 | 114-116 | 126 (7/8) |

- Diagnosis path: METIS run on *our* coarse graphs gives separators of the same weight as
  ours at each coarse level, and METIS's coarse separator projected + our CPU FM reaches
  ~100 on lat3_10, so coarsening and initial separators were fine; the refinement was the
  gap. A CPU reference FM (`cpufm.jl`) reproduced the GPU numbers (no kernel bug), and
  switching both to METIS's 1-sided rules closed it.
- Bug found on the way: atomics on the gain array bypass L1, so later plain loads in the
  same warp read stale gains; gain updates are now serialized per pulled vertex (no atomics).

### Fill (laptop, GND + serial MF1 leaves, Lb = 1024, 8 replicas; flops ratio)
| graph | vs MF1 | vs METIS-ND |
|---|---|---|
| lat2_100 | 0.999 | 1.026 |
| lat3_20 | 0.526 | 1.065 |
| lat3_25 | 0.418 | 1.020 |
| thermal1 | 0.952 | 1.007 |
| G2_circuit | 0.616 | 1.003 |
| apache1 | 0.869 | 0.883 |
| bcsstk17 (11k) | 1.542 | 0.977 |
bcsstk17 is the MF1-favoured case (Lb 4096: 1.14 vs MF1).

### Speed work (B200 profiles prof1-prof6, jobs 46788902 ... 46797485)
Where the time went, in order of discovery (ND stage only; leaves were always 20-400 ms):
| change | effect |
|---|---|
| first version (2-sided FM, replicas of the full graph) | parabolic_fem ND 12 s, lat3_64 186 s: coarsening stalled at ~4k-vertex star-like coarse graphs, so the lane-serial initial FM ran O(m^2) on big graphs |
| 2-hop pairing in the matching (sort-based, then atomicMin-based, deterministic) | coarsening reaches C0; lat3_64 ND 186 s -> 5.3 s |
| 1-sided FM, incremental gains (separator list + gain per list position) | 2-4x on the FM; a list-rebuild bug (pulled-then-moved vertices re-added) found and fixed |
| replicas only after 4 shared coarsening levels | coarsening 0.3-1.0 s -> 0.05-0.15 s per graph |
| no lane FM in the initial separator (`npass=0`) | initsep 0.1-0.25 s -> < 5 ms |
| FM limit 150, skip FM at levels 3-4, pre-thinning by 3 parallel positive-gain passes at parts >= 20k | ND -30..-45% (lat3_140 7.8 -> 5.3 s), fill +0-4% (lat3 up to +10%, see below) |
| chunked argmax, warp-cooperative gains of pulled vertices, pulled data kept in registers | small (per move 4.2k -> 3.7k cycles on the laptop) |

- Per-move cost of the warp FM, laptop clock counters (`fm_cyc`): argmax 1.35k cycles,
  bookkeeping 0.2k, pulls 0.65k, gain updates 1.4-1.6k, chunk maxima 0.5k = ~3.7k cycles
  (~2-3.5 us per move on the B200 as well). A move is a chain of ~15 dependent global loads.
- Moves on the critical path (sum over FM launches of the slowest part): lat3_40 192k moves.
  65% of FM moves are exploration that is rolled back (METIS rule); METIS does the same on one
  CPU core at ~0.1 us per move. So the GPU wins only where many parts run at once (deep ND
  levels); the top 2-3 ND levels (1-4 parts) are latency bound and dominate (lat3_140: top
  level 1.9 s of 5.3 s ND).
- Tried and not kept:
  - jet-style parallel refinement at big levels (`tfm`): 5-15% larger separators, lat3_64 fill
    +34%;
  - parallel FM with K warps per part (`pfmk`, CAS on the label word for pulls, per-warp best
    prefix + repair of rolled-back pulls next to kept moves, undo of a worse pass): 2-5x faster
    on top separators but 5-30% larger separators (stale gains across warp segments);
  - second-level argmax (max of 32 chunk maxima): left disabled, the list became inconsistent
    in tests (`cm2`, not debugged further; chunking alone did not change lat3_40 timings);
  - matching priority by small degree sum (`mdeg`), 8 matching sweeps, C0 50/200: within
    seed noise.

## Results (Oct 20; B200 jobs 46799448/46799449/46800471/46805365, milan CPU job 46788297)

Setup: GPU rows on one B200 (time = host CSR upload .. permutation on the host, min of 2
runs, includes the grounds_batch leaf call); serial MF1 = Richard's new MF1 from main_table
(milan, 1 core, job 46754897); METIS-ND = METIS_NodeND and "ND+MF1 (CT)" = CliqueTrees
`ND{3}(MF(strategy=1), METISND(); width=2048|8192)` (best of the two), both rmain + Metis on
milan, 1 core (`nd/results/cpu_milan.tsv`); "CPU-METIS + GPU leaves" = recursive
METIS_ComputeVertexSeparator on the host down to leaves of 4096 + the same grounds_batch call
(`cpund.jl`, B200 node CPU). Fill = sbench.jl recount (unit weights, nnz(L) with diagonal,
flops = sbench formula). Default GND options (`GNDOpts()`, Lb = 1024). Tables:
`python3 nd/analyze.py` with MAIN=gnd:1024 on `nd/results/full_{d,a,b,c}_b200.tsv`
(`nd/results/final_tables.md`).

Main config (gnd, Lb 1024, speed defaults):

### Per size bucket (geometric means; fill ratios vs min(MF1, METIS-ND))

| bucket | graphs | nnz vs best (max) | flops vs best (max) | nnz vs MF1 | nnz vs METIS | x MF1 (min-max) | x METIS-ND | x ND+MF1 | x CPU-METIS+GPU leaves |
|---|---|---|---|---|---|---|---|---|---|
| 100k-1M | 12 | 1.036 (1.070) | 1.073 (1.286) | 0.897 | 1.030 | 1.46 (0.38-4.98) | 4.12 | 12.8 | 3.63 |
| 1-3M | 12 | 1.038 (1.115) | 1.057 (1.224) | 0.649 | 1.038 | 5.59 (2.11-87.08) | 8.02 | 76.2 | 10.28 |
| >=3M | 4 | 1.054 (1.112) | 1.094 (1.142) | 0.711 | 1.054 | 4.46 (3.19-6.94) | 14.02 | 146.7 | 13.56 |


Per graph:

| graph | n | nnz vs best | flops vs best | best | t GPU s (nd / leaves) | x MF1 | x METIS-ND | x ND+MF1 (CT, best L) | x CPU-METIS+GPU leaves | fill CPU-METIS+GPU leaves vs best |
|---|---|---|---|---|---|---|---|---|---|---|
| lat2_224 | 99904 | 1.024 | 1.165 | METIS | 0.144 (0.128 / 0.014) | 0.54 | 2.51 | 2.5 | 1.82 | 1.011 |
| thermomech_TC | 102158 | 1.046 | 1.057 | METIS | 0.148 (0.130 / 0.016) | 1.44 | 2.80 | 3.3 | 2.22 | 1.075 |
| Dubcova3 | 146689 | 1.003 | 1.085 | METIS | 0.131 (0.115 / 0.012) | 0.78 | 5.30 | 3.6 | 5.44 | 1.015 |
| G2_circuit | 150102 | 1.042 | 1.040 | METIS | 0.231 (0.209 / 0.020) | 1.68 | 2.57 | 15.8 | 2.16 | 1.074 |
| lat3_40 | 187200 | 1.026 | 0.984 | METIS | 0.478 (0.458 / 0.017) | 0.38 | 2.83 | 17.8 | 2.27 | 1.022 |
| thermomech_dM | 204316 | 1.043 | 1.062 | METIS | 0.174 (0.148 / 0.024) | 2.84 | 4.89 | 6.2 | 3.89 | 1.072 |
| lat2_362 | 261364 | 1.055 | 1.286 | METIS | 0.277 (0.252 / 0.021) | 1.03 | 3.60 | 5.9 | 2.77 | 1.013 |
| lat2_512 | 523264 | 1.029 | 1.103 | METIS | 0.385 (0.349 / 0.031) | 2.10 | 5.75 | 10.8 | 3.88 | 0.986 |
| parabolic_fem | 525825 | 1.049 | 1.000 | METIS | 0.569 (0.519 / 0.044) | 2.87 | 4.20 | 43.1 | 3.18 | 1.097 |
| lat3_64 | 774144 | 1.070 | 1.134 | METIS | 1.285 (1.233 / 0.043) | 1.21 | 5.05 | 54.7 | 4.36 | 1.045 |
| audikw_1 | 943695 | 0.991 | 0.974 | METIS | 0.851 (0.637 / 0.073) | 4.98 | 8.84 | 70.0 | 22.05 | 1.024 |
| ecology2 | 999999 | 1.062 | 1.022 | MF1 | 0.807 (0.737 / 0.060) | 2.27 | 4.64 | 33.6 | 3.82 | 1.134 |
| lat2_724 | 1046904 | 1.000 | 0.934 | METIS | 0.509 (0.451 / 0.049) | 3.36 | 9.14 | 18.9 | 7.27 | 0.965 |
| thermal2 | 1228045 | 1.028 | 1.008 | METIS | 0.608 (0.520 / 0.076) | 6.81 | 10.52 | 33.6 | 8.40 | 1.052 |
| Serena | 1391349 | 1.012 | 1.009 | METIS | 1.219 (0.985 / 0.093) | 4.52 | 7.30 | 126.0 | 13.31 | 1.012 |
| Geo_1438 | 1437960 | 1.034 | 1.074 | METIS | 1.384 (1.234 / 0.085) | 4.81 | 5.90 | 99.2 | 11.62 | 1.041 |
| StocF-1465 | 1465137 | 1.045 | 1.070 | METIS | 1.923 (1.743 / 0.160) | 8.96 | 7.17 | 250.6 | 5.17 | 1.066 |
| Hook_1498 | 1498023 | 1.035 | 1.054 | METIS | 1.148 (1.013 / 0.080) | 5.28 | 7.93 | 77.7 | 14.26 | 1.044 |
| af_shell10 | 1508065 | 1.022 | 0.997 | METIS | 0.538 (0.329 / 0.072) | 2.11 | 4.37 | 19.8 | 19.04 | 1.056 |
| Flan_1565 | 1564794 | 1.010 | 1.013 | METIS | 1.075 (0.711 / 0.115) | 3.42 | 10.01 | 104.2 | 24.18 | 1.040 |
| G3_circuit | 1585478 | 1.066 | 1.069 | METIS | 0.864 (0.758 / 0.093) | 7.15 | 9.29 | 124.0 | 6.44 | 1.066 |
| lat2_1024 | 2095104 | 1.061 | 1.224 | METIS | 0.903 (0.788 / 0.095) | 3.68 | 11.44 | 32.2 | 7.49 | 1.005 |
| Bump_2911 | 2911419 | 1.039 | 1.065 | METIS | 2.717 (2.322 / 0.142) | 87.08 | 6.59 | 178.6 | 12.80 | 1.028 |
| lat3_100 | 2970000 | 1.115 | 1.200 | METIS | 2.910 (2.726 / 0.145) | 2.38 | 9.81 | 169.3 | 7.03 | 0.993 |
| Queen_4147 | 4147110 | 1.008 | 1.011 | METIS | 3.365 (2.273 / 0.313) | 3.27 | 9.27 | 252.3 | 24.64 | 1.052 |
| lat2_1448 | 4190512 | 1.041 | 1.101 | METIS | 1.455 (1.181 / 0.181) | 5.47 | 15.09 | 63.2 | 10.11 | 1.004 |
| lat3_140 | 8173200 | 1.112 | 1.142 | METIS | 6.021 (5.372 / 0.404) | 3.19 | 14.04 | 349.2 | 10.23 | 0.965 |
| lat2_2048 | 8384512 | 1.056 | 1.127 | METIS | 2.475 (1.873 / 0.392) | 6.94 | 19.66 | 83.1 | 13.28 | 0.995 |


Quality preset (`fmlimit=300, fmfine=99`: METIS's FM limit and FM at every coarse level),
job 46805365, 10 graphs where the default lags most:
| graph | n | nnz vs best | flops vs best | t s | x MF1 | x METIS-ND | default: nnz vs best / x MF1 |
|---|---|---|---|---|---|---|---|
| lat3_40 | 187k | 1.031 | 1.006 | 1.08 | 0.17 | 1.25 | 1.026 / 0.38 |
| lat3_64 | 774k | 1.077 | 1.157 | 2.51 | 0.62 | 2.59 | 1.070 / 1.21 |
| thermal2 | 1.23M | 1.024 | 0.998 | 1.13 | 3.68 | 5.68 | 1.028 / 6.81 |
| G3_circuit | 1.59M | 1.041 | 1.021 | 1.60 | 3.86 | 5.00 | 1.066 / 7.15 |
| lat2_1024 | 2.10M | 1.046 | 1.159 | 1.55 | 2.15 | 6.66 | 1.061 / 3.68 |
| lat3_100 | 2.97M | 1.035 | 1.056 | 5.29 | 1.31 | 5.40 | 1.115 / 2.38 |
| Queen_4147 | 4.15M | 1.010 | 1.021 | 4.68 | 2.36 | 6.67 | 1.008 / 3.27 |
| lat2_1448 | 4.19M | 1.010 | 0.999 | 2.25 | 3.54 | 9.78 | 1.041 / 5.47 |
| lat3_140 | 8.17M | 1.078 | 1.106 | 9.51 | 2.02 | 8.89 | 1.112 / 3.19 |
| lat2_2048 | 8.38M | 1.019 | 1.006 | 3.57 | 4.81 | 13.65 | 1.056 / 6.94 |

Other configurations (all graphs, `nd/results/full_*`): Lb 2048 / 4096 give the same time
(+-3%) and 1-2% more nnz than Lb 1024. Run-to-run (node-to-node) time variation seen up to
1.6x on lat3_64 (1.28 s job 46800471 vs 2.0 s job 46805365), +-10% elsewhere.

## Verdict

- Fill: within 0-12% of min(serial MF1, METIS-ND) in nnz(L) on all 28 graphs (geomean
  1.04), better than METIS on audikw_1; flops geomean 1.06-1.09, worst on 2D lattices
  (lat2_362 1.29, lat2_1024 1.22: leaves ordered by MF1 instead of further dissection).
  Worst nnz: 3D lattices with the speed defaults (lat3_100 1.115, lat3_140 1.112); the
  quality preset brings them to 1.035 / 1.078 at ~1.7x the time. 0.65-0.90x serial MF1's nnz
  on 1M+ graphs (METIS-ND is the better one on 27 of 28; MF1 only on ecology2).
- Speed vs serial MF1 (B200 vs 1 milan core): 100k-1M geomean 1.5x (0.4x lat3_40 .. 5x
  audikw_1; below ~300k the GPU ND is slower than serial MF1), 1-3M 5.6x (2.1x af_shell10 ..
  9x StocF; 87x Bump where MF1 is pathological), >=3M 4.5x (3.2-6.9x). The >=10x target at
  1M+ is NOT met except Bump.
- vs METIS-ND (1 core): 4x / 8x / 14x by bucket; vs CliqueTrees METIS-ND + serial MF1 leaves:
  13x / 76x / 147x (and that baseline's fill is 1.5-4x METIS); vs CPU METIS separators + the
  same GPU leaves: 3.6x / 10x / 14x, with similar fill (METIS separators are 3-13% better on 3D/2D
  lattices, worse on ecology2/parabolic/Queen).
- Why not 10x: the node FM is a sequential chain of ~3.7k-cycle moves (~15 dependent global
  loads each); deep ND levels run thousands of parts in parallel, but the top 2-3 ND levels
  (1-4 parts) are latency bound; the top 3 ND levels take 40-60% of the ND time on 1M+ graphs. Parallel
  replacements tried (jet-style, multi-warp FM) lose 5-30% separator size, which costs more
  fill than the 1.1-1.5x they save. What is left: shared-memory labels/gains for the FM (est.
  1.3-1.5x on FM), a block-cooperative FM for the top levels, fewer ND levels in the first
  place (a k-way first split), and the grounds_batch device-input API (10-60 ms per graph).

### API requests for GSmall.jl (grounds_batch)
1. Device input: take the leaf union CSR as device arrays (ptr, adj, per-leaf offsets) and
   return the orders on the device. Now the leaves go device -> host (leaf CSR) -> host
   packing -> device and back (`leafcut` + `leafprep` + batch pre/post: 10-60 ms per graph,
   up to 15% of the end-to-end time at 1M).
2. Constrained (halo) ordering: leaf + adjacent separator vertices, the halo never eliminated
   (CAMD-style), so leaf MF1 sees the fill it causes in the separators. METIS-ND leaves are
   ~100-200 vertices; with 1-2k MF1 leaves the halo is where the remaining fill gap to METIS on
   meshes is expected (to be measured).

## Open items / how to continue
- Leaf bound: fixed `Lb = 1024` after the sweep (1024/2048/4096: same end-to-end time within
  +-3% on the B200 because the batch is 3-15% of the total; fill improves monotonically with
  smaller leaves). The replica count is from the device (4 x SM count / parts, <= 8). A
  cost model for Lb only matters once the ND stage gets much cheaper.
- Nothing is running on HPG for this task. Re-run: `sbatch --export=ALL code/hpg/ndgpu.sbatch`
  in `/blue/fairbanksj/itaykadosh/gpu/rounds/nd` with NAMES / CFGS / OUT / GRDIR (see
  `ndbench.jl` header); CPU baselines `code/hpg/ndcpu.sbatch`; tables `python3 nd/analyze.py`.
- Laptop tests use `nd/snap/` (v2 GRounds, GRDIR=snap) when the rounds/ copy does not compile
  on sm_120; the B200 runs used the rounds/ GRounds.jl / GSmall.jl of Oct 20 (new
  grounds_batch API, nw = 0, weights = nothing).
- Results are not bit-reproducible run to run: the per-part balance budget in the greedy
  pre-thinning is claimed with atomics (separator sizes vary by a few vertices).

## Closing the AMD gap (Oct 20, second session)

Goal: GPU ND + MF1 leaves at serial-AMD speed for n >= ~500k without losing its fill.
New files: `nd/hybfm.jl` (CPU+GPU hybrid refinement, staging arena, calibration), `nd/kway.jl` (top-k
variant), `nd/policy.jl` (small-graph policy: cost model `auto`, race `race`), `nd/analyze3.py`,
`nd/hpg/ndhyb.sbatch`; `GND.jl` gains options `hyb`, `cfmlimit`, `cfmpass`, `hybmin`, `kway`, `kwc0`,
`kwtrials`, kwargs `leafdev` / `abort`; `GSmall.jl` gains `grounds_batch_dev` (appended; existing API
untouched); `small/check.jl` gains mode `batchdev`.

### 1. CPU+GPU hybrid refinement (`hyb=1`)
- Where the time was (laptop RTX 5060 profile, parabolic_fem): GPU FM kernel 74% of GPU time; the top
  level's finest FM alone 55 ms for 24.8k moves on its critical path (2.2 us/move, one warp, one part).
- Design: the GPU coarsens/projects as before. During uncoarsening, at each level the GPU would refine,
  a cost model decides whether the CPU threads refine this and every finer level of the ND level:
  labels are downloaded once, the host projects (`hproject!`) and runs METIS FM_2WayNodeRefine1Sided
  (`cpu_fm1_part!`: binary max-heap of gains, incremental gains of pulled vertices, rollback to the best
  prefix; same rules as `k_wfm1b!`) on all parts in parallel (dynamic queue, largest part first), every
  level (no skipped levels, METIS limit min(3|S|, 300)), and uploads the final labels once.
  Level graphs come from a pinned staging arena (`CUDA.pin`, persistent, bump-allocated); the finest graph
  of an ND level is prefetched on a helper task/stream as soon as the level starts (overlaps the GPU
  coarsening); coarser levels stream to the host in refinement order while the CPU works.
- Cost model (generic; constants measured on the machine by `hyb_calibrate!` after warm-up, grids
  300^2, 30^3, 500^2): GPU per refined level = glaunch + gsep * (largest part's separator); CPU =
  cfix + sum over remaining levels of csep * max(spmax, spsum / T) + cvert * n / T + 5 lat + bytes / bw.
  The GPU rate is re-estimated online from CUDA events around the GPU levels (read at the end of each ND
  level, no extra syncs); the CPU rate from the measured part times. (A first version calibrated with
  plain least squares picked up a 150 ms compile outlier: medians now.)
- Laptop (RTX 5060 + Ryzen AI 7 350, -t8), ND stage only, min of 4: parabolic_fem 0.339 -> 0.187 s,
  ecology2 0.465 -> 0.271, lat2_724 0.387 -> 0.299, thermomech_dM 0.136 -> 0.100, lat3_40 0.272 -> 0.169.
  "CPU refines every level" (`hyb=2`) is worse than the model's choice on most graphs (deep levels with
  hundreds of parts stay faster on the GPU).
- Fill (4 seeds, MF1 leaves, Lb 1024): nnz parabolic +0.0%, ecology2 +1.2%, lat2_724 -1.2%,
  thermomech_dM -0.2% (within seed spread: ecology2 seeds span 35.5-36.8M). With the quality GPU
  settings for the remaining GPU levels (`fmlimit=300;fmfine=99;pregreedy=0`, "hybq") nnz -0.2..-2.5%,
  flops -1..-7% for +10-15% time.

### 2. Device-resident leaves (`gndd`, `GRounds.grounds_batch_dev`)
- `grounds_batch_dev(uptr, uadj, loffH, alg)`: the leaf union CSR stays on the device; packing
  (`k_dscatter!`), live lists (`k_dalive!`/scan/`k_dlive!`) and the order expansion (`k_demit!`, one thread
  per graph, same DFS as `emit!`) are kernels; same arena sizes / kernels / parameters as
  `grounds_batch_csr(arena=:auto)`; graphs with a CPU tail fall back to the host unpacking.
- Checks: `small/check.jl` (QUICK, block8/batch/batchdev, lat2_30 lat3_8 bcsstk17 thermal1): existing
  API identical fill (all modes same as GRounds_pre_small), batchdev 92/92 same fill as the reference;
  per-leaf nnz identical to grounds_batch_csr on the GND leaves of thermomech_dM / parabolic_fem (orders
  differ only inside rounds, where the host path is itself nondeterministic: atomic order positions).

### 3. Top-k variant (`kway=d`)
- One GPU coarsening to ~kwc0 * 2^d vertices; on the coarsest graph the CPU computes a depth-d nested
  dissection (BFS-grown separators, best of 8, node FM); every vertex gets a tree code (heap numbering);
  uncoarsening on the CPU: projection, then per level and per depth the node FM on each tree node's
  region (vertex-list version of the same FM); a vertex leaving S goes into its child's separator (always
  valid, the next depth's FM moves it down; an earlier version that only re-coded vertices whose code was
  the node itself produced A-B edges, fixed). The ND driver turns the codes into A/B/S labels for its first
  d levels (`k_kwlab!`). Validity checked (no edge between different subtrees).
- Laptop: replaces the top 2/3 levels' 57/84 ms by 41/63 ms (parabolic) but nnz +0.7/+3%, flops +5/+20%
  (lat2_724 kw3: +5% nnz): the deep tree nodes' separators come from a whole-graph coarse hierarchy.

### 4. 2D lattices
- Separator order inside a separator does not matter for fill (it is a clique when eliminated); the
  flops gap was refinement (limit 150 + skipped levels) and leaf size. With the hybrid (CPU refines every
  level with METIS limits) lat2_362 flops / METIS 1.27 -> 1.09 (laptop), -> 1.07 with Lb 256; parabolic
  nnz 27.0M (Lb 1024) -> 26.2M (256) -> 25.8M (128), METIS 25.6M; leaf batch cost +1-5%.

### 5. Small-graph policy
- `auto` (cost model from grid calibration, t = c (n+m) serial, a0 + a1 (n+m) GPU ND): mispredicts AMD
  on FEM graphs by ~2x (grids are AMD's cheap case), so it is only a fallback.
- `race` (policy.jl `race_perm`): serial AMF on one CPU thread and GPU ND at once; when AMF finishes at
  ts, GPU ND may run until slack * ts (1.25) and is then abandoned at its next ND-level boundary
  (`abort` kwarg). Time <= ~slack * t(AMF) + one ND level; fill = GND's whenever GND makes the deadline.

### B200 results (jobs 46872484-87, 46873507-09; Xeon Platinum 8570 host, --cpus-per-task=32)
Code `/blue/fairbanksj/itaykadosh/gpu/rounds/nd/code3` (GRDIR `nd/grd3`, env `nd/henv` = rounds/env + AMD.jl),
raw rows `nd/results3/*_b200.tsv` (copied locally), tables: `python3 nd/analyze3.py nd/results3/*_b200.tsv`
(env AFTER / BEFORE pick configs; key = cfg:Lb@threads) -> `nd/results3/tables.md`. Baselines: AMD and METIS-ND
= pbase serial.tsv (milan, rmain + AMD.jl, job 46781988), MF1 / AMF = main_table (milan), "AMD (node)" = SuiteSparse
AMD through CliqueTrees on the B200 host CPU in the same job (it is 1.1-2.4x faster than milan on the 0.5-1M graphs).
ParAMD / ParMETIS: jobs 46781984-87 are still pending (no rows). BEFORE = `gnd:1024` (old default, re-run in the
same job), AFTER = `gndd:256:hyb=1;rmax=4` (hybrid + device leaves + Lb 256 + at most 4 replicas), 32 threads.

| bucket | graphs | before/AMD | after/AMD | after/AMD(node) | after/METIS | after/MF1 | nnz before/METIS | nnz after/METIS | nnz after/MF1 | nnz after/AMD | flops after/METIS | after <= AMD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100k-500k | 7 | 5.25 | 3.08 | 3.41 | 0.188 | 0.611 | 1.034 | 1.002 | 0.916 | 0.865 | 1.006 | 0/7 |
| 500k-1M | 5 | 2.01 | 0.78 | 1.17 | 0.074 | 0.164 | 1.024 | 1.003 | 0.817 | 0.716 | 0.991 | 3/5 |
| 1-3M | 12 | 0.86 | 0.35 | 0.48 | 0.053 | 0.075 | 1.037 | 1.010 | 0.631 | 0.505 | 1.012 | 11/12 |
| >=3M | 4 | 1.00 | 0.45 | 0.55 | 0.031 | 0.095 | 1.055 | 1.015 | 0.685 | 0.627 | 1.037 | 4/4 |

(columns: bucket, graphs, before/AMD(milan), after/AMD(milan), after/AMD(node), after/METIS, after/MF1 (time
ratios), nnz before/METIS, nnz after/METIS, nnz after/MF1, nnz after/AMD, flops after/METIS, #graphs with after <=
AMD(milan)).

Per graph (s; after vs AMD milan / node): parabolic_fem 0.600 -> 0.221 (AMD 0.427 / 0.174), ecology2 0.827 -> 0.283
(0.341 / 0.187), lat2_724 0.574 -> 0.342 (0.270 / 0.154), thermomech_dM 0.179 -> 0.138 (0.104 / 0.088), lat2_512
0.400 -> 0.194 (0.111 / 0.081), lat3_64 1.326 -> 0.396 (0.262 / 0.182), audikw_1 0.855 -> 0.395 (1.61 / 1.82),
lat3_140 5.81 -> 2.12 (3.83 / 2.48), lat2_2048 2.27 -> 1.38 (2.32 / 2.02). nnz vs METIS-ND geomean 1.002-1.015 per
bucket (was 1.024-1.055); vs serial MF1 0.62-0.92; vs AMD 0.50-0.87.

Other configurations (same jobs, bucket geomeans of time / AMD(milan) for 100k-500k | 500k-1M | 1-3M | >=3M):
- `gndd:256:hyb=1` (rmax 8): 3.37 | 0.82 | 0.39 | 0.57; rmax 2: 2.84 | 0.71 | 0.33 | 0.43 (nnz +0.7%, flops
  +2%); rmax 1: 2.63 | 0.68 | 0.33 | 0.43 (nnz +1-3%, flops +3-10%). Replicas were not free: they multiply the
  coarse work; 4 keeps the fill.
- Lb 512 with rmax 4: 2.92 | 0.75 | 0.34 | 0.43, nnz +0.3-0.6%; Lb 128 about as Lb 256.
- 8 vs 32 threads: within +-3% (the critical path is the single-part top levels; deep levels stay on the GPU).
- hybq (GPU levels with METIS limits): +10-25% time, fill not better on average on the B200 set.
- device leaves alone (`gndd:1024` vs `gnd:1024`): -2..-20% (audikw_1 0.855 -> 0.776, af_shell10 0.541 -> 0.435,
  Queen 3.14 -> 2.80); host-leaf hybrid `gnd:1024:hyb=1` vs `gndd:1024:hyb=1`: the device path is 5-25% faster on most graphs (one noisy outlier, lat2_724).
- top-k (kway 2 / 3): no faster than the hybrid on the B200 (its CPU refinement of all depths at every level
  is on the critical path), fill +0.5..+5% nnz, and pathological on StocF-1465 / Bump_2911 (3-9 s; not debugged).
- policies: `auto` (cost model, B200 crossover n+m = 2.07M, i.e. n ~ 0.5-0.7M) picks AMF below it: small bucket
  0.82x AMD(milan) / 0.90x AMD(node) with AMF's fill (nnz vs METIS 1.09); `race` (cap n+m < 6.0M from the
  calibration): 1.57 | 0.65 | 0.32 | 0.45; racing costs GPU ND time on the graphs where both run (GC pauses from
  the serial thread's allocations), so the cost-model choice is the better default below the crossover.

Where GPU ND starts winning (B200 vs on-node AMD, same machine): FEM / circuit graphs from ~0.9-1.2M
(parabolic_fem 0.5M is 1.27x), 2D/3D lattices only from ~3-4M (lat2_1024 1.32x, lat3_100 1.11x, lat2_1448 0.90x).
vs milan AMD (the quoted baseline): from ~0.5M except lattices (lat2_512 1.76x, lat3_64 1.51x, lat2_724 1.27x).

### What is left
- The ND now costs ~15-30 ms per ND level on the B200 (13-14 levels at Lb 256): GPU coarsening 0.06-0.10 s,
  CPU FM 0.04-0.15 s, transfers 0.01-0.03 s. Lattices of 0.5-1M (AMD's cheapest case) need ~2x more.
  Candidates: overlap the leaf batches of finished ND levels with the deeper levels (separate stream), reuse
  the parent's coarse hierarchy for the children instead of re-coarsening each level, fewer launches per
  coarsening step (a warp-per-row contraction was tried: identical output but 2.4-4x slower, reverted).
- Defaults in GNDOpts are unchanged (hyb=0, Lb=1024, rmax=8); the recommended configuration is
  `gndd:256:hyb=1;rmax=4` (GNDOpts(Lb=256, hyb=1, rmax=4) + leafdev = grounds_batch_dev).
- Nothing is running on HPG for this task.
