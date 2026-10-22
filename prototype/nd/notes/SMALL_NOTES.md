# GPU-MF1 on small graphs: persistent and batched modes (GSmall.jl)

Changelog for making GPU minimum-average-fill (`RoundsAlg(score=:mf1, relax=0.1)`, the
fill config) fast below ~100k vertices. Scripts, raw rows and logs are in `small/`.
Baseline: Richard's new serial MF1 (`MF(strategy=1)`, rmain), timed on the same B200
node CPU (Xeon Platinum 8570) in the same job.

## Oct 19-20

### What was there
The replayed CUDA-graph driver (`grounds`, now `mode=:replay`) launches ~10 full-device
kernels per round. On the B200 a round costs 70-100 µs at every size below ~100k
vertices, and serial MF1 needs only ~0.3-1 µs per vertex, so GPU-MF1 lost below ~45k
vertices (MF_SUMMARY: 0.60x of new serial MF1 below 30k).

### Modes implemented
- **`mode=:block`** (`grounds(...; mode=:block, nw)`, `grounds_small`): the whole
  ordering in ONE kernel launch on one thread block of `32nw` threads.
  - Phases separated by `sync_threads`.
  - The control block (`ctl`, `kmin`) lives in shared memory.
- **`mode=:grid`** (`G` blocks): one cooperative launch (`@cuda cooperative=true`, so
  all blocks are resident).
  - Phases separated by a software grid barrier: the cooperative-groups scheme, arrive
    with `atom.add.release.gpu`, poll with `ld.acquire.gpu` (fences below sm_70).
  - CUDA.jl's `CG.sync` needs the driver workspace; the own barrier does not.
- **`mode=:auto`** (`pick_mode`, device attributes only):
  - one block (`auto_nw`: the most warps that keep ≥128 registers per thread) while
    n ≤ 2 × its thread count;
  - a cooperative grid of one block per SM while n ≤ 128 vertices per SM;
  - else replay.
  - The thresholds come from the B200 v1 sweep below; the default of `grounds` stays
    `:replay`.
- **`grounds_batch(graphs, alg; weights, nw=8)`**: many graphs in one launch, one block
  per graph, launched largest first.
  - Each block runs the round loop of its own graph to the end.
  - The graphs share one vertex numbering (disjoint union), so the setup kernels
    (twins, degrees, deficiencies, keys) run once on the union.
  - Per-graph slices: lists, arenas, pool region, control block. They are device
    views built on the host (`dsub`); no per-graph allocation.
  - A graph that runs out of arena or pool space (or of outside-mask space in dense
    mode) exits with NEED. The host grows only that graph's arrays (same rules as
    replay: grow, repack into its own pool) and relaunches only the unfinished graphs.
  - Graphs with a dense tail stay on the GPU (serial rule) unless `alg.tailalg` is set.
  - Returns `(orders, info)`; info has per-graph `nnz`, `flops`, `tw`, `rounds`.
- **`grounds_race(weights, graph, alg; serial=MF(strategy=1), cap, gpu)`**: serial
  MF1 on a CPU thread against the GPU (one `grounds` run, e.g. `mode=:auto`, or a
  `grounds_ens` ensemble) at the same time.
  - Returns the order with the better fill (`metric`) when both finish, or the first
    to finish once `cap` s have passed.
  - The loser keeps running in the background (Julia cannot kill tasks).
  - Needs `-t2` or more.

### How the round code is shared (GRounds.jl)
- The round kernels take an execution context `cx`:
  - `GridCx{NW}`: the grid works on one graph (the launched kernels, and `:grid`);
  - `BlockCx{NW}`: every block works on its own graph (`:block` and batch).
- `gwarp/nwarps/gthread/nthreads/vblock/vgrid/lastblock` dispatch on it. The replay
  driver passes `GCX = GridCx{8}()`, so its code path is unchanged.
- `k_whg!`'s staging buffers are sized from NW, with the same totals as before.
- `k_detect!` / `k_apply!` bodies became `detect_one!` / `apply_one!`.
- The persistent loop uses warp versions (`k_detectw!`, `k_applyw!`: a warp per
  selected vertex whose clique has ≤ 64 members, the block for bigger cliques).
  Without them, a few blocks would serialise the selected vertices.
- Dense mode (grouped Wing-Huang) is decided per round on the device, for rounds with
  a clique ≥ dgroup. Same updates as the host switch.
- The tie hash uses the vertex number inside its own graph (`voff`), so a batched
  graph is ordered exactly as on its own.

### Latency work for the persistent kernels (v2, after the v1 job)
- **Inlined phases**: `@inline` on the round kernels; they are still kernels for replay.
  - Before: twelve device-function calls per round, each passing ~25 arrays through
    local memory (5-6 KB of local memory per thread).
- **Field-wise descriptor loads** (`DRef`).
  - Before: the whole 1.7 KB descriptor was copied, or held across the loop, which
    meant 255 registers plus spills.
  - A first version loaded fields byte by byte (`unsafe_load` defaults to align 1): fixed.
  - Now: 224-240 registers, 32-56 B of local memory.
- **`k_cand!`**: a lane per candidate with a short list (≤ 16, 4 loads in flight).
  - Before: the warp walked the up to 32 candidates of its chunk one after another.
  - lat2_20 one-block cand phase: 86k → 4.6k cycles/round (laptop).
  - This kernel is shared with replay.
- **`k_elim12!`** (shared with replay):
  - N(a) staged in shared memory for the membership tests;
  - the bit-matrix row built in shared memory (≤ 32 words);
  - the missing pairs listed per warp, with their data loaded in parallel (≤ 64);
  - stt/kown/w loaded together; 2 entries per lane in flight.
- **`k_elim34!`**: K staged in shared memory for the merge ranks.
- One shared buffer (`wbuf`) for the staging of `k_elim12!`, `k_elim34!` and
  `k_whg!` in the persistent kernels, so static shared memory stays < 48 KB at nw ≤ 32.
- **Host side of batch**:
  - type-stable descriptor construction (before: NamedTuple splat per graph);
  - `pfinish` without per-graph allocations, with concretely typed downloads.
  - Before, `PState.U` was abstractly typed, so the expand loop dispatched per element.
  - 116 leaves / 75k vertices on the laptop: post 7 → 1 ms, setup 4 → 1.8 ms.
- Tried, no gain:
  - `dgroup` 1-16 (grouped updates for all cliques): slower;
  - acq_rel atomics instead of fences: as slow as a fence (~850 cycles);
  - `maxregs` 128/96 for 2 blocks/SM in batch: the spills cost more than the
    occupancy gains (laptop).

### Exactness (laptop RTX 5060)
- `small/check.jl` compares each mode against GRounds before this work
  (`small/GRounds_pre_small.jl`). It compares nnz(L), flops and tw (CliqueTrees
  recount), width and log2 cost with the instance weights, and the device counters
  against the recount.
- Configurations:
  - md/mf1 × relax 0/0.1/0.5 × compress on/off × TN/unit weights;
  - modes replay, block4/8/16/32, grid1/3/4/5, auto, batch;
  - batch also with tiny arenas (poolsmall=1, kmin0=16, ocap0=16, dgroup=16): 9-16
    launches and 23-49 per-graph growths per batch;
  - tails: GPU tail (cpualpha=Inf), CPU tail (tailalg), `tail=50`;
  - `small/` dgtest: dense TN graphs with dgroup off/2/64, which exercises pair lists
    over 32/64 and rows over 32 words.
- Graphs: lat2_20, lat3_8, qaoaamp_N240_p3, rqc8x8_m16, mc3sat_N300/N1000, cbuckle,
  bcsstk17, crystm03, qaoaamp_N1000_p5, rndhyp_T3000, lat2_100.
- **All identical**, about 1,900 comparisons over the runs. Every order is a valid
  permutation.
- Edge cases pass: n = 0 and 1, isolated vertices, path, star, K40, K300 (batch
  included).
- B200 v1 job: every mode had the same nnz as replay on all 45 single graphs. Batch
  matched single-graph runs on the first 20 leaves of all 15 leaf sets.
- Only CPU tail handoffs depend on timing, as before. Use cpualpha=Inf to compare tails.

### B200 v1 (job 46770084; results/small_b200_v1.tsv, small_cpu_v1.tsv; `python3 small/analyze.py`)
v1 is the first version: persistent modes without the latency work above.

**Single graphs.** GPU time is the whole `grounds` call; serial is new MF1 on 1 core.
| n | serial/replay | serial/best persistent | notes |
|---|---|---|---|
| 0.5-3k (14 graphs) | 0.05-0.26 | 0.05-0.30 | the GPU takes ≥ 5 ms for any graph: ~70 µs per round × ≥ 60 rounds |
| 3-30k sparse | 0.19-0.86 | same ± 10% | grid ≈ replay; block 1.5-4× slower than replay above 2k |
| 30-100k sparse | 0.35-2.2 | grid 0.33-1.6 | replay wins from ~45k |
| dense TN | mc3sat_N1000 1.3, rndhyp_T10000 2.0, mc3sat_N3000 4.0, qaoaamp_N4000_p5 1.6 | grid same or slightly better | — |

- v1 persistent rounds cost as much as replayed ones on the B200 (lat2_100: 172k
  cycles/round in grid148, 590k in block16).
- The phase profile (prof rows) showed every phase at 10-30k cycles.
- The one-block kernel ran per-row work on 8-16 warps of one SM.
- Fill: GPU nnz/serial nnz 0.95-1.17 (lat3_32 1.39), as in MF_SUMMARY.

**ND leaves.** Recursive METIS separators of ecology2, thermal2, G3_circuit, lat2_724 and
audikw_1, with leaf bounds 50k/10k/2k: 28-1110 leaves per set.
| leaves | GPU batch, kernel only, vs serial 1 thread / 8 threads | GPU batch, whole call, vs 1t / 8t |
|---|---|---|
| ≤ 2k (538-1110 leaves) | 8.1-21× / 1.5-4.5× | 2.6-3.5× / 0.44-0.66× |
| ≤ 10k (122-230 leaves) | 10-15× / 1.9-3.0× | 2.2-4.4× / 0.50-0.89× |
| ≤ 50k (28-44 leaves) | 2.0-6.0× / 0.42-1.24× | 1.4-3.5× / 0.24-0.72× |

- The same leaves one `grounds` call at a time (auto mode) took 3-23 s. The batch is
  5-13× faster than that loop at leaf bound 50k, and 20-35× at 10k and 2k.
- Fill: batch nnz/serial nnz 0.998-1.027.
- In v1 the whole-call time was 2-4× the kernel time: CSR 0.02-0.18 s, setup 0.02-0.2 s,
  orders 0.1-0.37 s. The orders step was the type-unstable loop fixed in v2.

### Expected for v2
- Laptop phase profile, cycles per round:
  - lat2_20: block16 116k → block16 76k, block8 70k;
  - bcsstk17: grid26 137k → 105k;
  - lat2_100: grid26 112k → 94k.
- So, roughly:
  - persistent single-graph rounds 1.2-1.6× faster than v1;
  - replay a little faster on sparse graphs, through `k_cand!`/`k_elim12!`;
  - batch whole-call time close to the kernel time.
- Single sparse graphs below ~30k will still lose to serial MF1 on one core. The round
  count is fixed by the selection rule, and a B200 round has ~8 phases of
  latency-bound global-memory chains (~0.4 µs per dependent access, ~2 µs per
  barrier). For those graphs `grounds_race` returns the CPU order.
- Batched leaves are where the GPU wins.

### B200 v2 (job 46776446; results/small_b200_v2.tsv, small_cpu_v2.tsv)
- The same sweep as v1, plus batch nw=4 and fewer grid sizes.
- First rows read while the job was running (`small/results/small_b200_v2_partial.tsv`).
- Phase profile (cycles per round, grid148): lat2_100 172k → 156k, bcsstk17 216k → 189k,
  jnlbrng1 160k → 146k.
- In the grid, `sizes` (~29k) and `scan` (~17k) cost 2-4× their one-block cost: the
  last-block decisions plus the barrier. That is the next target.

| graph | serial | v1 best | v2 replay | v2 grid (best G) | v2 auto | serial/GPU v1 → v2 |
|---|---|---|---|---|---|---|
| bcsstk17 (11k) | 0.0159 | 0.0640 | 0.0563 | 0.0534 | 0.0535 | 0.25 → 0.30 |
| cbuckle (14k) | 0.0351 | 0.1457 | 0.1223 | 0.1251 | 0.1250 | 0.24 → 0.29 |
| olafu (16k) | 0.0421 | 0.2098 | 0.1720 | 0.1820 | 0.1831 | 0.20 → 0.24 |
| crystm03 (25k) | 0.0707 | 0.2054 | 0.1538 | 0.1626 | 0.1558 | 0.34 → 0.46 |
| jnlbrng1 (40k) | 0.0486 | 0.1389 | 0.1278 | 0.1206 | 0.1312 | 0.35 → 0.40 |
| ct20stif (52k) | 0.1069 | 0.2249 | 0.1954 | 0.2129 | 0.1951 | 0.48 → 0.55 |
| nasasrb (55k) | 0.0924 | 0.1954 | 0.1661 | 0.1670 | 0.1647 | 0.47 → 0.56 |

- v2 is 13-25% faster than v1 in both drivers. Replay gained as much, from the shared
  `k_cand!`/`k_elim12!` changes.
- The persistent grid is still ≈ replay (±7%).
- Batch v2 rows (whole call vs kernel only, nw 4/8/16): run
  `python3 small/analyze.py small_b200_v2.tsv small_cpu_v2.tsv` on the job output.

### Files
- `GSmall.jl` (included by GRounds.jl):
  - persistent kernels: `k_pblock!`, `k_pgrid!`, `prounds!`, `gbar!`;
  - warp versions of detect and apply;
  - host side: `pinit`, `prun!`, `pgrow!`, `pfinish`;
  - API: `grounds_small`, `grounds_batch`, `grounds_race`, `pick_mode`, `auto_nw`.
- `GRounds.jl`:
  - the context refactor;
  - `mode` / `G` / `nw` / `prof` kwargs of `grounds_csr`;
  - `k_cand!`, `k_elim12!`, `k_elim34!` changes.
- `small/`:
  - `check.jl`: exactness;
  - `GRounds_pre_small.jl`: GRounds before this work, module GRoundsPS;
  - `analyze.py`;
  - `hpg/`: `sgpu.jl` (B200 sweep and ND leaves), `sser.jl` (serial baselines),
    `small.sbatch` / `small2.sbatch`;
  - `results/`: v1 rows and logs.
- HPG: `/blue/fairbanksj/itaykadosh/gpu/rounds/small/`:
  - `code/` for v1, `code2/` for v2;
  - `results/`, `leaves/` (serialized ND leaves), `logs/`.

## End to end (Oct 20)

Goal: the whole `grounds_batch` call (host graphs in, host permutations out) faster than
Richard's serial MF1 on 8 CPU threads (one graph per thread). All numbers: B200 node
c1001a-s5 (Xeon 8570). GPU process on 8 cores (`-t8`); serial MF1 = T worker tasks pulling
graphs largest first from a shared counter, min of 3 runs (1 thread: 2), BipartiteGraph
conversion excluded (the strongest baseline), min over two job samples. Fill: per-graph
nnz/flops/tw of every new configuration equal to the v2 path on every set (no FILLDIFF);
GPU nnz / serial nnz 0.98-1.03.

### Where the v2 time went (stage breakdown, ms; v2 nw16 vs new, 1 chunk, timed run)
| set | v2 csr / pre / gpu (launches) / post = total | new pack / h2d / twins / live / alloc = pre; kernel; relaunch; post |
|---|---|---|
| ecology2_L2000 (684) | 18 / 39 / 131 (2) / 9 = 160 | 2.6 / 1.4 / 0.1 / 2.4 / 0.5 = 7; 46; 0; 1.5 |
| G3_circuit_L2000 (1110) | 29 / 50 / 250 (4) / 15 = 288 | 5.0 / 1.5 / 0.1 / 4.5 / 0.5 = 12; 75; 0; 1.9 |
| thermal2_L10000 (174) | 44 / 40 / 123 (3) / 11 = 193 | 3.4 / 1.7 / 0.1 / 5.1 / 0.3 = 11; 82; 0; 3.0 |
| audikw_1_L2000 (538) | 109 / 127 / 38 (1) / 12 = 221 | 9.1 / 4.2 / 0.8 / 14.9 / 0.4 = 29; 39; 0; 4.1 |
| audikw_1_L50000 (28) | 161 / 202 / 1106 (4) / 8 = 1380 | 30 / 6.6 / 1.3 / 13.5 / 0.7 = 52; 515; 0; 2.7 |
- v2: CSR conversion 11-161 ms, packing/upload/setup 15-216 ms, and 2-4 launches: with the
  growth-on-demand start sizes nearly every ND leaf ran out of pool (ecology2/thermal2/G3:
  every leaf), and a relaunch waits for the whole first launch.
- Julia GC: 0 in every call (1-140 MB allocated per call, mostly the output buffers).

### What changed (GSmall.jl only; GRounds.jl untouched)
- **One-pass packing** (`ppack`): lists packed straight from `SimpleGraph.fadjlist` (or CSR)
  into a pinned host buffer, threaded over graphs; self loops detected in the same pass
  (that graph is repacked without them). No per-graph `csr`.
- **Pinned staging** (`pin_take`/`pin_give`): a pool of pinned buffers, checked out per
  batch/chunk, kept across calls; uploads are async DMA copies. Concurrent calls from
  several tasks are safe.
- **Arenas sized up front** (`arena = :auto`): pool = pfac × the lists (pfac from a quarter
  of the free device memory, 8-32; measured totals 11-16×, a few G3 leaves 38×), clique
  arena = max(kmin0, 2 maxdeg, n/2) (measured peaks ≤ 0.34 n).
- **Device-side growth** (`dgrow!`, batch only): a graph that still runs out takes the
  space from a spare arena (bump counter, `spare` bytes, default min(free/8,
  max(256 MB, 128 B × lists))) and redoes the round in the same launch. Same new sizes and
  same repacked pool layout as `pgrow!`, so the run is the same; if the spare arena is
  exhausted the graph exits with NEED and the host path takes over. Before: G3 relaunches
  cost 9-32 ms, audikw_1_L50000 203 ms, sweep_L50000_x100 202 ms; now 0 (19-39 device
  growths on G3, 9 on audikw_1_L50000).
- **Twins without a sort** (`k_hins!`/`k_twinsh!`): open-addressing table, representative
  = smallest vertex with the hash = the run head of the stable sortperm, so identical
  parents (checked on ND leaves, TN graphs, blown-up graphs with 1970 twins). 3.4-15 ms →
  0.15-1.2 ms.
- **Threaded unpack** into one `Vector{Int}`; `orders[b]` are views into it.
- **Chunked pipeline** (`chunks = 0`: 1-4 by volume): the size-sorted graphs are cut into
  chunks of growing volume, each set up/launched/unpacked in its own task (own stream),
  so packing chunk k+1 overlaps chunk k's kernel. Effect ±5-10% either way (packing is now
  small next to the kernel).
- **Block shape from the batch** (`nw = 0`, `batch_cfg`): occupancy from device attributes.
  Measured (job 46787933, `occ.jl`): for B ≫ SMs, 2-4 blocks per SM (8 warps or 4 warps at
  128 registers) run 1.3-1.8× faster than one 16-warp block per SM (sweep_L1000_x3000
  0.173 → 0.095 s, G3_L2000 0.135 → 0.094); for B ≈ SMs and 10k graphs 16 warps; for 30-50k
  graphs 32 warps (64 registers; ecology2_L50000 0.23 → 0.21, audikw 0.68 → 0.55).
  Rule: nmed ≥ 16k → 32 warps; else the widest of (16, 0), (8, 128), (4, 128) whose resident
  blocks ≥ B/3 with ≥ nmed/32 threads. (The v5 run used B/2 and nmed/64; the change to
  B/3, nmed/32 follows its rows: 10k × 300-800 and 2k × 300 were 5-25% faster at 16 warps.
  Not re-measured.)
- Instrumentation: `timing` gets every stage, `occ`, `regs`, `lmem`, `gc`, `alloc`, growth
  reasons (`gk/gb/gp/go`), `dgrows`, chosen `nw/maxregs`.

### API (for the GPU ND caller)
- `grounds_batch(graphs, alg; weights=nothing, nw=0, maxregs=0, timing, arena=:auto, chunks=0,
  spare=-1, twins=:hash, <old kwargs>) -> (orders, info)`. Signature compatible; changes:
  default `nw` 8 → 0 (auto); `orders[b]` is a `SubArray` view (collect to own);
  `info.dgrows` added; `grounds_batch_csr(csrs, wts, alg; ...)` takes `wts = nothing` and
  SimpleGraphs or `(ptr, tgt)` CSRs.
- Old behaviour: `arena=:min, chunks=1, spare=0, twins=:sort, nw=16`.

### Results, end to end (v5 = job 46791127; GPU default config)
ND leaf sets (time s; ×T = serial T threads / GPU):
| set | B | v2 | new | ×1t | ×8t | ×32t |
|---|---|---|---|---|---|---|
| ecology2 L2000 / L10000 / L50000 | 684/129/32 | 0.160/0.127/0.317 | 0.058/0.076/0.214 | 14.5/11.2/4.5 | 1.75/1.82/0.77 | 0.56/0.49/0.17 |
| thermal2 | 888/174/32 | 0.213/0.193/0.520 | 0.071/0.095/0.245 | 20.4/16.8/7.9 | 2.57/2.49/1.15 | 0.81/0.96/0.36 |
| G3_circuit | 1110/230/44 | 0.288/0.250/0.522 | 0.084/0.146/0.386 | 20.6/14.6/6.9 | 2.67/2.09/1.02 | 0.86/0.81/0.30 |
| lat2_724 | 733/137/32 | 0.079/0.084/0.318 | 0.041/0.044/0.129 | 14.1/12.4/4.7 | 1.68/1.46/0.72 | 0.60/0.82/0.21 |
| audikw_1 | 538/122/28 | 0.221/0.334/1.380 | 0.054/0.118/0.525 | 15.6/12.1/4.2 | 1.98/2.01/0.69 | 0.61/0.88/0.27 |
- New vs v2: 1.35-4.1× (median ~2.3×).

Sweep (random ND leaves of bound L from the five graphs; ×8t | ×32t):
| L \ B | 10 | 30 | 100 | 300 | 800-1000 | 3000 |
|---|---|---|---|---|---|---|
| 1k | 0.09 / 0.08 | 0.23 / 0.12 | 0.52 / 0.18 | 1.20 / 0.38 | 2.31 / 0.76 | 2.56 / 0.76 |
| 2k | 0.12 / 0.11 | 0.26 / 0.14 | 0.57 / 0.20 | 1.05 / 0.34 | 2.29 / 0.71 | 2.89 / 0.98 |
| 10k | 0.14 / 0.15 | 0.30 / 0.14 | 1.03 / 0.33 | 1.75 / 0.71 | 2.37 / 0.98 (800) | - |
| 50k | 0.22 / 0.22 | 0.58 / 0.23 | 1.48 / 0.52 | - | - | - |
(×1t at the largest B: 17-21×.)

### Crossover (GPU batch end to end vs 8 CPU threads)
- 1k-2k graphs: B ≈ 250-300 (≈ 0.3-0.4 M vertices in the batch); 10k graphs: B ≈ 100
  (≈ 0.7 M); 50k graphs: B ≈ 50-60 (≈ 2 M).
- Rule of thumb: the batch must give every SM at least one graph (B ≳ 2 × #SMs for small
  graphs, ≳ 0.7 × #SMs for 10k). Below that the time is one graph's single-block latency
  (2k: ~20 ms, 10k: ~75 ms, 50k: ~300 ms) while 8 threads finish in B/8 serial times.
- ND leaf sets: GPU wins vs 8 threads at L ≤ 10k on all five graphs (1.46-2.67×) and at
  L = 50k on thermal2 (1.15) and G3 (1.02); loses at L = 50k on ecology2/lat2/audikw
  (0.69-0.77: 28-32 leaves use 28-32 of 148 SMs).
- vs 32 threads: never ahead (best 0.96-0.98 at 10k × 174-800 and 2k × 3000). In steady
  state (B ≫ SMs) the batch equals ~20-30 serial threads: per SM-slot a 2k leaf takes
  ~15 ms per wave vs ~1.2 ms serial. Beating 32 threads needs faster rounds inside a block
  (the kernel work) or more resident graphs per SM, not more host work: host stages are now
  10-30 ms of 40-500 ms.

### Single graph (one `grounds` call, 30k-180k; job 46786424) vs serial MF1, 1 thread
- Host share is small: csr 0.6-7 ms, pre 0.7-3 ms, post ≤ 1.3 ms; the rest is the GPU.
- Serial / GPU (replay ≈ auto): jnlbrng1 0.30, ct20stif 0.40, nasasrb 0.42, apache1 0.48,
  reg3_N30000 0.86, lat2_181 1.08, thermal1 1.25 (auto), lat2_224 1.51, lat3_32 2.07,
  lat2_300 2.23.
- Suggested for GRounds.jl (not done, other owner): `grounds` → `csr(graph)` then pageable
  uploads; packing straight from `fadjlist` into the pinned pool (`ppack`) would save the
  csr step (≤ 7 ms, ≤ 4% here). Also: keep the args of the persistent loop free of Union
  types — a `Union{Nothing,Tuple}` argument made Julia compile `prounds!` twice and doubled
  the static shared memory (64 KB, over the 48 KB limit on sm_120).

### Exactness (laptop RTX 5060)
- `small/check.jl` (MODES=batch, QUICK=1; lat2_20, bcsstk17, qaoaamp_N240_p3, rqc8x8_m16, lat3_8):
  135 batch comparisons identical to GRounds before the small-graph work, now including
  arena :min/:auto, chunks 1/2/3, spare 0 (host growth only), -1 (device growth: 26-38
  growths in one launch), 3000 B (spare too small). Edge cases: self loops, CSR input,
  four concurrent calls from tasks, n = 0/1. Tails identical.
- Mixed growth (scratch mix.jl): spare 20 kB-2 MB so that the device grows some graphs and
  the host the rest: identical fill in all 24 runs.
- Orders differ between runs within a round (atomics), as before; fill is what is compared.

### Files (small/e2e/)
- `e2e_gpu.jl` (GPU sweep: leaf sets, sweep sets, single graphs; v2 vs new), `e2e_cpu.jl`
  (serial MF1 1/8/32 threads), `occ.jl` (block shape vs occupancy), `analyze_e2e.py`.
- `e2e.sbatch` (job 46786424: v4 = host work + sizing, CFGS v4), `e2e5.sbatch` (job 46791127:
  v5 defaults), `occ.sbatch` (job 46787933).
- `results/`: `e2e_gpu.tsv` (v4), `e2e_gpu_v5.tsv`, `e2e_cpu.tsv`, `e2e_cpu_v5.tsv`, `occ.tsv`, logs.
  Analyze: `python3 small/e2e/analyze_e2e.py small/e2e/results/e2e_gpu_v5.tsv
  small/e2e/results/e2e_cpu.tsv,small/e2e/results/e2e_cpu_v5.tsv` (v4: e2e_gpu.tsv).
- HPG: `/blue/fairbanksj/itaykadosh/gpu/rounds/small/e2e/` (`code3` v4, `code5` v5, `code4`
  occ; `sets/` the sweep sets; `results/`).
