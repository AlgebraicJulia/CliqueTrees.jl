# GPU rounds (GRounds.jl) performance notes

Changelog for the GPU round-based minimum-fill ordering, mainly score `:mf1`
(minimum average fill, Richard's `MF(strategy=1)`). Version 1 is kept as
`perf/GRounds_before.jl` (module `GRoundsBefore`) for before/after runs.
Generic GPU code only: launch grids come from the device (SM count × occupancy
from `CUDA.active_blocks`); the active-list size comes from the device's resident
thread count; the CPU/GPU tail split is a cost model with a measured round time.

## Oct 19: version 2

### Profile of version 1 (laptop RTX 5060; B200 numbers from results/hpg/scale_b200.tsv)
- Round-latency bound below ~1M vertices. Each round made 2 blocking host syncs
  (`Array(cnts)`, `Array(info)`), 2 `accumulate!` scans and ~13 launches.
  - jnlbrng1 (40k, fill config): 1723 rounds at 77 µs each. The GPU was busy 31% of the time.
  - B200: the `sel` field (time to the 2nd sync) is 85-95% of `gpu` on every lattice.
- O(live vertices) work per round: score, candidates and compaction all scan the
  full live list.
  - ecology2 (1M): 10064 rounds at 210 µs each = 2.1 s.
- Large cliques ran on one block each:
  - phase 4 (merging the new lists) was serial per thread;
  - supervariable detection was O(d²) per block.
  - Dense-end graphs therefore crawled. With relax 0.1, no tail:
    - qaoaamp_N4000_p5: 5.5 s on the laptop;
    - rndhyp_T10000: 5.6 s on the laptop;
    - Bump_2911: 221 s on the B200.
- Scale config (relax 0.5, tailsel 128): on TN/regk/rnd/mc3sat graphs the CPU
  serial MF1 tail was nearly all of the time.
  - Examples: qaoaamp_N4000_p5 2.4 s of 2.45 s on the B200; rndhyp_T30000 42 s; mc3sat_N3000 16 s.
- Repacks: pool factor 3, doubling. Arrays that grew were not freed; the GC freed them later.

### Changes
1. **Device-driven round loop.**
   - Kernels read their counts from a device control block and loop with grid stride.
   - All decisions are made on the device by the last block of a kernel (threadfence + counter):
     thresholds, done/tail, and capacity checks.
   - Two rounds are captured into one CUDA graph. The host replays it in batches (1, 2, 4 … 16 graphs).
   - The host reads the control block only between batches (~1 sync per 30 rounds instead of 2 per round).
   - Capacity overflow sets a flag; the round is abandoned, the host grows or repacks, re-captures, and redoes the round. The redo is deterministic.
2. **Arena space by atomics** (kar/bar offsets) instead of two prefix scans per round.
3. **Active candidate list L = {live v : key(v) ≤ T}**, kept exact:
   - Keys are cached per vertex (`kv`).
   - Only touched vertices are re-keyed:
     - the clique K, in `k_apply!`;
     - the outside neighbours, in `k_elim34!`. A round stamp skips vertices in another block's clique.
   - When the candidate threshold passes T, the live list is rebuilt from an 11-bit key histogram. T is the top of the bin below the target count, at least the bin of thr; the target is wave/4 or n/32.
   - Per-round cost is O(|L| + work).
4. **Elimination split** so clique rows spread over the whole GPU:
   - `k_sizes!` (warp per selected vertex) also builds K. It was fused with the old `k_elim0!`.
   - `k_elim12!`: warp per clique row, building the bit-matrix row and the Wing-Huang updates.
   - `k_elim34!`: warp per row. A warp-parallel merge writes the new list; each outside entry knows its rank in K\x and writes the K entries before it. It also hashes the outside list for detection.
   - `k_detect!` uses a per-clique hash table (O(d)) for d > 64, then warp-parallel verification.
   - Same detection result as version 1 (smallest equal (hash, count) index, then verify).
5. **Dense tail on the GPU (scale config).**
   - When version 1 would hand off to the CPU, version 2 re-captures with relax 0 (the serial rule) and continues on the GPU.
   - It hands the rest to the CPU once the cost model says the CPU is cheaper:
     - dense serial MF ≈ 1.1e-9·t³ s per core (fitted on mc3sat tails);
     - GPU ≈ t × round time over the last ≥16 rounds;
     - hand off only when the CPU is 2× cheaper.
   - An explicit `tailalg` still means a CPU tail.
6. Memory and host side:
   - The pool starts at max(2×, min(8×, 2^26)) of the input, which gives fewer repacks and re-captures.
   - Arrays that grow are freed explicitly.
   - Tail extraction is done on the device; only the tail CSR is downloaded, not the whole pool.
   - Fast CSR path for SimpleGraph.
   - The expand step uses a preallocated stack.
   - n = 0 is handled.

### Exactness
- Same selection rule and state updates as version 1. Without a tail the orderings have identical fill (nnz(L), flops, tw):
  - perf/check.jl: scores md/mf/mf1 × relax 0/0.1/0.5/Inf × compress on/off × 5 graphs with TN weights. All 120 identical. Edge cases (n = 0, 1, isolated vertices, star, clique) pass.
  - perf/lsweep.tsv: all 14 fill-config graphs are identical to version 1.
- With a tail, version 1 already depended on atomic order (the tail's vertex numbering). Version 2 numbers the tail by vertex id. Its GPU tail uses the hash tie-break instead of CliqueTrees' index order, so tail cases differ by tie-breaking only (± a few %, both signs).

### Laptop results (RTX 5060, loaded machine, `-t1`, min of 2-3 runs; perf/lsweep.tsv)
Fill config (relax 0.1, unit weights). Speedup is v1/v2; "vs MF1" and "vs AMF" are serial time / v2 time.

| graph | n | v2 s | speedup | vs MF1 | vs AMF |
|---|---|---|---|---|---|
| lat2_100 | 20k | 0.011 | 2.9 | 1.95 | 0.45 |
| qaoaamp_N4000_p5 | 20k | 0.51 | 10.9 | 4.9 | 0.04 |
| lat3_20 | 23k | 0.028 | 1.3 | 0.76 | 0.12 |
| crystm03 | 25k | 0.072 | 3.5 | 1.8 | 0.24 |
| jnlbrng1 | 40k | 0.055 | 3.7 | 1.6 | 0.29 |
| apache1 | 81k | 0.26 | 3.8 | 4.2 | 0.49 |
| thermal1 | 83k | 0.065 | 2.2 | 4.4 | 1.10 |
| lat3_32 | 95k | 0.030 | 3.1 | 7.0 | 1.8 |
| lat2_224 | 100k | 0.023 | 2.5 | 7.1 | 1.7 |
| G2_circuit | 150k | 0.15 | 3.5 | 5.2 | 1.0 |
| thermomech_dM | 204k | 0.10 | 3.0 | 9.5 | 2.0 |
| parabolic_fem | 526k | 0.19 | 2.3 | 15.0 | 3.6 |
| ecology2 | 1M | 0.57 | 4.1 | 4.4 | 0.87 |
| lat2_724 | 1M | 0.21 | 2.0 | 11.8 | 2.1 |

- Per-round time on jnlbrng1: 77 → ~30 µs. ecology2: 210 → ~45 µs.
- The rest is a dependent-latency chain of ~10 kernels per round (~0.75 µs per graph node, ~2-5 µs per kernel).
- ecology2 (1M rows) does not OOM. The pool high-water mark is ~0.4 GB at 1M on an 8 GB card. The old OOM was not reproducible here; it was probably other tenants on the card plus garbage that was never freed.

### What did not work, or was not done
- **Relaxation** (perf/selq.tsv, CPU reference, 12 graphs, MF1):
  - greedy (maximal) selection: −10-15% rounds, nnz(L) median +1.5%;
  - Luby relax 0.2: 1.5-3× fewer rounds, nnz(L) median ~+2% (jnlbrng1 +13%).
  - Both are outside the ±1% budget, so the candidate rule is unchanged.
- Cooperative persistent kernel: a grid sync measured 0.9-1.2 µs against 0.75 µs per graph node, so there is no gain over graph replay.
- Smaller active-list targets (wave/100 … 2 waves): no measurable difference. The per-kernel cost is latency, not bytes.
- 4× unrolled loads in `k_elim12!` phase 1: no change.
- Narrow grids (1-2 blocks/SM) for late rounds: lost in the noise on the laptop. Full occupancy is kept.

### Files
- GRounds.jl: version 2. The API is unchanged; new kwargs are `gputail`, `tailrelax`, `cpualpha` and `maxbatch`.
- perf/:
  - GRounds_before.jl: version 1;
  - cmp.jl, check.jl, prof.jl, lsweep.jl, selq.jl, tailtest.jl, mem.jl, micro.jl: measurement scripts;
  - hpg/: the B200 sweep (bsweep.jl, gpur.sbatch).
- B200 job: see below.
- B200 before/after sweep: job **46736366** (itay-gpur), /blue/fairbanksj/itaykadosh/gpu/rounds/v2.
  - Outputs: results/fill.tsv and results/scale.tsv.
  - Sets: the 16 small and 19 large SuiteSparse/lattice fill graphs, plus the scale.jl families.
  - Version 1 is rerun in the same job, except Bump_2911 (221 s in job 46702561).
  - Serial baselines: results/hpg/serial_milan.tsv for the scale set, and fill CPU job 46702562 (pending at submission).

### Laptop, scale config (relax 0.5, tailsel 128, unit weights; perf/lsweep.tsv)
- Speedup v1/v2: 1.2-2.2× on sparse graphs, 4× on rndhyp_T10000 (dense tail).
- Same fill except on the tail graphs:
  - nnz(L) v2/v1 is 0.987-1.014;
  - flops v2/v1 is 0.973-1.026.
- vs serial MF1: v2 is faster from 12k vertices up (1.4× at 20k; 3-4× at 100k; 11× at 1M).
- vs AMF: v2 wins only on lat3_32 (1.3×) and lat2_724 (1.2×).
- nnz(L) vs serial MF1 is unchanged from v1: 0.99-1.37, median ~1.07.
