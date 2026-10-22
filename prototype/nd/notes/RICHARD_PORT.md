# Porting Richard's serial MF speedups to GPU-MF1 (GRounds.jl)

Changelog. Richard's new serial code (CliqueTrees origin/main, 598b013) is now the
baseline to beat. Scripts and raw rows are in `rport/`.

## Oct 19

### Richard's changes (`git log 57ce749..origin/main`)
- **41ef420 "Improve MF." and 9048041 "Update MF"**: `MLFLib.jl/src/mfupd_def.jl`, plus genmf_wh.jl and mlfwh.jl.
  - The Wing-Huang deficiency update now takes the new clique members in groups of 64.
  - Each vertex gets a 64-bit UMASK of the group members it is adjacent to.
  - Each earlier member w scans N(w) once per group, not once per fill edge. The update is `defncy[x] -= q(w) * sum q(mask & partners)`.
  - Deficiencies are unchanged; only the heap tie order can change.
  - The UMARK array is gone: its storage is reused for UMASK.
- **99686b9 "aggressive absorbtion"**: `elmtra.jl`.
  - Adds AMD-style element absorption in the quotient graph.
  - An element whose unabsorbed members all lie in the new clique is removed from the members' element lists.
  - `ecount` (shares storage with heapinv) counts the unabsorbed members of each element.
- Other commits:
  - b78ed7b "Update MD": an MD `dense` parameter;
  - 9bda22c "Update AMF" and ad9f0eb "QG": an AMF rewrite and a quotient-graph utility;
  - 5adce40 "FND": a flow-based ND.

  None of them touch the MF/MF1 path. AMF does change: new AMF fill differs from old AMF by ±10% (below).

### Baseline env and claim check
- Worktree `~/Projects/GATAS/CliqueTrees.jl-rmain` (detached at origin/main, read-only use).
- Env `rounds/renv` devs that worktree. On HPG it is `rounds/rr/renv`, pointing at a copy in `rounds/ct-rmain`.
- Old serial baseline: CliqueTrees.jl-mt (`genv` locally; `rounds/env`, i.e. `rounds/ct`, on HPG; its MLFLib, AMFLib and MDLib are md5-identical to mt).
- Driver: `rport/sbench.jl TAG OUT ALGS names...`.
  - nnz(L), flops and tw come from the supernodes of `cliquetree(g; alg=order)`, using only the public API, so the same code runs on both versions.
  - These metrics match evalutil's colweights on the mt rows.
- Laptop, `-t1`, unit weights, min of 3 (`rport/local_serial.tsv`). MF1 old (mt) → new (rmain):

| graph | t old | t new | speedup | nnz(L) new/old |
|---|---|---|---|---|
| bcsstk17 | 0.0127 | 0.0090 | 1.41 | 0.999 |
| cbuckle | 0.0247 | 0.0198 | 1.25 | 1.015 |
| crystm03 | 0.0805 | 0.0460 | 1.75 | 0.993 |
| jnlbrng1 | 0.0409 | 0.0342 | 1.20 | 1.015 |
| lat2_100 | 0.0080 | 0.0075 | 1.07 | 1.007 |
| lat3_20 | 0.0170 | 0.0124 | 1.37 | 1.000 |
| ct20stif | 0.0911 | 0.0720 | 1.27 | 0.986 |
| apache1 | 0.554 | 0.294 | 1.88 | 0.962 |
| qaoaamp_N4000_p5 | 1.827 | 0.315 | **5.81** | 0.992 |

- The fill moves by at most ±4%, in both directions. That is inside serial MF1's own relabeling spread (ENSEMBLE_NOTES: up to +5%), which is consistent with "same deficiencies, different ties".
- The speedup grows with clique size: 1.1-1.9× on sparse graphs, 5.8× on a dense TN graph.
- New AMF: about the same speed. Its fill differs from old AMF: bcsstk17 +12.7%, crystm03 -11%, cbuckle +6.5%.

### Mapping onto GRounds
GRounds keeps an explicit elimination graph: every live vertex has its full adjacency
list, and K is merged into each member's list. There are no elements. Its deficiency
update (k_elim12!, warp per clique row a) scans N(b) once per missing pair {a, b}, with a
binary search of each entry in N(a). On qaoaamp_N4000_p5, k_elim12! was 82% of GPU time
(laptop profile).

- **Aggressive absorption: not applicable.** It shortens element lists in a quotient
  graph. GRounds has no element lists to shorten: the lists of K members already hold
  plain vertices, and eliminated vertices are dropped when a list is rewritten
  (k_elim34!). Nothing was ported.
- **Grouped mask updates: ported (dense mode).** Groups are 32 clique members, one word
  of the existing K×K bit matrix `bar`.
  - **Inside K**, the masks are the bit-matrix rows themselves. For row a with partners P
    (j > i, not adjacent), each K-neighbour c needs one update:
    `def[c] -= w(a) * w(bar[c] & P)`, word by word. There is one atomic per (a, c),
    not one per (pair, c).
  - **Outside K**, mask rows are claimed per round and per vertex.
    - In k_elim12!, a row with partners claims a W-word mask row for each outside
      neighbour c. The claim is a CAS on `lown[c] = (round << 32) | clique`, plus an
      atomic arena offset.
    - New kernel k_ob! ORs each member's group bit into the rows its clique owns.
    - New kernel k_whg! stages O(a) in shared memory once (up to 256 per chunk). It
      walks only the words that hold partners; that list is in shared memory when there
      are ≤128 such words. For each staged c it computes `m = ob[c][g] & P` and applies:
      - `def[c] -= w(a) w(m)`;
      - `def[a] -= w(c) w(m)`;
      - `def[b] -= w(a) w(c)` for each b in m, accumulated in per-warp shared memory and
        flushed once per word.
    - An outside vertex without a row falls back to binary searches of c in N(b) for the
      ≤32 partners of the word. That happens when another clique owns it this round or
      when the arena is full. The arena grows to 2× the requested size at the next host sync.
    - k_elim34! zeroes the used arena. The outside re-key in k_elim34! still sees final
      deficiencies, because k_whg! runs before it.
  - **Uniform-weight cliques** (unit weights): popcount instead of a weighted bit sum. The
    flag and the per-K weights `kwt` come from k_sizes!.
  - **When dense mode is on.** The host turns it on at the first batch whose last round had
    a clique with d ≥ `dgroup` (kwarg, default 64), and re-captures with k_ob! and k_whg!
    added. Within dense mode, cliques with d < dgroup keep the old per-pair path. Before
    dense mode the CUDA graph is unchanged: no extra kernels and no extra latency on sparse
    graphs.
  - Tried first and dropped: inside part from `bar` plus outside part in k_elim12! with
    O(a) staged. qaoaamp went 0.48 → 0.59 s, slower: O(a) is large there, so the outside
    part needs the masks too.
  - Integer atomics are order-independent, so the deficiencies, keys and selections match
    version 2 exactly. Only the order numbering inside a round can differ, and it already
    came from atomics.
- New kwargs: `dgroup` (typemax = off) and `ocap` (initial arena size, for tests).
  - `timing[:dense]` gives the round where dense mode started.
  - New control slots: C_OTOP, C_OCAP, C_OWANT, C_DACC, C_DLAST.
- The CPU tail still runs the old serial MF1. genv devs CliqueTrees.jl-mt, and `tailcw`
  uses `bestwork`, which origin/main no longer has. A tail is used only in the scale
  config, as the CPU handoff.

### Quality checks (`rport/check.jl`: GRounds vs `rport/GRounds_pre.jl`, the code before the port)
- The comparison is nnz(L), flops and tw with unit weights, plus width and log2 cost with
  the instance weights. Orders inside a round are numbered by atomics, so the order
  vectors themselves are not compared.
- Every run gave a valid permutation, and all edge cases passed (n=0, 1, isolated, path,
  star, K40, K300).
- Runs and results:
  - Default dgroup=64, scores md/mf/mf1 × relax 0/0.1/0.5 × compress on/off × TN/unit
    weights, on lat2_30, lat3_8, bcsstk17, qaoaamp_N240_p3, rqc8x8_m16, cbuckle:
    **216/216 identical**.
  - dgroup=2, which puts nearly every clique through k_ob!/k_whg!: 216/216 identical
    (144 runs in dense mode), plus crystm03 144/144.
  - dgroup=2 with ocap=64 (most claims fail, so the fallback path and arena growth run):
    144/144 identical.
  - Default, mf1 relax 0.1, on qaoaamp_N4000_p5, mc3sat_N3000, rndhyp_T10000 and apache1:
    identical. These cliques reach d ≈ 4400 and hit the >128-partner-words path.
  - The scale config (relax 0.5, tailsel 128, GPU tail plus CPU handoff) gives identical
    nnz(L), flops and tw on all 5 graphs tried.

### Local before/after (laptop RTX 5060, loaded machine, min of 3; `rport/cmp.jl`, `rport/tune.jl`)
Fill config (relax 0.1, unit weights):

| graph | v2 s | rr s | speedup | dense from round |
|---|---|---|---|---|
| qaoaamp_N4000_p5 | 0.476 | 0.163 | 2.9 | 539/936 |
| mc3sat_N3000 | 1.101 | 0.313 | 3.5 | 195/444 |
| rndhyp_T10000 | 0.439 | 0.118 | 3.7 | 265/617 |
| apache1 | 0.257 | 0.255 | 1.0 | 5428/5750 |
| G2_circuit | 0.143 | 0.144 | 1.0 | 2660/2767 |
| ct20stif, crystm03, jnlbrng1, thermal1, bcsstk17, lat2_100, lat3_20 | | | 1.0 ± noise | off |

- Scale config: qaoaamp_N4000_p5 0.79 → 0.19 s (4.1×), rndhyp_T10000 0.47 → 0.13 s (3.7×),
  mc3sat_N3000 1.09 → 0.32 s (3.4×). Same fill.
- qaoaamp kernel split after the port: k_whg! 70 ms and k_elim12! 26 ms, against
  411 ms for k_elim12! before. k_detect! (23 ms) is next.
- Choice of dgroup:
  - 16-32 helps apache1 and G2_circuit a little (up to 10%) and is equal on the dense graphs;
  - 16 costs about 10-20% on jnlbrng1 and lat3_20 (two extra kernels per round, earlier);
  - the default stays 64, and the B200 job also times 32 (`rr-dg32` rows).
- Against Richard's new serial MF1 on the laptop:
  - qaoaamp: GPU 0.163 s vs 0.315 s, **1.9×** (before the port the GPU was 0.65×, slower);
  - apache1 1.15×; lat3_20 1.3×;
  - crystm03 0.7×, jnlbrng1 0.6×, ct20stif 0.8×, lat2_100 0.8×: small sparse graphs,
    where the GPU is bound by round latency and the new serial code is 1.2-1.9× faster
    than before.

### HPG jobs (submitted Oct 19; outputs in /blue/fairbanksj/itaykadosh/gpu/rounds/rr/)
- **itay-gpurr, job 46754896** (B200; rport/hpg/gpurr.sbatch, driver rport/hpg/rsweep.jl, code in rr/code)
  - Rows: gpu-mf1-rr (dgroup 64), gpu-mf1-rr-dg32 and gpu-mf1-v2.
  - Fill config with unit weights: the 16 small fill graphs, the lattices
    lat2_100…1448 and lat3_16/20/32, the dense TN graphs (qaoaamp_N4000_p5,
    rndhyp_T10000/T30000, mc3sat_N3000), and the 19 large graphs (1M-10M).
  - Output: results/fill_b200.tsv.
  - Then the scale config with TN weights: lat2, lat3, corpus lattice2d/3d, qaoaamp
    and rnd ≤ 40k, mc3sat ≤ 3k. Output: results/scale_b200.tsv.
- **itay-rmf, job 46754897** (hpg-milan, 16 cpu; rport/hpg/rmf.sbatch)
  - Richard's new MF1, AMF and AMD, and our old MF1, on the same fill-config graphs.
  - One process per (graph, version, method), 12 in parallel, 3600 s cap.
  - Output: results/rmf_milan.tsv, with per-task parts in results/rmf_parts/.

### Files
- `GRounds.jl`: new kernels k_ob! and k_whg!, plus `pmask` and `gsum`; k_elim12! gains the
  claims, k_elim34! the arena zeroing, k_sizes! `kwt`, `wun` and the clique-size stat;
  the host loop gains the dense-mode switch and arena growth.
- `rport/GRounds_pre.jl`: GRounds before the port (module GRoundsPre = v2 plus the
  ensemble work).
- `rport/check.jl`, `cmp.jl`, `tune.jl`, `sbench.jl`, `local_serial.tsv`, `hpg/`.
- `renv/`: Julia env for Richard's serial code.
