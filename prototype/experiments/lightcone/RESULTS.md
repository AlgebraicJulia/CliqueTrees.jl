# Light-cone QAOA energy: is batched GPU path search worth building?

Feasibility measurement, laptop only (RTX 5060 Laptop, shared 16-thread CPU, numpy/BLAS limited to 4 threads). No HPG, no CliqueTrees changes, nothing pushed.
Everything is reproducible from `scripts/` (gen.py -> data/*.pkl, bench.py -> results/*.json, ours.jl = our methods, agg.py -> tables.md); venv has quimb 1.15, cotengra 0.8.2, kahypar, cupy.

## Verdict (go/no-go)

- **Conditional GO, but not for a GPU search kernel.** Search is the dominant cost even after dedup in three regimes, all with hundreds of distinct nets of 20-400 line-graph vertices (so a batch would have enough work, and almost all fit <=512 vertices, most <=128 for p<=3): (a) 3-regular p=4, N>=200 (about 300-580 distinct nets, 91-196 vertices, contraction about 10 ms/net on CPU): cotengra auto-hq search 540-770 s total vs 3-6 s contraction (share 99%), auto 64-105 s vs 9-10 s (88-91%); (b) Erdos-Renyi (avg deg 5) p=2, any N (129 / ~500 / ~470 distinct, no dedup gain): auto-hq 68-105 s vs 0.5-0.8 s contraction (99%), auto 7-11 s vs 0.4-0.9 s (91-94%); (c) ER N=1000 p=3 (about 325 distinct, 108-434 vertices, still contractible: 21 CPU-s / 3 GPU-s per net): auto-hq search 5.2e4 s (14 h) vs 3.8e3 s contraction (93%).
- **But the fix is already in hand on CPU:** our HBT at 1 s/net (single thread) replaces that search with 0.3-585 s total (e.g. ER N=1000 p=3: 339 s instead of 5.2e4 s; reg3 N=1000 p=4: 585 s instead of 771 s, only a 1.3-3x saving there because HBT uses its whole 1 s budget on these 100-200-vertex nets; a smaller budget would help) at equal or lower contraction cost, and the nets are independent, so 16 CPU threads give another 16x. Quality: our best-of (HBT1 / HBT-ND1 / exact) beats the best of cotengra greedy/auto/auto-hq/hyper0.5s on log10 C in 58 of 93 sampled nets, ties 9, loses 26 (losses are all small, <=0.25 log10 units = <=1.8x, mostly on p<=2 nets that cost milliseconds); at p=4 on 3-regular it is cheaper on 15/15 nets, median -0.47 log10 C (3x), mean -0.68 (4.8x). The gaps of -3 to -60 log10 units on ER p>=3 compare against capped/poor cotengra trees on nets that stay un-contractible (>1 h) either way, so they do not matter in practice.
- **Exact / provably-optimal width adds almost nothing.** PIDBT-based exact solves finished on 63/93 sampled nets (all lg n <=~60 in < 0.1 s; tree-like nets of 124-182 vertices in 1-3 s (ER N=1000 p=3, 3-regular N>=200 p=4); wrapped nets of 108+ vertices (ER N<=200 p>=3, 3-regular N=50 p=4) all hit the 12 s cap). Where it finished, HBT at 1 s found the same width 63/63 times, and the exact-width tree was cheaper than the HBT tree in only 5 cases, by 0.01-0.04 log10 C (<=10%). So a GPU exact solver (the only thing a GPU batch would buy over CPU HBT) could prove optimality on more of the 100-400-vertex nets but would not change the paths in any way that matters.
- **The pipeline bottleneck at scale is neither search nor contraction but quimb's light-cone extraction/simplification:** 0.01-0.3 s/term at p=1, up to 1.7 s/term (3-regular N=1000 p=4), 1.5 / 3.6 / 6.4 s/term (ER N=1000 p=2/3/4); totals 2.5e3 s for reg3 N=1000 p=4, 3.8e3 / 9e3 / 1.6e4 s for ER N=1000 p=2/3/4 (vs HBT search 0.9 / 339 / 405 s), i.e. larger than HBT search and than GPU contraction. Also, the optimizer loop of a real QAOA run reuses the path across angles, so search is paid once while contraction is paid per iteration (x100-1000): search share drops by that factor. Recommendation: do not build a GPU batched path search; the useful, cheap step is wiring CPU HBT (multi-thread, per-net 0.1-1 s) into the light-cone pipeline, and only then revisit GPU contraction (where the GPU wins 2-11x on nets with logC>=7 in T3, and loses on tiny nets because of launch overhead of about 30 us per pairwise contraction).

Hard regime note: ER N=50/200 at p>=3 (mostly) and ER N=1000 p=4 are not contractible by anyone (best-known trees 1e15-1e100 multiply-adds, line graphs 100-1850 vertices, cone wraps the whole graph): path search is irrelevant there; slicing/approximate methods would be needed. Dedup is decisive only for 3-regular at p<=3 (1500 terms -> 1-7 nets at p<=2, ~56 at p=3 for N=1000) and p=1 anywhere; at p=4 or on ER it removes ~0-60%. 3-regular is easy at p<=4 (treewidth stays ~10-25, contraction milliseconds) because cones are tree-like with short cycles only.

## Tables
(Tables follow after the caveats.)

## Method and caveats

- Graphs: random 3-regular and G(N, p) with average degree 5 (one seed, seed=1). N in {50, 200, 1000}, p in {1..4}, MaxCut QAOA, Hamiltonian terms Z_i Z_j on every edge (so #terms = #edges: 75/300/1500 for 3-regular, 130/500/2487 for ER), gammas/betas fixed. Network per term: `Circuit.local_expectation(ZZ, (i,j), rehearse='tn')`, quimb default simplification ('ADCRS').
- Dedup: Weisfeiler-Leman hash (6 rounds) of the tensor-index incidence graph with tensor rank / index dimension labels; data is identical for identical structure (same angles). WL equal-hash can merge non-isomorphic nets, so distinct counts are lower bounds (generous to dedup caching). Isomorphism not checked.
- Extraction was capped at 90 s per setting; when not all terms were reached (N=200/1000 at larger p, ER) the distinct count over all terms is estimated (distinct seen + Chao1 unseen, capped by remaining terms; "est." column). For ER p>=3 the estimate is essentially "all terms distinct"; for 3-regular N=1000 p=3,4 it is rough (56 and ~580; true values could differ by about 2x).
- cotengra per-net optimizers on a quantile sample of <=8 distinct nets per setting (<=3-4 at the largest), weighted back to all distinct nets (and to all terms for the no-dedup totals). All single-threaded (`parallel=False`; the default 'auto' presets would spawn a process pool). auto / auto-hq capped by an alarm (60 s; the first-run settings er N=1000 p=2,3 had the cap ineffective and ran uncapped, so their search times are complete); capped nets count their cap as search time (lower bound) and the greedy tree's stats. HyperOptimizer(max_time=0.5) overshoots its budget on big nets (up to 50 s on 700-1800 tensors); its trees at 0.5 s are erratic, so it is shown in T2 but excluded from T4.
- Nets with more than 2500 tensors (ER N=1000 p=4: 19/20 nets) are not optimised (too slow for any method here); that setting rests on one net.
- Contraction: random complex128 data, the optimizer's tree, numpy (4 BLAS threads) and cupy (RTX 5060 Laptop, fp64 which is slow on this consumer GPU) timed on every sampled net with logC<=9.3 and W<=24 (285 measurements). Time model fitted on those: t = a*(#pairwise contractions) + b*(multiply-adds) (T0 below, mean relative error 25-30%), used to extrapolate to all distinct nets and to nets too expensive to run. Contraction cost of hard nets (logC>9.3) is model-only. torch not installed; cupy installed in the venv (works).
- Our methods (Julia, 1 thread, via `ours.jl` calling CliqueTrees mtenv common.jl pipeline): bw (min-fill-type heuristics), HBT T=0.2 s, HBT T=1 s, HBT-ND T=1 s (nd init), exact = SafeRules + SafeSeparators + PIDBT (12 s cap, stop at first timeout), on <=5 nets/setting spread over the size range (<=2 in ER N=1000 p=4). Orderings are converted to contraction trees by the existing bridge `frontier/scripts/ftn.py:tree_from_order` (bucket elimination, auto-hq for binarising; no subtree refinement), then costed like cotengra's trees. Widths in T5 'width proven' and 'HBT1 = exact' are bag widths of the line graph (log2 of the largest bag; tw+1 with unit weights); the beat/tie/lose counts use cotengra's own metrics (logC, W = log2 of largest intermediate).
- Search times of ours are CPU wall times excluding Julia startup/JIT (15-45 s, once per process); cotengra times include its own Python overhead.
- One random graph per setting; no repeats, so per-setting differences of ~0.1 log10 C between methods are noise.

### T0. Contraction-time model (fit on all measured contractions, relative error)

- t_cpu: t = 16.60 us * (#pairwise contractions) + 0.210 ns * (scalar mult-adds); mean rel. error 0.30 over 285 measurements
- t_gpu: t = 32.37 us * (#pairwise contractions) + 0.032 ns * (scalar mult-adds); mean rel. error 0.25 over 285 measurements

### T1. Terms, distinct light-cone networks, line-graph sizes (distinct nets)

| graph | N | p | terms | terms sampled | distinct (sampled) | distinct (est. all terms) | quimb extract s/term | lg n min/med/max | lg m min/med/max | <=128 v | <=512 v |
|---|---|---|---|---|---|---|---|---|---|---|---|
| er | 50 | 1 | 130 | 130 | 4 | 4 | 0.033 | 5/6/7 | 8/12/16 | 3 | 3 |
| er | 50 | 2 | 130 | 130 | 129 | 129 | 0.094 | 23/45/67 | 50/127/205 | 129 | 129 |
| er | 50 | 3 | 130 | 130 | 129 | 129 | 0.174 | 102/145/173 | 305/468/562 | 23 | 129 |
| er | 50 | 4 | 130 | 130 | 129 | 129 | 0.260 | 202/246/274 | 661/828/922 | 0 | 129 |
| er | 200 | 1 | 500 | 500 | 3 | 3 | 0.115 | 5/6/6 | 8/10/12 | 2 | 2 |
| er | 200 | 2 | 500 | 217 | 215 | 498 | 0.414 | 12/41/77 | 22/98/202 | 215 | 215 |
| er | 200 | 3 | 500 | 93 | 93 | 500 | 0.962 | 84/242/347 | 206/704/1062 | 7 | 93 |
| er | 200 | 4 | 500 | 61 | 61 | 500 | 1.463 | 364/635/745 | 1082/2029/2438 | 0 | 10 |
| er | 1000 | 1 | 2487 | 341 | 2 | 2 | 0.265 | 5/5/5 | 8/8/8 | 1 | 1 |
| er | 1000 | 2 | 2487 | 60 | 54 | 471 | 1.515 | 18/40/68 | 36/86/167 | 54 | 54 |
| er | 1000 | 3 | 2487 | 25 | 25 | 325 | 3.606 | 108/197/434 | 247/476/1154 | 2 | 25 |
| er | 1000 | 4 | 2487 | 20 | 20 | 210 | 6.359 | 665/1064/1853 | 1794/3048/5703 | 0 | 0 |
| reg3 | 50 | 1 | 75 | 75 | 2 | 2 | 0.013 | 5/5/5 | 8/8/8 | 1 | 1 |
| reg3 | 50 | 2 | 75 | 75 | 6 | 6 | 0.043 | 17/20/21 | 35/40/42 | 6 | 6 |
| reg3 | 50 | 3 | 75 | 75 | 62 | 62 | 0.088 | 42/57/60 | 87/118/125 | 62 | 62 |
| reg3 | 50 | 4 | 75 | 75 | 75 | 75 | 0.144 | 91/129/155 | 197/278/335 | 37 | 75 |
| reg3 | 200 | 1 | 300 | 300 | 2 | 2 | 0.017 | 5/5/5 | 8/8/8 | 1 | 1 |
| reg3 | 200 | 2 | 300 | 300 | 4 | 4 | 0.085 | 17/20/21 | 35/39/42 | 4 | 4 |
| reg3 | 200 | 3 | 300 | 300 | 50 | 50 | 0.233 | 42/58/60 | 87/118/124 | 50 | 50 |
| reg3 | 200 | 4 | 300 | 171 | 170 | 299 | 0.529 | 93/137/171 | 196/285/352 | 17 | 170 |
| reg3 | 1000 | 2 | 1500 | 509 | 5 | 7 | 0.177 | 20/21/26 | 38/42/48 | 5 | 5 |
| reg3 | 1000 | 3 | 1500 | 142 | 20 | 56 | 0.634 | 50/57/59 | 102/116/120 | 20 | 20 |
| reg3 | 1000 | 4 | 1500 | 54 | 49 | 578 | 1.670 | 119/143/196 | 247/293/387 | 1 | 49 |

(lg = line graph: one vertex per tensor index; edge = two indices sharing a tensor. The <=128/<=512 columns count among the sampled-term distinct nets.)

### T2. cotengra per distinct network (quantile sample of <=10 nets per setting; mean over the sample, dedup-weighted)

search s = wall seconds of the optimizer (auto / auto-hq are capped at 60 s, 90 s in the first runs; a capped net contributes its cap as search time and the stats of the greedy tree); logC = log10 scalar mult-adds; W = log2 max intermediate. Hyper = HyperOptimizer(max_time=0.5).

| graph N p | #sampled | greedy s / logC / W | auto s / logC / W | auto-hq s / logC / W | hyper0.5 s / logC / W |
|---|---|---|---|---|---|
| er 50 1 | 3/3 | 0.000 / 2.11 / 3.7 | 0.157 / 2.11 / 3.7 | 20.155 / 2.11 / 3.7 (1 capped) | 0.536 / 2.08 / 3.7 |
| er 50 2 | 8/8 | 0.006 / 6.92 / 17.4 | 0.055 / 5.84 / 14.6 | 0.528 / 5.15 / 12.7 | 0.807 / 5.23 / 13.2 |
| er 50 3 | 8/8 | 0.072 / 25.02 / 66.7 | 59.821 / 24.15 / 65.0 (7 capped) | 60.009 / 25.02 / 66.7 (8 capped) | 6.023 / 17.78 / 51.8 |
| er 50 4 | 8/8 | 0.194 / 33.76 / 87.0 | 60.009 / 33.76 / 87.0 (8 capped) | 60.002 / 33.76 / 87.0 (8 capped) | 16.664 / 35.08 / 109.5 |
| er 200 1 | 2/2 | 0.000 / 1.98 / 3.5 | 0.169 / 1.98 / 3.5 | 0.172 / 1.98 / 3.5 | 0.518 / 1.98 / 3.5 |
| er 200 2 | 8/8 | 0.002 / 5.02 / 11.9 | 0.021 / 4.90 / 11.7 | 0.210 / 4.12 / 9.5 | 0.634 / 3.86 / 8.6 |
| er 200 3 | 8/8 | 0.067 / 28.19 / 73.7 | 47.080 / 27.64 / 72.4 (6 capped) | 51.700 / 27.15 / 71.1 (6 capped) | 7.481 / 27.30 / 84.3 |
| er 200 4 | 8/8 | 1.030 / 104.53 / 282.7 | 60.005 / 104.53 / 282.7 (8 capped) | 60.004 / 104.53 / 282.7 (8 capped) | 49.967 / 73.93 / 239.9 |
| er 1000 1 | 1/1 | 0.000 / 1.83 / 3.0 | 0.007 / 1.83 / 3.0 | 0.004 / 1.83 / 3.0 | 0.551 / 1.83 / 3.0 |
| er 1000 2 | 8/8 | 0.002 / 3.48 / 6.9 | 0.015 / 4.13 / 8.7 | 0.164 / 3.46 / 6.6 | 0.616 / 3.30 / 5.8 |
| er 1000 3 | 8/8 | 0.027 / 13.65 / 34.7 | 21.072 / 11.69 / 31.1 | 159.216 / 8.19 / 20.9 | 4.249 / 16.87 / 50.5 |
| er 1000 4 | 1/3 | 0.287 / 96.33 / 246.0 | 60.000 / 96.33 / 246.0 (1 capped) | 60.002 / 96.33 / 246.0 (1 capped) | 77.608 / 123.70 / 403.0 |
| reg3 50 1 | 1/1 | 0.000 / 1.83 / 3.0 | 0.004 / 1.83 / 3.0 | 0.003 / 1.83 / 3.0 | 0.554 / 1.83 / 3.0 |
| reg3 50 2 | 6/6 | 0.001 / 3.08 / 5.8 | 0.005 / 3.11 / 5.8 | 0.058 / 2.92 / 5.0 | 0.531 / 2.89 / 4.8 |
| reg3 50 3 | 8/8 | 0.003 / 5.28 / 12.7 | 0.025 / 5.23 / 12.5 | 0.363 / 5.33 / 12.7 | 0.717 / 4.99 / 12.1 |
| reg3 50 4 | 8/8 | 0.009 / 10.41 / 25.2 | 1.636 / 8.45 / 21.1 | 2.842 / 7.77 / 19.3 | 1.052 / 10.30 / 29.4 |
| reg3 200 1 | 1/1 | 0.000 / 1.83 / 3.0 | 0.004 / 1.83 / 3.0 | 0.003 / 1.83 / 3.0 | 0.545 / 1.83 / 3.0 |
| reg3 200 2 | 4/4 | 0.001 / 2.97 / 5.5 | 0.005 / 3.06 / 5.8 | 0.064 / 3.09 / 6.0 | 0.518 / 2.85 / 4.5 |
| reg3 200 3 | 8/8 | 0.002 / 4.55 / 9.8 | 0.018 / 4.85 / 10.7 | 0.242 / 4.38 / 9.7 | 0.714 / 4.07 / 8.4 |
| reg3 200 4 | 8/8 | 0.008 / 7.35 / 17.6 | 0.213 / 7.64 / 19.0 | 1.791 / 6.68 / 16.4 | 0.861 / 8.07 / 21.6 |
| reg3 1000 2 | 5/5 | 0.001 / 3.06 / 5.8 | 0.007 / 3.19 / 6.2 | 0.109 / 3.27 / 6.4 | 0.559 / 2.89 / 4.8 |
| reg3 1000 3 | 8/8 | 0.003 / 4.31 / 9.7 | 0.023 / 4.61 / 10.2 | 0.233 / 5.44 / 13.3 | 0.643 / 4.03 / 8.5 |
| reg3 1000 4 | 8/8 | 0.008 / 5.50 / 12.9 | 0.182 / 6.57 / 15.9 | 1.335 / 5.95 / 14.4 | 0.911 / 7.38 / 19.4 |

### T3. Actual contraction time (numpy complex128, 4 BLAS threads; cupy complex128 on RTX 5060 Laptop), nets with logC<=9.3 and W<=24, auto-hq tree

| graph N p | #measured | cpu ms (mean) | gpu ms (mean) | mean logC measured | note |
|---|---|---|---|---|---|
| er 50 1 | 3/3 | 1.39 | 1.28 | 2.11 | |
| er 50 2 | 8/8 | 34.27 | 10.07 | 5.15 | |
| er 50 3 | 0/8 | - | - | - | too expensive for laptop numpy (median logC 26.0) |
| er 50 4 | 0/8 | - | - | - | too expensive for laptop numpy (median logC 33.4) |
| er 200 1 | 2/2 | 0.21 | 0.27 | 1.98 | |
| er 200 2 | 8/8 | 1.40 | 3.91 | 4.12 | |
| er 200 3 | 1/8 | 125.27 | 23.02 | 8.68 | |
| er 200 4 | 0/8 | - | - | - | too expensive for laptop numpy (median logC 112.4) |
| er 1000 1 | 1/1 | 0.19 | 0.20 | 1.83 | |
| er 1000 2 | 8/8 | 1.17 | 2.33 | 3.47 | |
| er 1000 3 | 6/8 | 50.73 | 26.85 | 7.30 | |
| er 1000 4 | 0/1 | - | - | - | too expensive for laptop numpy (median logC 96.3) |
| reg3 50 1 | 1/1 | 0.19 | 0.29 | 1.83 | |
| reg3 50 2 | 6/6 | 0.60 | 1.07 | 2.92 | |
| reg3 50 3 | 8/8 | 2.81 | 4.73 | 5.32 | |
| reg3 50 4 | 8/8 | 35.04 | 16.84 | 7.77 | |
| reg3 200 1 | 1/1 | 0.25 | 0.28 | 1.83 | |
| reg3 200 2 | 4/4 | 0.63 | 1.09 | 3.09 | |
| reg3 200 3 | 8/8 | 1.83 | 3.39 | 4.37 | |
| reg3 200 4 | 8/8 | 13.22 | 10.86 | 6.68 | |
| reg3 1000 2 | 5/5 | 1.02 | 1.81 | 3.27 | |
| reg3 1000 3 | 8/8 | 5.20 | 4.92 | 5.33 | |
| reg3 1000 4 | 8/8 | 213.65 | 18.55 | 5.95 | |

### T4. Total path-search time vs total contraction time

Per setting: totals over all terms (no dedup) and over all distinct nets (dedup; distinct count extrapolated for partially-sampled settings, see T1). search = optimizer wall time per net (single thread); contract = modelled numpy-CPU / cupy-GPU time of the optimizer's own tree (a net whose search was capped uses the cheapest tree known for it, so its contraction term is optimistic and its search term a lower bound). 'infeas' = fraction of distinct nets whose cheapest known tree needs > 1 h of modelled CPU contraction (exact light-cone contraction is out of reach on the laptop regardless of search). ratio = search / contract; shares are search/(search+contract) over the FEASIBLE nets only, dedup. Hyper-0.5s is omitted (its trees are erratic on big nets, see T2).

| graph N p | opt | infeas | search no-dedup s | search dedup s | contract-CPU dedup s (feasible) | contract-GPU dedup s (feasible) | ratio CPU, no dedup | ratio CPU, dedup | ratio GPU, dedup | search share CPU dedup | search share GPU dedup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| er 50 1 | auto | 0.00 | 15.3 | 0.629 | 0.000731 | 0.00142 | 838 | 861 | 442 | 0.999 | 0.998 |
| er 50 1 | auto-hq | 0.00 | 457 | 80.6 | 0.000731 | 0.00142 | 2.51e+04 | 1.1e+05 | 5.66e+04 | 1.000 | 1.000 |
| er 50 2 | auto | 0.00 | 7.07 | 7.04 | 0.427 | 0.548 | 16.5 | 16.5 | 12.8 | 0.943 | 0.928 |
| er 50 2 | auto-hq | 0.00 | 68.6 | 68.1 | 0.528 | 0.563 | 130 | 129 | 121 | 0.992 | 0.992 |
| er 50 3 | auto | 0.74 | 7.78e+03 | 7.72e+03 | 7.83e+04 | 1.2e+04 | 8.6e-11 | 9.07e-11 | 5.93e-10 | 0.024 | 0.140 |
| er 50 3 | auto-hq | 0.74 | 7.8e+03 | 7.74e+03 | 1.11e+03 | 171 | 8.63e-11 | 9.1e-11 | 5.95e-10 | 0.640 | 0.921 |
| er 50 4 | auto | 1.00 | 7.8e+03 | 7.74e+03 | nan | nan | 2.7e-23 | 2.68e-23 | 1.75e-22 | nan | nan |
| er 50 4 | auto-hq | 1.00 | 7.8e+03 | 7.74e+03 | nan | nan | 2.7e-23 | 2.68e-23 | 1.75e-22 | nan | nan |
| er 200 1 | auto | 0.00 | 6.37 | 0.506 | 0.000448 | 0.000874 | 108 | 1.13e+03 | 579 | 0.999 | 0.998 |
| er 200 1 | auto-hq | 0.00 | 6.43 | 0.515 | 0.000448 | 0.000874 | 109 | 1.15e+03 | 589 | 0.999 | 0.998 |
| er 200 2 | auto | 0.00 | 10.7 | 10.7 | 0.857 | 1.59 | 12.5 | 12.5 | 6.74 | 0.926 | 0.871 |
| er 200 2 | auto-hq | 0.00 | 105 | 105 | 0.812 | 1.58 | 129 | 129 | 66.3 | 0.992 | 0.985 |
| er 200 3 | auto | 0.48 | 2.35e+04 | 2.35e+04 | 1.09e+05 | 1.67e+04 | 1.86e-27 | 1.86e-27 | 1.21e-26 | 0.076 | 0.350 |
| er 200 3 | auto-hq | 0.48 | 2.58e+04 | 2.58e+04 | 8.21e+04 | 1.26e+04 | 2.04e-27 | 2.04e-27 | 1.33e-26 | 0.121 | 0.474 |
| er 200 4 | auto | 1.00 | 3e+04 | 3e+04 | nan | nan | 5.03e-75 | 5.03e-75 | 3.29e-74 | nan | nan |
| er 200 4 | auto-hq | 1.00 | 3e+04 | 3e+04 | nan | nan | 5.03e-75 | 5.03e-75 | 3.29e-74 | nan | nan |
| er 1000 1 | auto | 0.00 | 17.4 | 0.014 | 0.000232 | 0.000453 | 60.1 | 60.1 | 30.8 | 0.984 | 0.969 |
| er 1000 1 | auto-hq | 0.00 | 9.16 | 0.00737 | 0.000232 | 0.000453 | 31.7 | 31.7 | 16.3 | 0.969 | 0.942 |
| er 1000 2 | auto | 0.00 | 36.1 | 7.15 | 0.698 | 1.34 | 10.1 | 10.3 | 5.32 | 0.911 | 0.842 |
| er 1000 2 | auto-hq | 0.00 | 390 | 77.4 | 0.689 | 1.34 | 110 | 112 | 57.7 | 0.991 | 0.983 |
| er 1000 3 | auto | 0.00 | 5.24e+04 | 6.85e+03 | 1.88e+12 | 2.87e+11 | 3.65e-09 | 3.65e-09 | 2.38e-08 | 0.000 | 0.000 |
| er 1000 3 | auto-hq | 0.00 | 3.96e+05 | 5.17e+04 | 3.78e+03 | 583 | 13.7 | 13.7 | 88.7 | 0.932 | 0.989 |
| er 1000 4 | auto | 1.00 | 1.49e+05 | 1.26e+04 | nan | nan | 3.46e-25 | 3.46e-25 | 2.26e-24 | nan | nan |
| er 1000 4 | auto-hq | 1.00 | 1.49e+05 | 1.26e+04 | nan | nan | 3.46e-25 | 3.46e-25 | 2.26e-24 | nan | nan |
| reg3 50 1 | auto | 0.00 | 0.267 | 0.00712 | 0.000232 | 0.000453 | 30.6 | 30.6 | 15.7 | 0.968 | 0.940 |
| reg3 50 1 | auto-hq | 0.00 | 0.252 | 0.00671 | 0.000232 | 0.000453 | 28.9 | 28.9 | 14.8 | 0.967 | 0.937 |
| reg3 50 2 | auto | 0.00 | 0.325 | 0.0273 | 0.00384 | 0.00748 | 6.89 | 7.11 | 3.65 | 0.877 | 0.785 |
| reg3 50 2 | auto-hq | 0.00 | 4.01 | 0.349 | 0.00384 | 0.00748 | 85.2 | 90.9 | 46.6 | 0.989 | 0.979 |
| reg3 50 3 | auto | 0.00 | 1.89 | 1.52 | 0.139 | 0.236 | 10.7 | 11 | 6.46 | 0.916 | 0.866 |
| reg3 50 3 | auto-hq | 0.00 | 27.8 | 22.5 | 0.132 | 0.235 | 174 | 170 | 95.6 | 0.994 | 0.990 |
| reg3 50 4 | auto | 0.00 | 123 | 123 | 23.6 | 4.24 | 5.2 | 5.2 | 28.9 | 0.839 | 0.967 |
| reg3 50 4 | auto-hq | 0.00 | 213 | 213 | 1.99 | 0.936 | 107 | 107 | 228 | 0.991 | 0.996 |
| reg3 200 1 | auto | 0.00 | 1.05 | 0.00703 | 0.000232 | 0.000453 | 30.2 | 30.2 | 15.5 | 0.968 | 0.939 |
| reg3 200 1 | auto-hq | 0.00 | 1 | 0.00667 | 0.000232 | 0.000453 | 28.7 | 28.7 | 14.7 | 0.966 | 0.936 |
| reg3 200 2 | auto | 0.00 | 1.32 | 0.0181 | 0.00251 | 0.00489 | 7.11 | 7.2 | 3.7 | 0.878 | 0.787 |
| reg3 200 2 | auto-hq | 0.00 | 16.3 | 0.256 | 0.00251 | 0.00489 | 87.9 | 102 | 52.3 | 0.990 | 0.981 |
| reg3 200 3 | auto | 0.00 | 5.6 | 0.912 | 0.099 | 0.19 | 9.64 | 9.21 | 4.79 | 0.902 | 0.827 |
| reg3 200 3 | auto-hq | 0.00 | 74.8 | 12.1 | 0.0993 | 0.191 | 127 | 122 | 63.6 | 0.992 | 0.985 |
| reg3 200 4 | auto | 0.00 | 63.5 | 63.6 | 9.09 | 3.96 | 6.99 | 6.99 | 16.1 | 0.875 | 0.941 |
| reg3 200 4 | auto-hq | 0.00 | 536 | 536 | 3.35 | 3.08 | 160 | 160 | 174 | 0.994 | 0.994 |
| reg3 1000 2 | auto | 0.00 | 8.06 | 0.051 | 0.00477 | 0.00929 | 8.73 | 10.7 | 5.49 | 0.915 | 0.846 |
| reg3 1000 2 | auto-hq | 0.00 | 222 | 0.763 | 0.00477 | 0.00929 | 240 | 160 | 82.1 | 0.994 | 0.988 |
| reg3 1000 3 | auto | 0.00 | 29 | 1.3 | 0.107 | 0.208 | 10.2 | 12.1 | 6.24 | 0.924 | 0.862 |
| reg3 1000 3 | auto-hq | 0.00 | 600 | 13.1 | 0.152 | 0.215 | 79.1 | 86.1 | 60.8 | 0.989 | 0.984 |
| reg3 1000 4 | auto | 0.00 | 252 | 105 | 10.2 | 6.72 | 10.2 | 10.3 | 15.6 | 0.911 | 0.940 |
| reg3 1000 4 | auto-hq | 0.00 | 1.98e+03 | 771 | 5.9 | 6.06 | 136 | 131 | 127 | 0.992 | 0.992 |

#### T4b. Our HBT (T=1 s, 1 thread) as the path source, same nets (subset of <=5 per setting; mean per net; contraction modelled)

| graph N p | #nets | HBT1 search s/net (+bucket tree s) | contract CPU s/net | contract GPU s/net | search/contract CPU | search/contract GPU | cotengra auto-hq on same nets: search s/net, contract CPU s/net (uncapped nets only) |
|---|---|---|---|---|---|---|---|
| er 50 1 | 3 | 0.000676 (0.00064) | 0.000183 | 0.000356 | 3.7 | 1.9 | 0.232, 0.000149 (2/3) |
| er 50 2 | 5 | 0.0195 (0.0078) | 0.00212 | 0.00408 | 9.22 | 4.79 | 0.453, 0.00216 (5/5) |
| er 50 3 | 5 | 1.27 (0.24) | 7.03e+04 | 1.08e+04 | 1.8e-05 | 0.000118 | all capped |
| er 50 4 | 5 | 1.29 (0.28) | 3.91e+08 | 5.98e+07 | 3.3e-09 | 2.16e-08 | all capped |
| er 200 1 | 2 | 0.000219 (0.0002) | 0.000149 | 0.000291 | 1.46 | 0.751 | 0.172, 0.000149 (2/2) |
| er 200 2 | 5 | 0.00446 (0.0027) | 0.00165 | 0.00321 | 2.7 | 1.39 | 0.225, 0.00165 (5/5) |
| er 200 3 | 5 | 1.1 (0.09) | 2.37e+07 | 3.63e+06 | 4.63e-08 | 3.02e-07 | 5.07, 0.105 (1/5) |
| er 200 4 | 5 | 2.43 (1.4) | 1.75e+43 | 2.68e+42 | 1.39e-43 | 9.06e-43 | all capped |
| er 1000 1 | 1 | 0.000209 (0.00018) | 0.000116 | 0.000227 | 1.79 | 0.921 | 0.00368, 0.000116 (1/1) |
| er 1000 2 | 5 | 0.00196 (0.0016) | 0.00152 | 0.00296 | 1.29 | 0.661 | 0.199, 0.00152 (5/5) |
| er 1000 3 | 5 | 1.04 (0.033) | 21 | 3.23 | 0.0497 | 0.323 | 157, 15.7 (5/5) |
| er 1000 4 | 1 | 1.93 (0.73) | 1.73e+26 | 2.65e+25 | 1.11e-26 | 7.26e-26 | all capped |
| reg3 50 1 | 1 | 0.000194 (0.00017) | 0.000116 | 0.000227 | 1.67 | 0.858 | 0.00336, 0.000116 (1/1) |
| reg3 50 2 | 5 | 0.00129 (0.00094) | 0.000638 | 0.00124 | 2.03 | 1.04 | 0.0581, 0.000638 (5/5) |
| reg3 50 3 | 5 | 0.027 (0.0024) | 0.00194 | 0.00377 | 13.9 | 7.16 | 0.311, 0.00217 (5/5) |
| reg3 50 4 | 5 | 1.01 (0.0074) | 0.0184 | 0.0113 | 55 | 89.4 | 3.81, 0.0357 (5/5) |
| reg3 200 1 | 1 | 0.000178 (0.00016) | 0.000116 | 0.000227 | 1.53 | 0.785 | 0.00333, 0.000116 (1/1) |
| reg3 200 2 | 4 | 0.00106 (0.00078) | 0.000627 | 0.00122 | 1.69 | 0.868 | 0.0639, 0.000627 (4/4) |
| reg3 200 3 | 5 | 0.00912 (0.0026) | 0.00196 | 0.00382 | 4.65 | 2.39 | 0.223, 0.00196 (5/5) |
| reg3 200 4 | 5 | 1.02 (0.0066) | 0.00555 | 0.00945 | 183 | 108 | 2.17, 0.00749 (5/5) |
| reg3 1000 2 | 5 | 0.00131 (0.00084) | 0.000681 | 0.00133 | 1.92 | 0.984 | 0.109, 0.000681 (5/5) |
| reg3 1000 3 | 5 | 0.00649 (0.0035) | 0.00191 | 0.00373 | 3.39 | 1.74 | 0.182, 0.00194 (5/5) |
| reg3 1000 4 | 5 | 1.01 (0.0083) | 0.00503 | 0.00979 | 201 | 103 | 0.804, 0.0051 (5/5) |

### T6. Where the wall time goes, per setting (dedup, totals over all distinct nets; HBT row from the <=5-net subset mean x #distinct)

extract = quimb light-cone extraction + simplification of ALL terms (needed even with dedup, to find the duplicates); search/contract columns as in T4/T4b; contract = modelled on the HBT1 tree, feasible nets only if any (else n/a: >1 h per net).

| graph N p | #terms | #distinct (est) | extract s | auto-hq search s | HBT1 search s | contract CPU s (HBT1 trees) | contract GPU s (HBT1 trees) | nets feasible (<=1 h CPU) |
|---|---|---|---|---|---|---|---|---|
| er 50 1 | 130 | 4 | 4.27 | 80.6 | 0.0027 | 0.000731 | 0.00142 | 3/3 |
| er 50 2 | 130 | 129 | 12.2 | 68.1 | 2.52 | 0.273 | 0.526 | 5/5 |
| er 50 3 | 130 | 129 | 22.6 | 7.74e+03 | 164 | 2.79e+03 | 427 | 2/5 |
| er 50 4 | 130 | 129 | 33.8 | 7.74e+03 | 166 | nan | nan | 0/5 |
| er 200 1 | 500 | 3 | 57.6 | 0.515 | 0.000656 | 0.000448 | 0.000874 | 2/2 |
| er 200 2 | 500 | 498 | 207 | 105 | 2.22 | 0.822 | 1.6 | 5/5 |
| er 200 3 | 500 | 500 | 481 | 2.58e+04 | 549 | 1.29e+05 | 1.97e+04 | 3/5 |
| er 200 4 | 500 | 500 | 732 | 3e+04 | 1.21e+03 | nan | nan | 0/5 |
| er 1000 1 | 2487 | 2 | 659 | 0.00737 | 0.000417 | 0.000232 | 0.000453 | 1/1 |
| er 1000 2 | 2487 | 471 | 3.77e+03 | 77.4 | 0.923 | 0.717 | 1.4 | 5/5 |
| er 1000 3 | 2487 | 325 | 8.97e+03 | 5.17e+04 | 339 | 6.83e+03 | 1.05e+03 | 5/5 |
| er 1000 4 | 2487 | 210 | 1.58e+04 | 1.26e+04 | 405 | nan | nan | 0/1 |
| reg3 50 1 | 75 | 2 | 0.944 | 0.00671 | 0.000389 | 0.000232 | 0.000453 | 1/1 |
| reg3 50 2 | 75 | 6 | 3.23 | 0.349 | 0.00775 | 0.00383 | 0.00746 | 5/5 |
| reg3 50 3 | 75 | 62 | 6.61 | 22.5 | 1.68 | 0.12 | 0.234 | 5/5 |
| reg3 50 4 | 75 | 75 | 10.8 | 213 | 75.9 | 1.38 | 0.849 | 5/5 |
| reg3 200 1 | 300 | 2 | 4.98 | 0.00667 | 0.000356 | 0.000232 | 0.000453 | 1/1 |
| reg3 200 2 | 300 | 4 | 25.4 | 0.256 | 0.00424 | 0.00251 | 0.00489 | 4/4 |
| reg3 200 3 | 300 | 50 | 70 | 12.1 | 0.456 | 0.0981 | 0.191 | 5/5 |
| reg3 200 4 | 300 | 299 | 159 | 536 | 304 | 1.66 | 2.83 | 5/5 |
| reg3 1000 2 | 1500 | 7 | 265 | 0.763 | 0.00914 | 0.00477 | 0.00929 | 5/5 |
| reg3 1000 3 | 1500 | 56 | 952 | 13.1 | 0.364 | 0.107 | 0.209 | 5/5 |
| reg3 1000 4 | 1500 | 578 | 2.5e+03 | 771 | 585 | 2.91 | 5.66 | 5/5 |

### T5. Our methods vs cotengra on the same nets (sample of <=5 nets per setting spanning the size range)

exact = SafeRules+SafeSeparators+PIDBT (12 s cap, stop after first timeout); 'proven' = exact finished OR HBT bag width == MMW lower bound. Costs are of the contraction trees obtained by bucket elimination from our order (ftn bridge, no subtree refinement). 'ours best' = cheaper of hbt1 / hbtnd1 / exact (by logC). beat/tie/lose counts use tol 0.01 in logC and 0.5 in W (W = log2 largest intermediate). vs hq: only nets where auto-hq finished within the cap. vs best-cot: best of greedy/auto/auto-hq/hyper0.5 among the uncapped ones.

| graph N p | #nets | lg n range | exact finished | width proven | exact time s (max) | HBT1 = exact width (of finished) | exact tree cheaper than HBT1 tree (of finished) | ours vs hq logC b/t/l (n) | ours vs best-cot logC b/t/l | ours vs best-cot W b/t/l | mean dlogC vs hq | mean dlogC vs best-cot | max dlogC vs best-cot (best/worst) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| er 50 1 | 3 | 5-7 | 3/3 | 3/3 | 6.01e-05 | 3/3 | 0/3 | 0/1/1 (2) | 0/1/2 | 0/3/0 | +0.05 | +0.05 | +0.00 / +0.09 |
| er 50 2 | 5 | 28-57 | 5/5 | 5/5 | 0.0676 | 5/5 | 1/5 | 5/0/0 (5) | 4/0/1 | 4/1/0 | -0.49 | -0.26 | -0.69 / +0.03 |
| er 50 3 | 5 | 119-164 | 0/5 | 0/5 | nan | 0/0 | 0/0 | - (0) | 5/0/0 | 5/0/0 | - | -3.89 | -6.29 / -1.76 |
| er 50 4 | 5 | 218-264 | 0/5 | 0/5 | nan | 0/0 | 0/0 | - (0) | 5/0/0 | 5/0/0 | - | -11.38 | -20.22 / -2.09 |
| er 200 1 | 2 | 5-6 | 2/2 | 2/2 | 5.39e-05 | 2/2 | 0/2 | 0/1/1 (2) | 0/1/1 | 0/2/0 | +0.06 | +0.06 | +0.00 / +0.11 |
| er 200 2 | 5 | 25-59 | 5/5 | 5/5 | 0.0396 | 5/5 | 1/5 | 3/0/2 (5) | 2/0/3 | 2/3/0 | -0.25 | -0.02 | -0.24 / +0.09 |
| er 200 3 | 5 | 125-308 | 0/5 | 0/5 | nan | 0/0 | 0/0 | 1/0/0 (1) | 5/0/0 | 5/0/0 | -2.07 | -8.39 | -17.09 / -2.07 |
| er 200 4 | 5 | 442-710 | 0/5 | 0/5 | nan | 0/0 | 0/0 | - (0) | 5/0/0 | 5/0/0 | - | -38.20 | -52.41 / -17.44 |
| er 1000 1 | 1 | 5-5 | 1/1 | 1/1 | 4.82e-05 | 1/1 | 0/1 | 0/1/0 (1) | 0/1/0 | 0/1/0 | +0.00 | +0.00 | +0.00 / +0.00 |
| er 1000 2 | 5 | 24-62 | 5/5 | 5/5 | 0.000573 | 5/5 | 0/5 | 3/1/1 (5) | 2/2/1 | 1/3/1 | -0.15 | -0.01 | -0.06 / +0.07 |
| er 1000 3 | 5 | 130-382 | 2/5 | 2/5 | 2.58 | 2/2 | 0/2 | 4/0/1 (5) | 4/0/1 | 2/2/1 | -0.65 | -0.45 | -0.99 / +0.13 |
| er 1000 4 | 1 | 836-836 | 0/1 | 0/1 | nan | 0/0 | 0/0 | - (0) | 1/0/0 | 1/0/0 | - | -60.41 | -60.41 / -60.41 |
| reg3 50 1 | 1 | 5-5 | 1/1 | 1/1 | 3.91e-05 | 1/1 | 0/1 | 0/1/0 (1) | 0/1/0 | 0/1/0 | +0.00 | +0.00 | +0.00 / +0.00 |
| reg3 50 2 | 5 | 17-21 | 5/5 | 5/5 | 0.000549 | 5/5 | 1/5 | 1/0/4 (5) | 0/0/5 | 0/3/2 | +0.03 | +0.07 | +0.02 / +0.15 |
| reg3 50 3 | 5 | 51-59 | 5/5 | 5/5 | 0.0533 | 5/5 | 0/5 | 4/1/0 (5) | 4/1/0 | 4/1/0 | -0.86 | -0.39 | -0.70 / +0.01 |
| reg3 50 4 | 5 | 108-149 | 0/5 | 0/5 | nan | 0/0 | 0/0 | 5/0/0 (5) | 5/0/0 | 4/0/1 | -0.89 | -0.78 | -1.22 / -0.13 |
| reg3 200 1 | 1 | 5-5 | 1/1 | 1/1 | 4.72e-05 | 1/1 | 0/1 | 0/1/0 (1) | 0/1/0 | 0/1/0 | +0.00 | +0.00 | +0.00 / +0.00 |
| reg3 200 2 | 4 | 17-21 | 4/4 | 4/4 | 0.00037 | 4/4 | 0/4 | 3/0/1 (4) | 0/0/4 | 0/2/2 | -0.16 | +0.08 | +0.03 / +0.13 |
| reg3 200 3 | 5 | 56-60 | 5/5 | 5/5 | 0.0584 | 5/5 | 0/5 | 4/0/1 (5) | 4/0/1 | 2/3/0 | -0.20 | -0.04 | -0.10 / +0.03 |
| reg3 200 4 | 5 | 124-150 | 4/5 | 4/5 | 1.02 | 4/4 | 0/4 | 5/0/0 (5) | 5/0/0 | 3/1/1 | -1.18 | -0.85 | -1.78 / -0.20 |
| reg3 1000 2 | 5 | 20-26 | 5/5 | 5/5 | 0.000579 | 5/5 | 0/5 | 3/1/1 (5) | 0/0/5 | 0/3/2 | -0.29 | +0.09 | +0.01 / +0.24 |
| reg3 1000 3 | 5 | 53-59 | 5/5 | 5/5 | 0.069 | 5/5 | 2/5 | 5/0/0 (5) | 2/1/2 | 3/1/1 | -0.72 | -0.03 | -0.20 / +0.09 |
| reg3 1000 4 | 5 | 132-175 | 5/5 | 5/5 | 0.0738 | 5/5 | 0/5 | 5/0/0 (5) | 5/0/0 | 5/0/0 | -1.02 | -0.40 | -1.11 / -0.09 |
