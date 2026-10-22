# Cost-aware selection among (near-)minimum-width TDs (Oct 20)

Goal: the exactness study (tn-work/exactness) showed that width predicts C only weakly near the optimum and that
width-optimal TDs spread by a median 0.08-0.15 log10 C. Method here: generate many TDs of width <= K (K = certified
optimum, else best known) and K+1, K+2, and select the cheapest by exact contraction cost.

HPG dir: `/blue/fairbanksj/itaykadosh/gpu/costsel` (= `$C`). Local copy of code: `tn-work/costsel/{src,hpg}`.
Study reference copied to `tn-work/costsel/ref/` (resA/cost, resA/td, resB/td, resB trees + GPU timings).

## Method (src/CostSel.jl, src/pmccost.jl, src/gen.jl)
1. Front end of HybridTW (safe rules, safe separators -> atoms). The width of the graph is the max over atoms, so
   every atom may take ANY TD of width <= K. (The pipeline keeps the first ordering that meets the bound: on
   non-critical atoms that is usually a greedy ordering, chosen without regard to cost.)
2. Per atom, a pool of candidate minimal triangulations (all minimalized with MinimalChordal):
   - the study's TDs of the network restricted to the atom (exact solver + 7 relabeled exact solves, MF/MF1/AMF/
     MMD/MCS/bw, random tie-break MF/MF1, METIS-ND, HBT 0.1/1/10 s x 3 seeds; resp. the B ladder). The restriction
     maps atom vertices to original vertices through the twin classes of the rules and the classes of the safe
     separators (`atommaps`, `restrict`); the exact TD restricts to width K on every atom;
   - MF/MF1/AMF/MMD and MF/MF1 on 8 random relabelings;
   - septw at width K on relabeled copies (the exact solver's random tie-breaking; 2-4 solves);
   - HBT 2 s x 4 seeds on atoms (n >= 16) with < 8 candidates of width <= K; RTW fallback.
3. PMC pool = union of the maximal cliques of all candidates (they are PMCs of the atom). When the minimal
   separators lighter than K+d can be listed (sepenum, <= 20k, 10 s) all PMCs from them are added (pmcenum).
4. `pmccost` (new, the in-solver cost-aware tie-break): the Bouchitte-Todinca/pmcdp DP with (min, +) over
   bag costs 2^w(bag) instead of (min, max), subject to bags <= K+d. On the full PMC list it is the exactly
   cheapest (by lcost = log2 sum 2^bag) width-<=K+d minimal triangulation; on the pool it recombines the
   candidates' pieces and is never worse (in lcost) than any of them. With random multiplicative noise
   (sigma 0.7) on the PMC costs it yields many distinct cheap TDs.
   Also committed on local branch `itay/costsel` (worktree `CliqueTrees.jl-costsel`, commit 20474b5, src/bt/pmccost.jl,
   not pushed); the experiments evaluate the same file into CliqueTrees at runtime.
5. Whole-graph TDs = one TD per atom, assembled by safe_pull/rule_pull. Modes per network:
   `ref`: the study's own whole-graph TDs unchanged (delta = 0/1/2 by width; t = their seconds);
   `uni` (width K): each atom a uniformly random raw candidate -> best-of-N selection of unguided opt-width TDs;
   `dp` (width <= K+d, d = 0, 1, 2): the DP's TD; `dpn`: DP + noise, N up to 1000 per d. Each TD: width, lcost,
   br cost (branchtree + branchrefine, Julia branchstats = study stage "br"), bucket C (Host.cost!, = GPU kernel).
   Selection (SEL lines): best of the first N (N = 1, 3, 10, ..., 1000, all) per group, by br and by bucket C.
6. GPU (src/score.jl, gpucost kernels): exact bucket C of every TD (check vs CPU), the width-capped GPU mass search
   (insertion moves from the 64 cheapest width-K TDs; a candidate is kept only if maxop <= K, and maxop of the
   bucket tree of a PEO = its TD width, verified on every TD), 10 s; GPU polish (reconf K=10, ILS, K=12; 10 s) of
   the br tree of the br-selected TD, the bucket tree of the GPU-selected TD, and the mass-search winner.
7. cotengra recost (src/recost.py): every SEL'd TD at stages bucket / br / br+ctg (as costA.py), mass-search winner
   (bucket, bucket+ctg), polished trees (raw, +ctg).
8. Real contraction (src/mkpklB.py + exactness contractB.py): for the 24 B networks where the new tree differs from
   ours60 by > 0.01 log10 C: csel (br+ctg of the selected TD) and csel_gpu (best polished +ctg), sliced to 2^30 like
   costB, executed on one B200 next to ours60 / ctg_hyper60 / td_exact re-timed in the same job.

Local checks (laptop, RTX 5060): GPU C = CPU C to 3e-6 bits; GPU maxop = TD width to 4e-6 bits; recost works.
Small examples (log2 C, br stage): mc_2021_075 (K=29) width-K pool best 31.13 vs exact solver TD 31.98; width K+1
DP 30.69; GPU polish -> 9.215 log10 (br+ctg; study best TD-tree 9.237, hyper10 13.5). qaoazz_N30_p3_s0: 14.50 vs
exact 14.61; polished 4.266 log10 vs study best 4.365.

## Jobs (final submission Oct 20 ~14:30 EDT, all named itay-csel)
| job | what |
|---|---|
| 46882800-03 | gen: 4 x 8 cpus (hpg-default/turin/milan, burst QOS fairbanksj-b), 32 workers on queue.txt (24 B nets first, then the rest of the 181 A nets; 186 total), 8 h cap -> pool/<net>.pool |
| 46882814 | gpu: 1 B200, score.jl (afterany gen), 4 h cap -> gpu/<net>.gpu |
| 46882815 | recost: cotengra re-costing (afterany gpu) -> rc/<net>.tsv, then mkpklB -> pklB/ |
| 46882816 | contract: 1 B200, contractB.py on pklB (afterany recost) -> res_contract.tsv |
Earlier attempts (prep 46878833 = genv precompile, OK; 46878834-36, 46879079-84, 46879113, 46881955-61) were
cancelled for code changes (study-TD seeding of the atoms, ref mode). The B200 env works (46881959 scored 2 nets
before it was cancelled).
Resume: every stage skips finished networks (gen: claims dir -> `rm -rf $C/pool/claims` first; score: existing .gpu;
recost: rc with #END). Resubmit with `sbatch hpg/<x>.sbatch` from $C.

## Analyze (when the jobs are done; partial results work)
    C=/blue/fairbanksj/itaykadosh/gpu/costsel; E=/blue/fairbanksj/itaykadosh/gpu/exactness
    ssh hpg "cd $C && $E/venv/bin/python src/analyze.py $E/resA/cost $E/resA/td pool gpu rc $E/resB/trees/trees.tsv tables.md - res_contract.tsv"
    rsync -a hpg:$C/tables.md tn-work/costsel/res/
    # section 8 = real contraction (csel/csel_gpu vs ours60/ctg_hyper60, timed in the same job)
tables.md: 1a/1b best-of-N vs exact TD / best study heuristic / best study opt TD / cotengra hyper10+ctg (stage br+ctg,
selection by br; stage bucket, selection by GPU C); 2 GPU mass search + polish; 3 spread recovered; 4 per family;
5 the 24 B networks vs hyper60; 6 time to quality; 7 per network.
Caveat for 1/3: the atom pools contain the study's TDs, so "vs exact" and "vs best heuristic" measure what selection
adds on top of them; the time of those study TDs is added in section 6 (t_ref). Section 3 also has
"study pool alone, selected by br" (= the study's own cost-aware pick) for comparison.

## Early signal (first 12 pools on HPG, br stage = Julia branchstats, log2 C; not yet recosted)
exact solver TD -> best width-K TD selected by br -> best over widths K..K+2:
gm_pedigree50 34.42 -> 34.38; lm_brackets_4_16d 37.95 -> 37.48; lm_first_last_4_16c 42.08 -> 41.69 (= best study TD);
qc_maxcut_n28 29.82 -> 29.80 -> 29.78; wmc_2021_187 35.29 -> 34.86 -> 34.58; str_nw_mera_closed_120 34.80 -> 34.79 -> 34.42;
qc_variational_29 35.00 -> 35.00 -> 32.59 (a width K+1/K+2 TD is 0.73 log10 cheaper); networks fully solved by
the safe rules (qc_qft_28, supremacy2d) have one TD per run, so only the study's relabeled exact TDs and the GPU
mass search add choice there.

## Results
(fill in from tables.md)
