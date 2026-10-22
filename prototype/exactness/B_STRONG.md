# Stress test of the B headline against stronger baselines (submitted Oct 20 ~15:30; results PENDING)

Headline under test (B_FAIR.md): ours60 vs cotengra HyperOptimizer(60 s, 1 proc, flops, slicing only if greedy W>30), cuTensorNet on B200:
20/2/2 on 24 nets, median 5.05x, 26.8x when cotengra slices. (Final fair job 46879308 finished: 328 rows, tables in res/B_fair_tables.md.)

## What runs (HPG /blue/fairbanksj/itaykadosh/gpu/exactness = $E; code mirrored in tn-work/exactness/{src,hpg})
1. 46901243 (array 0-5, itay-strongctg, qos fairbanksj-b, 16 cpus + 24 G each, hpg-default/turin/milan, 8 h): src/costStrong.py.
   Per net (the 37 with trees), HyperOptimizer(parallel=16, slicing_reconf_opts={'target_size':2^30} ALWAYS, max_repeats 1e6), then
   slice_and_reconfigure if still W>30, ftn.refine if unsliced (as costB). Configs: sf60 (flops), sc60 (combo), sc256_60 (combo-256),
   sr60 (ReusableHyperOptimizer minimize=combo, in-memory cache = the docs' memory-capped recipe), sc600, sc256_600, sf600.
   Re-costed by cotengra (contraction_cost log10, contraction_width, multiplicity). Output $E/resBS/trees/{<net>.json,strong.tsv}
   (json = ours60 + ctg_hyper60 + strong trees, in contract_cutn.py layout). NOTE: with slicing_reconf on, cotengra slices even
   when W << 30 (qc_maxcut_n28: 2 slices at W 21); this is cotengra's behaviour, kept and visible in the nslices column.
2. cuTensorNet path finder. Baseline job 46872173/46873980/1 still RUNNING at 4 h (partial harvest done, see below). Its rows are
   UNSLICED (cap 0), only 28/38 B nets, no trees saved, so they cannot be executed and are not the 2^30 setting.
   => 46901433 (itay-exact, B200, 32 cpus, 12 h): src/cutn_trees.py = cuTensorNet contract_path with memory_limit=2^30*8 B*memf
   (memf 1,2,4), threads=32, hyper-sample ladder to ~60 s (final S = S*60/dt), keepalive thread (HPG idle-GPU policy); best cap-met
   tree -> tag cutn60 in $E/resBS/cutntrees/<net>.json (all candidates in cutn_paths.tsv); then contract_cutn.py (--own 0) executes
   cutn60 + ours60 + ctg_hyper60 -> $E/resBS/gpu_cutn_strong1.tsv.
3. 46901434 (afterany:46901243): contract_cutn.py on resBS/trees (all strong configs + ours60 + ctg_hyper60) -> resBS/gpu_cutn_strong2.tsv.
   "Best strong cotengra" reported two ways: min predicted C, and ORACLE min measured time over the 7 configs (generous to cotengra).

## Harvest (when jobs finish)
    E=/blue/fairbanksj/itaykadosh/gpu/exactness
    ssh hpg "squeue -u itaykadosh -n itay-strongctg,itay-exact; tail -3 $E/logs/BStrong1-46901433.out $E/logs/BStrong2-46901434.out; cat $E/resBS/trees/strong.tsv | wc -l"
    rsync -a hpg:$E/resBS/gpu_cutn_strong1.tsv hpg:$E/resBS/gpu_cutn_strong2.tsv hpg:$E/resBS/trees/strong.tsv hpg:$E/resBS/cutntrees/cutn_paths.tsv tn-work/exactness/res/
    cd tn-work/exactness && python3 -I src/analyzeStrong.py res/gpu_cutn_strong1.tsv res/gpu_cutn_strong2.tsv > res/B_strong_tables.md
(partial files are fine; job 3 can be started by hand before the array ends: `cd $E; sbatch hpg/strong_gpu2.sbatch`, it is resumable. If a cpu array
task dies/times out, `sbatch --array=<k> hpg/strong_cpu.sbatch` resumes: finished configs are skipped.)
Table output: per comparator (ctg_old, cutn60, ctg_strong_bestC, ctg_strong_bestT, ctg_strong60_bestT) x class (a both unsliced,
b baseline sliced, c other): n, W/T/L (win >1.05, loss <0.95), median, geomean of t_cmp/t_ours, plus a per-net row table.
Predicted-cost view without GPU: join strong.tsv (log10C, W, nslices per config) with trees.tsv (ours60) in res/.

## Already available: cuTensorNet baseline (partial harvest, cap 0 = unsliced, T=100 s single thread, 22 B nets), predicted log10 C
(baselines/harvest, baselines/harvest.py --fetch --local harvest; 1845 ok rows -> 647 budget rows)
cutn100 minus ours60: median +0.50 dex, mean +0.65, min -0.20, max +2.04 (cutn worse on 20/22; better only lm_first_last (-0.20));
equal on qc_qft_28, qc_variational_29. cutn100 vs ctg_hyper60: median -0.27 (cutn mostly better where ctg60 sliced/blew up:
Promedas_33, arjun_102, maxcut n27/n29, google53_27 have ctg60 C 10^35-66 vs cutn 10^11-13). cutn widths up to 34 (unsliced).
Caveat: cutn rows ignore the 2^30 cap (7 had fallback_sliced), so executed numbers need the new job.

## Network selection (NOTES.md "Experiment B design"; queueB.txt, 38 nets)
Chosen as: einsum-benchmark qc/lm/str/rnd/gm/mc/wmc networks whose PUBLISHED best flops lie in 10^9.4-10^13.3 (i.e. executable on one GPU),
plus quimb Sycamore-54 m12/m14, rqc 7x7 m14/m16, QAOA zz/regular/amp (hand-picked families). Then the GPU job executes a net only if its
cheapest tree has C <= 10^13.5 and each tree only if C <= 10^13.5. Possible bias toward ours:
 - The cost window comes from the einsum benchmark's published flops (a cotengra-style searcher), so it favours networks where cotengra is
   already reasonable, which if anything hurts ours. But it excludes the hard networks (published > 10^13.3) where a stronger orderer would win
   the most and where sliced/unsliced behaviour differs; headline says nothing there.
 - Headline "24 networks" = those where BOTH ours60 and ctg_hyper60 executed. 14 of 38 dropped: ctg60 trees above 10^13.5 (Promedas_33 10^35.7,
   arjun_102 10^38.9, maxcut n27 10^66, ...) cannot be executed (exact list of the 14: networks without both rows in res/gpu_cutn_fair.tsv). Dropping them removes the cases where ours wins by
   orders of magnitude, so this is conservative for the speedup medians (but it also means the headline is conditional on cotengra finding an
   executable tree).
 - Nothing indicates networks were removed after
   seeing ours results, but queueB.txt was not recorded with a selection script: I could not verify the history. Recommend stating the rule
   (published flops window + families) in the paper and giving a second, rule-based sample (e.g. all einsum networks n<=800 with published flops
   in the window, not hand-picked families) to rule out family cherry-picking. (Experiment A's 181-network set is selected by n<=800, not by outcome.)
 - 6 of the 24 are class (a) where the claim is parity; the 5x median is driven by 16 class (b) nets where cotengra's 60 s single-process search
   was handicapped by slicing (lm_* 1900 slices of width 14-18). The stronger-cotengra runs above test exactly that.
