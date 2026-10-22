# Definitive einsum benchmark on HPG (submitted 2026-10-19)

Jobs (all `itay-ein`, no dependency, partition hpg-milan, 64 cores, 12 h limit), outdir `/blue/fairbanksj/itaykadosh/gpu/einsumbench/run1`:
- 46743040  cfg c1 (equal single core), kinds both: 36 ours workers (julia -t1) + 28 cotengra workers (parallel=False). Est. 2-3 h.
- 46743041  cfg c8 (equal 8 cores), ours: 8 workers x 8 cores (hbt/hbtnd threads=8, TreeSA ntrials=8). Est. ~6.5 h.
- 46743042  cfg c8, cotengra: 8 workers x 8 cores (hyper/hyper-sa/auto/auto-hq parallel=8). Est. ~5 h.
Logs: `logs/ein-<jobid>.out`, `run1/logs/worker_<cfg>_<tag>.log`. Results: `run1/res/<kind>_<cfg>_<tag>.tsv`
(columns in worker.py header: network method budget seed cfg status wall search post logC width maxop pf_logC pf_width greedy_s extra).
Queues are copied per job to `run1/queue_<kind>_<cfg>.<jobid>`; after a kill, the remaining lines are what is left to run
(copy to `tasks_<kind>.txt` and resubmit the same command: `sbatch einsum.sbatch <outdir> <cfg> <both|ours|ctg>`).

## Analyze
    rsync -a hpg:/blue/fairbanksj/itaykadosh/gpu/einsumbench/run1 .       # partial results are fine
    python analyze.py run1 run1/plots      # needs matplotlib; run with any python (tncorpus venv has it)
-> `plots/{c1,c8}_{pure,portfolio}/{profiles.png,curves.png,summary.tsv}`. x-axis = total wall (search+post); summary.tsv has
median search_s / post_s columns. pure = method's own tree; portfolio = min(tree, cotengra greedy), wall + greedy_s.
cotengra budget overrun is in `extra` (overrun=wall-T); cap 2T+30 s (killed = timeout = failure).

## What changed vs the draft harness
- ordering->tree conversion uses `greedy` for bucket binarization (was auto-hq: 4-7 s on mc_2021_074, qc_qft_27, mc_2023_arjun_133);
  refinement is chunked with its own cap max(0.1 T, 0.05) s. Caveat: on qc_qft_27 auto-hq conversion did cotengra's own search (logC 8.98 vs 9.61).
- Both modes (pure / portfolio) come from the same run. Remaining wall above T is Julia-side (line graph, nd8 / SafeRules preprocessing, e.g. 5 s on gm_pedigree50, 36 s on qc_maxcut_n18_r17_p100 at T=1) and is in search_s.
- Env: julia `ctgplugin/jlenv-hpg` (mt-65e72a3 + JSON, OMEinsumContractionOrders, Metis); python `frontier/venv` (cotengra 0.8.2, kahypar, optuna).
