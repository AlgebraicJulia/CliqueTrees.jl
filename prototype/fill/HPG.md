# Large-scale fill on HPG (GPU round-based MF/MD vs serial orderings)
Scripts: tn-work/fill/hpg/ (code/, *.sbatch, instances.sh); synced to /blue/fairbanksj/itaykadosh/gpu/fill/. Env: fill/env (rounds env + AMD).
Submitted Oct 19 2026:
- 46702560 itay-dl   (hpg-milan, 4 cpu): download SuiteSparse from www.cise.ufl.edu mirror (sparse.tamu.edu is 403) + convert to data/*.bin; lattices generated.
- 46702561 itay-gpu  (hpg-b200, afterok dl, singleton): GPU-MF/GPU-MD timing (min of 3, 2 if n>4M) + perms in data/perms/*.perm (Int32). 3h limit.
- 46702562 itay-ct   (hpg-milan exclusive, afterok dl, singleton): AMD AMF MMD MF MF1 METIS-ND, one julia proc per (instance,method), 10 parallel, 1800s kill cap; 8h limit. Fill evaluated in-process.
- 46702563 itay-ev   (hpg-milan, afterok gpu): nnz(L)/flops/width of the GPU perms.
Instances: lat3_25,40,64,100,140; lat2_512,1024,2048 (n up to ~8.4M); ecology2, thermal2, G3_circuit, audikw_1, StocF-1465, Flan_1565, Serena, af_shell10, Geo_1438, Hook_1498, Bump_2911, Queen_4147.
Outputs (results/): results.tsv (name n m meth nnzL flops tw time, same columns as local fill/results.tsv), gpu_times.tsv, status.tsv (timeouts/errors/download failures). Logs: logs/itay-*-<id>.out. Timeouts = missing row for that method.
Aggregate:
  rsync -a hpg:/blue/fairbanksj/itaykadosh/gpu/fill/results/ ~/Projects/GATAS/tn-work/fill/hpg/results/
  cd ~/Projects/GATAS/tn-work/fill && cp hpg/results/results.tsv results_hpg.tsv && <adapt agg.jl to read results_hpg.tsv>
Note: if GPU job dies mid-way, rerun itay-gpu (skips existing perms), then ev.sbatch (rows may duplicate; dedupe by name+meth).
