#!/bin/bash
#SBATCH --job-name=itay-gpu
#SBATCH --dependency=singleton
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-rtx6000 --gpus=rtx_pro_6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/pace/logs/gpu-%j.out
# GPU PID-BT on all 200 PACE 2017 exact instances, first 30 s per instance, then 1800 s per instance.
# (to use a B200 instead: sbatch --partition=hpg-b200 --gpus=b200:1 hpg_gpu.sbatch)
# Results: $P/results_hpg/gpu_30.tsv, gpu_1800.tsv; td files in $P/results_hpg/td_{30,1800}/ (validated with td-validate).
set -u
source /blue/fairbanksj/itaykadosh/gpu/gpupid/common.sh   # JULIA_DEPOT_PATH, julia module
P=/blue/fairbanksj/itaykadosh/gpu/pace
export GPUPID_SRC=$P/gpupid_src PACE_DATA=$P/data/plain PACE_VALIDATE=$P/tools/src/td-validate/td-validate
OUT=$P/results_hpg; mkdir -p $OUT $P/logs
seq -f "ex%03g" 1 200 > $OUT/list_all.txt
nvidia-smi --query-gpu=name,memory.total --format=csv
for T in 30 1800; do
  tsv=$OUT/gpu_$T.tsv; export PACE_TD_DIR=$OUT/td_$T
  for attempt in 1 2 3 4 5 6 7 8; do
    julia --project=/blue/fairbanksj/itaykadosh/gpu/gpupid/env -t 2 $P/gpu/run_pace_gpu.jl $tsv $T @$OUT/list_all.txt
    rc=$?; echo "T=$T attempt $attempt exit $rc"; [ $rc -eq 0 ] && break
    if [ $rc -ne 3 ]; then   # hard crash (segfault etc.): no row was written, record the instance as crashed so it is not retried forever
      nxt=$(comm -23 <(sort $OUT/list_all.txt) <(cut -f1 $tsv | sort) | head -1)
      [ -n "$nxt" ] && printf "%s\t0\t0\t0\t0\t-1\t0\tcrash\t%d\t-\texit code %d\n" $nxt $T $rc >> $tsv
    fi
  done
done
echo "done $(date)"
