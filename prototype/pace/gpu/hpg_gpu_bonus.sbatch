#!/bin/bash
#SBATCH --job-name=itay-gpub
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-b200 --gpus=b200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/pace/logs/gpub-%j.out
# GPU PID-BT pipeline on the 100 PACE 2017 BONUS instances, 1800 s per instance.
# (rtx6000 instead: sbatch --partition=hpg-rtx6000 --gpus=rtx_pro_6000:1 hpg_gpu_bonus.sbatch)
# Results: $P/results_hpg/gpu_bonus_1800.tsv (col known_ub, flag), tds in $P/results_hpg/td_bonus_1800/ (validated).
set -u
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
source /blue/fairbanksj/itaykadosh/gpu/gpupid/common.sh
P=/blue/fairbanksj/itaykadosh/gpu/pace
export GPUPID_SRC=$P/gpupid_src PACE_DATA=$P/data/plain PACE_VALIDATE=$P/tools/src/td-validate/td-validate PACE_SET=bonus
OUT=$P/results_hpg; mkdir -p $OUT $P/logs
ls $P/data/plain/bonus | sed 's/\.gr$//' | sort > $OUT/list_bonus.txt
nvidia-smi --query-gpu=name,memory.total --format=csv
T=1800; tsv=$OUT/gpu_bonus_$T.tsv; export PACE_TD_DIR=$OUT/td_bonus_$T
for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
  julia --project=/blue/fairbanksj/itaykadosh/gpu/gpupid/env -t 2 $P/gpu/run_pace_gpu.jl $tsv $T @$OUT/list_bonus.txt
  rc=$?; echo "attempt $attempt exit $rc"; [ $rc -eq 0 ] && break
  if [ $rc -ne 3 ]; then
    nxt=$(comm -23 <(sort $OUT/list_bonus.txt) <(cut -f1 $tsv | sort) | head -1)
    [ -n "$nxt" ] && printf "%s\t0\t0\t0\t0\t-1\t0\tcrash\t%d\t-\texit code %d\t-\n" $nxt $T $rc >> $tsv
  fi
done
echo "done $(date)"
