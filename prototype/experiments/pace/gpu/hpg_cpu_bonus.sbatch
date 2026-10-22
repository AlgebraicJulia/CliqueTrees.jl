#!/bin/bash
#SBATCH --job-name=itay-ctb
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-milan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/pace/logs/ctb-%j.out
# CPU pipeline (SafeRules+SafeSeparators + CPU septw per atom, run_pace_cpu.jl) on the 100 PACE 2017 BONUS instances,
# 1800 s each, 1 thread each, 32 instances in parallel. Per-instance tsv in $OUT/parts/, merged into cpu_bonus_1800.tsv.
# Results: $P/results_hpg/cpu_bonus_1800.tsv (col known_ub, flag), tds in td_cpu_bonus_1800/ (validated).
set -u
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
P=/blue/fairbanksj/itaykadosh/gpu/pace
export MTENV=/blue/fairbanksj/itaykadosh/gpu/ct-mt/envs/mt-65e72a3
source /blue/fairbanksj/itaykadosh/gpu/ct-mt/common.sh
export PACE_DATA=$P/data/plain PACE_VALIDATE=$P/tools/src/td-validate/td-validate PACE_SET=bonus SEPTW_NT=1
export PACE_TD_DIR=$P/results_hpg/td_cpu_bonus_1800 P OUT=$P/results_hpg/cpu_bonus_parts
mkdir -p $OUT $P/logs $PACE_TD_DIR
ls $P/data/plain/bonus | sed 's/\.gr$//' | sort > $OUT/list.txt
echo "node $(hostname) start $(date)"
run1() { n=$1; echo "$n" > $OUT/$n.list
  julia --project=$MTENV -t 1 $P/gpu/run_pace_cpu.jl $OUT/$n.tsv 1800 @$OUT/$n.list > $OUT/$n.log 2>&1; }
export -f run1
cat $OUT/list.txt | xargs -P 32 -I{} bash -c 'run1 "$1"' _ {}
f=$P/results_hpg/cpu_bonus_1800.tsv
head -1 $(ls $OUT/*.tsv | head -1) > $f; for t in $OUT/*.tsv; do tail -n +2 $t; done >> $f
echo "all done $(date)"
