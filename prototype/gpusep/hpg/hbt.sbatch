#!/bin/bash
#SBATCH --job-name=itay-gsep-hbt
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-b200 --gpus=b200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/gpusep/logs/hbt-%j.out
# usage: sbatch hbt.sbatch <names file> <seconds> <tag>
# Per network, two processes at once on disjoint cores (8 threads each): GPU-escalated HBT
# (esc-septw, then esc-pid) and CPU-only HBT (cpu).
set -u
source /blue/fairbanksj/itaykadosh/gpu/gpusep/common.sh
cd $GS
LIST=$1; T=$2; TAG=$3
nvidia-smi --query-gpu=name,memory.total --format=csv
cores=($(taskset -cp $$ | awk -F': ' '{print $2}' | tr ',' '\n' | awk -F- '{if(NF==2){for(i=$1;i<=$2;i++)print i}else print $1}'))
A=$(IFS=,; echo "${cores[*]:0:8}"); B=$(IFS=,; echo "${cores[*]:8:8}")
for name in $(cat $LIST); do
  taskset -c $A julia --project=$GS/env -t 8 $GS/scripts/bench_hbt.jl $name $T $GS/results/hbt_${TAG}.tsv esc-septw,esc-pid >> $GS/logs/hbt-$SLURM_JOB_ID-gpu.log 2>&1 &
  taskset -c $B julia --project=$GS/env -t 8 $GS/scripts/bench_hbt.jl $name $T $GS/results/hbt_${TAG}.tsv cpu >> $GS/logs/hbt-$SLURM_JOB_ID-cpu.log 2>&1 &
  wait
done
echo "done $(date)"
