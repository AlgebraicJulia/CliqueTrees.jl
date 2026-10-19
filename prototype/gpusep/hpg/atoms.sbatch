#!/bin/bash
#SBATCH --job-name=itay-gsep-atoms
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-b200 --gpus=b200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/gpusep/logs/atoms-%j.out
# usage: sbatch atoms.sbatch <names file> <seconds> <tag>
# Two processes on disjoint cores of the same node (same CPU for both sides, the GPU kept busy):
#   cores 0-7:  esc-septw,esc-pid,esc-pid:2(GPUSeptw, CPU first then GPU), 8 tasks
#   cores 8-15: cpu,rtw            (CPU septw and RTW, CPUSeptw), 8 tasks
set -u
source /blue/fairbanksj/itaykadosh/gpu/gpusep/common.sh
cd $GS
LIST=$1; T=$2; TAG=$3
nvidia-smi --query-gpu=name,memory.total --format=csv
cores=($(taskset -cp $$ | awk -F': ' '{print $2}' | tr ',' '\n' | awk -F- '{if(NF==2){for(i=$1;i<=$2;i++)print i}else print $1}'))
A=$(IFS=,; echo "${cores[*]:0:8}"); B=$(IFS=,; echo "${cores[*]:8:8}")
taskset -c $A julia --project=$GS/env -t 8 $GS/scripts/bench_atoms.jl $GS/atoms $LIST $T $GS/results/atoms_${TAG}_gpu.tsv esc-septw,esc-pid,esc-pid:2 > $GS/logs/atoms-$SLURM_JOB_ID-gpu.log 2>&1 &
taskset -c $B julia --project=$GS/env -t 8 $GS/scripts/bench_atoms.jl $GS/atoms $LIST $T $GS/results/atoms_${TAG}_cpu.tsv cpu,rtw > $GS/logs/atoms-$SLURM_JOB_ID-cpu.log 2>&1 &
wait
echo "done $(date)"
