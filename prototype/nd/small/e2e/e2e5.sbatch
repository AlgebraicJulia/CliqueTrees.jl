#!/bin/bash
#SBATCH --job-name=itay-gpus
#SBATCH --account=fairbanksj
#SBATCH --qos=fairbanksj
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=42
#SBATCH --mem=160gb
#SBATCH --time=02:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/rounds/small/logs/%x-%j.out
#SBATCH --error=/blue/fairbanksj/itaykadosh/gpu/rounds/small/logs/%x-%j.err
# v5 of the end-to-end runs (SMALL_NOTES.md "End to end"): defaults = block shape from batch_cfg, device-side growth, hash twins;
# vs the v2 host path (code3/old), on the 15 ND leaf sets, the sweep sets (sizes 1k/2k/10k/50k x graph counts), and the
# single-graph call (30k-100k). Phase A: GPU alone (8 cores, -t8). Phase B: Richard's serial MF1 at 1/8/32 threads on 32
# other cores, while the GPU repeats phase A (second sample, REP=b) so that it stays busy.
set -u
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
V=$R/small
E=$V/e2e
mkdir -p $V/logs $E/results $E/sets
cd $E/code5
export CORPUS=/blue/fairbanksj/itaykadosh/gpu/tncorpus
export DATA=$R/v2/data:/blue/fairbanksj/itaykadosh/gpu/fill/data
export LEAVES=$V/leaves SETS=$E/sets
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv
lscpu | grep -E "Model name|^CPU\(s\)"
# the job's cores: the first 8 for the GPU process, the next 32 for the serial baselines
ALL=$(python3 -c "
import os; print(','.join(map(str, sorted(os.sched_getaffinity(0)))))")
GC8=$(echo $ALL | cut -d, -f1-8); CC32=$(echo $ALL | cut -d, -f9-40)
echo "cores: gpu=$GC8 cpu=$CC32"
NDG=ecology2,thermal2,G3_circuit,lat2_724,audikw_1
LTAGS=$(for g in ${NDG//,/ }; do for l in 50000 10000 2000; do echo -n "${g}_L$l,"; done; done)
export LTAGS
export SINGLE= CFGS=v5 SETSREADY=1
STOP=$E/results/stop_$SLURM_JOB_ID
OUT=$E/results/e2e_gpu_v5.tsv REP=a taskset -c $GC8 julia --project=$R/env -t8 e2e_gpu.jl
echo "PHASE A DONE $(date)"
OUT=$E/results/e2e_gpu_v5.tsv REP=b SKIPOLD=1 STOPFILE=$STOP taskset -c $GC8 julia --project=$R/env -t8 e2e_gpu.jl > $E/results/keeper_$SLURM_JOB_ID.log 2>&1 &
KP=$!
OUT=$E/results/e2e_cpu_v5.tsv NTH=1,8,32 taskset -c $CC32 julia --project=$R/rr/renv -t32 e2e_cpu.jl
touch $STOP
wait $KP
echo DONE
